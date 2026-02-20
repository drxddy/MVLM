/*
 * MVLM - GEMM/GEMV Kernels for Adreno GPUs
 * Phase 1: GEMM Kernel (Critical Path)
 *
 * Multiple versions with progressive optimization:
 *   v1: Naive GEMM — correctness baseline
 *   v2: Tiled GEMM — workgroup-level tiling with local memory
 *   v3: Image-based GEMM — weights in 2D image for TP/L1 cache hits
 *   v4: GEMV — optimized for M=1 single-token decode (memory-bound)
 *
 * Adreno optimization conventions used throughout:
 *   - int/uint indexing instead of size_t (saves 2 regs per variable, §8.7)
 *   - mad24/mul24 for index arithmetic (native 24-bit multiply HW)
 *   - half (fp16) primary data type (2x ALU throughput on Adreno)
 *   - Branch-free code paths where possible
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* ============================================================================
 * v1: Naive GEMM — C[M,N] = A[M,K] * B[K,N]
 *
 * One work-item per output element. No tiling, no local memory.
 * Purpose: correctness reference, performance baseline.
 * ========================================================================= */
__kernel void gemm_naive(
    __global const half* restrict A,   // [M, K]
    __global const half* restrict B,   // [K, N]
    __global half* restrict C,         // [M, N]
    const int M,
    const int N,
    const int K)
{
    // Use int instead of size_t — saves registers on Adreno (§8.7)
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    // Accumulate in float to preserve precision across K reduction
    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        // mad24: native 24-bit multiply-add, faster than 32-bit mul+add
        const float a_val = vload_half(0, A + mad24(row, K, k));
        const float b_val = vload_half(0, B + mad24(k, N, col));
        acc = fma(a_val, b_val, acc);
    }

    vstore_half(acc, 0, C + mad24(row, N, col));
}

/* ============================================================================
 * v2: Tiled GEMM — workgroup-level tiling with __local memory
 *
 * Each workgroup loads a TILE_M x TILE_K tile of A and TILE_K x TILE_N tile
 * of B into local memory, then each work-item accumulates from the shared
 * tiles. This reuses each loaded element TILE_N (or TILE_M) times, cutting
 * global memory traffic by ~TILE_SIZE x.
 *
 * Tile sizes are compile-time defines so we can auto-tune them per device.
 * ========================================================================= */

#ifndef TILE_M
#define TILE_M 8
#endif
#ifndef TILE_N
#define TILE_N 8
#endif
#ifndef TILE_K
#define TILE_K 8
#endif

__kernel void gemm_tiled(
    __global const half* restrict A,   // [M, K]
    __global const half* restrict B,   // [K, N]
    __global half* restrict C,         // [M, N]
    const int M,
    const int N,
    const int K)
{
    // Local tile indices within the workgroup
    const int local_row = get_local_id(0);   // [0, TILE_M)
    const int local_col = get_local_id(1);   // [0, TILE_N)

    // Global output indices
    const int global_row = mad24((int)get_group_id(0), TILE_M, local_row);
    const int global_col = mad24((int)get_group_id(1), TILE_N, local_col);

    // Shared tiles in local memory (fast on-chip SRAM)
    __local half tile_A[TILE_M][TILE_K];
    __local half tile_B[TILE_K][TILE_N];

    // Accumulate in float for precision across the K-dimension reduction
    float acc = 0.0f;

    // Number of tiles along the K dimension
    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative load: each work-item loads one element of each tile
        const int a_col = mad24(t, TILE_K, local_col);
        const int b_row = mad24(t, TILE_K, local_row);

        // Bounds-checked loads — out-of-bounds reads zero
        if (global_row < M && a_col < K) {
            tile_A[local_row][local_col] = A[mad24(global_row, K, a_col)];
        } else {
            tile_A[local_row][local_col] = (half)0.0h;
        }

        if (b_row < K && global_col < N) {
            tile_B[local_row][local_col] = B[mad24(b_row, N, global_col)];
        } else {
            tile_B[local_row][local_col] = (half)0.0h;
        }

        // Barrier: ensure the entire tile is loaded before computation
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply-accumulate from the shared tiles
        for (int k = 0; k < TILE_K; ++k) {
            acc = fma((float)tile_A[local_row][k],
                      (float)tile_B[k][local_col],
                      acc);
        }

        // Barrier: ensure all work-items are done reading before next tile load
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result — bounds check for edge workgroups
    if (global_row < M && global_col < N) {
        vstore_half(acc, 0, C + mad24(global_row, N, global_col));
    }
}

/* ============================================================================
 * v3: Image-based GEMM — weights stored as 2D image (CL_RGBA, CL_HALF_FLOAT)
 *
 * On Adreno, image reads go through the Texture Processor (TP) → L1 texture
 * cache, which provides hardware prefetch and 2D spatial locality.
 * This typically gives 2-3x faster weight loads vs buffer reads.
 *
 * Weight layout in image: each pixel stores 4 half values (RGBA).
 *   - Image width  = N / 4  (4 output columns packed per pixel)
 *   - Image height = K      (one row per K-dimension element)
 *
 * So B_img(x, y) = { B[y, 4x+0], B[y, 4x+1], B[y, 4x+2], B[y, 4x+3] }
 * ========================================================================= */

__constant sampler_t weight_sampler =
    CLK_NORMALIZED_COORDS_FALSE |   // Integer pixel coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE |     // Clamp OOB reads
    CLK_FILTER_NEAREST;             // No interpolation — exact texel fetch

__kernel void gemm_image(
    __global const half* restrict A,   // [M, K] activations (buffer)
    __read_only image2d_t B_img,       // [K, N] weights as image (N/4 wide, K tall)
    __global half* restrict C,         // [M, N] output
    const int M,
    const int N,
    const int K)
{
    const int row = get_global_id(0);  // output row
    const int col4 = get_global_id(1); // output column / 4 (each WI computes 4 cols)

    if (row >= M) return;

    const int col_base = col4 << 2;  // col4 * 4
    if (col_base >= N) return;

    // Accumulate 4 output columns simultaneously (matches RGBA image width)
    float4 acc = (float4)(0.0f);

    for (int k = 0; k < K; ++k) {
        // Load one activation value (scalar)
        const float a_val = (float)A[mad24(row, K, k)];

        // read_imageh returns half4 → promoted to float4 for accumulation
        // Coordinates: (col4, k) — x is the pixel column, y is the row
        const half4 b_val = read_imageh(B_img, weight_sampler, (int2)(col4, k));

        // fma: fused multiply-add for each of the 4 packed columns
        acc = fma((float4)(a_val), convert_float4(b_val), acc);
    }

    // Write 4 output elements (or fewer at the boundary)
    const int out_offset = mad24(row, N, col_base);
    const int remaining = N - col_base;

    if (remaining >= 4) {
        vstore_half4(acc, 0, C + out_offset);
    } else {
        // Edge case: fewer than 4 columns remaining
        for (int i = 0; i < remaining; ++i) {
            vstore_half(acc[i], 0, C + out_offset + i);
        }
    }
}

/* ============================================================================
 * v4: GEMV — optimized for M=1 single-token decode phase
 *
 * y[1, N] = x[1, K] * W[K, N]
 *
 * This is memory-bound (bandwidth-limited), not compute-bound.
 * Strategy: each workgroup computes one output element. Work-items within
 * the group cooperatively load chunks of the K dimension, do partial
 * dot products, then reduce via local memory.
 *
 * Weights stored as image for TP/L1 cache bandwidth.
 * ========================================================================= */

#ifndef GEMV_WG_SIZE
#define GEMV_WG_SIZE 256
#endif

__kernel void gemv(
    __global const half* restrict x,     // [1, K] input vector
    __read_only image2d_t W_img,         // [K, N] weights as image (N/4 wide, K tall)
    __global half* restrict y,           // [1, N] output vector
    const int N,
    const int K)
{
    // Each workgroup handles 4 adjacent output elements (one image column)
    const int col4 = get_group_id(0);    // which group of 4 output columns
    const int lid = get_local_id(0);     // local work-item ID within group

    const int col_base = col4 << 2;
    if (col_base >= N) return;

    // Each work-item accumulates a partial sum over a stride of the K dimension
    // Stride access pattern: WI 0 does k=0, WG_SIZE, 2*WG_SIZE, ...
    //                       WI 1 does k=1, WG_SIZE+1, 2*WG_SIZE+1, ...
    float4 partial = (float4)(0.0f);

    for (int k = lid; k < K; k += GEMV_WG_SIZE) {
        const float x_val = (float)x[k];
        const half4 w_val = read_imageh(W_img, weight_sampler, (int2)(col4, k));
        partial = fma((float4)(x_val), convert_float4(w_val), partial);
    }

    // Reduction via local memory — sum partial results across all work-items
    __local float4 scratch[GEMV_WG_SIZE];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction: log2(GEMV_WG_SIZE) steps
    for (int stride = GEMV_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Work-item 0 writes the final result
    if (lid == 0) {
        const float4 result = scratch[0];
        const int remaining = N - col_base;

        if (remaining >= 4) {
            vstore_half4(result, 0, y + col_base);
        } else {
            for (int i = 0; i < remaining; ++i) {
                vstore_half(result[i], 0, y + col_base + i);
            }
        }
    }
}
