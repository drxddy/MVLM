/*
 * MVLM - RMSNorm Kernel for Adreno GPUs
 * Phase 2: Transformer Primitives
 *
 * Implements Root Mean Square Layer Normalization:
 *   y = x * rsqrt(mean(x^2) + eps) * weight
 *
 * Used in modern transformer architectures (LLaMA, Phi, etc.) instead of
 * standard LayerNorm because it skips the mean-subtraction step.
 *
 * Adreno optimizations:
 *   - Subgroup reductions (cl_khr_subgroups) to avoid local memory barriers
 *   - Local memory fallback for devices without subgroup support
 *   - __constant memory for norm weights (small, read-only, broadcast)
 *   - int/uint indexing (saves 2 regs per variable vs size_t)
 *   - native_rsqrt for fast reciprocal square root
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Workgroup size for the reduction. Must match host-side local_work_size. */
#ifndef NORM_WG_SIZE
#define NORM_WG_SIZE 256
#endif

/* ============================================================================
 * RMSNorm: y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 *
 * Dispatch: one workgroup per row (each row is `hidden_size` elements).
 *   global_work_size  = { num_rows * NORM_WG_SIZE }
 *   local_work_size   = { NORM_WG_SIZE }
 *
 * Two-pass within a single kernel launch:
 *   Pass 1: Compute sum of squares across hidden_size (parallel reduction)
 *   Pass 2: Normalize each element using the computed RMS
 * ========================================================================= */

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void rms_norm(
    __global const half* restrict input,     // [num_rows, hidden_size]
    __global half* restrict output,          // [num_rows, hidden_size]
    __constant const half* restrict weight,  // [hidden_size]
    const int hidden_size,
    const float eps)
{
    const int lid = get_local_id(0);
    const int row = get_group_id(0);
    const int row_offset = mul24(row, hidden_size);

    // --- Pass 1: Compute sum of squares ---
    // Each work-item accumulates over a strided portion of the row
    float sum_sq = 0.0f;
    for (int i = lid; i < hidden_size; i += NORM_WG_SIZE) {
        const float val = (float)input[row_offset + i];
        sum_sq = fma(val, val, sum_sq);
    }

    // Subgroup reduction: hardware-accelerated warp/wave-level reduce
    // Avoids local memory barrier — faster on Adreno with cl_khr_subgroups
    sum_sq = sub_group_reduce_add(sum_sq);

    // Cross-subgroup reduction via local memory (only needed if WG > subgroup)
    const int sub_id = get_sub_group_id();
    const int num_subs = get_num_sub_groups();
    const int sub_lid = get_sub_group_local_id();

    __local float sub_sums[32]; // Max 32 subgroups per workgroup
    if (sub_lid == 0) {
        sub_sums[sub_id] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Final reduction across subgroups (only first subgroup participates)
    if (sub_id == 0 && sub_lid < num_subs) {
        sum_sq = sub_sums[sub_lid];
    } else {
        sum_sq = 0.0f;
    }
    sum_sq = sub_group_reduce_add(sum_sq);

    // Broadcast the final RMS scaling factor to all work-items
    // rms_scale = rsqrt(mean(x^2) + eps)
    const float rms_scale = native_rsqrt(sum_sq / (float)hidden_size + eps);

    // --- Pass 2: Normalize and scale ---
    for (int i = lid; i < hidden_size; i += NORM_WG_SIZE) {
        const float val = (float)input[row_offset + i];
        const float w = (float)weight[i];
        const float normed = val * rms_scale * w;
        output[row_offset + i] = (half)normed;
    }
}

#else
/* ============================================================================
 * Fallback: Local memory reduction for devices without cl_khr_subgroups
 * ========================================================================= */

__kernel void rms_norm(
    __global const half* restrict input,
    __global half* restrict output,
    __constant const half* restrict weight,
    const int hidden_size,
    const float eps)
{
    const int lid = get_local_id(0);
    const int row = get_group_id(0);
    const int row_offset = mul24(row, hidden_size);

    // --- Pass 1: Compute sum of squares via local memory reduction ---
    float sum_sq = 0.0f;
    for (int i = lid; i < hidden_size; i += NORM_WG_SIZE) {
        const float val = (float)input[row_offset + i];
        sum_sq = fma(val, val, sum_sq);
    }

    __local float scratch[NORM_WG_SIZE];
    scratch[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = NORM_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // rms_scale = rsqrt(mean(x^2) + eps)
    const float rms_scale = native_rsqrt(scratch[0] / (float)hidden_size + eps);

    // Barrier to ensure rms_scale is visible before all WIs read scratch[0]
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Pass 2: Normalize and scale ---
    for (int i = lid; i < hidden_size; i += NORM_WG_SIZE) {
        const float val = (float)input[row_offset + i];
        const float w = (float)weight[i];
        const float normed = val * rms_scale * w;
        output[row_offset + i] = (half)normed;
    }
}

#endif /* cl_khr_subgroups */
