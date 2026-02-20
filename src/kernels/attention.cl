/*
 * MVLM - Attention Kernels for Adreno GPUs
 * Phase 2: Transformer Primitives
 *
 * Implements multi-head attention for two distinct phases:
 *   - Prefill: batch of tokens, compute full Q @ K^T attention matrix
 *   - Decode: single new token against KV-cache (memory-bound)
 *
 * Attention formula:
 *   scores = Q @ K^T / sqrt(head_dim)
 *   weights = softmax(scores + causal_mask)
 *   output = weights @ V
 *
 * Adreno optimizations:
 *   - int/uint indexing (saves registers vs size_t)
 *   - mad24/mul24 for index calculations
 *   - native_exp / native_recip for softmax
 *   - Local memory for shared tiles and reductions
 *   - Separate kernels for prefill (compute-bound) and decode (memory-bound)
 *   - Branch-free causal masking using fmin (avoids warp divergence)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef ATTN_WG_SIZE
#define ATTN_WG_SIZE 256
#endif

/* ============================================================================
 * Prefill Attention: standard multi-head attention for a batch of tokens
 *
 * Computes the full attention for all positions simultaneously. This is
 * compute-bound (dominated by the Q @ K^T GEMM and score @ V GEMM).
 *
 * Each workgroup handles one (head, query_position) pair.
 * Within the workgroup, work-items cooperatively compute the dot product
 * against all key positions, apply causal mask + softmax, then compute
 * the weighted sum over V.
 *
 * Dispatch:
 *   global_work_size  = { seq_len * num_heads * ATTN_WG_SIZE }
 *   local_work_size   = { ATTN_WG_SIZE }
 *
 * Layout:  Q/K/V[seq_pos * num_heads * head_dim + head * head_dim + d]
 * ========================================================================= */
__kernel void attention_prefill(
    __global const half* restrict Q,       // [seq_len, num_heads, head_dim]
    __global const half* restrict K,       // [seq_len, num_heads, head_dim]
    __global const half* restrict V,       // [seq_len, num_heads, head_dim]
    __global half* restrict output,        // [seq_len, num_heads, head_dim]
    const int seq_len,
    const int num_heads,
    const int head_dim)
{
    const int lid = get_local_id(0);
    const int group_idx = get_group_id(0);

    // Decode group_idx → (query_pos, head)
    const int query_pos = group_idx / num_heads;
    const int head = group_idx - mul24(query_pos, num_heads);

    if (query_pos >= seq_len) return;

    const int head_stride = mul24(num_heads, head_dim);
    const int q_offset = mad24(query_pos, head_stride, mul24(head, head_dim));

    // Scaling factor: 1 / sqrt(head_dim)
    const float scale = native_rsqrt((float)head_dim);

    // --- Phase 1: Compute attention scores for all key positions ---
    // Each work-item handles a subset of key positions
    // We need seq_len scores, stored in local memory for softmax

    __local float scores[ATTN_WG_SIZE];  // Reused for different passes
    __local float shared_max;
    __local float shared_sum;

    // --- Pass 1a: Compute Q·K^T scores and find max (for stable softmax) ---
    float local_max = -INFINITY;

    for (int kv_pos = lid; kv_pos < seq_len; kv_pos += ATTN_WG_SIZE) {
        const int k_offset = mad24(kv_pos, head_stride, mul24(head, head_dim));

        // Dot product: Q[query_pos, head, :] · K[kv_pos, head, :]
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fma((float)Q[q_offset + d], (float)K[k_offset + d], dot);
        }

        // Scale and apply causal mask
        // Causal: future positions (kv_pos > query_pos) get -inf
        // Branch-free: use fmin with a mask value
        float score = dot * scale;
        // If kv_pos > query_pos, mask = -INFINITY, else mask = 0.0f
        // select(a, b, cond): returns b if cond is true (non-zero), a otherwise
        const float mask = (kv_pos > query_pos) ? -INFINITY : 0.0f;
        score += mask;

        local_max = fmax(local_max, score);
    }

    // Reduce to find global max across workgroup
    scores[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = ATTN_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scores[lid] = fmax(scores[lid], scores[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) shared_max = scores[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    const float row_max = shared_max;

    // --- Pass 1b: Compute exp(score - max) and sum ---
    float local_sum = 0.0f;

    for (int kv_pos = lid; kv_pos < seq_len; kv_pos += ATTN_WG_SIZE) {
        const int k_offset = mad24(kv_pos, head_stride, mul24(head, head_dim));

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fma((float)Q[q_offset + d], (float)K[k_offset + d], dot);
        }

        float score = dot * scale;
        const float mask = (kv_pos > query_pos) ? -INFINITY : 0.0f;
        score += mask;

        local_sum += native_exp(score - row_max);
    }

    scores[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = ATTN_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scores[lid] += scores[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) shared_sum = scores[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    const float inv_sum = native_recip(shared_sum);

    // --- Phase 2: Compute weighted sum of V ---
    // Each work-item computes a subset of the head_dim output dimensions
    // For each output dimension d, sum over all kv_pos: weight[kv_pos] * V[kv_pos, head, d]

    for (int d = lid; d < head_dim; d += ATTN_WG_SIZE) {
        float acc = 0.0f;

        for (int kv_pos = 0; kv_pos < seq_len; ++kv_pos) {
            // Recompute attention weight for this kv_pos
            const int k_offset = mad24(kv_pos, head_stride, mul24(head, head_dim));

            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                dot = fma((float)Q[q_offset + dd], (float)K[k_offset + dd], dot);
            }

            float score = dot * scale;
            const float mask = (kv_pos > query_pos) ? -INFINITY : 0.0f;
            score += mask;

            const float weight = native_exp(score - row_max) * inv_sum;

            const int v_offset = mad24(kv_pos, head_stride, mul24(head, head_dim));
            acc = fma(weight, (float)V[v_offset + d], acc);
        }

        output[q_offset + d] = (half)acc;
    }
}

/* ============================================================================
 * Decode Attention: single query token against KV-cache
 *
 * During autoregressive decode, we generate one token at a time. The query
 * is a single vector Q[1, num_heads, head_dim] that attends to the entire
 * KV-cache K_cache[cache_len, num_heads, head_dim].
 *
 * This is memory-bound: we read the entire KV-cache for each token.
 * Strategy: each workgroup handles one attention head. Work-items cooperatively
 * compute dot products with all cache positions, reduce for softmax, then
 * compute the weighted V sum.
 *
 * Dispatch:
 *   global_work_size  = { num_heads * ATTN_WG_SIZE }
 *   local_work_size   = { ATTN_WG_SIZE }
 * ========================================================================= */
__kernel void attention_decode(
    __global const half* restrict Q,         // [1, num_heads, head_dim]
    __global const half* restrict K_cache,   // [cache_len, num_heads, head_dim]
    __global const half* restrict V_cache,   // [cache_len, num_heads, head_dim]
    __global half* restrict output,          // [1, num_heads, head_dim]
    const int cache_len,
    const int num_heads,
    const int head_dim)
{
    const int lid = get_local_id(0);
    const int head = get_group_id(0);

    if (head >= num_heads) return;

    const int head_stride = mul24(num_heads, head_dim);
    const int q_offset = mul24(head, head_dim);

    // Load Q vector for this head into registers (small, reused many times)
    // head_dim is typically 64 or 128 — fits in register file
    const float scale = native_rsqrt((float)head_dim);

    // --- Pass 1: Compute scores and find max ---
    __local float scratch[ATTN_WG_SIZE];
    float local_max = -INFINITY;

    for (int pos = lid; pos < cache_len; pos += ATTN_WG_SIZE) {
        const int k_offset = mad24(pos, head_stride, q_offset);

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fma((float)Q[q_offset + d], (float)K_cache[k_offset + d], dot);
        }

        local_max = fmax(local_max, dot * scale);
    }

    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = ATTN_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float row_max = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Pass 2: Compute exp(score - max) and sum ---
    float local_sum = 0.0f;

    for (int pos = lid; pos < cache_len; pos += ATTN_WG_SIZE) {
        const int k_offset = mad24(pos, head_stride, q_offset);

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fma((float)Q[q_offset + d], (float)K_cache[k_offset + d], dot);
        }

        local_sum += native_exp(dot * scale - row_max);
    }

    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = ATTN_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float inv_sum = native_recip(scratch[0]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Pass 3: Weighted sum of V ---
    // Each work-item handles a subset of head_dim output dimensions
    for (int d = lid; d < head_dim; d += ATTN_WG_SIZE) {
        float acc = 0.0f;

        for (int pos = 0; pos < cache_len; ++pos) {
            const int k_offset = mad24(pos, head_stride, q_offset);

            // Recompute attention weight for this position
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                dot = fma((float)Q[q_offset + dd], (float)K_cache[k_offset + dd], dot);
            }

            const float weight = native_exp(dot * scale - row_max) * inv_sum;

            const int v_offset = mad24(pos, head_stride, q_offset);
            acc = fma(weight, (float)V_cache[v_offset + d], acc);
        }

        output[q_offset + d] = (half)acc;
    }
}
