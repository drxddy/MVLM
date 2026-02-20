/*
 * MVLM - Rotary Position Embedding (RoPE) Kernel for Adreno GPUs
 * Phase 2: Transformer Primitives
 *
 * Implements RoPE as described in "RoFormer: Enhanced Transformer with Rotary
 * Position Embedding" (Su et al., 2021). Used in LLaMA, Phi, Qwen, etc.
 *
 * For each pair of elements (x0, x1) at positions (2i, 2i+1) in the head:
 *   y0 = x0 * cos(θ) - x1 * sin(θ)
 *   y1 = x0 * sin(θ) + x1 * cos(θ)
 *
 * where θ = position * freq_i, and freq_i = 1 / (10000^(2i/head_dim))
 *
 * We use precomputed sin/cos tables (stored in __constant memory) to avoid
 * computing transcendentals on-device per inference.
 *
 * Adreno optimizations:
 *   - __constant memory for sin/cos tables (small, broadcast to all WIs)
 *   - int/uint indexing (saves registers vs size_t)
 *   - mad24/mul24 for index calculations
 *   - In-place modification (no extra output buffer needed)
 *   - Branch-free rotation (pure ALU, no conditionals)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* ============================================================================
 * RoPE Apply
 *
 * Applied in-place to Q or K tensors.
 *
 * Tensor layout: qk[seq_pos, head, dim_pair*2 + {0,1}]
 *   - Flattened as: qk[seq_pos * num_heads * head_dim + head * head_dim + d]
 *
 * Dispatch:
 *   global_work_size  = { seq_len, num_heads, head_dim / 2 }
 *   Each work-item handles one (x0, x1) pair.
 *
 * The `offset` parameter supports decode phase: during autoregressive decode,
 * each new token has position = offset + seq_pos (where offset = cache_len).
 * ========================================================================= */
__kernel void rope_apply(
    __global half* restrict qk,                // [seq_len, num_heads, head_dim] — modified in-place
    __constant const half* restrict cos_table,  // [max_seq_len, head_dim/2]
    __constant const half* restrict sin_table,  // [max_seq_len, head_dim/2]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int offset)                           // position offset for decode phase
{
    const int seq_pos  = get_global_id(0);  // which token in the sequence
    const int head     = get_global_id(1);  // which attention head
    const int pair_idx = get_global_id(2);  // which rotation pair [0, head_dim/2)

    if (seq_pos >= seq_len || head >= num_heads) return;

    const int half_dim = head_dim >> 1;
    if (pair_idx >= half_dim) return;

    // Absolute position (for decode: position in the full sequence, not just this batch)
    const int pos = seq_pos + offset;

    // Index into the sin/cos tables: tables are [max_seq_len, head_dim/2]
    const int table_idx = mad24(pos, half_dim, pair_idx);

    // Load precomputed cos and sin for this position and dimension pair
    const float cos_val = (float)cos_table[table_idx];
    const float sin_val = (float)sin_table[table_idx];

    // Index into the qk tensor: qk[seq_pos, head, 2*pair_idx] and qk[seq_pos, head, 2*pair_idx + 1]
    const int base_idx = mad24(seq_pos, mul24(num_heads, head_dim),
                               mad24(head, head_dim, pair_idx << 1));

    // Load the pair (x0, x1) — contiguous in memory
    const float x0 = (float)qk[base_idx];
    const float x1 = (float)qk[base_idx + 1];

    // Apply rotation — branch-free, pure ALU
    //   y0 = x0 * cos - x1 * sin
    //   y1 = x0 * sin + x1 * cos
    const float y0 = fma(x0, cos_val, -(x1 * sin_val));
    const float y1 = fma(x0, sin_val, x1 * cos_val);

    // Write back in-place
    qk[base_idx]     = (half)y0;
    qk[base_idx + 1] = (half)y1;
}
