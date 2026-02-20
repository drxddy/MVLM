/*
 * MVLM - Embedding Lookup Kernel for Adreno GPUs
 * Phase 2: Transformer Primitives
 *
 * Simple indexed lookup into an embedding table. Each token ID maps to a
 * row in the embedding table, which is copied to the output buffer.
 *
 * This is a trivial, memory-bound kernel — the optimization focus is on
 * maximizing bandwidth utilization via vectorized loads/stores.
 *
 * Adreno optimizations:
 *   - int/uint indexing (saves registers vs size_t)
 *   - mad24/mul24 for index calculations
 *   - Vectorized 128-bit loads/stores (vload_half4/vstore_half4)
 *   - Each work-item copies one element (simple, cache-friendly)
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* ============================================================================
 * Embedding Lookup
 *
 * For each token in token_ids, copy the corresponding row from embed_table
 * to the output buffer.
 *
 * embed_table[token_id, :] → output[seq_pos, :]
 *
 * Dispatch:
 *   global_work_size  = { seq_len, ceil(embed_dim / 4) }
 *   Each work-item copies 4 contiguous half values (64 bits) for one token.
 *
 * Using vectorized loads (half4) to maximize memory bandwidth utilization.
 * Each half4 load/store is 64 bits; on Adreno this achieves good bus
 * utilization without requiring 128-bit alignment.
 * ========================================================================= */
__kernel void embedding_lookup(
    __global const half* restrict embed_table,  // [vocab_size, embed_dim]
    __global const int* restrict token_ids,     // [seq_len]
    __global half* restrict output,             // [seq_len, embed_dim]
    const int embed_dim)
{
    const int seq_pos = get_global_id(0);  // which token
    const int d4 = get_global_id(1);       // which group of 4 embedding dims
    const int d_base = d4 << 2;            // d4 * 4

    if (d_base >= embed_dim) return;

    // Look up the token ID for this sequence position
    const int token_id = token_ids[seq_pos];

    // Source: embed_table[token_id, d_base .. d_base+3]
    const int src_offset = mad24(token_id, embed_dim, d_base);
    // Destination: output[seq_pos, d_base .. d_base+3]
    const int dst_offset = mad24(seq_pos, embed_dim, d_base);

    // Vectorized copy: 4 halfs at a time
    if (d_base + 4 <= embed_dim) {
        const half4 val = vload_half4(0, embed_table + src_offset);
        vstore_half4(convert_float4(val), 0, output + dst_offset);
    } else {
        // Scalar tail for non-multiple-of-4 embed_dim
        for (int i = 0; i < embed_dim - d_base; ++i) {
            output[dst_offset + i] = embed_table[src_offset + i];
        }
    }
}
