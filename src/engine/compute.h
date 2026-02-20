#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "device.h"

namespace mgpu {

// --- GEMM / GEMV ---

// C[M,N] = A[M,K] * B[K,N] — naive, one work-item per output element
cl_event dispatch_gemm_naive(const DeviceInfo* dev, cl_program program,
                             cl_mem A, cl_mem B, cl_mem C,
                             int M, int N, int K);

// C[M,N] = A[M,K] * B[K,N] — tiled with local memory
cl_event dispatch_gemm_tiled(const DeviceInfo* dev, cl_program program,
                             cl_mem A, cl_mem B, cl_mem C,
                             int M, int N, int K);

// C[M,N] = A[M,K] * B_img[K,N] — A buffer, B as image2d_t
cl_event dispatch_gemm_image(const DeviceInfo* dev, cl_program program,
                             cl_mem A, cl_mem B_img, cl_mem C,
                             int M, int N, int K);

// y[1,N] = x[1,K] * W_img[K,N] — decode-phase GEMV, weights as image
cl_event dispatch_gemv(const DeviceInfo* dev, cl_program program,
                       cl_mem x, cl_mem W_img, cl_mem y,
                       int N, int K);

// --- Layer Normalization ---

// RMSNorm: output = input * rsqrt(mean(input^2) + eps) * weight
cl_event dispatch_rms_norm(const DeviceInfo* dev, cl_program program,
                           cl_mem input, cl_mem output, cl_mem weight,
                           int num_rows, int hidden_size, float eps);

// --- Activations ---

// SiLU (Swish): y = x * sigmoid(x)
cl_event dispatch_silu(const DeviceInfo* dev, cl_program program,
                       cl_mem input, cl_mem output, int n);

// GELU (approximate): y = 0.5 * x * (1 + tanh(...))
cl_event dispatch_gelu(const DeviceInfo* dev, cl_program program,
                       cl_mem input, cl_mem output, int n);

// Softmax over rows: y_i = exp(x_i - max) / sum(exp)
cl_event dispatch_softmax(const DeviceInfo* dev, cl_program program,
                          cl_mem input, cl_mem output,
                          int seq_len, int num_elements);

// Fused SiLU-gated MLP: output = silu(gate) * up
cl_event dispatch_silu_gate_multiply(const DeviceInfo* dev, cl_program program,
                                     cl_mem gate, cl_mem up, cl_mem output,
                                     int n);

// --- Attention ---

// Multi-head attention for prefill (full sequence)
cl_event dispatch_attention_prefill(const DeviceInfo* dev, cl_program program,
                                    cl_mem Q, cl_mem K, cl_mem V, cl_mem output,
                                    int seq_len, int num_heads, int head_dim);

// Single-token decode attention against KV-cache
cl_event dispatch_attention_decode(const DeviceInfo* dev, cl_program program,
                                   cl_mem Q, cl_mem K_cache, cl_mem V_cache,
                                   cl_mem output,
                                   int cache_len, int num_heads, int head_dim);

// --- RoPE ---

// Apply rotary position embeddings in-place
cl_event dispatch_rope_apply(const DeviceInfo* dev, cl_program program,
                             cl_mem qk, cl_mem cos_table, cl_mem sin_table,
                             int seq_len, int num_heads, int head_dim,
                             int offset);

// --- Embedding ---

// Lookup token embeddings from table
cl_event dispatch_embedding_lookup(const DeviceInfo* dev, cl_program program,
                                   cl_mem embed_table, cl_mem token_ids,
                                   cl_mem output,
                                   int seq_len, int embed_dim);

// --- Vision ---

// Preprocess image: resize + normalize RGBA → fp16 CHW
cl_event dispatch_preprocess_image(const DeviceInfo* dev, cl_program program,
                                   cl_mem input_image, cl_mem output,
                                   int target_h, int target_w,
                                   float mean_r, float mean_g, float mean_b,
                                   float std_r, float std_g, float std_b);

// Patch embedding: extract patches and project via GEMM
cl_event dispatch_patch_embed(const DeviceInfo* dev, cl_program program,
                              cl_mem image, cl_mem proj_weight,
                              cl_mem proj_bias, cl_mem patches,
                              int C, int H, int W,
                              int patch_h, int patch_w, int embed_dim);

// Element-wise vector addition: output = a + b
cl_event dispatch_vector_add(const DeviceInfo* dev, cl_program program,
                             cl_mem a, cl_mem b, cl_mem output, int n);

} // namespace mgpu
