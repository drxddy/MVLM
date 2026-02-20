#pragma once

#include "../engine/device.h"
#include "../engine/memory.h"
#include "gguf_loader.h"

namespace mgpu {

// Moondream2 architecture constants
struct Moondream2Config {
    // Vision encoder (SigLIP)
    int image_size = 378;       // input image resolution
    int patch_size = 14;        // patch size
    int vision_layers = 27;     // SigLIP encoder layers
    int vision_dim = 1152;      // vision hidden dimension
    int vision_heads = 16;      // vision attention heads
    int num_patches = 729;      // (378/14)^2 = 27^2 = 729

    // Projection
    int proj_dim = 2048;        // projection output dim (matches LLM dim)

    // Language model (Phi-1.5)
    int vocab_size = 51200;
    int llm_layers = 24;
    int llm_dim = 2048;
    int llm_heads = 32;
    int head_dim = 64;          // llm_dim / llm_heads
    int llm_intermediate = 8192; // MLP intermediate size
    int max_seq_len = 2048;
};

struct TransformerLayerWeights {
    // Attention projections (stored as images for TP/L1 cache)
    cl_mem q_proj_weight;      // image2d: [dim, dim]
    cl_mem k_proj_weight;      // image2d: [dim, dim]
    cl_mem v_proj_weight;      // image2d: [dim, dim]
    cl_mem o_proj_weight;      // image2d: [dim, dim]

    // MLP (SwiGLU)
    cl_mem gate_proj_weight;   // image2d: [dim, intermediate]
    cl_mem up_proj_weight;     // image2d: [dim, intermediate]
    cl_mem down_proj_weight;   // image2d: [intermediate, dim]

    // Norms (small vectors — buffers)
    cl_mem input_norm_weight;  // buffer: [dim]
    cl_mem post_norm_weight;   // buffer: [dim]
};

struct Moondream2Weights {
    cl_mem token_embed;        // buffer: [vocab_size, dim]
    cl_mem final_norm_weight;  // buffer: [dim]
    cl_mem lm_head_weight;     // image2d: [dim, vocab_size]

    TransformerLayerWeights* layers;
    int num_layers;

    // RoPE tables
    cl_mem cos_table;          // buffer: [max_seq_len, head_dim/2]
    cl_mem sin_table;          // buffer: [max_seq_len, head_dim/2]
};

struct KVCache {
    cl_mem k_cache;  // buffer: [max_seq_len, num_heads, head_dim]
    cl_mem v_cache;  // buffer: [max_seq_len, num_heads, head_dim]
    int length;      // current number of cached positions
    int capacity;    // max_seq_len
};

struct Moondream2Model {
    Moondream2Config config;
    GGUFFile weights;

    // OpenCL programs (compiled kernels)
    cl_program gemm_program;
    cl_program attention_program;
    cl_program norm_program;
    cl_program activation_program;
    cl_program rope_program;
    cl_program embedding_program;
    cl_program vision_program;

    // GPU weights and caches
    Moondream2Weights gpu_weights;
    KVCache kv_cache;

    // Scratch buffers for activations
    cl_mem scratch_a;     // [max_seq_len * dim]
    cl_mem scratch_b;     // [max_seq_len * dim]
    cl_mem scratch_gate;  // [max_seq_len * intermediate]
    cl_mem scratch_up;    // [max_seq_len * intermediate]
    cl_mem scratch_q;     // [max_seq_len * dim]
    cl_mem scratch_k;     // [max_seq_len * dim]
    cl_mem scratch_v;     // [max_seq_len * dim]
    cl_mem scratch_attn;  // [max_seq_len * dim]

    bool initialized;
};

// Load model: open GGUF, compile kernels, upload weights, allocate buffers
bool moondream2_load(Moondream2Model* model, const DeviceInfo* device,
                     const char* gguf_path, const char* kernel_dir);
void moondream2_destroy(Moondream2Model* model);

// Upload weights from GGUF to GPU
bool moondream2_upload_weights(Moondream2Model* model, const DeviceInfo* device);

// Precompute RoPE sin/cos tables
bool moondream2_init_rope(Moondream2Model* model, const DeviceInfo* device);

// Allocate KV-cache and scratch buffers
bool moondream2_alloc_buffers(Moondream2Model* model, const DeviceInfo* device);

// Run the full LLM forward pass, returns logits buffer [vocab_size]
cl_mem moondream2_forward(Moondream2Model* model, const DeviceInfo* device,
                          const int* tokens, int seq_len);

// Greedy autoregressive text generation
// Encodes prompt, runs prefill, then decodes token-by-token
// Prints generated tokens to stdout as they are produced
// Returns total number of tokens generated (excluding prompt)
int moondream2_generate(Moondream2Model* model, const DeviceInfo* device,
                        const char* prompt, int max_new_tokens,
                        const char* vocab_path);

// Reset KV-cache (for new conversation)
void moondream2_reset_cache(Moondream2Model* model);

// Release GPU resources
void moondream2_release_gpu(Moondream2Model* model);

} // namespace mgpu
