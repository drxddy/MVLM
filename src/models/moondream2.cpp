#include "moondream2.h"
#include "tokenizer.h"
#include "../engine/compute.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace mgpu {

// --- Helpers ---

static cl_program load_kernel(const DeviceInfo* device, const char* kernel_dir,
                              const char* filename, const char* build_opts) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", kernel_dir, filename);
    cl_program prog = build_program_from_file(device, path, build_opts);
    if (!prog)
        fprintf(stderr, "Warning: Failed to build kernel: %s\n", path);
    return prog;
}

// Try several naming conventions for GGUF tensor lookup
static const TensorInfo* find_weight(const GGUFFile* file, const char* name) {
    const TensorInfo* t = gguf_find_tensor(file, name);
    if (t) return t;

    // Try with "model." prefix
    char buf[300];
    snprintf(buf, sizeof(buf), "model.%s", name);
    t = gguf_find_tensor(file, buf);
    if (t) return t;

    // Try with "transformer." prefix
    snprintf(buf, sizeof(buf), "transformer.%s", name);
    t = gguf_find_tensor(file, buf);
    return t;
}

static const TensorInfo* find_layer_weight(const GGUFFile* file, int layer,
                                           const char* suffix) {
    char name[300];

    // Try: model.layers.{i}.{suffix}
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    const TensorInfo* t = gguf_find_tensor(file, name);
    if (t) return t;

    // Try: blk.{i}.{suffix}
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    t = gguf_find_tensor(file, name);
    if (t) return t;

    // Try: transformer.h.{i}.{suffix}
    snprintf(name, sizeof(name), "transformer.h.%d.%s", layer, suffix);
    return gguf_find_tensor(file, name);
}

// Upload a 2D weight matrix as an image object (for fp16 data)
static cl_mem upload_weight_image(const DeviceInfo* device, const GGUFFile* file,
                                  const TensorInfo* tensor) {
    if (!tensor) return nullptr;

    if (tensor->type != GGMLType::F16) {
        fprintf(stderr, "  Warning: tensor '%s' is not F16 (type=%d), storing as buffer\n",
                tensor->name, (int)tensor->type);
        const void* data = gguf_tensor_data(file, tensor);
        return create_buffer(device, tensor->data_size,
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             (void*)data);
    }

    int rows = (int)tensor->dims[1];
    int cols = (int)tensor->dims[0];
    if (tensor->n_dims == 1) {
        rows = 1;
        cols = (int)tensor->dims[0];
    }

    const cl_half* data = (const cl_half*)gguf_tensor_data(file, tensor);
    return create_weight_image(device, rows, cols, data);
}

// Upload a 1D weight vector as a buffer
static cl_mem upload_weight_buffer(const DeviceInfo* device, const GGUFFile* file,
                                   const TensorInfo* tensor) {
    if (!tensor) return nullptr;

    const void* data = gguf_tensor_data(file, tensor);
    return create_buffer(device, tensor->data_size,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         (void*)data);
}

// --- Weight Upload ---

bool moondream2_upload_weights(Moondream2Model* model, const DeviceInfo* device) {
    const GGUFFile* f = &model->weights;
    Moondream2Weights* w = &model->gpu_weights;
    const Moondream2Config& cfg = model->config;

    printf("Uploading weights to GPU...\n");

    // Token embeddings — large matrix, use buffer
    const TensorInfo* embed = find_weight(f, "embed_tokens.weight");
    if (!embed) embed = find_weight(f, "token_embd.weight");
    if (embed) {
        const void* data = gguf_tensor_data(f, embed);
        w->token_embed = create_buffer(device, embed->data_size,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       (void*)data);
        printf("  token_embed: %llu x %llu (%.1f MB)\n",
               (unsigned long long)embed->dims[1], (unsigned long long)embed->dims[0],
               (double)embed->data_size / (1024.0 * 1024.0));
    } else {
        fprintf(stderr, "Error: token embedding weight not found\n");
        return false;
    }

    // Final norm
    const TensorInfo* fnorm = find_weight(f, "norm.weight");
    if (!fnorm) fnorm = find_weight(f, "output_norm.weight");
    if (fnorm) w->final_norm_weight = upload_weight_buffer(device, f, fnorm);

    // LM head
    const TensorInfo* lmh = find_weight(f, "lm_head.weight");
    if (!lmh) lmh = find_weight(f, "output.weight");
    if (lmh) w->lm_head_weight = upload_weight_image(device, f, lmh);

    // Transformer layers
    w->num_layers = cfg.llm_layers;
    w->layers = (TransformerLayerWeights*)calloc(w->num_layers, sizeof(TransformerLayerWeights));
    if (!w->layers) return false;

    int loaded = 0;
    for (int i = 0; i < w->num_layers; i++) {
        TransformerLayerWeights* lw = &w->layers[i];

        // Attention projections
        const TensorInfo* t;

        t = find_layer_weight(f, i, "self_attn.q_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn.q_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn_q.weight");
        lw->q_proj_weight = upload_weight_image(device, f, t);

        t = find_layer_weight(f, i, "self_attn.k_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn.k_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn_k.weight");
        lw->k_proj_weight = upload_weight_image(device, f, t);

        t = find_layer_weight(f, i, "self_attn.v_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn.v_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn_v.weight");
        lw->v_proj_weight = upload_weight_image(device, f, t);

        t = find_layer_weight(f, i, "self_attn.dense.weight");
        if (!t) t = find_layer_weight(f, i, "self_attn.o_proj.weight");
        if (!t) t = find_layer_weight(f, i, "attn_output.weight");
        lw->o_proj_weight = upload_weight_image(device, f, t);

        // MLP projections
        t = find_layer_weight(f, i, "mlp.fc1.weight");
        if (!t) t = find_layer_weight(f, i, "mlp.gate_proj.weight");
        if (!t) t = find_layer_weight(f, i, "ffn_gate.weight");
        lw->gate_proj_weight = upload_weight_image(device, f, t);

        t = find_layer_weight(f, i, "mlp.fc1.weight"); // Phi uses single fc1 for gate+up packed
        if (!t) t = find_layer_weight(f, i, "mlp.up_proj.weight");
        if (!t) t = find_layer_weight(f, i, "ffn_up.weight");
        lw->up_proj_weight = upload_weight_image(device, f, t);

        t = find_layer_weight(f, i, "mlp.fc2.weight");
        if (!t) t = find_layer_weight(f, i, "mlp.down_proj.weight");
        if (!t) t = find_layer_weight(f, i, "ffn_down.weight");
        lw->down_proj_weight = upload_weight_image(device, f, t);

        // Norms
        t = find_layer_weight(f, i, "input_layernorm.weight");
        if (!t) t = find_layer_weight(f, i, "attn_norm.weight");
        lw->input_norm_weight = upload_weight_buffer(device, f, t);

        t = find_layer_weight(f, i, "post_attention_layernorm.weight");
        if (!t) t = find_layer_weight(f, i, "ffn_norm.weight");
        lw->post_norm_weight = upload_weight_buffer(device, f, t);

        loaded++;
    }

    printf("  Uploaded %d/%d transformer layers\n", loaded, w->num_layers);
    return true;
}

// --- RoPE Initialization ---

bool moondream2_init_rope(Moondream2Model* model, const DeviceInfo* device) {
    const Moondream2Config& cfg = model->config;
    int half_dim = cfg.head_dim / 2;
    int max_len = cfg.max_seq_len;
    size_t table_size = (size_t)max_len * half_dim;

    cl_half* cos_data = (cl_half*)malloc(table_size * sizeof(cl_half));
    cl_half* sin_data = (cl_half*)malloc(table_size * sizeof(cl_half));
    if (!cos_data || !sin_data) {
        free(cos_data);
        free(sin_data);
        return false;
    }

    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            double freq = 1.0 / pow(10000.0, 2.0 * i / cfg.head_dim);
            double theta = pos * freq;
            float c = (float)cos(theta);
            float s = (float)sin(theta);

            // Quick fp32→fp16 conversion
            int idx = pos * half_dim + i;
            uint32_t cf, sf;
            memcpy(&cf, &c, 4);
            memcpy(&sf, &s, 4);
            uint32_t c_sign = (cf >> 16) & 0x8000;
            int32_t c_exp = ((cf >> 23) & 0xFF) - 127 + 15;
            uint32_t c_mant = (cf >> 13) & 0x3FF;
            uint32_t s_sign = (sf >> 16) & 0x8000;
            int32_t s_exp = ((sf >> 23) & 0xFF) - 127 + 15;
            uint32_t s_mant = (sf >> 13) & 0x3FF;

            cos_data[idx] = (c_exp <= 0) ? (cl_half)c_sign :
                            (c_exp >= 31) ? (cl_half)(c_sign | 0x7C00) :
                            (cl_half)(c_sign | (c_exp << 10) | c_mant);
            sin_data[idx] = (s_exp <= 0) ? (cl_half)s_sign :
                            (s_exp >= 31) ? (cl_half)(s_sign | 0x7C00) :
                            (cl_half)(s_sign | (s_exp << 10) | s_mant);
        }
    }

    size_t bytes = table_size * sizeof(cl_half);
    model->gpu_weights.cos_table = create_buffer(device, bytes,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cos_data);
    model->gpu_weights.sin_table = create_buffer(device, bytes,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sin_data);

    free(cos_data);
    free(sin_data);

    printf("  RoPE tables: %d positions x %d dims (%.1f KB)\n",
           max_len, half_dim, (double)(bytes * 2) / 1024.0);

    return model->gpu_weights.cos_table && model->gpu_weights.sin_table;
}

// --- Buffer Allocation ---

bool moondream2_alloc_buffers(Moondream2Model* model, const DeviceInfo* device) {
    const Moondream2Config& cfg = model->config;
    size_t half_size = sizeof(cl_half);

    // KV-cache: [max_seq_len * num_heads * head_dim]
    size_t kv_size = (size_t)cfg.max_seq_len * cfg.llm_heads * cfg.head_dim * half_size;
    model->kv_cache.k_cache = create_buffer(device, kv_size, CL_MEM_READ_WRITE);
    model->kv_cache.v_cache = create_buffer(device, kv_size, CL_MEM_READ_WRITE);
    model->kv_cache.length = 0;
    model->kv_cache.capacity = cfg.max_seq_len;

    if (!model->kv_cache.k_cache || !model->kv_cache.v_cache) {
        fprintf(stderr, "Error: Failed to allocate KV-cache (%.1f MB each)\n",
                (double)kv_size / (1024.0 * 1024.0));
        return false;
    }

    // Scratch buffers for activations
    size_t act_size = (size_t)cfg.max_seq_len * cfg.llm_dim * half_size;
    size_t mlp_size = (size_t)cfg.max_seq_len * cfg.llm_intermediate * half_size;

    model->scratch_a    = create_buffer(device, act_size, CL_MEM_READ_WRITE);
    model->scratch_b    = create_buffer(device, act_size, CL_MEM_READ_WRITE);
    model->scratch_q    = create_buffer(device, act_size, CL_MEM_READ_WRITE);
    model->scratch_k    = create_buffer(device, act_size, CL_MEM_READ_WRITE);
    model->scratch_v    = create_buffer(device, act_size, CL_MEM_READ_WRITE);
    model->scratch_attn = create_buffer(device, act_size, CL_MEM_READ_WRITE);
    model->scratch_gate = create_buffer(device, mlp_size, CL_MEM_READ_WRITE);
    model->scratch_up   = create_buffer(device, mlp_size, CL_MEM_READ_WRITE);

    printf("  KV-cache: %.1f MB, scratch: %.1f MB\n",
           (double)(kv_size * 2) / (1024.0 * 1024.0),
           (double)(act_size * 6 + mlp_size * 2) / (1024.0 * 1024.0));

    return model->scratch_a && model->scratch_b && model->scratch_q &&
           model->scratch_k && model->scratch_v && model->scratch_attn &&
           model->scratch_gate && model->scratch_up;
}

// --- Residual Add ---

static cl_event dispatch_residual_add(const DeviceInfo* dev, cl_program act_program,
                                       cl_mem a, cl_mem b, cl_mem out, int n) {
    return dispatch_vector_add(dev, act_program, a, b, out, n);
}

// --- KV-cache append ---

static void kv_cache_append(const DeviceInfo* dev, KVCache* cache,
                            cl_mem new_k, cl_mem new_v,
                            int seq_len, int num_heads, int head_dim) {
    size_t row_bytes = (size_t)num_heads * head_dim * sizeof(cl_half);
    size_t offset = (size_t)cache->length * row_bytes;
    size_t copy_bytes = (size_t)seq_len * row_bytes;

    clEnqueueCopyBuffer(dev->queue, new_k, cache->k_cache, 0, offset,
                        copy_bytes, 0, nullptr, nullptr);
    clEnqueueCopyBuffer(dev->queue, new_v, cache->v_cache, 0, offset,
                        copy_bytes, 0, nullptr, nullptr);
    cache->length += seq_len;
}

// --- Forward Pass ---

cl_mem moondream2_forward(Moondream2Model* model, const DeviceInfo* device,
                          const int* tokens, int seq_len) {
    if (!model->initialized) {
        fprintf(stderr, "Error: model not initialized\n");
        return nullptr;
    }

    const Moondream2Config& cfg = model->config;
    Moondream2Weights* w = &model->gpu_weights;
    int pos_offset = model->kv_cache.length;

    printf("[forward] seq_len=%d, pos_offset=%d\n", seq_len, pos_offset);

    // 1. Upload token IDs to GPU
    cl_int err;
    cl_mem d_tokens = clCreateBuffer(device->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)seq_len * sizeof(int), (void*)tokens, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: failed to create token buffer\n");
        return nullptr;
    }

    // 2. Embedding lookup: tokens → scratch_a [seq_len, dim]
    cl_event ev = dispatch_embedding_lookup(device, model->embedding_program,
                                            w->token_embed, d_tokens,
                                            model->scratch_a, seq_len, cfg.llm_dim);
    if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }
    clReleaseMemObject(d_tokens);

    printf("[forward] embedding lookup done\n");

    // Current hidden state is in scratch_a
    cl_mem hidden = model->scratch_a;
    cl_mem residual_buf = model->scratch_b;

    // 3. Transformer layers
    for (int layer = 0; layer < cfg.llm_layers; layer++) {
        TransformerLayerWeights* lw = &w->layers[layer];
        bool is_decode = (seq_len == 1);

        // --- Attention block ---

        // RMSNorm(hidden) → scratch_b
        ev = dispatch_rms_norm(device, model->norm_program,
                               hidden, residual_buf, lw->input_norm_weight,
                               seq_len, cfg.llm_dim, 1e-5f);
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Q = norm_out @ q_proj  [seq_len, dim]
        if (is_decode && lw->q_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               residual_buf, lw->q_proj_weight,
                               model->scratch_q, cfg.llm_dim, cfg.llm_dim);
        } else if (lw->q_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     residual_buf, lw->q_proj_weight,
                                     model->scratch_q, seq_len, cfg.llm_dim, cfg.llm_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // K = norm_out @ k_proj  [seq_len, dim]
        if (is_decode && lw->k_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               residual_buf, lw->k_proj_weight,
                               model->scratch_k, cfg.llm_dim, cfg.llm_dim);
        } else if (lw->k_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     residual_buf, lw->k_proj_weight,
                                     model->scratch_k, seq_len, cfg.llm_dim, cfg.llm_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // V = norm_out @ v_proj  [seq_len, dim]
        if (is_decode && lw->v_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               residual_buf, lw->v_proj_weight,
                               model->scratch_v, cfg.llm_dim, cfg.llm_dim);
        } else if (lw->v_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     residual_buf, lw->v_proj_weight,
                                     model->scratch_v, seq_len, cfg.llm_dim, cfg.llm_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // RoPE on Q and K
        if (model->rope_program) {
            ev = dispatch_rope_apply(device, model->rope_program,
                                     model->scratch_q, w->cos_table, w->sin_table,
                                     seq_len, cfg.llm_heads, cfg.head_dim, pos_offset);
            if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

            ev = dispatch_rope_apply(device, model->rope_program,
                                     model->scratch_k, w->cos_table, w->sin_table,
                                     seq_len, cfg.llm_heads, cfg.head_dim, pos_offset);
            if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }
        }

        // Append K, V to KV-cache
        kv_cache_append(device, &model->kv_cache, model->scratch_k, model->scratch_v,
                        seq_len, cfg.llm_heads, cfg.head_dim);

        // Attention: Q against full KV-cache → scratch_attn
        int cache_len = model->kv_cache.length;
        if (is_decode) {
            ev = dispatch_attention_decode(device, model->attention_program,
                                           model->scratch_q,
                                           model->kv_cache.k_cache,
                                           model->kv_cache.v_cache,
                                           model->scratch_attn,
                                           cache_len, cfg.llm_heads, cfg.head_dim);
        } else {
            ev = dispatch_attention_prefill(device, model->attention_program,
                                            model->scratch_q,
                                            model->kv_cache.k_cache,
                                            model->kv_cache.v_cache,
                                            model->scratch_attn,
                                            cache_len, cfg.llm_heads, cfg.head_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Output projection: attn_out @ o_proj → scratch_b
        if (is_decode && lw->o_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               model->scratch_attn, lw->o_proj_weight,
                               residual_buf, cfg.llm_dim, cfg.llm_dim);
        } else if (lw->o_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     model->scratch_attn, lw->o_proj_weight,
                                     residual_buf, seq_len, cfg.llm_dim, cfg.llm_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Residual: hidden = hidden + attn_output
        dispatch_residual_add(device, model->activation_program, hidden, residual_buf, hidden,
                               seq_len * cfg.llm_dim);

        // --- MLP block ---

        // RMSNorm(hidden) → scratch_b
        ev = dispatch_rms_norm(device, model->norm_program,
                               hidden, residual_buf, lw->post_norm_weight,
                               seq_len, cfg.llm_dim, 1e-5f);
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Gate projection: norm_out @ gate_proj → scratch_gate
        if (is_decode && lw->gate_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               residual_buf, lw->gate_proj_weight,
                               model->scratch_gate, cfg.llm_intermediate, cfg.llm_dim);
        } else if (lw->gate_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     residual_buf, lw->gate_proj_weight,
                                     model->scratch_gate,
                                     seq_len, cfg.llm_intermediate, cfg.llm_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Up projection: norm_out @ up_proj → scratch_up
        if (is_decode && lw->up_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               residual_buf, lw->up_proj_weight,
                               model->scratch_up, cfg.llm_intermediate, cfg.llm_dim);
        } else if (lw->up_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     residual_buf, lw->up_proj_weight,
                                     model->scratch_up,
                                     seq_len, cfg.llm_intermediate, cfg.llm_dim);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Fused SiLU gate multiply: silu(gate) * up → scratch_gate
        int mlp_n = seq_len * cfg.llm_intermediate;
        ev = dispatch_silu_gate_multiply(device, model->activation_program,
                                         model->scratch_gate, model->scratch_up,
                                         model->scratch_gate, mlp_n);
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Down projection: mlp_out @ down_proj → scratch_b
        if (is_decode && lw->down_proj_weight) {
            ev = dispatch_gemv(device, model->gemm_program,
                               model->scratch_gate, lw->down_proj_weight,
                               residual_buf, cfg.llm_dim, cfg.llm_intermediate);
        } else if (lw->down_proj_weight) {
            ev = dispatch_gemm_image(device, model->gemm_program,
                                     model->scratch_gate, lw->down_proj_weight,
                                     residual_buf,
                                     seq_len, cfg.llm_dim, cfg.llm_intermediate);
        }
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

        // Residual: hidden = hidden + mlp_output
        dispatch_residual_add(device, model->activation_program, hidden, residual_buf, hidden,
                              seq_len * cfg.llm_dim);

        if (layer % 8 == 0 || layer == cfg.llm_layers - 1) {
            printf("[forward] layer %d/%d done\n", layer + 1, cfg.llm_layers);
        }
    }

    // 4. Final RMSNorm
    ev = dispatch_rms_norm(device, model->norm_program,
                           hidden, residual_buf, w->final_norm_weight,
                           seq_len, cfg.llm_dim, 1e-5f);
    if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }

    // 5. LM head: last_hidden @ lm_head_weight → logits [1, vocab_size]
    // Only compute for the last token position
    size_t last_offset = (size_t)(seq_len - 1) * cfg.llm_dim * sizeof(cl_half);
    cl_buffer_region region = { last_offset, (size_t)cfg.llm_dim * sizeof(cl_half) };
    cl_mem last_hidden = clCreateSubBuffer(residual_buf, CL_MEM_READ_ONLY,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            &region, &err);

    cl_mem logits = create_buffer(device, (size_t)cfg.vocab_size * sizeof(cl_half),
                                  CL_MEM_READ_WRITE);

    if (last_hidden && logits && w->lm_head_weight) {
        ev = dispatch_gemv(device, model->gemm_program,
                           last_hidden, w->lm_head_weight,
                           logits, cfg.vocab_size, cfg.llm_dim);
        if (ev) { clWaitForEvents(1, &ev); clReleaseEvent(ev); }
    }

    if (last_hidden) clReleaseMemObject(last_hidden);

    printf("[forward] complete, logits ready\n");
    clFinish(device->queue);
    return logits;
}

// --- Argmax on GPU logits ---

static int argmax_logits(const DeviceInfo* device, cl_mem logits, int vocab_size) {
    // Read logits back to CPU for argmax
    // TODO: implement GPU-side argmax kernel
    cl_half* host_logits = (cl_half*)malloc((size_t)vocab_size * sizeof(cl_half));
    if (!host_logits) return -1;

    cl_int err = clEnqueueReadBuffer(device->queue, logits, CL_TRUE, 0,
                                      (size_t)vocab_size * sizeof(cl_half),
                                      host_logits, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        free(host_logits);
        return -1;
    }

    // fp16 argmax — convert to float for comparison
    int best_id = 0;
    float best_val = -1e30f;
    for (int i = 0; i < vocab_size; i++) {
        // Quick fp16→fp32 decode
        uint16_t h = host_logits[i];
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp_val = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        float val;
        if (exp_val == 0) {
            val = 0.0f;
        } else if (exp_val == 31) {
            val = (sign ? -1.0f : 1.0f) * 1e30f;
        } else {
            uint32_t f32 = sign | ((exp_val - 15 + 127) << 23) | (mant << 13);
            memcpy(&val, &f32, 4);
        }
        if (val > best_val) {
            best_val = val;
            best_id = i;
        }
    }

    free(host_logits);
    return best_id;
}

// --- Text Generation ---

int moondream2_generate(Moondream2Model* model, const DeviceInfo* device,
                        const char* prompt, int max_new_tokens,
                        const char* vocab_path) {
    if (!model->initialized) {
        fprintf(stderr, "Error: model not initialized\n");
        return -1;
    }

    // Load tokenizer
    TokenizerVocab vocab;
    memset(&vocab, 0, sizeof(vocab));
    bool has_tokenizer = false;

    if (vocab_path) {
        has_tokenizer = tokenizer_load_from_file(&vocab, vocab_path);
        if (has_tokenizer) {
            printf("Tokenizer loaded: %d tokens\n", vocab.vocab_size);
        } else {
            fprintf(stderr, "Warning: failed to load tokenizer from %s\n", vocab_path);
        }
    }

    // Encode prompt
    int prompt_tokens[2048];
    int prompt_len = 0;

    if (has_tokenizer && prompt) {
        prompt_len = tokenizer_encode(&vocab, prompt, prompt_tokens, 2048);
        printf("Prompt encoded: %d tokens\n", prompt_len);
    } else if (prompt) {
        // Fallback: use raw bytes as token IDs (only works for testing)
        fprintf(stderr, "Warning: no tokenizer — using raw byte encoding\n");
        const char* p = prompt;
        while (*p && prompt_len < 2048) {
            prompt_tokens[prompt_len++] = (unsigned char)*p++;
        }
    }

    if (prompt_len == 0) {
        fprintf(stderr, "Error: empty prompt\n");
        if (has_tokenizer) tokenizer_free(&vocab);
        return -1;
    }

    // Reset KV-cache for fresh generation
    moondream2_reset_cache(model);

    printf("\n--- Generation ---\n");
    if (has_tokenizer && prompt) {
        printf("Prompt: %s\n", prompt);
    }
    printf("Output: ");
    fflush(stdout);

    struct timespec t_start, t_prefill_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Prefill: process all prompt tokens at once
    cl_mem logits = moondream2_forward(model, device, prompt_tokens, prompt_len);
    if (!logits) {
        fprintf(stderr, "Error: prefill forward pass failed\n");
        if (has_tokenizer) tokenizer_free(&vocab);
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_prefill_end);

    // Decode loop: generate one token at a time
    int generated = 0;
    int next_token = argmax_logits(device, logits, model->config.vocab_size);
    clReleaseMemObject(logits);

    if (next_token < 0) {
        fprintf(stderr, "Error: argmax failed\n");
        if (has_tokenizer) tokenizer_free(&vocab);
        return -1;
    }

    for (int i = 0; i < max_new_tokens; i++) {
        // Check for EOS
        if (has_tokenizer && next_token == vocab.eos_id) {
            break;
        }

        // Print the generated token
        if (has_tokenizer) {
            const char* tok_str = tokenizer_decode(&vocab, next_token);
            if (tok_str) {
                printf("%s", tok_str);
                fflush(stdout);
            }
        } else {
            if (next_token >= 32 && next_token < 127) {
                putchar(next_token);
                fflush(stdout);
            } else {
                printf("[%d]", next_token);
                fflush(stdout);
            }
        }

        generated++;

        // Forward pass with single token
        int token_arr[1] = { next_token };
        logits = moondream2_forward(model, device, token_arr, 1);
        if (!logits) {
            fprintf(stderr, "\nError: decode forward pass failed at token %d\n", i);
            break;
        }

        next_token = argmax_logits(device, logits, model->config.vocab_size);
        clReleaseMemObject(logits);

        if (next_token < 0) {
            fprintf(stderr, "\nError: argmax failed at token %d\n", i);
            break;
        }
    }

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double prefill_ms = (t_prefill_end.tv_sec - t_start.tv_sec) * 1000.0 +
                        (t_prefill_end.tv_nsec - t_start.tv_nsec) / 1e6;
    double total_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e6;
    double decode_ms = total_ms - prefill_ms;
    double tok_per_sec = (generated > 0) ? (generated / (decode_ms / 1000.0)) : 0.0;

    printf("\n\n--- Stats ---\n");
    printf("  Prompt tokens:  %d\n", prompt_len);
    printf("  Generated:      %d tokens\n", generated);
    printf("  Prefill:        %.1f ms (%.1f ms/token)\n",
           prefill_ms, prompt_len > 0 ? prefill_ms / prompt_len : 0.0);
    printf("  Decode:         %.1f ms (%.1f tok/s)\n", decode_ms, tok_per_sec);
    printf("  Total:          %.1f ms\n", total_ms);

    if (has_tokenizer) tokenizer_free(&vocab);
    return generated;
}

// --- Reset ---

void moondream2_reset_cache(Moondream2Model* model) {
    model->kv_cache.length = 0;
}

// --- GPU Resource Cleanup ---

static void release_mem(cl_mem* m) {
    if (*m) { clReleaseMemObject(*m); *m = nullptr; }
}

void moondream2_release_gpu(Moondream2Model* model) {
    Moondream2Weights* w = &model->gpu_weights;

    release_mem(&w->token_embed);
    release_mem(&w->final_norm_weight);
    release_mem(&w->lm_head_weight);
    release_mem(&w->cos_table);
    release_mem(&w->sin_table);

    if (w->layers) {
        for (int i = 0; i < w->num_layers; i++) {
            TransformerLayerWeights* lw = &w->layers[i];
            release_mem(&lw->q_proj_weight);
            release_mem(&lw->k_proj_weight);
            release_mem(&lw->v_proj_weight);
            release_mem(&lw->o_proj_weight);
            release_mem(&lw->gate_proj_weight);
            release_mem(&lw->up_proj_weight);
            release_mem(&lw->down_proj_weight);
            release_mem(&lw->input_norm_weight);
            release_mem(&lw->post_norm_weight);
        }
        free(w->layers);
        w->layers = nullptr;
        w->num_layers = 0;
    }

    release_mem(&model->kv_cache.k_cache);
    release_mem(&model->kv_cache.v_cache);
    model->kv_cache.length = 0;

    release_mem(&model->scratch_a);
    release_mem(&model->scratch_b);
    release_mem(&model->scratch_q);
    release_mem(&model->scratch_k);
    release_mem(&model->scratch_v);
    release_mem(&model->scratch_attn);
    release_mem(&model->scratch_gate);
    release_mem(&model->scratch_up);
}

// --- Load / Destroy ---

bool moondream2_load(Moondream2Model* model, const DeviceInfo* device,
                     const char* gguf_path, const char* kernel_dir) {
    memset(model, 0, sizeof(Moondream2Model));
    model->config = Moondream2Config{};

    // Load GGUF weights
    printf("Loading model weights from: %s\n", gguf_path);
    if (!gguf_open(&model->weights, gguf_path)) {
        fprintf(stderr, "Error: Failed to load GGUF file: %s\n", gguf_path);
        return false;
    }

    printf("Model loaded: %llu tensors\n",
           (unsigned long long)model->weights.tensor_count);
    gguf_print_tensors(&model->weights);

    // Build kernel programs
    const char* build_opts = "-cl-mad-enable -cl-fast-relaxed-math";

    if (kernel_dir) {
        printf("Building kernels from: %s\n", kernel_dir);
        model->gemm_program       = load_kernel(device, kernel_dir, "gemm.cl", build_opts);
        model->attention_program  = load_kernel(device, kernel_dir, "attention.cl", build_opts);
        model->norm_program       = load_kernel(device, kernel_dir, "layernorm.cl", build_opts);
        model->activation_program = load_kernel(device, kernel_dir, "activations.cl", build_opts);
        model->rope_program       = load_kernel(device, kernel_dir, "rope.cl", build_opts);
        model->embedding_program  = load_kernel(device, kernel_dir, "embedding.cl", build_opts);
        model->vision_program     = load_kernel(device, kernel_dir, "vision.cl", build_opts);
    }

    // Print model configuration
    printf("\n=== Moondream2 Configuration ===\n");
    printf("  Vision encoder: SigLIP (%d layers, dim=%d, heads=%d)\n",
           model->config.vision_layers, model->config.vision_dim, model->config.vision_heads);
    printf("  LLM decoder:    Phi-1.5 (%d layers, dim=%d, heads=%d)\n",
           model->config.llm_layers, model->config.llm_dim, model->config.llm_heads);
    printf("  Vocab size:     %d, Max seq len: %d\n",
           model->config.vocab_size, model->config.max_seq_len);

    // Upload weights to GPU
    if (!moondream2_upload_weights(model, device)) {
        fprintf(stderr, "Error: Failed to upload weights\n");
        moondream2_destroy(model);
        return false;
    }

    // Initialize RoPE tables
    if (!moondream2_init_rope(model, device)) {
        fprintf(stderr, "Error: Failed to initialize RoPE tables\n");
        moondream2_destroy(model);
        return false;
    }

    // Allocate scratch and KV-cache buffers
    if (!moondream2_alloc_buffers(model, device)) {
        fprintf(stderr, "Error: Failed to allocate buffers\n");
        moondream2_destroy(model);
        return false;
    }

    model->initialized = true;
    printf("\n=== Model ready for inference ===\n\n");
    return true;
}

void moondream2_destroy(Moondream2Model* model) {
    if (!model) return;

    moondream2_release_gpu(model);

    if (model->gemm_program)       clReleaseProgram(model->gemm_program);
    if (model->attention_program)  clReleaseProgram(model->attention_program);
    if (model->norm_program)       clReleaseProgram(model->norm_program);
    if (model->activation_program) clReleaseProgram(model->activation_program);
    if (model->rope_program)       clReleaseProgram(model->rope_program);
    if (model->embedding_program)  clReleaseProgram(model->embedding_program);
    if (model->vision_program)     clReleaseProgram(model->vision_program);

    gguf_close(&model->weights);
    model->initialized = false;
}

} // namespace mgpu
