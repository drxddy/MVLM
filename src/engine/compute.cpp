#include "compute.h"

#include <cstdio>

#ifdef MGPU_ANDROID
#include <android/log.h>
#define MGPU_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "MGPU", __VA_ARGS__)
#else
#define MGPU_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define CL_CHECK_NULL(err) do { \
    if ((err) != CL_SUCCESS) { \
        MGPU_ERR("OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
        return nullptr; \
    } \
} while(0)

namespace mgpu {

// Round up `value` to the next multiple of `multiple`
static inline size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// --- GEMM / GEMV ---

cl_event dispatch_gemm_naive(const DeviceInfo* dev, cl_program program,
                             cl_mem A, cl_mem B, cl_mem C,
                             int M, int N, int K) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "gemm_naive", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    if (err != CL_SUCCESS) {
        MGPU_ERR("gemm_naive: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    size_t global[2] = { round_up((size_t)M, 16), round_up((size_t)N, 16) };
    size_t local[2]  = { 16, 16 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_gemm_tiled(const DeviceInfo* dev, cl_program program,
                             cl_mem A, cl_mem B, cl_mem C,
                             int M, int N, int K) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "gemm_tiled", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    if (err != CL_SUCCESS) {
        MGPU_ERR("gemm_tiled: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t TILE_M = 8;
    const size_t TILE_N = 8;
    size_t global[2] = { round_up((size_t)M, TILE_M), round_up((size_t)N, TILE_N) };
    size_t local[2]  = { TILE_M, TILE_N };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_gemm_image(const DeviceInfo* dev, cl_program program,
                             cl_mem A, cl_mem B_img, cl_mem C,
                             int M, int N, int K) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "gemm_image", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_img);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    if (err != CL_SUCCESS) {
        MGPU_ERR("gemm_image: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    // Each work-item computes 4 output columns; global_id(1) = col4
    size_t n_div4 = ((size_t)N + 3) / 4;
    size_t global[2] = { round_up((size_t)M, 16), round_up(n_div4, 4) };
    size_t local[2]  = { 16, 4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_gemv(const DeviceInfo* dev, cl_program program,
                       cl_mem x, cl_mem W_img, cl_mem y,
                       int N, int K) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "gemv", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &W_img);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &y);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &K);
    if (err != CL_SUCCESS) {
        MGPU_ERR("gemv: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    // Each workgroup handles 4 output elements (one image column)
    const size_t WG_SIZE = 256;
    size_t num_groups = ((size_t)N + 3) / 4;
    size_t global[1] = { num_groups * WG_SIZE };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Layer Normalization ---

cl_event dispatch_rms_norm(const DeviceInfo* dev, cl_program program,
                           cl_mem input, cl_mem output, cl_mem weight,
                           int num_rows, int hidden_size, float eps) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "rms_norm", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &hidden_size);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &eps);
    if (err != CL_SUCCESS) {
        MGPU_ERR("rms_norm: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t global[1] = { (size_t)num_rows * WG_SIZE };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Activations ---

cl_event dispatch_silu(const DeviceInfo* dev, cl_program program,
                       cl_mem input, cl_mem output, int n) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "silu", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        MGPU_ERR("silu: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t num_wis = ((size_t)n + 3) / 4;
    size_t global[1] = { round_up(num_wis, WG_SIZE) };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_gelu(const DeviceInfo* dev, cl_program program,
                       cl_mem input, cl_mem output, int n) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        MGPU_ERR("gelu: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t num_wis = ((size_t)n + 3) / 4;
    size_t global[1] = { round_up(num_wis, WG_SIZE) };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_softmax(const DeviceInfo* dev, cl_program program,
                          cl_mem input, cl_mem output,
                          int seq_len, int num_elements) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "softmax", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &seq_len);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &num_elements);
    if (err != CL_SUCCESS) {
        MGPU_ERR("softmax: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t global[1] = { (size_t)seq_len * WG_SIZE };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_silu_gate_multiply(const DeviceInfo* dev, cl_program program,
                                     cl_mem gate, cl_mem up, cl_mem output,
                                     int n) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "silu_gate_multiply", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gate);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &up);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        MGPU_ERR("silu_gate_multiply: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t num_wis = ((size_t)n + 3) / 4;
    size_t global[1] = { round_up(num_wis, WG_SIZE) };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Attention ---

cl_event dispatch_attention_prefill(const DeviceInfo* dev, cl_program program,
                                    cl_mem Q, cl_mem K, cl_mem V, cl_mem output,
                                    int seq_len, int num_heads, int head_dim) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "attention_prefill", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &Q);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &K);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &V);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &seq_len);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &num_heads);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &head_dim);
    if (err != CL_SUCCESS) {
        MGPU_ERR("attention_prefill: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t global[1] = { (size_t)seq_len * (size_t)num_heads * WG_SIZE };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_attention_decode(const DeviceInfo* dev, cl_program program,
                                   cl_mem Q, cl_mem K_cache, cl_mem V_cache,
                                   cl_mem output,
                                   int cache_len, int num_heads, int head_dim) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "attention_decode", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &Q);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &K_cache);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_cache);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &cache_len);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &num_heads);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &head_dim);
    if (err != CL_SUCCESS) {
        MGPU_ERR("attention_decode: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t global[1] = { (size_t)num_heads * WG_SIZE };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- RoPE ---

cl_event dispatch_rope_apply(const DeviceInfo* dev, cl_program program,
                             cl_mem qk, cl_mem cos_table, cl_mem sin_table,
                             int seq_len, int num_heads, int head_dim,
                             int offset) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "rope_apply", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &qk);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cos_table);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &sin_table);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &seq_len);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &num_heads);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &head_dim);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &offset);
    if (err != CL_SUCCESS) {
        MGPU_ERR("rope_apply: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    // 3D dispatch: (seq_len, num_heads, head_dim/2)
    size_t global[3] = { (size_t)seq_len, (size_t)num_heads, (size_t)(head_dim / 2) };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 3, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Embedding ---

cl_event dispatch_embedding_lookup(const DeviceInfo* dev, cl_program program,
                                   cl_mem embed_table, cl_mem token_ids,
                                   cl_mem output,
                                   int seq_len, int embed_dim) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "embedding_lookup", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &embed_table);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &token_ids);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &embed_dim);
    if (err != CL_SUCCESS) {
        MGPU_ERR("embedding_lookup: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    // 2D dispatch: (seq_len, ceil(embed_dim/4))
    size_t dim4 = ((size_t)embed_dim + 3) / 4;
    size_t global[2] = { (size_t)seq_len, dim4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Vision ---

cl_event dispatch_preprocess_image(const DeviceInfo* dev, cl_program program,
                                   cl_mem input_image, cl_mem output,
                                   int target_h, int target_w,
                                   float mean_r, float mean_g, float mean_b,
                                   float std_r, float std_g, float std_b) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "preprocess_image", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &target_h);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &target_w);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &mean_r);
    err |= clSetKernelArg(kernel, 5, sizeof(float), &mean_g);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &mean_b);
    err |= clSetKernelArg(kernel, 7, sizeof(float), &std_r);
    err |= clSetKernelArg(kernel, 8, sizeof(float), &std_g);
    err |= clSetKernelArg(kernel, 9, sizeof(float), &std_b);
    if (err != CL_SUCCESS) {
        MGPU_ERR("preprocess_image: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    // 2D dispatch: (target_w, target_h)
    size_t global[2] = { (size_t)target_w, (size_t)target_h };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

cl_event dispatch_patch_embed(const DeviceInfo* dev, cl_program program,
                              cl_mem image, cl_mem proj_weight,
                              cl_mem proj_bias, cl_mem patches,
                              int C, int H, int W,
                              int patch_h, int patch_w, int embed_dim) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "patch_embed", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &proj_weight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &proj_bias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &patches);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &C);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &H);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &W);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &patch_h);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &patch_w);
    err |= clSetKernelArg(kernel, 9, sizeof(int), &embed_dim);
    if (err != CL_SUCCESS) {
        MGPU_ERR("patch_embed: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    // 2D dispatch: (num_patches, ceil(embed_dim/4))
    size_t num_patches = (size_t)(H / patch_h) * (size_t)(W / patch_w);
    size_t embed4 = ((size_t)embed_dim + 3) / 4;
    size_t global[2] = { num_patches, embed4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Vision RMSNorm ---

cl_event dispatch_vision_rmsnorm(const DeviceInfo* dev, cl_program program,
                                 cl_mem input, cl_mem output, cl_mem weight,
                                 int num_patches, int hidden_dim, float eps) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "vision_rmsnorm", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &num_patches);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &hidden_dim);
    err |= clSetKernelArg(kernel, 5, sizeof(float), &eps);
    if (err != CL_SUCCESS) {
        MGPU_ERR("vision_rmsnorm: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    size_t hidden4 = ((size_t)hidden_dim + 3) / 4;
    size_t global[2] = { (size_t)num_patches, hidden4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Vision Attention ---

cl_event dispatch_vision_attention(const DeviceInfo* dev, cl_program program,
                                    cl_mem input, cl_mem qkv_weight,
                                    cl_mem qkv_bias, cl_mem out_weight,
                                    cl_mem out_bias, cl_mem output,
                                    int num_patches, int hidden_dim,
                                    int num_heads, float scale) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "vision_attention", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &qkv_weight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &qkv_bias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_weight);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_bias);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &num_patches);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &hidden_dim);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &num_heads);
    err |= clSetKernelArg(kernel, 9, sizeof(float), &scale);
    if (err != CL_SUCCESS) {
        MGPU_ERR("vision_attention: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    size_t hidden4 = ((size_t)hidden_dim + 3) / 4;
    size_t global[2] = { (size_t)num_patches, hidden4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Vision MLP ---

cl_event dispatch_vision_mlp(const DeviceInfo* dev, cl_program program,
                              cl_mem input, cl_mem gate_weight,
                              cl_mem up_weight, cl_mem down_weight,
                              cl_mem output,
                              int num_patches, int hidden_dim, int intermediate) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "vision_mlp", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gate_weight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &up_weight);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &down_weight);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &num_patches);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &hidden_dim);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &intermediate);
    if (err != CL_SUCCESS) {
        MGPU_ERR("vision_mlp: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    size_t hidden4 = ((size_t)hidden_dim + 3) / 4;
    size_t global[2] = { (size_t)num_patches, hidden4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Vision Projection ---

cl_event dispatch_vision_proj(const DeviceInfo* dev, cl_program program,
                              cl_mem visual_tokens, cl_mem proj_weight,
                              cl_mem proj_bias, cl_mem output,
                              int num_patches, int vision_dim, int llm_dim) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "vision_proj", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &visual_tokens);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &proj_weight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &proj_bias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &num_patches);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &vision_dim);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &llm_dim);
    if (err != CL_SUCCESS) {
        MGPU_ERR("vision_proj: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    size_t llm4 = ((size_t)llm_dim + 3) / 4;
    size_t global[2] = { (size_t)num_patches, llm4 };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 2, nullptr,
                                 global, nullptr, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

// --- Vector Add ---

cl_event dispatch_vector_add(const DeviceInfo* dev, cl_program program,
                             cl_mem a, cl_mem b, cl_mem output, int n) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CL_CHECK_NULL(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        MGPU_ERR("vector_add: failed to set kernel args (err=%d)\n", err);
        clReleaseKernel(kernel);
        return nullptr;
    }

    const size_t WG_SIZE = 256;
    size_t num_wis = ((size_t)n + 3) / 4;
    size_t global[1] = { round_up(num_wis, WG_SIZE) };
    size_t local[1]  = { WG_SIZE };

    cl_event event;
    err = clEnqueueNDRangeKernel(dev->queue, kernel, 1, nullptr,
                                 global, local, 0, nullptr, &event);
    clReleaseKernel(kernel);
    CL_CHECK_NULL(err);
    return event;
}

} // namespace mgpu
