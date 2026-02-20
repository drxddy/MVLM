#include "../src/engine/device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

struct BenchConfig {
    const char* name;
    int M;
    int K;
    int N;
};

static const BenchConfig bench_configs[] = {
    { "small",       256,  256,  256  },
    { "medium",      2048, 2048, 2048 },
    { "decode_gemv", 1,    2048, 2048 },
    { "llm_ffn",     1,    2048, 8192 },
    { "prefill_32",  32,   2048, 2048 },
};

static const int num_configs = sizeof(bench_configs) / sizeof(bench_configs[0]);

// Fill buffer with random fp16 values (stored as uint16_t)
static void fill_random_fp16(uint16_t* buf, int count) {
    for (int i = 0; i < count; i++) {
        // Simple random float in [-1, 1], convert to fp16
        // Use a rough fp16 encoding: sign=0, exponent=15 (bias), small mantissa
        float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        // Quick and dirty fp32->fp16 via bit manipulation
        uint32_t f32;
        memcpy(&f32, &val, 4);
        uint32_t sign = (f32 >> 16) & 0x8000;
        int32_t exp_val = ((f32 >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (f32 >> 13) & 0x3FF;
        if (exp_val <= 0) {
            buf[i] = (uint16_t)sign; // flush to zero
        } else if (exp_val >= 31) {
            buf[i] = (uint16_t)(sign | 0x7C00); // inf
        } else {
            buf[i] = (uint16_t)(sign | (exp_val << 10) | mant);
        }
    }
}

static double compute_gflops(int M, int K, int N, double time_ms) {
    double flops = 2.0 * M * K * N;
    return flops / (time_ms * 1e6); // GFLOPS
}

struct KernelVariant {
    const char* name;
    const char* kernel_func;
};

static const KernelVariant kernel_variants[] = {
    { "naive",  "gemm_naive"  },
    { "tiled",  "gemm_tiled"  },
    { "image",  "gemm_image"  },
    { "gemv",   "gemv"        },
};

static const int num_variants = sizeof(kernel_variants) / sizeof(kernel_variants[0]);

static bool run_gemm_bench(mgpu::DeviceInfo* device, cl_program program,
                           const KernelVariant* variant, const BenchConfig* config,
                           int warmup_iters, int bench_iters) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, variant->kernel_func, &err);
    if (err != CL_SUCCESS) {
        printf("  %-8s  %-12s  (kernel not found, skipping)\n", variant->name, config->name);
        return false;
    }

    int M = config->M;
    int K = config->K;
    int N = config->N;

    size_t size_a = (size_t)M * K * sizeof(uint16_t);
    size_t size_b = (size_t)K * N * sizeof(uint16_t);
    size_t size_c = (size_t)M * N * sizeof(uint16_t);

    // Allocate host buffers
    uint16_t* h_a = (uint16_t*)malloc(size_a);
    uint16_t* h_b = (uint16_t*)malloc(size_b);
    if (!h_a || !h_b) {
        fprintf(stderr, "Error: Failed to allocate host buffers\n");
        free(h_a);
        free(h_b);
        clReleaseKernel(kernel);
        return false;
    }
    fill_random_fp16(h_a, M * K);
    fill_random_fp16(h_b, K * N);

    // Create device buffers
    cl_mem d_a = clCreateBuffer(device->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                size_a, h_a, &err);
    cl_mem d_b = clCreateBuffer(device->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                size_b, h_b, &err);
    cl_mem d_c = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, size_c, nullptr, &err);

    if (!d_a || !d_b || !d_c) {
        fprintf(stderr, "Error: Failed to create device buffers\n");
        if (d_a) clReleaseMemObject(d_a);
        if (d_b) clReleaseMemObject(d_b);
        if (d_c) clReleaseMemObject(d_c);
        free(h_a);
        free(h_b);
        clReleaseKernel(kernel);
        return false;
    }

    // Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &K);
    clSetKernelArg(kernel, 5, sizeof(int), &N);

    size_t global_work_size[2] = { (size_t)N, (size_t)M };
    size_t local_work_size[2] = { 16, 16 };

    // Clamp local work size for small matrices or GEMV
    if (M == 1) {
        global_work_size[0] = (size_t)N;
        global_work_size[1] = 1;
        local_work_size[0] = 256;
        local_work_size[1] = 1;
        // Round up global to multiple of local
        global_work_size[0] = ((N + 255) / 256) * 256;
    } else {
        // Round up to multiples of local work size
        global_work_size[0] = ((N + 15) / 16) * 16;
        global_work_size[1] = ((M + 15) / 16) * 16;
    }

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        clEnqueueNDRangeKernel(device->queue, kernel, 2, nullptr,
                               global_work_size, local_work_size, 0, nullptr, nullptr);
    }
    clFinish(device->queue);

    // Benchmark with profiling events
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;

    for (int i = 0; i < bench_iters; i++) {
        cl_event event;
        err = clEnqueueNDRangeKernel(device->queue, kernel, 2, nullptr,
                                      global_work_size, local_work_size,
                                      0, nullptr, &event);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Kernel enqueue failed (err=%d)\n", err);
            break;
        }
        clWaitForEvents(1, &event);

        cl_ulong t_start, t_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
        clReleaseEvent(event);

        double ms = (double)(t_end - t_start) / 1e6;
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    double avg_ms = total_ms / bench_iters;
    double gflops = compute_gflops(M, K, N, avg_ms);

    // Estimate theoretical peak: compute_units * clock * 2 (FMA) * vector_width
    // This is a rough estimate; actual peak depends on architecture
    double peak_gflops = (double)device->compute_units * device->max_clock_freq * 2.0 / 1000.0;
    if (device->has_fp16) peak_gflops *= 2.0; // fp16 doubles throughput
    double efficiency = (peak_gflops > 0) ? (gflops / peak_gflops * 100.0) : 0.0;

    printf("  %-8s  %-12s  %4dx%4dx%4d  %8.3f ms  %8.2f GFLOPS  %5.1f%% eff  (min=%.3f max=%.3f)\n",
           variant->name, config->name, M, K, N,
           avg_ms, gflops, efficiency, min_ms, max_ms);

    // Cleanup
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    free(h_a);
    free(h_b);
    clReleaseKernel(kernel);
    return true;
}

int main(int argc, char** argv) {
    const char* kernel_file = "src/kernels/gemm.cl";
    int warmup_iters = 5;
    int bench_iters = 20;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kernels") == 0 && i + 1 < argc) {
            kernel_file = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            bench_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--kernels <gemm.cl>] [--warmup N] [--iters N]\n", argv[0]);
            return 0;
        }
    }

    printf("=== MGPU GEMM Benchmark ===\n\n");

    srand((unsigned)time(nullptr));

    mgpu::DeviceInfo device;
    if (!mgpu::init_device(&device)) {
        fprintf(stderr, "Error: Failed to initialize OpenCL device\n");
        return 1;
    }
    mgpu::print_device_info(&device);

    // Build GEMM program
    printf("\nBuilding GEMM kernels from: %s\n", kernel_file);
    const char* build_opts = "-cl-mad-enable -cl-fast-relaxed-math";
    cl_program program = mgpu::build_program_from_file(&device, kernel_file, build_opts);
    if (!program) {
        fprintf(stderr, "Error: Failed to build GEMM kernels\n");
        fprintf(stderr, "Make sure %s exists with kernel functions.\n", kernel_file);
        mgpu::destroy_device(&device);
        return 1;
    }

    // Estimate theoretical peak
    double peak_gflops = (double)device.compute_units * device.max_clock_freq * 2.0 / 1000.0;
    if (device.has_fp16) peak_gflops *= 2.0;
    printf("\nEstimated peak FP16: %.1f GFLOPS (rough estimate)\n", peak_gflops);
    printf("Warmup: %d iters, Bench: %d iters\n\n", warmup_iters, bench_iters);

    printf("  %-8s  %-12s  %-14s  %8s    %8s        %5s       %s\n",
           "Kernel", "Config", "Size", "Time", "GFLOPS", "Eff", "Min/Max");
    printf("  %-8s  %-12s  %-14s  %8s    %8s        %5s       %s\n",
           "--------", "------------", "--------------", "--------",
           "--------", "-----", "-------------------");

    for (int v = 0; v < num_variants; v++) {
        for (int c = 0; c < num_configs; c++) {
            run_gemm_bench(&device, program, &kernel_variants[v], &bench_configs[c],
                          warmup_iters, bench_iters);
        }
        printf("\n");
    }

    clReleaseProgram(program);
    mgpu::destroy_device(&device);

    printf("Benchmark complete.\n");
    return 0;
}
