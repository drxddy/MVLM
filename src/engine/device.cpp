#include "device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef MGPU_ANDROID
#include <android/log.h>
#define MGPU_LOG(...) __android_log_print(ANDROID_LOG_INFO, "MGPU", __VA_ARGS__)
#define MGPU_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "MGPU", __VA_ARGS__)
#else
#define MGPU_LOG(...) fprintf(stdout, __VA_ARGS__)
#define MGPU_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

// Qualcomm extension constants (not always in standard headers)
#ifndef CL_CONTEXT_PERF_HINT_QCOM
#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#endif
#ifndef CL_PERF_HINT_HIGH_QCOM
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#endif
#ifndef CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM
#define CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM 0x40CB
#endif

#define CL_CHECK(err) do { \
    if ((err) != CL_SUCCESS) { \
        MGPU_ERR("OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define CL_CHECK_NULL(err) do { \
    if ((err) != CL_SUCCESS) { \
        MGPU_ERR("OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
        return nullptr; \
    } \
} while(0)

namespace mgpu {

bool has_extension(cl_device_id device, const char* ext_name) {
    size_t ext_size = 0;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    if (err != CL_SUCCESS || ext_size == 0) return false;

    char* extensions = (char*)malloc(ext_size);
    if (!extensions) return false;

    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, extensions, nullptr);
    if (err != CL_SUCCESS) {
        free(extensions);
        return false;
    }

    // Search for exact extension name (avoid substring matches)
    const char* start = extensions;
    size_t name_len = strlen(ext_name);
    bool found = false;
    while (*start) {
        // Skip leading spaces
        while (*start == ' ') start++;
        if (*start == '\0') break;

        const char* end = start;
        while (*end && *end != ' ') end++;

        size_t token_len = (size_t)(end - start);
        if (token_len == name_len && strncmp(start, ext_name, name_len) == 0) {
            found = true;
            break;
        }
        start = end;
    }

    free(extensions);
    return found;
}

static cl_device_id find_gpu_device(cl_platform_id* out_platform) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return nullptr;

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms) return nullptr;

    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
    if (err != CL_SUCCESS) {
        free(platforms);
        return nullptr;
    }

    cl_device_id best_device = nullptr;
    cl_platform_id best_platform = nullptr;
    bool found_adreno = false;

    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        if (!devices) continue;

        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }

        for (cl_uint d = 0; d < num_devices; d++) {
            char name[256] = {0};
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(name), name, nullptr);

            // Prefer Adreno GPU
            if (strstr(name, "Adreno") || strstr(name, "QUALCOMM") || strstr(name, "adreno")) {
                best_device = devices[d];
                best_platform = platforms[p];
                found_adreno = true;
                break;
            }

            // Take the first GPU as fallback
            if (!best_device) {
                best_device = devices[d];
                best_platform = platforms[p];
            }
        }

        free(devices);
        if (found_adreno) break;
    }

    if (out_platform) *out_platform = best_platform;
    free(platforms);
    return best_device;
}

bool init_device(DeviceInfo* info) {
    memset(info, 0, sizeof(DeviceInfo));

    info->device = find_gpu_device(&info->platform);
    if (!info->device) {
        MGPU_ERR("No GPU device found\n");
        return false;
    }

    cl_int err;

    // Query device name and driver version
    err = clGetDeviceInfo(info->device, CL_DEVICE_NAME,
                          sizeof(info->device_name), info->device_name, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_VENDOR,
                          sizeof(info->vendor), info->vendor, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DRIVER_VERSION,
                          sizeof(info->driver_version), info->driver_version, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_VERSION,
                          sizeof(info->opencl_version), info->opencl_version, nullptr);
    CL_CHECK(err);

    // Query capabilities
    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(info->compute_units), &info->compute_units, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(info->max_workgroup_size), &info->max_workgroup_size, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_LOCAL_MEM_SIZE,
                          sizeof(info->local_mem_size), &info->local_mem_size, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(info->global_mem_size), &info->global_mem_size, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                          sizeof(info->max_clock_freq), &info->max_clock_freq, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(info->max_alloc_size), &info->max_alloc_size, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                          sizeof(info->max_constant_size), &info->max_constant_size, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                          sizeof(info->max_work_item_dims), &info->max_work_item_dims, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(info->max_work_item_sizes), info->max_work_item_sizes, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_IMAGE_SUPPORT,
                          sizeof(info->image_support), &info->image_support, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                          sizeof(info->preferred_vector_width_half), &info->preferred_vector_width_half, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                          sizeof(info->native_vector_width_half), &info->native_vector_width_half, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                          sizeof(info->max_image2d_width), &info->max_image2d_width, nullptr);
    CL_CHECK(err);

    err = clGetDeviceInfo(info->device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                          sizeof(info->max_image2d_height), &info->max_image2d_height, nullptr);
    CL_CHECK(err);

    // Check extensions
    info->has_fp16 = has_extension(info->device, "cl_khr_fp16");
    info->has_subgroups = has_extension(info->device, "cl_khr_subgroups");
    info->has_qcom_subgroup_shuffle = has_extension(info->device, "cl_qcom_subgroup_shuffle");
    info->has_qcom_onchip_global_memory = has_extension(info->device, "cl_qcom_onchip_global_memory");
    info->has_qcom_recordable_queues = has_extension(info->device, "cl_qcom_recordable_queues");
    info->has_qcom_perf_hint = has_extension(info->device, "cl_qcom_perf_hint");
    info->has_qcom_dot_product8 = has_extension(info->device, "cl_qcom_dot_product8");
    info->has_qcom_ahb = has_extension(info->device, "cl_qcom_android_ahardwarebuffer_host_ptr");
    info->has_int_dot_product = has_extension(info->device, "cl_khr_integer_dot_product");
    info->has_image = (info->image_support == CL_TRUE);

    // Query preferred subgroup size (if subgroups supported)
    info->preferred_subgroup_size = 0;
    if (info->has_subgroups) {
        // CL_DEVICE_SUB_GROUP_SIZES_INTEL is not standard; use a generic query
        // On Adreno, wave size is typically 64 or 128
        // Try querying via cl_khr_subgroups: get all supported subgroup sizes, pick first
        size_t sizes_ret = 0;
        err = clGetDeviceInfo(info->device, 0x4108 /* CL_DEVICE_SUB_GROUP_SIZES_INTEL fallback */,
                              0, nullptr, &sizes_ret);
        if (err == CL_SUCCESS && sizes_ret > 0) {
            size_t* sizes = (size_t*)malloc(sizes_ret);
            if (sizes) {
                err = clGetDeviceInfo(info->device, 0x4108, sizes_ret, sizes, nullptr);
                if (err == CL_SUCCESS) {
                    // Pick the largest supported subgroup size (Adreno prefers larger waves)
                    size_t count = sizes_ret / sizeof(size_t);
                    info->preferred_subgroup_size = sizes[0];
                    for (size_t i = 1; i < count; i++) {
                        if (sizes[i] > info->preferred_subgroup_size)
                            info->preferred_subgroup_size = sizes[i];
                    }
                }
                free(sizes);
            }
        }
        // Fallback: typical Adreno wave size
        if (info->preferred_subgroup_size == 0)
            info->preferred_subgroup_size = 64;
    }

    // Query on-chip global memory size (Qualcomm extension)
    info->onchip_global_mem_size = 0;
    if (info->has_qcom_onchip_global_memory) {
        err = clGetDeviceInfo(info->device, CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM,
                              sizeof(info->onchip_global_mem_size),
                              &info->onchip_global_mem_size, nullptr);
        if (err != CL_SUCCESS) {
            MGPU_ERR("Warning: failed to query on-chip global memory size (err=%d)\n", err);
            info->onchip_global_mem_size = 0;
        }
    }

    // Create context — with perf hint if available
    if (info->has_qcom_perf_hint) {
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)info->platform,
            CL_CONTEXT_PERF_HINT_QCOM, (cl_context_properties)CL_PERF_HINT_HIGH_QCOM,
            0
        };
        info->context = clCreateContext(props, 1, &info->device, nullptr, nullptr, &err);
    } else {
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)info->platform,
            0
        };
        info->context = clCreateContext(props, 1, &info->device, nullptr, nullptr, &err);
    }
    CL_CHECK(err);

    // Create in-order command queue with profiling enabled
    cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;
    info->queue = clCreateCommandQueue(info->context, info->device, queue_props, &err);
    CL_CHECK(err);

    MGPU_LOG("MGPU: initialized device '%s'\n", info->device_name);
    return true;
}

void destroy_device(DeviceInfo* info) {
    if (info->queue) {
        clFinish(info->queue);
        clReleaseCommandQueue(info->queue);
        info->queue = nullptr;
    }
    if (info->context) {
        clReleaseContext(info->context);
        info->context = nullptr;
    }
    // Do NOT release device — we did not retain it
    info->device = nullptr;
}

void print_device_info(const DeviceInfo* info) {
    MGPU_LOG("╔══════════════════════════════════════════════════════════╗\n");
    MGPU_LOG("║               MGPU Device Information                   ║\n");
    MGPU_LOG("╠══════════════════════════════════════════════════════════╣\n");
    MGPU_LOG("║ Device Name          : %-33s ║\n", info->device_name);
    MGPU_LOG("║ Driver Version       : %-33s ║\n", info->driver_version);
    MGPU_LOG("║ Compute Units        : %-33u ║\n", info->compute_units);
    MGPU_LOG("║ Max Workgroup Size   : %-33zu ║\n", info->max_workgroup_size);
    MGPU_LOG("║ Local Memory         : %-30llu KB ║\n",
             (unsigned long long)(info->local_mem_size / 1024));
    MGPU_LOG("║ Global Memory        : %-30llu MB ║\n",
             (unsigned long long)(info->global_mem_size / (1024 * 1024)));
    MGPU_LOG("║ Max Clock Freq       : %-30u MHz ║\n", info->max_clock_freq);
    MGPU_LOG("║ Max Image2D          : %-15zu x %-16zu ║\n",
             info->max_image2d_width, info->max_image2d_height);
    MGPU_LOG("║ Preferred Subgroup   : %-33zu ║\n", info->preferred_subgroup_size);
    if (info->has_qcom_onchip_global_memory) {
        MGPU_LOG("║ On-Chip Global Mem   : %-30llu KB ║\n",
                 (unsigned long long)(info->onchip_global_mem_size / 1024));
    }
    MGPU_LOG("╠══════════════════════════════════════════════════════════╣\n");
    MGPU_LOG("║ Extension Support                                       ║\n");
    MGPU_LOG("╠══════════════════════════════════════════════════════════╣\n");

    #define PRINT_EXT(name, flag) \
        MGPU_LOG("║   %-35s : %s          ║\n", name, (flag) ? "YES" : " NO")

    PRINT_EXT("cl_khr_fp16", info->has_fp16);
    PRINT_EXT("cl_khr_subgroups", info->has_subgroups);
    PRINT_EXT("cl_khr_integer_dot_product", info->has_int_dot_product);
    PRINT_EXT("cl_qcom_subgroup_shuffle", info->has_qcom_subgroup_shuffle);
    PRINT_EXT("cl_qcom_onchip_global_memory", info->has_qcom_onchip_global_memory);
    PRINT_EXT("cl_qcom_recordable_queues", info->has_qcom_recordable_queues);
    PRINT_EXT("cl_qcom_perf_hint", info->has_qcom_perf_hint);
    PRINT_EXT("cl_qcom_dot_product8", info->has_qcom_dot_product8);
    PRINT_EXT("cl_qcom_android_ahb", info->has_qcom_ahb);

    #undef PRINT_EXT

    MGPU_LOG("╚══════════════════════════════════════════════════════════╝\n");
}

cl_program build_program_from_source(const DeviceInfo* info, const char* source,
                                     size_t length, const char* build_opts) {
    cl_int err;
    cl_program program = clCreateProgramWithSource(info->context, 1, &source, &length, &err);
    if (err != CL_SUCCESS) {
        MGPU_ERR("clCreateProgramWithSource failed (err=%d)\n", err);
        return nullptr;
    }

    // Build default options
    char full_opts[1024];
    int written = snprintf(full_opts, sizeof(full_opts),
                           "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math");
    if (build_opts && build_opts[0]) {
        snprintf(full_opts + written, sizeof(full_opts) - (size_t)written, " %s", build_opts);
    }

    err = clBuildProgram(program, 1, &info->device, full_opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        MGPU_ERR("clBuildProgram failed (err=%d)\n", err);

        // Print build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, info->device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        if (log_size > 0) {
            char* log = (char*)malloc(log_size + 1);
            if (log) {
                clGetProgramBuildInfo(program, info->device, CL_PROGRAM_BUILD_LOG,
                                     log_size, log, nullptr);
                log[log_size] = '\0';
                MGPU_ERR("Build log:\n%s\n", log);
                free(log);
            }
        }

        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}

cl_program build_program_from_file(const DeviceInfo* info, const char* filepath,
                                   const char* build_opts) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        MGPU_ERR("Failed to open kernel file: %s\n", filepath);
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        MGPU_ERR("Empty or invalid kernel file: %s\n", filepath);
        fclose(f);
        return nullptr;
    }

    char* source = (char*)malloc((size_t)file_size + 1);
    if (!source) {
        MGPU_ERR("Failed to allocate %ld bytes for kernel source\n", file_size);
        fclose(f);
        return nullptr;
    }

    size_t read_bytes = fread(source, 1, (size_t)file_size, f);
    fclose(f);
    source[read_bytes] = '\0';

    cl_program program = build_program_from_source(info, source, read_bytes, build_opts);
    free(source);
    return program;
}

} // namespace mgpu
