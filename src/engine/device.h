#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>

namespace mgpu {

struct DeviceInfo {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    char device_name[256];
    char vendor[256];
    char driver_version[256];
    char opencl_version[256];

    cl_uint compute_units;
    cl_uint max_clock_freq;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_ulong max_alloc_size;
    cl_ulong max_constant_size;
    size_t max_workgroup_size;
    cl_uint max_work_item_dims;
    size_t max_work_item_sizes[3];
    cl_uint preferred_vector_width_half;
    cl_uint native_vector_width_half;

    size_t max_image2d_width;
    size_t max_image2d_height;
    cl_bool image_support;

    // Extension flags
    bool has_fp16;
    bool has_subgroups;
    bool has_image;
    bool has_qcom_subgroup_shuffle;
    bool has_qcom_onchip_global_memory;
    bool has_qcom_recordable_queues;
    bool has_qcom_perf_hint;
    bool has_qcom_dot_product8;
    bool has_qcom_ahb;
    bool has_int_dot_product;

    // Adreno-specific capabilities
    size_t preferred_subgroup_size;
    cl_ulong onchip_global_mem_size;
};

// Initialize OpenCL device (prefers Adreno GPU, falls back to first GPU)
bool init_device(DeviceInfo* info);

// Print device info summary
void print_device_info(const DeviceInfo* info);

// Check if a specific extension is supported
bool has_extension(cl_device_id device, const char* ext_name);

// Build an OpenCL program from source string
cl_program build_program_from_source(const DeviceInfo* info, const char* source,
                                     size_t length, const char* build_opts);

// Build an OpenCL program from a .cl source file
cl_program build_program_from_file(const DeviceInfo* info, const char* filepath,
                                   const char* build_opts);

// Release all device resources
void destroy_device(DeviceInfo* info);

} // namespace mgpu
