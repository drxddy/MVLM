#include "memory.h"
#include "device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef MGPU_ANDROID
#include <android/log.h>
#define MGPU_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "MGPU", __VA_ARGS__)
#else
#define MGPU_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef CL_MEM_QUALCOMM_ONCHIP_GLOBAL
#define CL_MEM_QUALCOMM_ONCHIP_GLOBAL 0x10000000
#endif

#define CL_CHECK(err) do { \
    if ((err) != CL_SUCCESS) { \
        MGPU_ERR("OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

namespace mgpu {

cl_mem create_weight_image(const DeviceInfo* info, int rows, int cols, const cl_half* data) {
    // Pack weight matrix into RGBA half-float image
    // Each texel holds 4 fp16 values → image width = ceil(cols / 4)
    int padded_cols = ((cols + 3) / 4) * 4;
    int img_width = padded_cols / 4;
    int img_height = rows;

    cl_image_format fmt;
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_HALF_FLOAT;

    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)img_width;
    desc.image_height = (size_t)img_height;
    desc.image_row_pitch = 0;  // Let OpenCL decide

    // If cols is not divisible by 4, we need to create a padded copy
    cl_half* padded_data = nullptr;
    const cl_half* src_data = data;

    if (cols != padded_cols) {
        size_t padded_size = (size_t)img_height * (size_t)padded_cols;
        padded_data = (cl_half*)calloc(padded_size, sizeof(cl_half));
        if (!padded_data) {
            MGPU_ERR("Failed to allocate padded weight buffer (%d x %d)\n", rows, padded_cols);
            return nullptr;
        }
        for (int r = 0; r < rows; r++) {
            memcpy(padded_data + r * padded_cols,
                   data + r * cols,
                   (size_t)cols * sizeof(cl_half));
            // Remaining elements are already zeroed by calloc
        }
        src_data = padded_data;
    }

    cl_int err;
    cl_mem image = clCreateImage(info->context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 &fmt, &desc, (void*)src_data, &err);

    if (padded_data) free(padded_data);

    if (err != CL_SUCCESS) {
        MGPU_ERR("clCreateImage failed for weight image %dx%d (err=%d)\n", rows, cols, err);
        return nullptr;
    }

    return image;
}

cl_mem create_buffer(const DeviceInfo* info, size_t size_bytes, cl_mem_flags flags, void* host_ptr) {
    cl_int err;
    cl_mem buf = clCreateBuffer(info->context, flags, size_bytes, host_ptr, &err);
    if (err != CL_SUCCESS) {
        MGPU_ERR("clCreateBuffer failed (size=%zu, err=%d)\n", size_bytes, err);
        return nullptr;
    }
    return buf;
}

cl_mem create_onchip_buffer(const DeviceInfo* info, size_t size_bytes) {
    if (!info->has_qcom_onchip_global_memory) {
        MGPU_ERR("On-chip global memory not supported on this device\n");
        return nullptr;
    }

    if ((cl_ulong)size_bytes > info->onchip_global_mem_size) {
        MGPU_ERR("Requested on-chip buffer (%zu) exceeds available on-chip memory (%llu)\n",
                 size_bytes, (unsigned long long)info->onchip_global_mem_size);
        return nullptr;
    }

    cl_int err;
    cl_mem_flags flags = CL_MEM_READ_WRITE | (cl_mem_flags)CL_MEM_QUALCOMM_ONCHIP_GLOBAL;
    cl_mem buf = clCreateBuffer(info->context, flags, size_bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        MGPU_ERR("clCreateBuffer (on-chip) failed (size=%zu, err=%d)\n", size_bytes, err);
        return nullptr;
    }
    return buf;
}

bool init_buffer_pool(BufferPool* pool, const DeviceInfo* info, size_t size_bytes) {
    pool->buffers[0] = nullptr;
    pool->buffers[1] = nullptr;
    pool->size = size_bytes;
    pool->current = 0;

    cl_int err;
    pool->buffers[0] = clCreateBuffer(info->context, CL_MEM_READ_WRITE,
                                      size_bytes, nullptr, &err);
    CL_CHECK(err);

    pool->buffers[1] = clCreateBuffer(info->context, CL_MEM_READ_WRITE,
                                      size_bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(pool->buffers[0]);
        pool->buffers[0] = nullptr;
        MGPU_ERR("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__);
        return false;
    }

    return true;
}

cl_mem get_current_buffer(BufferPool* pool) {
    return pool->buffers[pool->current];
}

cl_mem get_next_buffer(BufferPool* pool) {
    return pool->buffers[1 - pool->current];
}

void swap_buffers(BufferPool* pool) {
    pool->current = 1 - pool->current;
}

void destroy_buffer_pool(BufferPool* pool) {
    for (int i = 0; i < 2; i++) {
        if (pool->buffers[i]) {
            clReleaseMemObject(pool->buffers[i]);
            pool->buffers[i] = nullptr;
        }
    }
    pool->size = 0;
    pool->current = 0;
}

} // namespace mgpu
