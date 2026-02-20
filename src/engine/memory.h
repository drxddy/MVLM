#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>

namespace mgpu {

struct DeviceInfo;

// Create a 2D image object from a weight matrix (for texture cache path)
cl_mem create_weight_image(const DeviceInfo* info, int rows, int cols, const cl_half* data);

// Create a regular buffer
cl_mem create_buffer(const DeviceInfo* info, size_t size_bytes, cl_mem_flags flags, void* host_ptr = nullptr);

// Create on-chip global memory buffer (if extension available)
cl_mem create_onchip_buffer(const DeviceInfo* info, size_t size_bytes);

// Simple buffer pool for activation reuse (ping-pong)
struct BufferPool {
    cl_mem buffers[2];
    size_t size;
    int current;
};

bool init_buffer_pool(BufferPool* pool, const DeviceInfo* info, size_t size_bytes);
cl_mem get_current_buffer(BufferPool* pool);
cl_mem get_next_buffer(BufferPool* pool);
void swap_buffers(BufferPool* pool);
void destroy_buffer_pool(BufferPool* pool);

} // namespace mgpu
