#include <gtest/gtest.h>
#include "../src/engine/device.h"
#include "../src/engine/memory.h"
#include <cstdio>

using namespace mgpu;

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        memset(&device_, 0, sizeof(device_));
    }
    void TearDown() override {
        if (device_.context) {
            destroy_device(&device_);
        }
    }
    DeviceInfo device_;
};

// Test device initialization (requires GPU)
TEST_F(DeviceTest, InitDevice) {
    // This test requires OpenCL GPU
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    EXPECT_NE(device_.context, nullptr);
    EXPECT_NE(device_.queue, nullptr);
    EXPECT_GT(device_.compute_units, 0u);
    EXPECT_GT(device_.max_workgroup_size, 0u);
}

// Test device info (requires GPU)
TEST_F(DeviceTest, DeviceInfo) {
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    // Print device info (just ensure no crash)
    print_device_info(&device_);

    // Check basic properties
    EXPECT_GT(device_.max_alloc_size, 0ull);
    EXPECT_GT(device_.max_work_item_dims, 0u);
}

// Test kernel loading (requires GPU)
TEST_F(DeviceTest, LoadKernel) {
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    // Try to build a simple kernel
    const char* kernel_src = R"(
        __kernel void test_add(__global float* a, __global float* b, __global float* c) {
            int id = get_global_id(0);
            c[id] = a[id] + b[id];
        }
    )";

    cl_program program = build_program_from_source(&device_, kernel_src, strlen(kernel_src), "-cl-fast-relaxed-math");
    EXPECT_NE(program, nullptr);

    if (program) {
        clReleaseProgram(program);
    }
}

// Test buffer creation (requires GPU)
TEST_F(DeviceTest, BufferOperations) {
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    const size_t buffer_size = 1024;

    // Create buffer using the helper
    cl_mem buffer = create_buffer(&device_, buffer_size, CL_MEM_READ_WRITE, nullptr);
    EXPECT_NE(buffer, nullptr);

    // Fill buffer with pattern
    cl_int err = clEnqueueFillBuffer(device_.queue, buffer, "\xAB", 1, 0, buffer_size, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS);

    // Read back
    uint8_t* host_buf = (uint8_t*)malloc(buffer_size);
    err = clEnqueueReadBuffer(device_.queue, buffer, CL_TRUE, 0, buffer_size, host_buf, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS);

    // Check filled value
    for (size_t i = 0; i < buffer_size; i++) {
        EXPECT_EQ(host_buf[i], 0xAB);
    }

    free(host_buf);
    clReleaseMemObject(buffer);
}

// Test buffer mapping (requires GPU)
TEST_F(DeviceTest, BufferMap) {
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    const size_t buffer_size = 256;
    cl_mem buffer = create_buffer(&device_, buffer_size, CL_MEM_READ_WRITE, nullptr);
    EXPECT_NE(buffer, nullptr);

    // Map buffer (using deprecated 10-arg version for OpenCL 1.2)
    cl_map_flags map_flags = CL_MAP_WRITE;
    cl_int err;
    void* mapped = clEnqueueMapBuffer(device_.queue, buffer, CL_TRUE, map_flags, 0, buffer_size, 0, nullptr, nullptr, &err);
    EXPECT_NE(mapped, nullptr);
    EXPECT_EQ(err, CL_SUCCESS);

    // Write to mapped memory
    memset(mapped, 0x42, buffer_size);

    // Unmap
    clEnqueueUnmapMemObject(device_.queue, buffer, mapped, 0, nullptr, nullptr);
    clFinish(device_.queue);

    // Read back and verify
    uint8_t* host_buf = (uint8_t*)malloc(buffer_size);
    clEnqueueReadBuffer(device_.queue, buffer, CL_TRUE, 0, buffer_size, host_buf, 0, nullptr, nullptr);

    for (size_t i = 0; i < buffer_size; i++) {
        EXPECT_EQ(host_buf[i], 0x42);
    }

    free(host_buf);
    clReleaseMemObject(buffer);
}

// Test simple kernel execution (requires GPU)
TEST_F(DeviceTest, KernelExecution) {
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    const char* kernel_src = R"(
        __kernel void test_add(__global float* a, __global float* b, __global float* c) {
            int id = get_global_id(0);
            c[id] = a[id] + b[id];
        }
    )";

    cl_program program = build_program_from_source(&device_, kernel_src, strlen(kernel_src), "");
    if (!program) {
        GTEST_SKIP() << "Failed to build test kernel";
    }

    cl_kernel kernel = clCreateKernel(program, "test_add", nullptr);
    if (!kernel) {
        clReleaseProgram(program);
        GTEST_SKIP() << "Failed to create test kernel";
    }

    const size_t n = 64;
    const size_t buffer_size = n * sizeof(float);

    // Create buffers
    cl_mem a = create_buffer(&device_, buffer_size, CL_MEM_READ_ONLY, nullptr);
    cl_mem b = create_buffer(&device_, buffer_size, CL_MEM_READ_ONLY, nullptr);
    cl_mem c = create_buffer(&device_, buffer_size, CL_MEM_WRITE_ONLY, nullptr);

    // Prepare data
    float* a_host = (float*)malloc(buffer_size);
    float* b_host = (float*)malloc(buffer_size);
    float* c_host = (float*)malloc(buffer_size);

    for (size_t i = 0; i < n; i++) {
        a_host[i] = (float)i;
        b_host[i] = (float)(i * 2);
    }

    // Write input
    clEnqueueWriteBuffer(device_.queue, a, CL_TRUE, 0, buffer_size, a_host, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(device_.queue, b, CL_TRUE, 0, buffer_size, b_host, 0, nullptr, nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);

    // Execute
    size_t global = n;
    clEnqueueNDRangeKernel(device_.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clFinish(device_.queue);

    // Read result
    clEnqueueReadBuffer(device_.queue, c, CL_TRUE, 0, buffer_size, c_host, 0, nullptr, nullptr);

    // Verify
    for (size_t i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(c_host[i], a_host[i] + b_host[i]);
    }

    // Cleanup
    free(a_host);
    free(b_host);
    free(c_host);
    clReleaseMemObject(a);
    clReleaseMemObject(b);
    clReleaseMemObject(c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

// Test buffer pool (requires GPU)
TEST_F(DeviceTest, BufferPool) {
    bool success = init_device(&device_);
    if (!success) {
        GTEST_SKIP() << "No OpenCL devices available";
    }

    BufferPool pool;
    memset(&pool, 0, sizeof(pool));

    // Initialize buffer pool
    bool pool_init = init_buffer_pool(&pool, &device_, 1024);
    EXPECT_TRUE(pool_init);

    if (pool_init) {
        // Get buffers
        cl_mem buf1 = get_current_buffer(&pool);
        EXPECT_NE(buf1, nullptr);

        // Swap
        swap_buffers(&pool);
        cl_mem buf2 = get_current_buffer(&pool);
        EXPECT_NE(buf2, nullptr);

        // Cleanup
        destroy_buffer_pool(&pool);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
