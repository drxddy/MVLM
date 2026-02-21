#pragma once

#include "../engine/device.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

namespace mgpu {

// Forward declarations
struct DeviceInfo;

// ============================================================================
// Qualcomm Performance Hints Extension
// ============================================================================

// Performance hint extensions for Adreno
// Reference: Qualcomm OpenCL Programming Guide Section 9.1.1

// Hint types for GPU performance optimization
enum class PerfHintType {
    GPU_PERF_HINT_QCOM = 0x10000,
};

// Priority levels for hint
enum class PerfPriority {
    PRIORITY_LOW_QCOM = 0x1,
    PRIORITY_MEDIUM_QCOM = 0x2,
    PRIORITY_HIGH_QCOM = 0x3,
};

// Apply performance hints to context/queue
bool apply_perf_hints(DeviceInfo* info);

// ============================================================================
// Qualcomm Recordable Queues Extension
// ============================================================================

// Recordable queue for decode loop optimization
// Reference: Qualcomm OpenCL Programming Guide Section 9.1.3
//
// For transformer decode, we run 24-32 layers repeatedly.
// Recording the kernel sequence once and replaying saves dispatch overhead.

struct RecordableQueue {
    cl_command_queue record_queue;   // Queue for recording
    cl_command_queue live_queue;     // Queue for execution
    void* recording;                 // cl_recording_qcom (opaque handle)
    bool is_recording;
    bool is_valid;

    // Kernel info for parameter updates
    cl_kernel kernel;
    cl_uint num_args;
    void** arg_values;
    size_t* arg_sizes;
};

// Create a recordable queue pair
// Returns true if extension is available and queues created
bool create_recordable_queue(const DeviceInfo* info, RecordableQueue* queue);

// Destroy recordable queue
void destroy_recordable_queue(RecordableQueue* queue);

// Start recording kernel sequence
// kernel: the kernel to record
// global_size: work dimensions
// local_size: workgroup size
bool start_recording(RecordableQueue* queue, cl_kernel kernel,
                     const size_t* global_size, const size_t* local_size);

// Stop recording
bool stop_recording(RecordableQueue* queue);

// Execute recorded sequence (replay)
// Can update kernel arguments between replays for KV-cache updates
bool replay_recording(RecordableQueue* queue,
                     cl_uint num_updates,
                     const cl_uint* arg_indices,
                     const void** arg_values,
                     const size_t* arg_sizes);

// ============================================================================
// Qualcomm On-Chip Global Memory Extension
// ============================================================================

// On-chip memory buffer for KV-cache and activations
// Reference: Qualcomm OpenCL Programming Guide Section 9.1.6
//
// Fast on-chip SRAM for intermediate data - stays in GPU, no DRAM round-trip

struct OnChipBuffer {
    cl_mem buffer;
    size_t size;
    bool is_valid;
};

// Create on-chip memory buffer (if extension available)
// Returns buffer in on-chip memory, falls back to regular if not
OnChipBuffer create_onchip_buffer(const DeviceInfo* info, size_t size_bytes);

// Free on-chip buffer
void destroy_onchip_buffer(OnChipBuffer* buf);

// Check if on-chip memory is available
bool has_onchip_memory(const DeviceInfo* info);

// Get recommended on-chip buffer size (leaves room for other allocations)
size_t get_recommended_onchip_size(const DeviceInfo* info);

// ============================================================================
// Qualcomm AHB (Android Hardware Buffer) Zero-Copy
// ============================================================================

// Zero-copy camera input via AHardwareBuffer
// Reference: Qualcomm OpenCL Programming Guide Section 7.4
//
// Camera ISP → GPU memory directly, no staging buffer needed

struct AHBImage {
    cl_mem image;           // OpenCL image from AHB
    void* ahb_handle;       // AHardwareBuffer handle
    int width;
    int height;
    bool is_valid;
};

// Create OpenCL image from Android Hardware Buffer
// This enables zero-copy camera → GPU pipeline
AHBImage create_ahb_image(const DeviceInfo* info,
                          int width, int height,
                          void* ahb_handle);

// Destroy AHB image
void destroy_ahb_image(AHBImage* img);

// Check if AHB extension is available
bool has_ahb_support(const DeviceInfo* info);

// ============================================================================
// Qualcomm Dot Product Extension (Int8/Int4 Inference)
// ============================================================================

// Enable hardware-accelerated int8 matrix multiplication
// Reference: Qualcomm OpenCL Programming Guide Section 9.4
//
// For quantized inference (Q4/Q8 weights)

struct DotProductConfig {
    bool has_extension;
    bool has_accelerated_dot8;    // cl_qcom_dot_product8
};

// Query dot product capabilities
DotProductConfig query_dot_product(const DeviceInfo* info);

// ============================================================================
// Subgroup Shuffle Extension
// ============================================================================

// Enable efficient cross-thread data exchange
// Reference: Qualcomm OpenCL Programming Guide Section 9.2.2

struct SubgroupConfig {
    bool has_shuffle;      // cl_qcom_subgroup_shuffle
    size_t max_subgroup_size;
    size_t preferred_size;
};

// Query subgroup capabilities
SubgroupConfig query_subgroup_config(const DeviceInfo* info);

// ============================================================================
// Pipeline State
// ============================================================================

// Complete inference pipeline state
struct InferencePipeline {
    DeviceInfo* device;

    // Performance optimization
    PerfHintType perf_hint;
    bool perf_hints_applied;

    // Recordable queue for decode
    RecordableQueue decode_queue;

    // On-chip buffers for KV-cache
    OnChipBuffer kv_cache;
    OnChipBuffer activations;

    // Camera input
    bool camera_connected;

    // Quantization
    DotProductConfig dot_product;

    // Subgroup config
    SubgroupConfig subgroup;

    // State
    bool initialized;
};

// Initialize pipeline with all Qualcomm extensions
bool init_pipeline(InferencePipeline* pipeline, DeviceInfo* device);

// Destroy pipeline and release resources
void destroy_pipeline(InferencePipeline* pipeline);

// ============================================================================
// High-Level Pipeline Functions
// ============================================================================

// Camera frame → GPU (zero-copy)
bool pipeline_process_camera_frame(InferencePipeline* pipeline,
                                   void* ahb_handle,
                                   int width, int height);

// Process vision encoder with on-chip memory
bool pipeline_process_vision(InferencePipeline* pipeline,
                            const AHBImage* image,
                            cl_mem output_tokens);

// Process LLM decode with recordable queue
bool pipeline_process_decode(InferencePipeline* pipeline,
                           const int* prompt_tokens,
                           int num_tokens,
                           int* output_token);

} // namespace mgpu
