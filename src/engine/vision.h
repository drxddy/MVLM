#pragma once

#include "device.h"
#include "pipeline.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace mgpu {

// Forward declarations
struct GGUFFile;
struct TensorInfo;

// ============================================================================
// Image Loading (stb_image)
// ============================================================================

// Load image from file using stb_image
// Returns buffer with RGB float32 data, or nullptr on failure
// Output: width, height, channels (always 3 for RGB)
float* load_image_from_file(const char* path, int* width, int* height, int* channels);

// Free image data
void free_image_data(float* data);

// ============================================================================
// Zero-Copy Camera AHB Integration
// ============================================================================

// Process camera frame directly to GPU via AHardwareBuffer
// This is the zero-copy path: camera ISP → GPU memory without staging buffer
//
// Architecture:
// 1. Camera captures frame to AHardwareBuffer (YUV/RGBA)
// 2. AHB is imported as OpenCL buffer/image via cl_qcom extension
// 3. Vision encoder processes directly from GPU memory
//
// Note: This requires Android 8.0+ and cl_qcom_android_ahardwarebuffer_host_ptr
struct CameraFrame {
    cl_mem gpu_buffer;     // OpenCL buffer containing frame data
    void* ahb_handle;      // AHardwareBuffer* handle
    int width;
    int height;
    int format;            // e.g., HAL_PIXEL_FORMAT_YCbCr_420_SP or similar
    bool is_zero_copy;    // true if using AHB zero-copy
};

// Import AHardwareBuffer as OpenCL memory (zero-copy)
// Returns CameraFrame with GPU-accessible buffer
// The frame data never leaves GPU memory - no staging buffer needed
CameraFrame import_camera_frame_ahb(const DeviceInfo* device,
                                   void* ahb_handle,
                                   int width, int height, int format);

// Alternative: Create GPU buffer from AHB for explicit copy (fallback)
CameraFrame create_camera_frame_from_ahb(const DeviceInfo* device,
                                        void* ahb_handle,
                                        int width, int height, int format);

// Free camera frame resources
void destroy_camera_frame(CameraFrame* frame);

// Check if zero-copy AHB is supported
bool is_ahb_zero_copy_supported(const DeviceInfo* device);

// ============================================================================
// Vision Weight Loading from GGUF
// ============================================================================

// Load vision encoder weights from GGUF file
// Parses and uploads: patch_embed, vision layers, projection
struct VisionWeights {
    // Patch embedding
    cl_mem patch_embed_weight;     // [patch_area * 3, vision_dim]
    cl_mem patch_embed_bias;       // [vision_dim]

    // Vision transformer layers
    struct Layer {
        cl_mem attn_qkv_weight;   // [vision_dim, vision_dim * 3]
        cl_mem attn_qkv_bias;     // [vision_dim * 3]
        cl_mem attn_out_weight;   // [vision_dim, vision_dim]
        cl_mem attn_out_bias;    // [vision_dim]

        cl_mem mlp_fc_weight;     // [vision_dim, mlp_hidden]
        cl_mem mlp_fc_bias;     // [mlp_hidden]
        cl_mem mlp_proj_weight;  // [mlp_hidden, vision_dim]
        cl_mem mlp_proj_bias;   // [vision_dim]

        cl_mem norm_weight;      // [vision_dim]
        cl_mem norm_bias;        // [vision_dim]
    };

    Layer* layers;
    int num_layers;

    // Output projection
    cl_mem norm_weight;
    cl_mem norm_bias;

    // Vision to language projection
    cl_mem proj_weight;           // [vision_dim, llm_dim]
    cl_mem proj_bias;            // [llm_dim]
};

// Load vision weights from GGUF
// Returns true if successful
bool load_vision_weights_from_gguf(const DeviceInfo* device,
                                  const GGUFFile* gguf,
                                  VisionWeights* weights,
                                  int vision_dim,
                                  int num_layers);

// Free vision weights
void free_vision_weights(VisionWeights* weights);

// ============================================================================
// Combined VLM Pipeline
// ============================================================================

// Full VLM pipeline: camera → vision encoder → LLM → text
struct VLMPipeline {
    DeviceInfo* device;
    InferencePipeline qpipeline;  // Qualcomm extensions
    VisionWeights vision_weights;
    bool initialized;
};

// Initialize VLM pipeline
bool init_vlm_pipeline(VLMPipeline* pipeline, DeviceInfo* device);

// Process camera frame to text
// ahb_handle: AHardwareBuffer from camera
// prompt: optional text prompt
// max_tokens: max tokens to generate
// output: buffer for generated text (caller allocates)
bool process_vlm_camera(VLMPipeline* pipeline,
                       void* ahb_handle,
                       const char* prompt,
                       int max_tokens,
                       char* output, int output_size);

// Process image file to text (uses stb_image)
bool process_vlm_image_file(VLMPipeline* pipeline,
                           const char* image_path,
                           const char* prompt,
                           int max_tokens,
                           char* output, int output_size);

// Destroy VLM pipeline
void destroy_vlm_pipeline(VLMPipeline* pipeline);

} // namespace mgpu
