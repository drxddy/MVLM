/*
 * MVLM - Vision Processing Kernels for Adreno GPUs
 * Phase 4: Vision Encoder
 *
 * Implements GPU-side image preprocessing and patch embedding for the
 * vision encoder (SigLIP / ViT). Keeping preprocessing on the GPU avoids
 * a CPU→GPU memcpy for the preprocessed image.
 *
 * Kernels:
 *   - preprocess_image: resize + normalize RGBA uint8 → fp16 CHW patches
 *   - patch_embed: extract patches and project via GEMM (conv2d with stride)
 *
 * Adreno optimizations:
 *   - Image objects for input (TP/L1 texture cache, hardware bilinear interp)
 *   - Image objects for projection weights (read-only, cached)
 *   - int/uint indexing (saves registers vs size_t)
 *   - mad24/mul24 for index calculations
 *   - native_recip for fast normalization
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* ============================================================================
 * Image Preprocessing: Resize + Normalize
 *
 * Takes an RGBA uint8 input image (from camera / AHardwareBuffer) and
 * produces fp16 normalized output in CHW format suitable for the ViT encoder.
 *
 * Operations:
 *   1. Bilinear resize from (src_h, src_w) to (target_h, target_w)
 *      — handled by the sampler with CLK_FILTER_LINEAR
 *   2. Convert uint8 [0,255] → float [0,1]
 *   3. Normalize: (pixel - mean) / std per channel
 *   4. Output in CHW layout (channel-first, as expected by the model)
 *
 * Input image is a 2D image object, so reads go through the Texture Processor
 * with hardware bilinear interpolation — no manual interpolation code needed.
 *
 * Dispatch:
 *   global_work_size = { target_w, target_h }
 *   Each work-item produces 3 output values (R, G, B) for one spatial position.
 * ========================================================================= */

__constant sampler_t img_sampler =
    CLK_NORMALIZED_COORDS_TRUE |    // Use [0,1] normalized coordinates for resize
    CLK_ADDRESS_CLAMP_TO_EDGE |     // Clamp out-of-bounds reads
    CLK_FILTER_LINEAR;              // Hardware bilinear interpolation

__kernel void preprocess_image(
    __read_only image2d_t input_image,
    __global half* restrict output,    // [3, target_h, target_w] CHW layout
    const int target_h,
    const int target_w,
    const float mean_r, const float mean_g, const float mean_b,
    const float std_r, const float std_g, const float std_b)
{
    const int x = get_global_id(0);  // output column
    const int y = get_global_id(1);  // output row

    if (x >= target_w || y >= target_h) return;

    // Normalized coordinates [0, 1] for bilinear sampling
    // Add 0.5 to sample at pixel centers (OpenCL convention)
    const float u = ((float)x + 0.5f) * native_recip((float)target_w);
    const float v = ((float)y + 0.5f) * native_recip((float)target_h);

    // Read RGBA from input image — hardware bilinear interpolation
    // read_imagef returns float4 in [0, 1] range for normalized formats
    const float4 pixel = read_imagef(input_image, img_sampler, (float2)(u, v));

    // Normalize per channel: (value - mean) / std
    // Using native_recip for fast division
    const float r = (pixel.x - mean_r) * native_recip(std_r);
    const float g = (pixel.y - mean_g) * native_recip(std_g);
    const float b = (pixel.z - mean_b) * native_recip(std_b);

    // Output in CHW layout: channel 0 = R, channel 1 = G, channel 2 = B
    const int spatial_size = mul24(target_h, target_w);
    const int spatial_idx = mad24(y, target_w, x);

    output[spatial_idx]                      = (half)r;  // Channel 0 (R)
    output[spatial_idx + spatial_size]        = (half)g;  // Channel 1 (G)
    output[spatial_idx + mul24(2, spatial_size)] = (half)b;  // Channel 2 (B)
}

/* ============================================================================
 * Patch Embedding: Extract patches and project via GEMM
 *
 * Equivalent to a conv2d with kernel_size=patch_size and stride=patch_size:
 *   - Extract non-overlapping patches of size (C, patch_h, patch_w)
 *   - Flatten each patch to a vector of length C * patch_h * patch_w
 *   - Project via matrix multiply: embed = patch_vec @ weight^T + bias
 *
 * For SigLIP with 384x384 input, patch_size=14:
 *   - num_patches = (384/14)^2 = 27^2 = 729
 *   - patch_dim = 3 * 14 * 14 = 588
 *   - embed_dim = 768 (SigLIP base) or 1152 (SigLIP-384)
 *
 * Projection weights stored as image for TP/L1 cache hits.
 *   Image layout: (embed_dim/4 wide, patch_dim tall) with RGBA half
 *   Each pixel = 4 consecutive output dimensions
 *
 * Dispatch:
 *   global_work_size  = { num_patches, embed_dim / 4 }
 *   Each work-item computes 4 output embedding dimensions for one patch.
 * ========================================================================= */

__constant sampler_t weight_sampler =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST;

__kernel void patch_embed(
    __global const half* restrict image,          // [C, H, W] preprocessed CHW fp16
    __read_only image2d_t proj_weight,            // [patch_dim, embed_dim] as image
    __global const half* restrict proj_bias,      // [embed_dim]
    __global half* restrict patches,              // [num_patches, embed_dim]
    const int C,
    const int H,
    const int W,
    const int patch_h,
    const int patch_w,
    const int embed_dim)
{
    const int patch_idx = get_global_id(0);   // which patch
    const int embed_col4 = get_global_id(1);  // which group of 4 embed dims

    const int patches_per_row = W / patch_w;
    const int num_patches_h = H / patch_h;
    const int num_patches = mul24(num_patches_h, patches_per_row);

    if (patch_idx >= num_patches) return;

    const int embed_base = embed_col4 << 2;
    if (embed_base >= embed_dim) return;

    // Patch grid position
    const int patch_row = patch_idx / patches_per_row;
    const int patch_col = patch_idx - mul24(patch_row, patches_per_row);

    // Top-left corner of this patch in the image
    const int y0 = mul24(patch_row, patch_h);
    const int x0 = mul24(patch_col, patch_w);

    // Accumulate dot product: patch_vector @ weight^T
    // patch_dim = C * patch_h * patch_w
    float4 acc = (float4)(0.0f);

    int patch_dim_idx = 0;
    const int spatial_size = mul24(H, W);

    for (int c = 0; c < C; ++c) {
        const int channel_offset = mul24(c, spatial_size);
        for (int py = 0; py < patch_h; ++py) {
            const int img_y = y0 + py;
            for (int px = 0; px < patch_w; ++px) {
                const int img_x = x0 + px;

                // Load one pixel value from the image
                const float pixel = (float)image[channel_offset + mad24(img_y, W, img_x)];

                // Load 4 weight values from the projection weight image
                // Image coords: (embed_col4, patch_dim_idx)
                const half4 w = read_imageh(proj_weight, weight_sampler,
                                            (int2)(embed_col4, patch_dim_idx));

                acc = fma((float4)(pixel), convert_float4(w), acc);
                patch_dim_idx++;
            }
        }
    }

    // Add bias
    if (embed_base + 4 <= embed_dim) {
        const half4 bias = vload_half4(0, proj_bias + embed_base);
        acc += convert_float4(bias);
    } else {
        for (int i = 0; i < embed_dim - embed_base; ++i) {
            acc[i] += (float)proj_bias[embed_base + i];
        }
    }

    // Write output
    const int out_offset = mad24(patch_idx, embed_dim, embed_base);
    if (embed_base + 4 <= embed_dim) {
        vstore_half4(acc, 0, patches + out_offset);
    } else {
        for (int i = 0; i < embed_dim - embed_base; ++i) {
            vstore_half(acc[i], 0, patches + out_offset + i);
        }
    }
}
