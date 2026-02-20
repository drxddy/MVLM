#include "../engine/device.h"

#include <cstdio>
#include <cstring>

// Extensions required or desired for MGPU on Adreno
struct ExtensionCheck {
    const char* name;
    const char* description;
    bool required;
};

static const ExtensionCheck mgpu_extensions[] = {
    { "cl_khr_fp16",                            "FP16 compute",                          true  },
    { "cl_khr_subgroups",                       "Subgroup operations",                   true  },
    { "cl_khr_integer_dot_product",             "Integer dot product (generic)",          false },
    { "cl_qcom_subgroup_shuffle",               "Qualcomm subgroup shuffle",             false },
    { "cl_qcom_onchip_global_memory",           "Qualcomm on-chip global memory",        false },
    { "cl_qcom_recordable_queues",              "Qualcomm recordable queues",            false },
    { "cl_qcom_dot_product8",                   "Qualcomm int8 dot product",             false },
    { "cl_qcom_perf_hint",                      "Qualcomm performance hints",            false },
    { "cl_qcom_android_ahardwarebuffer_host_ptr", "Qualcomm AHardwareBuffer zero-copy", false },
    { "cl_khr_image2d_from_buffer",             "Image2D from buffer",                   false },
    { "cl_khr_gl_sharing",                      "OpenGL sharing",                        false },
};

static const int num_mgpu_extensions = sizeof(mgpu_extensions) / sizeof(mgpu_extensions[0]);

static void print_image_formats(const mgpu::DeviceInfo* info) {
    cl_uint num_formats = 0;
    cl_int err = clGetSupportedImageFormats(info->context, CL_MEM_READ_ONLY,
                                            CL_MEM_OBJECT_IMAGE2D, 0, nullptr, &num_formats);
    if (err != CL_SUCCESS || num_formats == 0) {
        printf("  No 2D image formats supported (or error querying)\n");
        return;
    }

    cl_image_format* formats = new cl_image_format[num_formats];
    clGetSupportedImageFormats(info->context, CL_MEM_READ_ONLY,
                               CL_MEM_OBJECT_IMAGE2D, num_formats, formats, nullptr);

    printf("  Supported 2D image formats (READ_ONLY): %u\n", num_formats);

    auto channel_order_str = [](cl_channel_order o) -> const char* {
        switch (o) {
            case CL_R:           return "R";
            case CL_A:           return "A";
            case CL_RG:          return "RG";
            case CL_RA:          return "RA";
            case CL_RGB:         return "RGB";
            case CL_RGBA:        return "RGBA";
            case CL_BGRA:        return "BGRA";
            case CL_ARGB:        return "ARGB";
            case CL_INTENSITY:   return "INTENSITY";
            case CL_LUMINANCE:   return "LUMINANCE";
#ifdef CL_Rx
            case CL_Rx:          return "Rx";
#endif
#ifdef CL_RGx
            case CL_RGx:         return "RGx";
#endif
#ifdef CL_RGBx
            case CL_RGBx:        return "RGBx";
#endif
            default:             return "UNKNOWN";
        }
    };

    auto channel_type_str = [](cl_channel_type t) -> const char* {
        switch (t) {
            case CL_SNORM_INT8:       return "SNORM_INT8";
            case CL_SNORM_INT16:      return "SNORM_INT16";
            case CL_UNORM_INT8:       return "UNORM_INT8";
            case CL_UNORM_INT16:      return "UNORM_INT16";
            case CL_UNORM_SHORT_565:  return "UNORM_SHORT_565";
            case CL_UNORM_SHORT_555:  return "UNORM_SHORT_555";
            case CL_UNORM_INT_101010: return "UNORM_INT_101010";
            case CL_SIGNED_INT8:      return "SIGNED_INT8";
            case CL_SIGNED_INT16:     return "SIGNED_INT16";
            case CL_SIGNED_INT32:     return "SIGNED_INT32";
            case CL_UNSIGNED_INT8:    return "UNSIGNED_INT8";
            case CL_UNSIGNED_INT16:   return "UNSIGNED_INT16";
            case CL_UNSIGNED_INT32:   return "UNSIGNED_INT32";
            case CL_HALF_FLOAT:       return "HALF_FLOAT";
            case CL_FLOAT:            return "FLOAT";
            default:                  return "UNKNOWN";
        }
    };

    for (cl_uint i = 0; i < num_formats; i++) {
        printf("    %-12s  %s\n",
               channel_order_str(formats[i].image_channel_order),
               channel_type_str(formats[i].image_channel_data_type));
    }

    delete[] formats;
}

int main() {
    printf("=== MGPU Device Info Tool ===\n\n");

    mgpu::DeviceInfo device;
    if (!mgpu::init_device(&device)) {
        fprintf(stderr, "Error: Failed to initialize OpenCL device\n");
        return 1;
    }

    // Print basic device info
    mgpu::print_device_info(&device);

    // Print full extension list
    printf("\n=== Extensions ===\n");
    char extensions[8192] = {};
    clGetDeviceInfo(device.device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);

    // Tokenize and print each extension
    int ext_count = 0;
    char* ext_copy = new char[strlen(extensions) + 1];
    strcpy(ext_copy, extensions);
    char* tok = strtok(ext_copy, " ");
    while (tok) {
        printf("  %s\n", tok);
        ext_count++;
        tok = strtok(nullptr, " ");
    }
    printf("  Total: %d extensions\n", ext_count);
    delete[] ext_copy;

    // Half-precision vector widths
    printf("\n=== Half-Precision Support ===\n");
    printf("  Preferred vector width (half): %u\n", device.preferred_vector_width_half);
    printf("  Native vector width (half):    %u\n", device.native_vector_width_half);

    // Image format support
    printf("\n=== Image Format Support ===\n");
    if (device.has_image) {
        print_image_formats(&device);
    } else {
        printf("  Image support: NOT AVAILABLE\n");
    }

    // MGPU extension readiness check
    printf("\n=== MGPU Readiness Report ===\n");
    int required_met = 0;
    int required_total = 0;
    int optional_met = 0;
    int optional_total = 0;

    for (int i = 0; i < num_mgpu_extensions; i++) {
        bool found = strstr(extensions, mgpu_extensions[i].name) != nullptr;
        const char* icon = found ? "\xE2\x9C\x85" : "\xE2\x9D\x8C";
        const char* req_str = mgpu_extensions[i].required ? "[REQUIRED]" : "[optional]";

        printf("  %s %-48s %s %s\n", icon, mgpu_extensions[i].name,
               req_str, mgpu_extensions[i].description);

        if (mgpu_extensions[i].required) {
            required_total++;
            if (found) required_met++;
        } else {
            optional_total++;
            if (found) optional_met++;
        }
    }

    // Additional checks
    bool image_ok = device.has_image;
    printf("  %s %-48s %s %s\n",
           image_ok ? "\xE2\x9C\x85" : "\xE2\x9D\x8C",
           "Image object support", "[REQUIRED]", "2D image objects for weight caching");
    required_total++;
    if (image_ok) required_met++;

    bool all_required = (required_met == required_total);

    printf("\n  Required:  %d/%d\n", required_met, required_total);
    printf("  Optional:  %d/%d\n", optional_met, optional_total);
    printf("\n  Overall: %s\n",
           all_required ? "\xE2\x9C\x85 READY for MGPU" : "\xE2\x9D\x8C NOT READY for MGPU");

    if (!all_required) {
        printf("  Missing required features — Adreno GPU with OpenCL 2.0+ recommended\n");
    }

    mgpu::destroy_device(&device);
    return 0;
}
