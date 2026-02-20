#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mgpu {

size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::Q4_0: return 18;   // 2 bytes scale + 16 bytes data (32 elements)
        case GGMLType::Q4_1: return 20;   // 2+2 bytes scale/min + 16 bytes data (32 elements)
        case GGMLType::Q5_0: return 22;   // 2 bytes scale + 4 bytes high bits + 16 bytes data (32 elements)
        case GGMLType::Q5_1: return 24;   // 2+2 bytes + 4 bytes high bits + 16 bytes data (32 elements)
        case GGMLType::Q8_0: return 34;   // 2 bytes scale + 32 bytes data (32 elements)
        case GGMLType::Q8_1: return 40;   // 4+4 bytes scale/sum + 32 bytes data (32 elements)
        case GGMLType::Q2_K: return 84;   // 256-element block
        case GGMLType::Q3_K: return 110;  // 256-element block
        case GGMLType::Q4_K: return 144;  // 256-element block
        case GGMLType::Q5_K: return 176;  // 256-element block
        case GGMLType::Q6_K: return 210;  // 256-element block
        case GGMLType::I8:   return 1;
        case GGMLType::I16:  return 2;
        case GGMLType::I32:  return 4;
        default:             return 0;
    }
}

int ggml_type_block_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 1;
        case GGMLType::F16:  return 1;
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q8_1: return 32;
        case GGMLType::Q2_K: return 256;
        case GGMLType::Q3_K: return 256;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        case GGMLType::I8:   return 1;
        case GGMLType::I16:  return 1;
        case GGMLType::I32:  return 1;
        default:             return 0;
    }
}

static const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::I8:   return "I8";
        case GGMLType::I16:  return "I16";
        case GGMLType::I32:  return "I32";
        default:             return "UNKNOWN";
    }
}

// Compute total number of elements in a tensor
static uint64_t tensor_num_elements(const TensorInfo* t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        n *= t->dims[i];
    }
    return n;
}

// Compute tensor data size in bytes
static size_t compute_tensor_size(const TensorInfo* t) {
    uint64_t n_elements = tensor_num_elements(t);
    int block_size = ggml_type_block_size(t->type);
    size_t type_size = ggml_type_size(t->type);
    if (block_size == 0 || type_size == 0) return 0;
    uint64_t n_blocks = (n_elements + block_size - 1) / block_size;
    return (size_t)(n_blocks * type_size);
}

// Read a GGUFString from the cursor, advancing it
static bool read_gguf_string(const uint8_t** cursor, const uint8_t* end, GGUFString* out) {
    if (*cursor + 8 > end) return false;
    uint64_t len;
    memcpy(&len, *cursor, 8);
    *cursor += 8;
    if (*cursor + len > end) return false;
    out->len = len;
    out->data = (const char*)*cursor;
    *cursor += len;
    return true;
}

// Skip a metadata value based on its type
static bool skip_metadata_value(const uint8_t** cursor, const uint8_t* end,
                                GGUFMetadataValueType vtype) {
    switch (vtype) {
        case GGUFMetadataValueType::UINT8:
        case GGUFMetadataValueType::INT8:
        case GGUFMetadataValueType::BOOL:
            if (*cursor + 1 > end) return false;
            *cursor += 1;
            return true;
        case GGUFMetadataValueType::UINT16:
        case GGUFMetadataValueType::INT16:
            if (*cursor + 2 > end) return false;
            *cursor += 2;
            return true;
        case GGUFMetadataValueType::UINT32:
        case GGUFMetadataValueType::INT32:
        case GGUFMetadataValueType::FLOAT32:
            if (*cursor + 4 > end) return false;
            *cursor += 4;
            return true;
        case GGUFMetadataValueType::UINT64:
        case GGUFMetadataValueType::INT64:
        case GGUFMetadataValueType::FLOAT64:
            if (*cursor + 8 > end) return false;
            *cursor += 8;
            return true;
        case GGUFMetadataValueType::STRING: {
            GGUFString s;
            return read_gguf_string(cursor, end, &s);
        }
        case GGUFMetadataValueType::ARRAY: {
            if (*cursor + 12 > end) return false;
            uint32_t arr_type;
            memcpy(&arr_type, *cursor, 4);
            *cursor += 4;
            uint64_t arr_len;
            memcpy(&arr_len, *cursor, 8);
            *cursor += 8;
            for (uint64_t i = 0; i < arr_len; i++) {
                if (!skip_metadata_value(cursor, end, (GGUFMetadataValueType)arr_type)) {
                    return false;
                }
            }
            return true;
        }
        default:
            fprintf(stderr, "Error: Unknown metadata value type %u\n", (uint32_t)vtype);
            return false;
    }
}

bool gguf_open(GGUFFile* file, const char* filepath) {
    memset(file, 0, sizeof(GGUFFile));

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open file: %s\n", filepath);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "Error: Cannot stat file: %s\n", filepath);
        close(fd);
        return false;
    }

    file->file_size = (size_t)st.st_size;
    if (file->file_size < sizeof(GGUFHeader)) {
        fprintf(stderr, "Error: File too small to be GGUF: %s\n", filepath);
        close(fd);
        return false;
    }

    file->mapped_data = mmap(nullptr, file->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (file->mapped_data == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed for: %s\n", filepath);
        file->mapped_data = nullptr;
        return false;
    }

    const uint8_t* data = (const uint8_t*)file->mapped_data;
    const uint8_t* end = data + file->file_size;
    const uint8_t* cursor = data;

    // Read header
    if (cursor + sizeof(GGUFHeader) > end) {
        fprintf(stderr, "Error: Truncated GGUF header\n");
        gguf_close(file);
        return false;
    }
    memcpy(&file->header, cursor, sizeof(GGUFHeader));
    cursor += sizeof(GGUFHeader);

    if (file->header.magic != GGUF_MAGIC) {
        fprintf(stderr, "Error: Invalid GGUF magic: 0x%08X (expected 0x%08X)\n",
                file->header.magic, GGUF_MAGIC);
        gguf_close(file);
        return false;
    }

    if (file->header.version < 2 || file->header.version > 3) {
        fprintf(stderr, "Error: Unsupported GGUF version: %u (supported: 2-3)\n",
                file->header.version);
        gguf_close(file);
        return false;
    }

    printf("GGUF: version=%u, tensors=%llu, metadata_kv=%llu\n",
           file->header.version,
           (unsigned long long)file->header.tensor_count,
           (unsigned long long)file->header.metadata_kv_count);

    // Skip metadata key-value pairs
    for (uint64_t i = 0; i < file->header.metadata_kv_count; i++) {
        // Read key (string)
        GGUFString key;
        if (!read_gguf_string(&cursor, end, &key)) {
            fprintf(stderr, "Error: Failed to read metadata key %llu\n", (unsigned long long)i);
            gguf_close(file);
            return false;
        }

        // Read value type
        if (cursor + 4 > end) {
            fprintf(stderr, "Error: Truncated metadata value type\n");
            gguf_close(file);
            return false;
        }
        uint32_t vtype;
        memcpy(&vtype, cursor, 4);
        cursor += 4;

        // Skip value
        if (!skip_metadata_value(&cursor, end, (GGUFMetadataValueType)vtype)) {
            fprintf(stderr, "Error: Failed to skip metadata value for key '%.*s'\n",
                    (int)key.len, key.data);
            gguf_close(file);
            return false;
        }
    }

    // Parse tensor info entries
    file->tensor_count = file->header.tensor_count;
    file->tensors = (TensorInfo*)calloc(file->tensor_count, sizeof(TensorInfo));
    if (!file->tensors) {
        fprintf(stderr, "Error: Failed to allocate tensor info array\n");
        gguf_close(file);
        return false;
    }

    for (uint64_t i = 0; i < file->tensor_count; i++) {
        TensorInfo* t = &file->tensors[i];

        // Read tensor name
        GGUFString name;
        if (!read_gguf_string(&cursor, end, &name)) {
            fprintf(stderr, "Error: Failed to read tensor name %llu\n", (unsigned long long)i);
            gguf_close(file);
            return false;
        }
        size_t name_len = name.len < 255 ? name.len : 255;
        memcpy(t->name, name.data, name_len);
        t->name[name_len] = '\0';

        // Read n_dims
        if (cursor + 4 > end) {
            fprintf(stderr, "Error: Truncated tensor n_dims\n");
            gguf_close(file);
            return false;
        }
        memcpy(&t->n_dims, cursor, 4);
        cursor += 4;

        if (t->n_dims > 4) {
            fprintf(stderr, "Error: Tensor '%s' has %u dims (max 4)\n", t->name, t->n_dims);
            gguf_close(file);
            return false;
        }

        // Read dims
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (cursor + 8 > end) {
                fprintf(stderr, "Error: Truncated tensor dims\n");
                gguf_close(file);
                return false;
            }
            memcpy(&t->dims[d], cursor, 8);
            cursor += 8;
        }
        for (uint32_t d = t->n_dims; d < 4; d++) {
            t->dims[d] = 1;
        }

        // Read type
        if (cursor + 4 > end) {
            fprintf(stderr, "Error: Truncated tensor type\n");
            gguf_close(file);
            return false;
        }
        uint32_t type_val;
        memcpy(&type_val, cursor, 4);
        cursor += 4;
        t->type = (GGMLType)type_val;

        // Read offset
        if (cursor + 8 > end) {
            fprintf(stderr, "Error: Truncated tensor offset\n");
            gguf_close(file);
            return false;
        }
        memcpy(&t->offset, cursor, 8);
        cursor += 8;

        // Compute data size
        t->data_size = compute_tensor_size(t);
    }

    // Data section starts at the next alignment boundary after all headers/tensor info
    // GGUF aligns the data section to the nearest multiple of 32 bytes
    size_t header_end = (size_t)(cursor - data);
    size_t alignment = 32;
    size_t aligned = (header_end + alignment - 1) & ~(alignment - 1);
    file->data_start = data + aligned;

    if (file->data_start >= end) {
        fprintf(stderr, "Error: Data section starts beyond file end\n");
        gguf_close(file);
        return false;
    }

    printf("GGUF: header_size=%zu, data_offset=%zu, file_size=%zu\n",
           header_end, aligned, file->file_size);

    return true;
}

const TensorInfo* gguf_find_tensor(const GGUFFile* file, const char* name) {
    for (uint64_t i = 0; i < file->tensor_count; i++) {
        if (strcmp(file->tensors[i].name, name) == 0) {
            return &file->tensors[i];
        }
    }
    return nullptr;
}

const void* gguf_tensor_data(const GGUFFile* file, const TensorInfo* tensor) {
    return file->data_start + tensor->offset;
}

void gguf_close(GGUFFile* file) {
    if (file->tensors) {
        free(file->tensors);
        file->tensors = nullptr;
    }
    if (file->mapped_data && file->mapped_data != MAP_FAILED) {
        munmap(file->mapped_data, file->file_size);
        file->mapped_data = nullptr;
    }
    file->data_start = nullptr;
    file->tensor_count = 0;
    file->file_size = 0;
}

void gguf_print_tensors(const GGUFFile* file) {
    printf("\n%-60s %-6s %-24s %12s\n", "Tensor Name", "Type", "Shape", "Size (bytes)");
    printf("%-60s %-6s %-24s %12s\n",
           "------------------------------------------------------------",
           "------", "------------------------", "------------");

    size_t total_size = 0;
    for (uint64_t i = 0; i < file->tensor_count; i++) {
        const TensorInfo* t = &file->tensors[i];

        char shape[64];
        if (t->n_dims == 1) {
            snprintf(shape, sizeof(shape), "[%llu]", (unsigned long long)t->dims[0]);
        } else if (t->n_dims == 2) {
            snprintf(shape, sizeof(shape), "[%llu, %llu]",
                     (unsigned long long)t->dims[0], (unsigned long long)t->dims[1]);
        } else if (t->n_dims == 3) {
            snprintf(shape, sizeof(shape), "[%llu, %llu, %llu]",
                     (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
                     (unsigned long long)t->dims[2]);
        } else {
            snprintf(shape, sizeof(shape), "[%llu, %llu, %llu, %llu]",
                     (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
                     (unsigned long long)t->dims[2], (unsigned long long)t->dims[3]);
        }

        printf("%-60s %-6s %-24s %12zu\n", t->name, ggml_type_name(t->type), shape, t->data_size);
        total_size += t->data_size;
    }

    printf("\nTotal: %llu tensors, %.2f MB\n",
           (unsigned long long)file->tensor_count,
           (double)total_size / (1024.0 * 1024.0));
}

// --- Metadata access implementation ---

// Re-parse metadata section to find specific keys
// This is called on-demand since we skip metadata during initial load
static const uint8_t* find_metadata_key(const GGUFFile* file, const char* key,
                                        GGUFMetadataValueType* out_vtype) {
    const uint8_t* data = (const uint8_t*)file->mapped_data;
    const uint8_t* cursor = data + sizeof(GGUFHeader);
    const uint8_t* end = data + file->file_size;

    for (uint64_t i = 0; i < file->header.metadata_kv_count; i++) {
        GGUFString key_str;
        if (!read_gguf_string(&cursor, end, &key_str)) {
            return nullptr;
        }

        uint32_t vtype;
        if (cursor + 4 > end) return nullptr;
        memcpy(&vtype, cursor, 4);
        cursor += 4;

        const uint8_t* value_start = cursor;

        if (key_str.len == strlen(key) && memcmp(key_str.data, key, key_str.len) == 0) {
            *out_vtype = (GGUFMetadataValueType)vtype;
            return value_start;
        }

        // Skip the value
        if (!skip_metadata_value(&cursor, end, (GGUFMetadataValueType)vtype)) {
            return nullptr;
        }
    }

    return nullptr;
}

bool gguf_get_metadata_u32(const GGUFFile* file, const char* key, uint32_t* out) {
    GGUFMetadataValueType vtype;
    const uint8_t* value = find_metadata_key(file, key, &vtype);
    if (!value) return false;

    if (vtype != GGUFMetadataValueType::UINT32 &&
        vtype != GGUFMetadataValueType::INT32) {
        return false;
    }

    memcpy(out, value, 4);
    return true;
}

bool gguf_get_metadata_string(const GGUFFile* file, const char* key,
                              const char** out, uint64_t* out_len) {
    GGUFMetadataValueType vtype;
    const uint8_t* value = find_metadata_key(file, key, &vtype);
    if (!value) return false;

    if (vtype != GGUFMetadataValueType::STRING) return false;

    GGUFString s;
    const uint8_t* cursor = value;
    const uint8_t* end = (const uint8_t*)file->mapped_data + file->file_size;
    if (!read_gguf_string(&cursor, end, &s)) return false;

    *out = s.data;
    *out_len = s.len;
    return true;
}

bool gguf_get_metadata_string_array(const GGUFFile* file, const char* key,
                                     const char*** out_strings,
                                     uint64_t* out_count) {
    GGUFMetadataValueType vtype;
    const uint8_t* value = find_metadata_key(file, key, &vtype);
    if (!value) return false;

    if (vtype != GGUFMetadataValueType::ARRAY) return false;

    const uint8_t* cursor = value;
    const uint8_t* end = (const uint8_t*)file->mapped_data + file->file_size;

    // Read array type and length
    uint32_t elem_type;
    if (cursor + 12 > end) return false;
    memcpy(&elem_type, cursor, 4);
    cursor += 4;
    uint64_t arr_len;
    memcpy(&arr_len, cursor, 8);
    cursor += 8;

    if (elem_type != (uint32_t)GGUFMetadataValueType::STRING) {
        return false;
    }

    // Allocate temporary array to hold string pointers
    const char** strings = (const char**)malloc(arr_len * sizeof(const char*));
    if (!strings) return false;

    for (uint64_t i = 0; i < arr_len; i++) {
        GGUFString s;
        if (!read_gguf_string(&cursor, end, &s)) {
            free(strings);
            return false;
        }
        strings[i] = s.data;
    }

    *out_strings = strings;
    *out_count = arr_len;
    return true;
}

bool gguf_get_metadata_float_array(const GGUFFile* file, const char* key,
                                   const float** out_floats,
                                   uint64_t* out_count) {
    GGUFMetadataValueType vtype;
    const uint8_t* value = find_metadata_key(file, key, &vtype);
    if (!value) return false;

    if (vtype != GGUFMetadataValueType::ARRAY) return false;

    const uint8_t* cursor = value;
    const uint8_t* end = (const uint8_t*)file->mapped_data + file->file_size;

    // Read array type and length
    uint32_t elem_type;
    if (cursor + 12 > end) return false;
    memcpy(&elem_type, cursor, 4);
    cursor += 4;
    uint64_t arr_len;
    memcpy(&arr_len, cursor, 8);
    cursor += 8;

    if (elem_type != (uint32_t)GGUFMetadataValueType::FLOAT32) {
        return false;
    }

    if (cursor + arr_len * 4 > end) return false;

    *out_floats = (const float*)cursor;
    *out_count = arr_len;
    return true;
}

void gguf_print_metadata(const GGUFFile* file) {
    printf("\nGGUF Metadata (%llu keys):\n",
           (unsigned long long)file->header.metadata_kv_count);

    const uint8_t* data = (const uint8_t*)file->mapped_data;
    const uint8_t* cursor = data + sizeof(GGUFHeader);
    const uint8_t* end = data + file->file_size;

    for (uint64_t i = 0; i < file->header.metadata_kv_count; i++) {
        GGUFString key;
        if (!read_gguf_string(&cursor, end, &key)) break;

        uint32_t vtype;
        if (cursor + 4 > end) break;
        memcpy(&vtype, cursor, 4);
        cursor += 4;

        const char* type_name = "?";
        switch ((GGUFMetadataValueType)vtype) {
            case GGUFMetadataValueType::UINT8: type_name = "uint8"; break;
            case GGUFMetadataValueType::INT8: type_name = "int8"; break;
            case GGUFMetadataValueType::UINT16: type_name = "uint16"; break;
            case GGUFMetadataValueType::INT16: type_name = "int16"; break;
            case GGUFMetadataValueType::UINT32: type_name = "uint32"; break;
            case GGUFMetadataValueType::INT32: type_name = "int32"; break;
            case GGUFMetadataValueType::FLOAT32: type_name = "float32"; break;
            case GGUFMetadataValueType::BOOL: type_name = "bool"; break;
            case GGUFMetadataValueType::STRING: type_name = "string"; break;
            case GGUFMetadataValueType::ARRAY: type_name = "array"; break;
            case GGUFMetadataValueType::UINT64: type_name = "uint64"; break;
            case GGUFMetadataValueType::INT64: type_name = "int64"; break;
            case GGUFMetadataValueType::FLOAT64: type_name = "float64"; break;
            default: type_name = "unknown"; break;
        }

        printf("  %.*s : %s\n", (int)key.len, key.data, type_name);

        if (!skip_metadata_value(&cursor, end, (GGUFMetadataValueType)vtype)) {
            break;
        }
    }
}

} // namespace mgpu
