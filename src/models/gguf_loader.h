#pragma once

#include <cstdint>
#include <cstddef>

namespace mgpu {

// GGUF file format constants
constexpr uint32_t GGUF_MAGIC = 0x46475547; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    COUNT
};

enum class GGUFMetadataValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12
};

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct GGUFString {
    uint64_t len;
    const char* data; // points into mmap'd region
};

struct TensorInfo {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    GGMLType type;
    uint64_t offset; // offset from start of data section
    size_t data_size; // computed size in bytes
};

struct GGUFFile {
    void* mapped_data;     // mmap'd file
    size_t file_size;
    GGUFHeader header;
    const uint8_t* data_start; // pointer to tensor data section
    TensorInfo* tensors;
    uint64_t tensor_count;
};

// Open and parse a GGUF file (memory-mapped)
bool gguf_open(GGUFFile* file, const char* filepath);

// Find a tensor by name, returns nullptr if not found
const TensorInfo* gguf_find_tensor(const GGUFFile* file, const char* name);

// Get raw pointer to tensor data
const void* gguf_tensor_data(const GGUFFile* file, const TensorInfo* tensor);

// Get size in bytes for a GGML type per element (for quantized types, per block)
size_t ggml_type_size(GGMLType type);

// Get block size for quantized types (number of elements per block)
int ggml_type_block_size(GGMLType type);

// Close and unmap the file
void gguf_close(GGUFFile* file);

// Print all tensor names, shapes, types
void gguf_print_tensors(const GGUFFile* file);

// --- Metadata access functions ---

// Read a uint32 metadata value by key name (e.g., "tokenizer.ggml.bos_token_id")
// Returns true if found and value stored in *out
bool gguf_get_metadata_u32(const GGUFFile* file, const char* key, uint32_t* out);

// Read a string metadata value by key name
// Returns true if found, string stored in *out (points into mmap'd region)
bool gguf_get_metadata_string(const GGUFFile* file, const char* key,
                              const char** out, uint64_t* out_len);

// Read a string array metadata value by key name
// Returns true if found, array data stored in pointers (point into mmap'd region)
bool gguf_get_metadata_string_array(const GGUFFile* file, const char* key,
                                     const char*** out_strings,
                                     uint64_t* out_count);

// Read a float array metadata value by key name
// Returns true if found, data stored in pointer (points into mmap'd region)
bool gguf_get_metadata_float_array(const GGUFFile* file, const char* key,
                                   const float** out_floats,
                                   uint64_t* out_count);

// Print all metadata keys (useful for debugging)
void gguf_print_metadata(const GGUFFile* file);

} // namespace mgpu
