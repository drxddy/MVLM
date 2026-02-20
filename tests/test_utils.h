#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

namespace mgpu {
namespace test {

// Test utilities for creating test GGUF files
struct TestGGUFBuilder {
    std::vector<uint8_t> data;

    void write_bytes(const void* ptr, size_t size) {
        const uint8_t* p = (const uint8_t*)ptr;
        data.insert(data.end(), p, p + size);
    }

    void write_u32(uint32_t v) { write_bytes(&v, 4); }
    void write_u64(uint64_t v) { write_bytes(&v, 8); }
    void write_string(const std::string& s) {
        write_u64(s.size());
        data.insert(data.end(), s.begin(), s.end());
    }

    // Write GGUF header
    void write_header(uint32_t version, uint64_t tensor_count, uint64_t metadata_count) {
        write_u32(0x46475547);  // GGUF magic
        write_u32(version);
        write_u64(tensor_count);
        write_u64(metadata_count);
    }

    // Write a metadata key-value pair
    void write_metadata_string(const std::string& key, const std::string& value) {
        write_string(key);
        write_u32(8);  // STRING type
        write_string(value);
    }

    void write_metadata_u32(const std::string& key, uint32_t value) {
        write_string(key);
        write_u32(4);  // UINT32 type
        write_u32(value);
    }

    void write_metadata_float_array(const std::string& key, const float* values, size_t count) {
        write_string(key);
        write_u32(9);  // ARRAY type
        write_u32(6);  // FLOAT32 element type
        write_u64(count);
        data.insert(data.end(), (const uint8_t*)values, (const uint8_t*)(values + count));
    }

    void write_metadata_string_array(const std::string& key, const std::string* values, size_t count) {
        write_string(key);
        write_u32(9);  // ARRAY type
        write_u32(8);  // STRING element type
        write_u64(count);
        for (size_t i = 0; i < count; i++) {
            write_string(values[i]);
        }
    }

    // Write a tensor info entry
    void write_tensor(const std::string& name, uint32_t n_dims, const uint64_t* dims, uint32_t type, uint64_t offset) {
        write_string(name);
        write_u32(n_dims);
        for (uint32_t i = 0; i < n_dims; i++) {
            write_u64(dims[i]);
        }
        write_u32(type);
        write_u64(offset);
    }

    // Pad to 32-byte alignment
    void pad_alignment() {
        size_t align = 32;
        size_t pad = (align - (data.size() % align)) % align;
        for (size_t i = 0; i < pad; i++) {
            data.push_back(0);
        }
    }

    // Save to file
    bool save_to_file(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write((char*)data.data(), data.size());
        return f.good();
    }
};

// Create a minimal GGUF file for testing
inline bool create_test_gguf_file(const std::string& path) {
    TestGGUFBuilder builder;

    // Write minimal header (version 3, 0 tensors, 0 metadata)
    builder.write_header(3, 0, 0);

    // Pad to alignment (needed even with 0 tensors)
    builder.pad_alignment();

    // Add at least 32 bytes of dummy data so file is valid
    uint8_t dummy[32] = {0};
    builder.data.insert(builder.data.end(), dummy, dummy + 32);

    return builder.save_to_file(path);
}

// Create GGUF file with tokenizer metadata for testing
inline bool create_tokenizer_test_gguf(const std::string& path) {
    TestGGUFBuilder builder;

    // Write header: version 3, 1 tensor, 3 metadata entries
    builder.write_header(3, 1, 3);

    // Metadata: tokenizer.ggml.tokens (string array)
    std::string tokens[] = {"a", "b", "ab", "c", "▁"};
    builder.write_metadata_string_array("tokenizer.ggml.tokens", tokens, 5);

    // Metadata: tokenizer.ggml.scores (float array)
    float scores[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    builder.write_metadata_float_array("tokenizer.ggml.scores", scores, 5);

    // Metadata: tokenizer.ggml.bos_token_id
    builder.write_metadata_u32("tokenizer.ggml.bos_token_id", 1);

    // Metadata: tokenizer.ggml.eos_token_id
    builder.write_metadata_u32("tokenizer.ggml.eos_token_id", 2);

    // Tensor info for a dummy weight tensor
    uint64_t dims[2] = {10, 20};
    builder.write_tensor("model.embed_tokens.weight", 2, dims, 1, 0);  // type 1 = F16

    // Pad to alignment
    builder.pad_alignment();

    // Write dummy tensor data
    uint8_t dummy_data[400] = {0};
    builder.data.insert(builder.data.end(), dummy_data, dummy_data + 400);

    return builder.save_to_file(path);
}

// Create GGUF file with different token names
inline bool create_tokenizer_test_gguf_v2(const std::string& path) {
    TestGGUFBuilder builder;

    // Write header: version 3, 1 tensor, 2 metadata entries
    builder.write_header(3, 1, 2);

    // Try alternative key name: tokenizer.tokens
    std::string tokens[] = {"hello", "world"};
    builder.write_metadata_string_array("tokenizer.tokens", tokens, 2);

    // Tensor info
    uint64_t dims[2] = {10, 20};
    builder.write_tensor("model.embed_tokens.weight", 2, dims, 1, 0);

    // Pad to alignment
    builder.pad_alignment();

    // Write tensor data (offset 0 = start of data section after alignment)
    uint8_t dummy_data[400] = {0};
    builder.data.insert(builder.data.end(), dummy_data, dummy_data + 400);

    return builder.save_to_file(path);
}

} // namespace test
} // namespace mgpu
