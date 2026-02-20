#include <gtest/gtest.h>
#include "test_utils.h"
#include "../src/models/gguf_loader.h"
#include <cstdio>
#include <filesystem>

using namespace mgpu;

class GGUFLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test GGML type size calculations
TEST_F(GGUFLoaderTest, GGMLTypeSize) {
    EXPECT_EQ(ggml_type_size(GGMLType::F32), 4);
    EXPECT_EQ(ggml_type_size(GGMLType::F16), 2);
    EXPECT_EQ(ggml_type_size(GGMLType::Q4_0), 18);
    EXPECT_EQ(ggml_type_size(GGMLType::Q8_0), 34);
    EXPECT_EQ(ggml_type_size(GGMLType::I8), 1);
    EXPECT_EQ(ggml_type_size(GGMLType::I16), 2);
    EXPECT_EQ(ggml_type_size(GGMLType::I32), 4);
}

// Test GGML block size calculations
TEST_F(GGUFLoaderTest, GGMLBlockSize) {
    EXPECT_EQ(ggml_type_block_size(GGMLType::F32), 1);
    EXPECT_EQ(ggml_type_block_size(GGMLType::F16), 1);
    EXPECT_EQ(ggml_type_block_size(GGMLType::Q4_0), 32);
    EXPECT_EQ(ggml_type_block_size(GGMLType::Q8_0), 32);
    EXPECT_EQ(ggml_type_block_size(GGMLType::Q2_K), 256);
    EXPECT_EQ(ggml_type_block_size(GGMLType::I8), 1);
}

// Test loading invalid file
TEST_F(GGUFLoaderTest, LoadInvalidFile) {
    GGUFFile file;
    EXPECT_FALSE(gguf_open(&file, "/nonexistent/file.gguf"));
}

// Test loading minimal valid GGUF file
TEST_F(GGUFLoaderTest, LoadMinimalGGUF) {
    std::string path = "/tmp/test_minimal.gguf";
    ASSERT_TRUE(test::create_test_gguf_file(path));

    GGUFFile file;
    EXPECT_TRUE(gguf_open(&file, path.c_str()));
    EXPECT_EQ(file.header.version, 3);
    EXPECT_EQ(file.header.tensor_count, 0);
    EXPECT_EQ(file.header.metadata_kv_count, 0);

    gguf_close(&file);
    std::remove(path.c_str());
}

// Test GGUF magic validation
TEST_F(GGUFLoaderTest, InvalidMagic) {
    std::vector<uint8_t> data;
    // Write invalid magic
    data.insert(data.end(), {0x00, 0x00, 0x00, 0x00});  // Invalid magic
    data.insert(data.end(), {0x03, 0x00, 0x00, 0x00});  // Version 3
    data.insert(data.end(), {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}); // 0 tensors
    data.insert(data.end(), {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}); // 0 metadata

    std::string path = "/tmp/test_invalid_magic.gguf";
    std::ofstream f(path, std::ios::binary);
    f.write((char*)data.data(), data.size());

    GGUFFile file;
    EXPECT_FALSE(gguf_open(&file, path.c_str()));

    std::remove(path.c_str());
}

// Test print metadata function (just ensure it doesn't crash)
TEST_F(GGUFLoaderTest, PrintMetadata) {
    std::string path = "/tmp/test_print.gguf";
    ASSERT_TRUE(test::create_test_gguf_file(path));

    GGUFFile file;
    EXPECT_TRUE(gguf_open(&file, path.c_str()));

    // This should not crash
    gguf_print_metadata(&file);

    gguf_close(&file);
    std::remove(path.c_str());
}

// Test print tensors function
TEST_F(GGUFLoaderTest, PrintTensors) {
    std::string path = "/tmp/test_tensors.gguf";
    ASSERT_TRUE(test::create_test_gguf_file(path));

    GGUFFile file;
    EXPECT_TRUE(gguf_open(&file, path.c_str()));

    // This should not crash
    gguf_print_tensors(&file);

    gguf_close(&file);
    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
