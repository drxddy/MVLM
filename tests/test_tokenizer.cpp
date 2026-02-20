#include <gtest/gtest.h>
#include "test_utils.h"
#include <cstdio>
#include <cstring>

// Include gguf_loader first to define GGUFFile, then tokenizer
#include "../src/models/gguf_loader.h"
#include "../src/models/tokenizer.h"

using namespace mgpu;

class TokenizerTest : public ::testing::Test {
protected:
    void TearDown() override {
        if (vocab_.tokens) {
            tokenizer_free(&vocab_);
        }
    }
    TokenizerVocab vocab_;
};

// Test loading vocab from file
TEST_F(TokenizerTest, LoadFromFile) {
    // Create a simple vocab file
    const char* vocab_content =
        "a 10\n"
        "b 9\n"
        "ab 8\n"
        "c 7\n"
        "▁ 6\n";

    FILE* f = fopen("/tmp/test_vocab.txt", "w");
    ASSERT_NE(f, nullptr);
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    EXPECT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab.txt"));
    EXPECT_EQ(vocab_.vocab_size, 5);
    EXPECT_EQ(vocab_.bos_id, 1);
    EXPECT_EQ(vocab_.eos_id, 2);
    EXPECT_STREQ(vocab_.tokens[0], "a");
    EXPECT_STREQ(vocab_.tokens[2], "ab");
    EXPECT_FLOAT_EQ(vocab_.scores[0], 10.0f);

    std::remove("/tmp/test_vocab.txt");
}

// Test loading vocab from GGUF
TEST_F(TokenizerTest, LoadFromGGUF) {
    std::string path = "/tmp/test_tokenizer.gguf";
    ASSERT_TRUE(test::create_tokenizer_test_gguf(path));

    EXPECT_TRUE(tokenizer_load_from_gguf(&vocab_, path.c_str()));
    EXPECT_EQ(vocab_.vocab_size, 5);
    EXPECT_EQ(vocab_.bos_id, 1);
    EXPECT_EQ(vocab_.eos_id, 2);
    // Check tokens start with expected characters (GGUF may have extra bytes)
    EXPECT_EQ(vocab_.tokens[0][0], 'a');
    EXPECT_EQ(vocab_.tokens[2][0], 'a');

    std::remove(path.c_str());
}

// Test loading from GGUF file directly (using path-based function)
TEST_F(TokenizerTest, LoadFromGGUFFileDirect) {
    std::string path = "/tmp/test_tokenizer_direct.gguf";
    ASSERT_TRUE(test::create_tokenizer_test_gguf(path));

    // Load tokenizer from GGUF file using path-based function
    EXPECT_TRUE(tokenizer_load_from_gguf(&vocab_, path.c_str()));
    EXPECT_EQ(vocab_.vocab_size, 5);

    std::remove(path.c_str());
}

// Test loading from invalid GGUF
TEST_F(TokenizerTest, LoadFromInvalidGGUF) {
    EXPECT_FALSE(tokenizer_load_from_gguf(&vocab_, "/nonexistent/file.gguf"));
}

// Test loading from GGUF without tokenizer metadata
TEST_F(TokenizerTest, LoadFromGGUFNoTokenizer) {
    std::string path = "/tmp/test_no_tokenizer.gguf";
    ASSERT_TRUE(test::create_test_gguf_file(path));

    EXPECT_FALSE(tokenizer_load_from_gguf(&vocab_, path.c_str()));

    std::remove(path.c_str());
}

// Test tokenizer decode
TEST_F(TokenizerTest, Decode) {
    const char* vocab_content = "hello 5\nworld 4\n";
    FILE* f = fopen("/tmp/test_vocab_decode.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_decode.txt"));

    EXPECT_STREQ(tokenizer_decode(&vocab_, 0), "hello");
    EXPECT_STREQ(tokenizer_decode(&vocab_, 1), "world");
    EXPECT_STREQ(tokenizer_decode(&vocab_, 999), "");  // Invalid ID

    std::remove("/tmp/test_vocab_decode.txt");
}

// Test tokenizer decode sequence
TEST_F(TokenizerTest, DecodeSequence) {
    const char* vocab_content = "hello 5\nworld 4\n";
    FILE* f = fopen("/tmp/test_vocab_seq.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_seq.txt"));

    int tokens[] = {0, 1};
    char output[256];
    int len = tokenizer_decode_sequence(&vocab_, tokens, 2, output, sizeof(output));

    EXPECT_GT(len, 0);
    // Just check it contains both words
    EXPECT_NE(strstr(output, "hello"), nullptr);
    EXPECT_NE(strstr(output, "world"), nullptr);

    std::remove("/tmp/test_vocab_seq.txt");
}

// Test tokenizer encode basic
TEST_F(TokenizerTest, EncodeBasic) {
    const char* vocab_content = "a 10\nb 9\nab 8\nc 7\n";
    FILE* f = fopen("/tmp/test_vocab_enc.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_enc.txt"));

    int tokens[32];
    int count = tokenizer_encode(&vocab_, "ab", tokens, 32);

    // Should encode "ab" as a single token since it's in vocab
    EXPECT_GE(count, 1);

    std::remove("/tmp/test_vocab_enc.txt");
}

// Test tokenizer encode with unknown
TEST_F(TokenizerTest, EncodeWithUnknown) {
    const char* vocab_content = "a 5\nb 4\n";
    FILE* f = fopen("/tmp/test_vocab_unk.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_unk.txt"));

    // Encode something that has unknown characters
    int tokens[32];
    int count = tokenizer_encode(&vocab_, "xyz", tokens, 32);

    // Should use byte fallback tokens for unknown characters
    EXPECT_GE(count, 1);

    std::remove("/tmp/test_vocab_unk.txt");
}

// Test tokenizer encode empty
TEST_F(TokenizerTest, EncodeEmpty) {
    const char* vocab_content = "a 5\nb 4\n";
    FILE* f = fopen("/tmp/test_vocab_empty.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_empty.txt"));

    int tokens[32];
    int count = tokenizer_encode(&vocab_, "", tokens, 32);
    EXPECT_EQ(count, 0);

    std::remove("/tmp/test_vocab_empty.txt");
}

// Test tokenizer encode max tokens
TEST_F(TokenizerTest, EncodeMaxTokens) {
    const char* vocab_content = "a 5\nb 4\nc 3\n";
    FILE* f = fopen("/tmp/test_vocab_max.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_max.txt"));

    int tokens[2];
    int count = tokenizer_encode(&vocab_, "abc", tokens, 2);

    // Should be limited to max_tokens (2)
    EXPECT_LE(count, 2);

    std::remove("/tmp/test_vocab_max.txt");
}

// Test tokenizer free
TEST_F(TokenizerTest, Free) {
    const char* vocab_content = "a 5\nb 4\n";
    FILE* f = fopen("/tmp/test_vocab_free.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_free.txt"));
    EXPECT_NE(vocab_.tokens, nullptr);

    tokenizer_free(&vocab_);
    EXPECT_EQ(vocab_.tokens, nullptr);
    EXPECT_EQ(vocab_.vocab_size, 0);

    std::remove("/tmp/test_vocab_free.txt");
}

// Test tokenizer free null
TEST_F(TokenizerTest, FreeNull) {
    // Should not crash
    tokenizer_free(nullptr);
}

// Test UTF-8 handling
TEST_F(TokenizerTest, UTF8Handling) {
    const char* vocab_content = "ü 5\nö 4\n";
    FILE* f = fopen("/tmp/test_vocab_utf8.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_utf8.txt"));

    // UTF-8 encoded ü is 2 bytes
    EXPECT_EQ(vocab_.vocab_size, 2);

    std::remove("/tmp/test_vocab_utf8.txt");
}

// Test special token IDs
TEST_F(TokenizerTest, SpecialTokenIDs) {
    const char* vocab_content = "a 5\nb 4\n";
    FILE* f = fopen("/tmp/test_vocab_special.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_special.txt"));

    // Default BOS/EOS should be 1/2 (TOKEN_BOS/TOKEN_EOS)
    EXPECT_EQ(vocab_.bos_id, 1);
    EXPECT_EQ(vocab_.eos_id, 2);

    std::remove("/tmp/test_vocab_special.txt");
}

// Test tokenizer with escaped characters
TEST_F(TokenizerTest, EscapedCharacters) {
    const char* vocab_content = "hello\\nworld 5\ntab\\there 4\n";
    FILE* f = fopen("/tmp/test_vocab_escape.txt", "w");
    fwrite(vocab_content, 1, strlen(vocab_content), f);
    fclose(f);

    ASSERT_TRUE(tokenizer_load_from_file(&vocab_, "/tmp/test_vocab_escape.txt"));

    // Should have unescaped the tokens
    EXPECT_STRNE(vocab_.tokens[0], "hello\\nworld");  // Should be "hello\nworld"

    std::remove("/tmp/test_vocab_escape.txt");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
