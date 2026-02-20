#pragma once

#include <cstdint>
#include <cstddef>

// Forward declaration for GGUFFile
struct GGUFFile;

namespace mgpu {

// Special token IDs for Phi-1.5 / Moondream2
constexpr int TOKEN_BOS = 1;        // <|endoftext|> or <s>
constexpr int TOKEN_EOS = 2;        // </s>
constexpr int TOKEN_PAD = 0;
constexpr int TOKEN_UNKNOWN = 0;

struct TokenizerVocab {
    char** tokens;          // token strings (UTF-8)
    float* scores;          // merge scores (for BPE)
    int vocab_size;
    int bos_id;
    int eos_id;
};

// Load vocabulary from a GGUF file's metadata
// GGUF stores tokenizer vocab under keys like:
//   tokenizer.ggml.tokens (string array)
//   tokenizer.ggml.scores (float array)
//   tokenizer.ggml.bos_token_id (uint32)
//   tokenizer.ggml.eos_token_id (uint32)
// If gguf_path is nullptr, uses the pre-loaded GGUF file (for internal use)
bool tokenizer_load_from_gguf(TokenizerVocab* vocab, const char* gguf_path);

// Load tokenizer from an already-opened GGUF file (internal use)
bool tokenizer_load_from_gguf_file(TokenizerVocab* vocab, const GGUFFile* gguf);

// Load vocabulary from a simple text file (one token per line)
// Format: "token_string score" per line (tab or space separated)
// This is a fallback for when GGUF metadata parsing is too complex
bool tokenizer_load_from_file(TokenizerVocab* vocab, const char* vocab_path);

// Encode text to token IDs using BPE
// Returns number of tokens written to output (at most max_tokens)
int tokenizer_encode(const TokenizerVocab* vocab, const char* text,
                     int* output, int max_tokens);

// Decode a single token ID to its string representation
// Returns pointer to static/internal string (do NOT free)
const char* tokenizer_decode(const TokenizerVocab* vocab, int token_id);

// Decode a sequence of token IDs to text
// Writes to output buffer, returns number of bytes written
int tokenizer_decode_sequence(const TokenizerVocab* vocab,
                              const int* tokens, int num_tokens,
                              char* output, int max_bytes);

// Free tokenizer resources
void tokenizer_free(TokenizerVocab* vocab);

} // namespace mgpu
