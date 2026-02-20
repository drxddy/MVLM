#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mgpu {

// --- Helpers ---

// Returns the number of bytes in a UTF-8 character given its leading byte
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

// Find a token string in the vocab, return its index or -1
static int vocab_lookup(const TokenizerVocab* vocab, const char* str, int len) {
    for (int i = 0; i < vocab->vocab_size; i++) {
        if (vocab->tokens[i] &&
            (int)strlen(vocab->tokens[i]) == len &&
            memcmp(vocab->tokens[i], str, len) == 0) {
            return i;
        }
    }
    return -1;
}

// Unescape a token string in-place: handle \n, \t, etc.
static void unescape_token(char* s) {
    char* dst = s;
    const char* src = s;
    while (*src) {
        if (*src == '\\' && *(src + 1)) {
            switch (*(src + 1)) {
                case 'n':  *dst++ = '\n'; src += 2; break;
                case 't':  *dst++ = '\t'; src += 2; break;
                case 'r':  *dst++ = '\r'; src += 2; break;
                case '\\': *dst++ = '\\'; src += 2; break;
                default:   *dst++ = *src++; break;
            }
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
}

// --- Load from GGUF (stub) ---

bool tokenizer_load_from_gguf(TokenizerVocab* /*vocab*/, const char* /*gguf_path*/) {
    fprintf(stderr, "tokenizer: GGUF tokenizer loading not yet implemented — "
                    "use tokenizer_load_from_file\n");
    return false;
}

// --- Load from text file ---

bool tokenizer_load_from_file(TokenizerVocab* vocab, const char* vocab_path) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open vocab file: %s\n", vocab_path);
        return false;
    }

    // First pass: count lines
    int count = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '\0' && line[0] != '\n') count++;
    }
    rewind(f);

    vocab->vocab_size = count;
    vocab->tokens = (char**)calloc(count, sizeof(char*));
    vocab->scores = (float*)calloc(count, sizeof(float));
    vocab->bos_id = TOKEN_BOS;
    vocab->eos_id = TOKEN_EOS;

    if (!vocab->tokens || !vocab->scores) {
        fprintf(stderr, "tokenizer: allocation failed for vocab_size=%d\n", count);
        fclose(f);
        return false;
    }

    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < count) {
        // Strip trailing newline/carriage return
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        // Try to split on last tab or space for score
        float score = 0.0f;
        char* sep = strrchr(line, '\t');
        if (!sep) sep = strrchr(line, ' ');

        if (sep && sep != line) {
            // Check if everything after separator looks like a number
            char* endptr = nullptr;
            float val = strtof(sep + 1, &endptr);
            if (endptr && endptr != sep + 1 && *endptr == '\0') {
                score = val;
                *sep = '\0';
            }
        }

        vocab->tokens[idx] = strdup(line);
        unescape_token(vocab->tokens[idx]);
        vocab->scores[idx] = score;
        idx++;
    }

    fclose(f);
    vocab->vocab_size = idx;
    printf("tokenizer: loaded %d tokens from %s\n", idx, vocab_path);
    return true;
}

// --- BPE Encode ---

int tokenizer_encode(const TokenizerVocab* vocab, const char* text,
                     int* output, int max_tokens) {
    if (!text || !vocab || max_tokens <= 0) return 0;

    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    // Allocate working arrays for token IDs and their string representations
    // Maximum possible tokens = number of UTF-8 characters in input
    int max_work = text_len; // at most one token per byte
    int* work_ids = (int*)malloc(max_work * sizeof(int));
    char** work_strs = (char**)malloc(max_work * sizeof(char*));
    if (!work_ids || !work_strs) {
        free(work_ids);
        free(work_strs);
        return 0;
    }

    // Step 1: Split input into individual UTF-8 characters and look up each
    int n = 0;
    const char* p = text;
    while (*p && n < max_work) {
        int clen = utf8_char_len((unsigned char)*p);
        if (p + clen > text + text_len) break;

        int tok_id = vocab_lookup(vocab, p, clen);
        if (tok_id < 0) {
            // Try single-byte fallback for each byte in this UTF-8 char
            for (int b = 0; b < clen && n < max_work; b++) {
                // Look for byte fallback token like <0xAB>
                char hex_token[8];
                snprintf(hex_token, sizeof(hex_token), "<0x%02X>",
                         (unsigned char)p[b]);
                int byte_id = vocab_lookup(vocab, hex_token, (int)strlen(hex_token));
                if (byte_id >= 0) {
                    work_ids[n] = byte_id;
                    work_strs[n] = strdup(hex_token);
                } else {
                    work_ids[n] = TOKEN_UNKNOWN;
                    work_strs[n] = strdup("");
                }
                n++;
            }
        } else {
            work_ids[n] = tok_id;
            work_strs[n] = (char*)malloc(clen + 1);
            memcpy(work_strs[n], p, clen);
            work_strs[n][clen] = '\0';
            n++;
        }
        p += clen;
    }

    // Step 2: Iteratively merge highest-scoring adjacent pairs
    char merge_buf[8192];
    while (n >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < n - 1; i++) {
            // Build concatenation of work_strs[i] + work_strs[i+1]
            int len_a = (int)strlen(work_strs[i]);
            int len_b = (int)strlen(work_strs[i + 1]);
            if (len_a + len_b + 1 > (int)sizeof(merge_buf)) continue;

            memcpy(merge_buf, work_strs[i], len_a);
            memcpy(merge_buf + len_a, work_strs[i + 1], len_b);
            merge_buf[len_a + len_b] = '\0';

            int merged_id = vocab_lookup(vocab, merge_buf, len_a + len_b);
            if (merged_id >= 0 && vocab->scores[merged_id] > best_score) {
                best_score = vocab->scores[merged_id];
                best_idx = i;
                best_id = merged_id;
            }
        }

        if (best_idx < 0) break; // no more merges possible

        // Merge: replace pair at best_idx with merged token
        int len_a = (int)strlen(work_strs[best_idx]);
        int len_b = (int)strlen(work_strs[best_idx + 1]);
        char* merged_str = (char*)malloc(len_a + len_b + 1);
        memcpy(merged_str, work_strs[best_idx], len_a);
        memcpy(merged_str + len_a, work_strs[best_idx + 1], len_b);
        merged_str[len_a + len_b] = '\0';

        free(work_strs[best_idx]);
        free(work_strs[best_idx + 1]);

        work_ids[best_idx] = best_id;
        work_strs[best_idx] = merged_str;

        // Shift remaining elements left
        for (int i = best_idx + 1; i < n - 1; i++) {
            work_ids[i] = work_ids[i + 1];
            work_strs[i] = work_strs[i + 1];
        }
        n--;
    }

    // Step 3: Copy to output
    int out_count = n < max_tokens ? n : max_tokens;
    for (int i = 0; i < out_count; i++) {
        output[i] = work_ids[i];
    }

    // Cleanup
    for (int i = 0; i < n; i++) {
        free(work_strs[i]);
    }
    free(work_ids);
    free(work_strs);

    return out_count;
}

// --- Decode ---

const char* tokenizer_decode(const TokenizerVocab* vocab, int token_id) {
    if (!vocab || token_id < 0 || token_id >= vocab->vocab_size) {
        return "";
    }
    return vocab->tokens[token_id] ? vocab->tokens[token_id] : "";
}

int tokenizer_decode_sequence(const TokenizerVocab* vocab,
                              const int* tokens, int num_tokens,
                              char* output, int max_bytes) {
    if (!vocab || !tokens || !output || max_bytes <= 0) return 0;

    int pos = 0;

    for (int t = 0; t < num_tokens; t++) {
        const char* tok_str = tokenizer_decode(vocab, tokens[t]);
        if (!tok_str || tok_str[0] == '\0') continue;

        // Handle byte fallback tokens like <0xAB>
        if (tok_str[0] == '<' && tok_str[1] == '0' && tok_str[2] == 'x' &&
            strlen(tok_str) == 6 && tok_str[5] == '>') {
            char hex[3] = {tok_str[3], tok_str[4], '\0'};
            unsigned int byte_val = 0;
            if (sscanf(hex, "%x", &byte_val) == 1 && pos < max_bytes - 1) {
                output[pos++] = (char)byte_val;
            }
            continue;
        }

        // Process token string, replacing ▁ with space
        const char* s = tok_str;
        while (*s && pos < max_bytes - 1) {
            if ((unsigned char)s[0] == 0xE2 &&
                (unsigned char)s[1] == 0x96 &&
                (unsigned char)s[2] == 0x81) {
                output[pos++] = ' ';
                s += 3;
            } else {
                output[pos++] = *s++;
            }
        }
    }

    if (pos < max_bytes) output[pos] = '\0';
    else output[max_bytes - 1] = '\0';

    return pos;
}

// --- Free ---

void tokenizer_free(TokenizerVocab* vocab) {
    if (!vocab) return;
    if (vocab->tokens) {
        for (int i = 0; i < vocab->vocab_size; i++) {
            free(vocab->tokens[i]);
        }
        free(vocab->tokens);
        vocab->tokens = nullptr;
    }
    if (vocab->scores) {
        free(vocab->scores);
        vocab->scores = nullptr;
    }
    vocab->vocab_size = 0;
}

} // namespace mgpu
