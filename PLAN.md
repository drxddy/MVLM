# MGPU Implementation Plan

On-device VLM inference engine for Qualcomm Adreno GPUs, targeting Moondream2 at <300ms latency.

## Phase 0: Environment Setup ✅
- [x] Project scaffolding (CMake, directory structure, scripts)
- [x] OpenCL device detection + extension probing (`mgpu_device_info`)
- [x] Android NDK cross-compilation support (`build_android.sh`)
- [x] ADB deploy/run script (`push_and_run.sh`)
- [x] GPU perf mode script (`perf_mode.sh`)
- [ ] Verify on Snapdragon 8 Gen 3/Elite hardware

## Phase 1: GEMM Kernels ✅
- [x] Naive GEMM (correctness baseline)
- [x] Tiled GEMM (local memory, workgroup tiling)
- [x] Image-based GEMM (TP/L1 texture cache for weights)
- [x] GEMV (M=1 decode, workgroup reduction)
- [x] Benchmark harness (`mgpu_bench`)
- [ ] Run benchmarks on Adreno, reach >50% peak FP16 GFLOPS
- [ ] Auto-tune tile sizes per device

## Phase 2: Transformer Primitives ✅
- [x] RMSNorm (subgroup-optimized + local memory fallback)
- [x] SiLU / GELU activations (vectorized)
- [x] Softmax (numerically stable, 3-pass)
- [x] Fused SiLU-gated MLP (`silu_gate_multiply`)
- [x] RoPE (3D dispatch, precomputed tables)
- [x] Embedding lookup (vectorized)
- [x] Attention prefill (full sequence)
- [x] Attention decode (single token vs KV-cache)

## Phase 3: Model Graph Integration ✅
- [x] Kernel dispatch layer (`engine/compute.h/cpp`)
- [x] GGUF weight loader (`models/gguf_loader.h/cpp`)
- [x] GPU weight upload (images for matrices, buffers for vectors)
- [x] RoPE table precomputation + upload
- [x] KV-cache allocation and management
- [x] Scratch buffer pool for activations
- [x] Transformer forward pass (embedding → layers → logits)
- [x] CLI integration (`mgpu_cli --model ... --kernels ...`)
- [x] Residual add kernel (`vector_add` in activations.cl)
- [ ] Verify forward pass end-to-end with actual GGUF weights

## Phase 4: Vision Encoder 🔲
- [x] Image preprocessing kernel (resize + normalize, hardware bilinear)
- [x] Patch embedding kernel (conv2d-like with image weights)
- [ ] SigLIP encoder layer wiring (27 transformer layers)
- [ ] Vision→LLM projection layer
- [ ] Zero-copy camera input via AHardwareBuffer

## Phase 5: End-to-End Pipeline 🟡
- [x] Tokenizer (BPE encoder/decoder, vocab file loading)
- [x] Greedy decode loop with argmax sampling
- [x] `moondream2_generate()` — full prompt→tokens→inference→output text
- [x] CLI: `--prompt`, `--vocab`, `--max-tokens` flags
- [x] Prefill/decode latency and tok/s statistics
- [x] GGUF metadata tokenizer loading (currently uses vocab file)
- [ ] Recordable queues for decode loop (Qualcomm extension)
- [ ] Pipeline event management (`engine/pipeline.h/cpp`)

## Phase 6: Optimization & Profiling 🔲
- [ ] Kernel fusion (RMSNorm + GEMM, attention score + softmax)
- [ ] Workgroup size auto-tuning
- [ ] On-chip global memory for KV-cache (Qualcomm extension)
- [ ] Quantized weight support (Q4_0, Q8_0 dequantize kernels)
- [ ] Profile session integration across full forward pass

## Phase 7: Demo App 🔲
- [ ] Android camera preview + real-time VLM inference
- [ ] End-to-end latency measurement
- [ ] Memory usage optimization for mobile

## Current Status

**Phase 5 Complete** — full text generation pipeline is wired up.

### Implemented
- ✅ GGUF tokenizer metadata loading
- ✅ Android NDK build support (build_android.sh)
- ✅ Android app with JNI wrapper (APK builds)
- ✅ Unit tests (test_gguf, test_tokenizer, test_device)
- ✅ OpenCL kernels (GEMM, Attention, RMSNorm, SiLU, RoPE, etc.)
- ✅ GGUF loader with metadata access
- ✅ BPE tokenizer with GGUF support

### Technical Details
- OpenCL 1.2+ with Qualcomm extensions
- Image-based GEMM (L1 texture cache for weights)
- Subgroup operations for reductions
- FP16 precision throughout
- Memory-mapped GGUF loading

### Next Steps
- Test on real Adreno device
- Complete vision encoder (Phase 4)
- Pipeline event management
