# Worker B - Infrastructure Developer Tasks

**Your Focus**: GPU optimization, LLM infrastructure, production features
**Your Directories**: `src/orchestration/local_llm/`, `src/gpu/`, `src/quantum/src/`, `src/production/`, `tests/`
**Your Time**: 130 hours over 3-4 weeks

---

## WEEK 1: GGUF Loader + Attention (40 hours)

### Day 1-3: GGUF Model Loader (24 hours)

**Monday Morning** (4h):
```bash
git checkout -b worker-b-infrastructure
cd src/orchestration/local_llm

# Create gguf_loader.rs
touch gguf_loader.rs
```

**Task 3.1.1**: GGUF v3 Parser
- [ ] Read file header (magic number, version, metadata count)
- [ ] Parse tensor metadata (name, shape, dtype)
- [ ] Read tensor data section
- [ ] Create `GGUFLoader` struct
- [ ] Test with dummy GGUF file

**Monday Afternoon + Tuesday** (12h):
**Task 3.1.2**: Quantization Handling
- [ ] Implement INT4 dequantization
- [ ] Implement INT8 dequantization
- [ ] FP16 conversion
- [ ] Test with quantized tensors

**Wednesday** (8h):
**Task 3.1.3**: GPU Upload
- [ ] Upload all weights to GPU as CudaSlice
- [ ] Organize for efficient access
- [ ] Create `GpuModelWeights` struct

**Task 3.1.4**: Architecture Detection
- [ ] Detect Llama vs Mistral vs Falcon
- [ ] Extract hyperparameters from metadata
- [ ] Validate model structure

### Day 4-5: Proper Attention (16 hours)

**Thursday** (8h):
**Task 3.2.1**: Q/K/V Projections
- [ ] Fix line 117 in `gpu_transformer.rs`
- [ ] Implement: `Q = input @ Wq` (use existing matmul kernel)
- [ ] Implement: `K = input @ Wk`
- [ ] Implement: `V = input @ Wv`
- [ ] Keep Q/K/V on GPU (no downloads)

**Task 3.2.2**: Validate Attention
- [ ] Test attention scores are correct
- [ ] Compare with reference implementation

**Friday** (8h):
**Task 3.2.3**: Attention Masking
- [ ] Implement causal mask for decoder
- [ ] Mask kernel or modify attention kernel
- [ ] Test autoregressive generation

---

## WEEK 2: KV-Cache + Sampling + BPE (40 hours)

### Day 6-7: KV-Cache (15 hours)

**Monday** (7h):
**Task 3.3.1**: Cache Data Structures
```bash
touch kv_cache.rs
```
- [ ] Create `KVCache` struct
- [ ] Cached K/V as `Vec<CudaSlice<f32>>`
- [ ] One cache per layer

**Task 3.3.2**: GPU Concatenation
- [ ] Implement cache append on GPU
- [ ] Create `concat_cache` kernel:
```cuda
__global__ void concat_cache(
    float* old_cache, float* new_item,
    float* updated_cache,
    int old_len, int new_len, int d_model
);
```
- [ ] Register and test kernel

**Tuesday** (8h):
**Task 3.3.3**: Cache Management
- [ ] LRU eviction policy
- [ ] Cache size limits
- [ ] Sliding window for long sequences

**Task 3.3.4**: Integration
- [ ] Wire into transformer forward pass
- [ ] Measure speedup (should be 5-10x)

### Day 8: Feed-Forward Optimization (6 hours)

**Task 3.4.1**: Use Fused Kernels
- [ ] Replace separate matmul + GELU with `fused_linear_gelu`
- [ ] Eliminate downloads in `feed_forward_gpu()`
- [ ] Keep everything on GPU

**Task 3.4.2**: SwiGLU Activation
- [ ] Implement SwiGLU (Llama uses this, not GELU)
- [ ] Create kernel or use existing GELU + multiply

### Day 9-10: Sampling + BPE (19 hours)

**Wednesday** (8h):
**Task 3.5.1-3.5.4**: Top-p Sampling
```bash
# Enhance gpu_transformer.rs
```
- [ ] Temperature scaling on GPU
- [ ] Top-k filtering (use existing kernel)
- [ ] Top-p (nucleus) filtering - NEW KERNEL:
```cuda
__global__ void nucleus_filtering(
    float* sorted_probs, int* indices,
    float* filtered_probs, float p_threshold,
    int vocab_size
);
```
- [ ] Repetition penalty
- [ ] cuRAND categorical sampling

**Thursday-Friday** (11h):
**Task 3.6.1-3.6.3**: BPE Tokenizer
```bash
touch bpe_tokenizer.rs
```
- [ ] Parse tokenizer.json (HuggingFace format)
- [ ] Extract vocab, merges, special tokens
- [ ] Implement BPE merge algorithm
- [ ] Handle special tokens (<|endoftext|>, etc.)
- [ ] UTF-8 encoding/decoding
- [ ] Test with actual Llama tokenizer file

---

## WEEK 3: Mixed Precision + Advanced Optimization (40 hours)

### Day 11-12: FP16 + Tensor Cores (16 hours)

**Task 3.7.1-3.7.4**: Mixed Precision
```bash
touch mixed_precision.rs
cd ../../gpu
touch tensor_core_ops.rs
```

**Monday** (8h):
- [ ] Convert weight storage to FP16
- [ ] Implement FP16 data structures
- [ ] Add conversion utilities

**Tuesday** (8h):
- [ ] Implement Tensor Core matmul using WMMA API:
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void matmul_fp16_tensor_core(
    __half* a, __half* b, float* c,
    int m, int n, int k
) {
    // Use wmma::fragment for 16x16x16 tiles
    // Load, compute, store using Tensor Cores
}
```
- [ ] Validate accuracy
- [ ] Benchmark speedup (should be 5-8x)

### Day 13-15: Advanced Kernel Fusion + Async (24 hours)

**Wednesday-Thursday** (16h):
**Task 6.2**: Advanced Kernel Fusion
```bash
touch kernel_fusion_advanced.rs
```
- [ ] Fused transformer block kernel (LayerNorm + Attention + FFN + Residual):
```cuda
__global__ void fused_transformer_block(
    float* input, float* wq, float* wk, float* wv,
    float* wo, float* w1, float* w2,
    float* ln_params, float* output,
    int seq_len, int d_model
);
```
- [ ] Fused TE computation pipeline
- [ ] Fused thermodynamic selection
- [ ] Test reduction in kernel launches

**Friday** (8h):
**Task 6.3**: Multi-Stream Async
```bash
touch async_executor.rs
```
- [ ] Create multiple CUDA streams
- [ ] Async execution framework
- [ ] Event-based synchronization
- [ ] Overlap transfer with computation

---

## WEEK 4: Production Features (50 hours)

### Day 16-17: Error Handling & Monitoring (14 hours)

**Monday** (8h):
**Task 5.1**: Error Handling
```bash
mkdir -p ../../production
touch ../../production/error_handling.rs
```
- [ ] Define specific error types
- [ ] GPU error recovery
- [ ] Automatic fallback
- [ ] Input validation

**Tuesday** (6h):
**Task 5.2**: Performance Monitoring
```bash
touch ../../production/monitoring.rs
```
- [ ] nvidia-smi integration
- [ ] Kernel time tracking
- [ ] Memory usage monitoring
- [ ] Performance regression detection

### Day 18: Configuration (4 hours)

**Task 5.3**: Config Management
```bash
touch ../../production/config.rs
```
- [ ] TOML config files
- [ ] Environment-based config
- [ ] Hot-reload capability
- [ ] Validation

### Day 19-20: Testing Framework (16 hours)

**Task 5.4**: Comprehensive Testing
```bash
mkdir -p ../../tests
```
- [ ] Unit tests for every GPU kernel
- [ ] Integration tests for pipelines
- [ ] Property-based testing
- [ ] Benchmark suite with regression tracking
- [ ] Stress testing (memory leaks, long runs)

### Day 21: Documentation (16 hours)

**Task 5.5**: Complete Documentation
- [ ] rustdoc for all public APIs
- [ ] Mathematical foundations document
- [ ] Deployment guide (Docker, cloud)
- [ ] Performance tuning guide
- [ ] Example notebooks/tutorials

---

## KERNELS YOU'LL CREATE

### Week 1:
1. `concat_cache` - KV-cache concatenation

### Week 2:
2. `nucleus_filtering` - Top-p sampling

### Week 3:
3. `matmul_fp16_tensor_core` - Tensor Core matmul
4. `fused_transformer_block` - Complete transformer in ONE kernel

### Week 4:
5. Various testing/monitoring utilities

**Total New Kernels**: ~5 major ones

---

## SUPPORT FOR WORKER A

**You Provide**:
- GPU kernel infrastructure
- Performance optimization
- cuRAND utilities
- Fused kernel templates

**Worker A Requests**:
- New kernels via GitHub issues
- Performance profiling help
- GPU debugging support

**How to Add Kernel for Worker A**:
1. Implement in `src/gpu/kernel_executor.rs`
2. Add to `kernels` module
3. Register in `register_standard_kernels()`
4. Add method to `GpuKernelExecutor`
5. Document usage
6. Commit with clear message
7. Notify Worker A

---

## SUCCESS METRICS - WORKER B

**Local LLM**:
- [ ] Loads Llama-7B from GGUF
- [ ] Proper BPE tokenization
- [ ] KV-cache 10x faster generation
- [ ] Top-p sampling produces diverse text
- [ ] 50-100 tokens/sec on RTX 5070
- [ ] Coherent outputs

**GPU Optimization**:
- [ ] Tensor Cores: 5-8x FP16 speedup
- [ ] Fused kernels: 10x fewer kernel launches
- [ ] Multi-stream: Transfer/compute overlap
- [ ] Overall 10-20x additional speedup

**Production**:
- [ ] 90%+ test coverage
- [ ] All APIs documented
- [ ] Error recovery works
- [ ] Monitoring operational

---

## TESTING CHECKLIST

**After Each Major Feature**:
```bash
# Build
cargo build --release --features cuda

# Test
cargo test --lib --features cuda -- [feature_name]

# Benchmark
cargo bench --features cuda

# Commit
git add -A
git commit -m "feat: [Feature name]"
git push origin worker-b-infrastructure
```

---

## RESOURCES

**GGUF Format**:
- HuggingFace GGUF spec
- llama.cpp reference implementation

**Tensor Cores**:
- NVIDIA WMMA documentation
- CUDA Programming Guide Chapter 7

**BPE**:
- SentencePiece documentation
- tokenizers library (HuggingFace)

**CUDA Optimization**:
- CUDA Best Practices Guide
- Nsight Profiler tutorials

---

**WORKER B - YOUR MISSION**: Make infrastructure fast and production-ready
**YOUR STRENGTH**: GPU optimization and systems engineering
**YOUR DELIVERABLE**: Enterprise-grade LLM inference and GPU infrastructure