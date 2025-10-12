# Tensor Core Optimization Plan

**Phase**: 2 - Performance Optimization
**Worker**: Worker 2 (GPU Infrastructure)
**Date**: 2025-10-12
**Goal**: Achieve **8x speedup** on matrix operations using FP16 Tensor Cores

---

## Executive Summary

The RTX 5070 (Ada Lovelace, Compute 12.0) has 4th generation Tensor Cores that can perform matrix multiplication at **8x** the throughput of standard CUDA cores when using FP16 (half precision). This phase will implement Tensor Core-accelerated kernels for our most compute-intensive operations.

---

## Why Tensor Cores?

### Performance Gains

**Standard CUDA Cores (FP32)**:
- 1 FP32 multiply-accumulate per CUDA core per clock
- RTX 5070: ~5,888 CUDA cores
- Peak: ~18 TFLOPS (FP32)

**Tensor Cores (FP16)**:
- Warp Matrix Multiply-Accumulate (WMMA)
- Processes 16x16x16 matrix tiles in parallel
- RTX 5070: 184 Tensor Cores (4th gen)
- Peak: ~144 TFLOPS (FP16) with Tensor Cores
- **Effective speedup: 8x over FP32 CUDA cores**

### Accuracy Considerations

**FP16 Range**:
- Exponent: 5 bits (-14 to +15)
- Mantissa: 10 bits
- Range: ±65,504
- Precision: ~3 decimal digits

**For Our Use Cases**:
- ✅ Neural networks: FP16 widely used, proven
- ✅ Matrix multiplication: Accumulation in FP32 preserves accuracy
- ✅ Attention mechanisms: Scores typically in [-10, 10]
- ⚠️ May need scaling for very small/large values

---

## Target Kernels (Priority Order)

### 1. Matrix Multiplication (Highest Impact)

**Current**: `matmul` kernel (FP32)
```cuda
// Standard CUDA matmul
c[row][col] = Σ a[row][k] * b[k][col]
```

**Performance**:
- 1024x1024 × 1024x1024: ~2ms (FP32)
- Bottleneck for many operations

**Target**: `tensor_core_matmul` kernel (FP16)
```cuda
// Tensor Core matmul using WMMA
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**Expected Performance**:
- 1024x1024 × 1024x1024: ~0.25ms (FP16)
- **Speedup: 8x**

**Priority**: ⭐⭐⭐ **HIGHEST**

---

### 2. Multi-Head Attention (High Impact)

**Current**: `multi_head_attention` kernel
- Q×K^T matmul: Bottleneck
- Attention scores × V: Second bottleneck

**Breakdown**:
```
Total time: ~10ms (batch=64, seq=128, d_model=512)
├─ Q×K^T: ~4ms (3 matmuls for Q, K, V projection)
├─ Softmax: ~1ms
└─ Attn×V: ~4ms (attention scores × values)
```

**Target**: `tensor_core_attention`
- All matmuls use Tensor Cores
- Fused softmax (stays FP32 for stability)

**Expected Performance**:
- Q×K^T: 4ms → 0.5ms (8x)
- Attn×V: 4ms → 0.5ms (8x)
- Total: 10ms → 2ms (~5x overall)

**Priority**: ⭐⭐ **HIGH**

---

### 3. Fused Linear Operations (Medium Impact)

**Current**:
- `fused_matmul_relu`: Matmul + ReLU
- `fused_linear_gelu`: Linear + GELU

**Target**: `tensor_core_fused_linear`
- Tensor Core matmul
- Inline activation in accumulation phase
- Single kernel launch

**Expected Performance**:
- Linear layer (1024→512): 2ms → 0.25ms
- **Speedup: 8x**

**Priority**: ⭐ **MEDIUM**

---

## Technical Implementation

### WMMA API Overview

**Warp Matrix Multiply-Accumulate (WMMA)**:
```cuda
#include <mma.h>
using namespace nvcuda;

// Fragment types
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

// Operations
wmma::load_matrix_sync(a_frag, a_ptr, lda);        // Load from memory
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // Multiply-accumulate
wmma::store_matrix_sync(c_ptr, c_frag, ldc, ...); // Store to memory
```

**Tile Sizes** (Compute 12.0):
- **16x16x16**: Standard tile (M=16, N=16, K=16)
- **8x32x16**: Tall tile
- **32x8x16**: Wide tile

**Best Performance**: 16x16x16 for square matrices

---

### Kernel Implementation Strategy

#### Step 1: Basic Tensor Core Matmul

```cuda
extern "C" __global__ void tensor_core_matmul(
    half* a, half* b, float* c,
    int m, int n, int k
) {
    // Warp-level operation (32 threads per warp)
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Tile across K dimension
    for (int tile_k = 0; tile_k < k; tile_k += 16) {
        // Load 16x16 tiles
        wmma::load_matrix_sync(a_frag, a + warp_row * 16 * k + tile_k, k);
        wmma::load_matrix_sync(b_frag, b + tile_k * n + warp_col * 16, n);

        // Compute C += A * B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    wmma::store_matrix_sync(c + warp_row * 16 * n + warp_col * 16, c_frag, n, wmma::mem_row_major);
}
```

**Key Points**:
- Each warp processes one 16x16 output tile
- Loop over K dimension in 16-element chunks
- Accumulation in FP32 for accuracy
- Output in FP32 (can cast to FP16 if needed)

---

#### Step 2: FP16 Conversion Utilities

**Rust Side**:
```rust
// Add to GpuKernelExecutor
impl GpuKernelExecutor {
    /// Convert FP32 to FP16 on GPU
    fn convert_f32_to_f16(&self, data: &[f32]) -> Result<Vec<u16>> {
        // FP16 is stored as u16 (half precision format)
        let stream = self.context.default_stream();

        // Upload FP32 data
        let f32_dev = stream.memcpy_stod(data)?;

        // Allocate FP16 buffer
        let mut f16_dev = stream.alloc_zeros::<u16>(data.len())?;

        // Convert using CUDA (curandGenerateNormal uses FP16 internally)
        // Or use custom kernel:
        // __global__ void fp32_to_fp16(float* in, half* out, int n)

        let f16_host = stream.memcpy_dtov(&f16_dev)?;
        Ok(f16_host)
    }

    /// Convert FP16 to FP32 on GPU
    fn convert_f16_to_f32(&self, data: &[u16]) -> Result<Vec<f32>> {
        // Reverse conversion
        // ...
    }
}
```

**CUDA Conversion Kernel**:
```cuda
extern "C" __global__ void fp32_to_fp16(
    float* input, half* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

extern "C" __global__ void fp16_to_fp32(
    half* input, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}
```

---

#### Step 3: Wrapper API Design

```rust
impl GpuKernelExecutor {
    /// Matrix multiplication using Tensor Cores (FP16 compute, FP32 output)
    /// Achieves ~8x speedup over standard FP32 matmul
    /// GPU ONLY - NO CPU LOOPS
    pub fn tensor_core_matmul(
        &self,
        a: &[f32],    // Input: FP32
        b: &[f32],    // Input: FP32
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(a.len() == m * k, "Matrix A size mismatch");
        anyhow::ensure!(b.len() == k * n, "Matrix B size mismatch");
        anyhow::ensure!(m % 16 == 0 && n % 16 == 0 && k % 16 == 0,
            "Dimensions must be multiples of 16 for Tensor Cores");

        let stream = self.context.default_stream();

        // Convert inputs to FP16
        let a_f16 = self.convert_f32_to_f16(a)?;
        let b_f16 = self.convert_f32_to_f16(b)?;

        // Upload FP16 data
        let a_dev = stream.memcpy_stod(&a_f16)?;
        let b_dev = stream.memcpy_stod(&b_f16)?;
        let mut c_dev = stream.alloc_zeros::<f32>(m * n)?;

        // Launch Tensor Core kernel
        // Each block: 8 warps = 256 threads
        // Each warp processes one 16x16 tile
        let warps_m = m / 16;
        let warps_n = n / 16;
        let cfg = LaunchConfig {
            grid_dim: ((warps_n + 7) / 8, (warps_m + 7) / 8, 1),
            block_dim: (256, 1, 1),  // 8 warps per block
            shared_mem_bytes: 0,
        };

        let kernel = self.get_kernel("tensor_core_matmul")?;
        unsafe {
            stream.launch_builder(kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&(m as i32))
                .arg(&(n as i32))
                .arg(&(k as i32))
                .launch(cfg)?;
        }

        // Download FP32 result
        let result = stream.memcpy_dtov(&c_dev)?;
        Ok(result)
    }
}
```

---

## Performance Validation

### Benchmarking Strategy

```rust
// Create: benches/tensor_core_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_ai::gpu::kernel_executor::GpuKernelExecutor;

fn bench_matmul_comparison(c: &mut Criterion) {
    let executor = GpuKernelExecutor::new(0).unwrap();

    let sizes = vec![256, 512, 1024, 2048];

    let mut group = c.benchmark_group("matmul");

    for size in sizes {
        // Standard FP32 matmul
        group.bench_with_input(
            BenchmarkId::new("fp32", size),
            &size,
            |b, &s| {
                let a = vec![1.0f32; s * s];
                let b = vec![1.0f32; s * s];
                b.iter(|| {
                    executor.matrix_multiply_gpu(&a, &b, s, s, s).unwrap()
                });
            }
        );

        // Tensor Core FP16 matmul
        group.bench_with_input(
            BenchmarkId::new("fp16_tensor_core", size),
            &size,
            |b, &s| {
                let a = vec![1.0f32; s * s];
                let b = vec![1.0f32; s * s];
                b.iter(|| {
                    executor.tensor_core_matmul(&a, &b, s, s, s).unwrap()
                });
            }
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul_comparison);
criterion_main!(benches);
```

**Expected Results**:
```
matmul/fp32/256           time: [150 µs]
matmul/fp16_tensor_core/256  time: [20 µs]   speedup: 7.5x

matmul/fp32/512           time: [600 µs]
matmul/fp16_tensor_core/512  time: [80 µs]   speedup: 7.5x

matmul/fp32/1024          time: [2.0 ms]
matmul/fp16_tensor_core/1024 time: [250 µs]  speedup: 8.0x

matmul/fp32/2048          time: [8.0 ms]
matmul/fp16_tensor_core/2048 time: [1.0 ms]  speedup: 8.0x
```

---

### Accuracy Validation

```rust
#[test]
fn test_tensor_core_accuracy() {
    let executor = GpuKernelExecutor::new(0).unwrap();

    let size = 256;
    let a = vec![1.0f32; size * size];
    let b = vec![2.0f32; size * size];

    // Compute with FP32
    let c_fp32 = executor.matrix_multiply_gpu(&a, &b, size, size, size).unwrap();

    // Compute with Tensor Cores (FP16)
    let c_fp16 = executor.tensor_core_matmul(&a, &b, size, size, size).unwrap();

    // Compare results
    let max_error = c_fp32.iter().zip(&c_fp16)
        .map(|(&fp32, &fp16)| (fp32 - fp16).abs())
        .fold(0.0f32, f32::max);

    let relative_error = max_error / c_fp32[0];

    println!("Max absolute error: {}", max_error);
    println!("Relative error: {}", relative_error);

    // FP16 should be accurate to ~0.1% for typical values
    assert!(relative_error < 0.001, "Tensor Core result too inaccurate");
}
```

---

## Implementation Timeline

### Week 2, Day 2 (Today)
- [x] Research Tensor Core requirements
- [x] Design implementation plan
- [ ] Create FP32↔FP16 conversion kernels
- [ ] Implement basic tensor_core_matmul kernel
- [ ] Add wrapper method to GpuKernelExecutor
- [ ] Test compilation

### Week 2, Day 3
- [ ] Optimize tile scheduling
- [ ] Handle edge cases (non-multiple-of-16 dimensions)
- [ ] Add accuracy validation tests
- [ ] Benchmark against FP32 matmul
- [ ] Document performance gains

### Week 2, Day 4
- [ ] Implement tensor_core_attention kernel
- [ ] Fuse Q/K/V projections
- [ ] Benchmark attention performance
- [ ] Integration with LLM kernels

### Week 2, Day 5
- [ ] Implement fused Tensor Core operations
- [ ] tensor_core_linear_relu
- [ ] tensor_core_linear_gelu
- [ ] Final benchmarking and validation

---

## Challenges and Solutions

### Challenge 1: FP16 Data Type in Rust

**Problem**: Rust doesn't have native FP16 type
**Solution**:
- Use `u16` to store FP16 bits
- Convert on GPU using `__float2half` / `__half2float`
- Or use `half` crate for host-side operations

### Challenge 2: Dimension Alignment

**Problem**: Tensor Cores require multiples of 16
**Solution**:
- Pad matrices to nearest multiple of 16
- Or use fallback for non-aligned dimensions
- Document alignment requirements

### Challenge 3: Numerical Stability

**Problem**: FP16 has limited range
**Solution**:
- Accumulate in FP32 (WMMA supports this)
- Scale attention scores if needed
- Use FP32 for softmax (stability critical)

### Challenge 4: Memory Bandwidth

**Problem**: FP16 doubles memory bandwidth
**Solution**:
- Convert on-device (don't transfer FP16 over PCIe)
- Use async streams for overlap
- Keep intermediate results in FP16 on GPU

---

## Success Metrics

### Performance Targets
- ✅ **8x speedup** on 1024x1024 matmul
- ✅ **5x speedup** on multi-head attention
- ✅ **<1% accuracy loss** compared to FP32

### Integration Targets
- ✅ Drop-in replacement for existing matmul
- ✅ Transparent to other workers
- ✅ Automatic fallback for non-aligned dimensions

### Documentation Targets
- ✅ Usage guide for Tensor Core APIs
- ✅ Performance comparison benchmarks
- ✅ Best practices for FP16 operations

---

## Next Steps

1. **Immediate**: Implement FP32↔FP16 conversion kernels
2. **Today**: Basic tensor_core_matmul implementation
3. **This Week**: Optimize and validate
4. **Next Week**: Attention and fused operations

---

## References

### NVIDIA Documentation
- [CUDA Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Tensor Core Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [FP16 Best Practices](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

### Ada Lovelace (RTX 5070)
- Compute Capability: 12.0
- Tensor Cores: 4th Generation
- FP16 Tensor Core Performance: ~144 TFLOPS
- FP32 CUDA Core Performance: ~18 TFLOPS

---

**Status**: Plan complete, ready for implementation
**Next Action**: Implement FP32↔FP16 conversion kernels
**Expected Completion**: Week 2, Day 5
**Target Performance**: 8x speedup on matrix operations
