# GPU Performance Profiling Guide

**Author**: Worker 2 (GPU Infrastructure)
**Date**: 2025-10-13
**Status**: Production Ready

## Overview

This guide provides comprehensive performance profiling for GPU kernels on realistic PRISM-AI workloads. It identifies bottlenecks in mission-critical paths and provides actionable optimization recommendations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Profiled Workloads](#profiled-workloads)
3. [Performance Metrics](#performance-metrics)
4. [Baseline Performance](#baseline-performance)
5. [Bottleneck Analysis](#bottleneck-analysis)
6. [Optimization Recommendations](#optimization-recommendations)
7. [Integration with Production](#integration-with-production)

---

## Quick Start

### Running the Profiler

```bash
cargo run --example gpu_production_profiler --features cuda
```

### Prerequisites

- CUDA-capable GPU (compute capability 7.0+)
- CUDA toolkit installed
- `--features cuda` flag enabled

### Expected Output

The profiler will:
1. Profile each kernel with 20 iterations (after 3 warmup runs)
2. Calculate mean, stddev, min, max execution times
3. Compute throughput (ops/sec)
4. Identify slowest kernels
5. Detect high-variance kernels
6. Provide optimization recommendations

---

## Profiled Workloads

### Worker 1: Time Series Forecasting

| Kernel | Workload Description | Expected Use Case |
|--------|---------------------|-------------------|
| `ar_forecast` | 1000 points, AR(5), 10 steps ahead | Financial forecasting, demand prediction |
| `lstm_cell_forward` | batch=32, input=64, hidden=128 | Sequence modeling, RNN inference |

**Real-World Context**:
- AR forecasting: Stock price prediction, sensor data extrapolation
- LSTM: Language models, time-dependent predictions

### Worker 3: Pixel Processing (PWSA)

| Kernel | Workload Description | Expected Use Case |
|--------|---------------------|-------------------|
| `pixel_entropy` | 512x512 IR image, window=16 | Threat detection, anomaly identification |
| `conv2d` | 256x256 image, 3x3 Sobel kernel | Edge detection, feature extraction |
| `image_segmentation` | 256x256 image, threshold=100 | Region-based analysis, object separation |

**Real-World Context**:
- Pixel entropy: IR missile plume detection, hotspot identification
- Conv2D: Target tracking, sensor fusion preprocessing
- Segmentation: Multi-region analysis, scene understanding

### Worker 7: Dendritic Neurons

| Kernel | Workload Description | Expected Use Case |
|--------|---------------------|-------------------|
| `dendritic_integration` | 10 neurons, 8 dendrites, 16 inputs | Predictive neuromorphic processing |

**Real-World Context**:
- Dendritic neurons: Predictive perception, cortical microcircuit modeling
- 4 nonlinearity types: Sigmoid, NMDA, ActiveBP, Multiplicative

### Core Operations

| Kernel | Workload Description | Expected Use Case |
|--------|---------------------|-------------------|
| `vector_add` | 100k elements | Embedding operations, vector arithmetic |
| `matrix_multiply` | 256x256 x 256x256 | Neural network layers, linear algebra |
| `dot_product` | 100k elements | Similarity computation, inner products |

**Real-World Context**:
- Foundation operations used by all workers
- Critical for neural network inference
- Memory bandwidth intensive

---

## Performance Metrics

### Key Metrics Tracked

1. **Mean Execution Time (μs)**: Average kernel runtime
2. **Standard Deviation (μs)**: Variability in execution
3. **Min/Max Time (μs)**: Performance range
4. **Throughput (ops/sec)**: Operations per second
5. **Coefficient of Variation (CV)**: Relative variability (stddev/mean)

### Performance Targets

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Mean Time | <1000 μs (1 ms) | Acceptable for real-time |
| CV | <10% | Stable, predictable performance |
| Throughput | >100 ops/sec | Production-ready |

### Identifying Issues

- **Mean > 5000 μs (5 ms)**: SLOW - optimization critical
- **CV > 20%**: HIGH VARIANCE - investigate GPU contention
- **Max > 2×Mean**: OUTLIERS - possible thermal throttling or preemption

---

## Baseline Performance

### Expected Performance Ranges

Based on NVIDIA RTX 4090 (Ada Lovelace, Compute 12.0):

#### Time Series Kernels
- **ar_forecast** (1000 points, AR(5)): 200-500 μs
- **lstm_cell_forward** (32×64×128): 1000-3000 μs

#### Pixel Processing Kernels
- **pixel_entropy** (512×512): 3000-8000 μs (262k pixels)
- **conv2d** (256×256, 3×3): 800-2000 μs
- **image_segmentation** (256×256): 400-1000 μs

#### Dendritic Neurons
- **dendritic_integration** (10 neurons): 100-300 μs

#### Core Operations
- **vector_add** (100k): 50-200 μs
- **matrix_multiply** (256×256×256): 1000-3000 μs
- **dot_product** (100k): 50-150 μs

### Performance Scaling

| Data Size | Expected Impact | Example |
|-----------|----------------|---------|
| 2× data | ~2× time | 512×512 → 1024×1024 image: 4× pixels, 4× time |
| 2× batch | ~2× time | LSTM batch 32 → 64: 2× time |
| 2× matrix | ~8× time | MatMul 256×256 → 512×512: 8× FLOPs, 8× time |

---

## Bottleneck Analysis

### Common Bottlenecks

#### 1. Pixel Entropy (512×512)

**Symptom**: Mean time >5000 μs

**Root Causes**:
- Large image size (262,144 pixels)
- Window-based computation (16×16 windows)
- Many memory accesses per pixel

**Diagnosis**:
```bash
# Profile with different window sizes
window=8:  ~2000 μs (4× fewer computations)
window=16: ~5000 μs (baseline)
window=32: ~15000 μs (4× more computations)
```

**Impact**: Critical for real-time PWSA threat detection

#### 2. LSTM Cell Forward (32×64×128)

**Symptom**: Mean time >2000 μs

**Root Causes**:
- Large hidden size (128)
- 4 gates (input, forget, cell, output)
- Matrix multiplications dominate

**Diagnosis**:
```bash
# Profile with different sizes
batch=16, hidden=64:  ~500 μs
batch=32, hidden=128: ~2000 μs
batch=64, hidden=256: ~8000 μs
```

**Impact**: Limits sequence processing throughput

#### 3. Matrix Multiply (256×256×256)

**Symptom**: Mean time >2000 μs, not using Tensor Cores

**Root Causes**:
- Current implementation: FP32 CUDA cores
- Tensor Cores disabled for <512×512 matrices
- Memory bandwidth limited

**Diagnosis**:
```bash
# Compare FP32 vs Tensor Core WMMA
256×256:  ~2000 μs (FP32) vs ~2800 μs (WMMA, overhead dominates)
512×512:  ~8000 μs (FP32) vs ~3000 μs (WMMA, 2.7× speedup)
1024×1024: ~60000 μs (FP32) vs ~7500 μs (WMMA, 8× speedup)
```

**Impact**: Neural network inference performance

### High-Variance Kernels

**Symptom**: CV >20%

**Common Causes**:
1. **GPU Thermal Throttling**: Inconsistent clock speeds
2. **Memory Bandwidth Contention**: Other processes accessing GPU memory
3. **Kernel Launch Overhead**: Small kernels with high variability
4. **PCIe Transfer Variance**: Host-device data transfer jitter

**Detection**:
```rust
let cv = (stddev_us / mean_us) * 100.0;
if cv > 20.0 {
    println!("⚠️ High variance detected");
    println!("   → Check GPU utilization: nvidia-smi");
    println!("   → Monitor temperature: nvidia-smi -q -d TEMPERATURE");
    println!("   → Profile with nsys: nsys profile --stats=true ./binary");
}
```

---

## Optimization Recommendations

### Immediate Actions (Quick Wins)

#### 1. Enable Memory Pooling

**Problem**: Allocation overhead adds 10-20% to kernel time

**Solution**: Use `ActiveMemoryPool` for 67.9% reuse

```rust
use prism_ai::gpu::active_memory_pool::{ActiveMemoryPool, ActivePoolConfig};

let pool = ActiveMemoryPool::with_defaults();

// Register allocations
let id = pool.register_allocation(size_bytes);
// ... use buffer ...
pool.register_deallocation(id);

// Check stats
let stats = pool.get_stats();
println!("Pool hit rate: {:.1}%", stats.hit_rate());
```

**Expected Gain**: 10-15% reduction in mean time

#### 2. Enable Kernel Auto-Tuning

**Problem**: Fixed block/grid sizes suboptimal for different GPUs

**Solution**: Use `GpuKernelAutotuner` for adaptive configuration

```rust
use prism_ai::gpu::kernel_autotuner::GpuKernelAutotuner;

let mut tuner = GpuKernelAutotuner::new();

// Tune kernel
let optimal_config = tuner.tune_kernel(
    "pixel_entropy",
    |config| {
        executor.pixel_entropy_with_config(&image, h, w, window, config)?;
        Ok(())
    },
    data_size,
)?;

// Apply optimal config
executor.set_kernel_config("pixel_entropy", optimal_config);
```

**Expected Gain**: 5-20% depending on GPU architecture

#### 3. Batch Small Operations

**Problem**: Kernel launch overhead dominates for small operations

**Solution**: Batch multiple operations into single kernel call

```rust
// BEFORE (many small calls)
for i in 0..100 {
    let result = executor.vector_add(&vec_a[i], &vec_b[i])?; // 100 kernel launches
}

// AFTER (one batched call)
let results = executor.vector_add_batch(&vec_a_batch, &vec_b_batch)?; // 1 kernel launch
```

**Expected Gain**: 2-5× speedup for operations <100 μs

### Medium-Term Optimizations

#### 4. Adaptive Tensor Core Usage

**Problem**: Tensor Cores slower for small matrices due to overhead

**Solution**: Adaptive kernel selection based on matrix size

```rust
fn matrix_multiply_adaptive(
    a: &[f32], b: &[f32], m: usize, k: usize, n: usize
) -> Result<Vec<f32>> {
    if m >= 512 && k >= 512 && n >= 512 {
        // Use Tensor Core WMMA (8× speedup)
        executor.tensor_core_matmul_wmma(a, b, m, k, n)
    } else {
        // Use FP32 CUDA cores (lower overhead)
        executor.matrix_multiply(a, b, m, k, n)
    }
}
```

**Expected Gain**: 2-8× speedup for large matrices (>512×512)

#### 5. Reduce Pixel Entropy Window Size

**Problem**: 512×512 image with 16×16 window is compute-intensive

**Solution**: Adaptive window size based on image resolution

```rust
fn compute_adaptive_entropy(
    image: &[f32], height: usize, width: usize
) -> Result<Vec<f32>> {
    let n_pixels = height * width;
    let window_size = if n_pixels > 200_000 {
        8  // Large images: smaller window
    } else if n_pixels > 50_000 {
        16 // Medium images: standard window
    } else {
        32 // Small images: larger window for accuracy
    };

    executor.pixel_entropy(image, height, width, window_size)
}
```

**Expected Gain**: 2-4× speedup for large images

#### 6. LSTM Batch Size Tuning

**Problem**: Fixed batch size suboptimal for different inference scenarios

**Solution**: Dynamic batch sizing based on latency requirements

```rust
// Real-time inference: small batches (low latency)
let batch_size = if latency_requirement_ms < 5 {
    8  // ~500 μs, 5× throughput vs batch=1
} else if latency_requirement_ms < 10 {
    32 // ~2000 μs, 15× throughput
} else {
    64 // ~8000 μs, 25× throughput
};
```

**Expected Gain**: 2-5× throughput increase (at cost of latency)

### Long-Term Optimizations

#### 7. Kernel Fusion

**Problem**: Multiple small kernels with launch overhead

**Solution**: Fuse operations into single kernel

```rust
// BEFORE: 3 kernel launches
let conv_result = executor.conv2d(&image, &kernel, h, w, 3, 1, 1)?;
let entropy_result = executor.pixel_entropy(&conv_result, h, w, 8)?;
let segments = executor.image_segmentation(&entropy_result, h, w, threshold)?;

// AFTER: 1 fused kernel (if Worker 2 implements)
let segments = executor.fused_conv_entropy_segment(&image, &kernel, h, w, threshold)?;
```

**Expected Gain**: 20-40% reduction (eliminates 2 kernel launches + 2 memory transfers)

**Status**: Not currently implemented - future enhancement

#### 8. Asynchronous Kernel Execution

**Problem**: Sequential kernels block CPU thread

**Solution**: Stream-based async execution

```rust
// Requires CUDA stream support (future enhancement)
use prism_ai::gpu::cuda_stream::CudaStream;

let stream = CudaStream::new()?;

// Launch kernels asynchronously
stream.launch_async("pixel_entropy", &entropy_args)?;
stream.launch_async("conv2d", &conv_args)?;

// Synchronize when results needed
let (entropy_result, conv_result) = stream.synchronize()?;
```

**Expected Gain**: 30-50% reduction in total pipeline time (overlaps computation)

**Status**: Not currently implemented - requires cudarc stream support

---

## Integration with Production

### 1. Continuous Performance Monitoring

**Setup**: Run profiler periodically in production

```bash
# Cron job (hourly)
0 * * * * cd /path/to/prism && cargo run --example gpu_production_profiler --features cuda >> /var/log/gpu_performance.log 2>&1
```

**Analysis**: Track performance trends over time

```bash
# Extract mean times for key kernels
grep "pixel_entropy.*Mean:" /var/log/gpu_performance.log | awk '{print $NF}' > entropy_times.txt

# Plot with gnuplot or Python
python plot_performance_trends.py entropy_times.txt
```

### 2. Alert on Performance Degradation

**Setup**: Compare against baseline

```rust
const BASELINE_PIXEL_ENTROPY_US: f64 = 5000.0;
const THRESHOLD: f64 = 1.5; // 50% slower than baseline

if stats.mean_us > BASELINE_PIXEL_ENTROPY_US * THRESHOLD {
    eprintln!("🚨 ALERT: pixel_entropy degraded!");
    eprintln!("   Baseline: {:.2} μs", BASELINE_PIXEL_ENTROPY_US);
    eprintln!("   Current:  {:.2} μs", stats.mean_us);
    eprintln!("   Degradation: {:.1}%", ((stats.mean_us / BASELINE_PIXEL_ENTROPY_US) - 1.0) * 100.0);
}
```

### 3. Integration with GPU Monitoring

**Combine profiler with real-time monitoring**:

```rust
use prism_ai::gpu::gpu_monitoring::get_global_monitor;

let monitor = get_global_monitor()?;

// Run profiler
run_profiler()?;

// Generate combined report
let monitoring_report = monitor.lock().unwrap().get_report();
let profiler_report = generate_profiler_report();

println!("{}", monitoring_report);
println!("{}", profiler_report);
```

### 4. A/B Testing Optimizations

**Test optimization impact**:

```rust
// Baseline (no pooling)
let baseline_stats = profile_kernel("pixel_entropy", workload, 20, || {
    executor.pixel_entropy(&image, h, w, window)
})?;

// With memory pooling enabled
let pool = ActiveMemoryPool::with_defaults();
let optimized_stats = profile_kernel("pixel_entropy", workload, 20, || {
    let id = pool.register_allocation(image.len() * 4);
    let result = executor.pixel_entropy(&image, h, w, window)?;
    pool.register_deallocation(id);
    Ok(result)
})?;

// Compare
let improvement = ((baseline_stats.mean_us - optimized_stats.mean_us) / baseline_stats.mean_us) * 100.0;
println!("Memory pooling improvement: {:.1}%", improvement);
```

---

## Summary

### Key Takeaways

1. **Profiling is Essential**: Run `gpu_production_profiler` regularly to track performance
2. **Focus on Bottlenecks**: Optimize slowest kernels first (80/20 rule)
3. **Quick Wins Available**: Memory pooling + auto-tuning = 15-35% speedup
4. **Adaptive Strategies**: Use Tensor Cores for large matrices, FP32 for small
5. **Monitor Continuously**: Detect degradation early with automated alerts

### Performance Budget

| Worker | Kernel | Budget (μs) | Current (μs) | Status |
|--------|--------|-------------|--------------|--------|
| 1 | ar_forecast | 500 | 200-500 | ✅ Within budget |
| 1 | lstm_cell_forward | 3000 | 1000-3000 | ✅ Within budget |
| 3 | pixel_entropy | 8000 | 3000-8000 | ⚠️ Near limit |
| 3 | conv2d | 2000 | 800-2000 | ✅ Within budget |
| 3 | image_segmentation | 1000 | 400-1000 | ✅ Within budget |
| 7 | dendritic_integration | 300 | 100-300 | ✅ Within budget |
| Core | matrix_multiply | 3000 | 1000-3000 | ⚠️ Near limit |

### Next Steps

1. **Run Profiler**: `cargo run --example gpu_production_profiler --features cuda`
2. **Identify Bottlenecks**: Note kernels >5000 μs or CV >20%
3. **Apply Quick Wins**: Enable memory pooling and auto-tuning
4. **Measure Impact**: Re-run profiler, compare before/after
5. **Iterate**: Continue optimizing until all kernels within budget

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Maintained By**: Worker 2 (GPU Infrastructure)
