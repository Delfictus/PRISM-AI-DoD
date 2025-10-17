# GPU Kernel Request - Worker 4 to Worker 2

**Requestor**: Worker 4 (Applications Domain)
**Target**: Worker 2 (GPU Infrastructure)
**Priority**: High
**Timeline**: Week 2-3

---

## Overview

Worker 4's Financial Portfolio Optimization module requires three GPU kernels to achieve 10-50x performance improvement for large-scale portfolio optimization.

---

## Requested Kernels

### 1. Covariance Matrix Kernel

**Purpose**: Calculate covariance matrix from asset return time series

**Signature**:
```cuda
__global__ void covariance_matrix(
    const float* returns,        // [n_assets x n_periods] return matrix
    float* covariance,           // [n_assets x n_assets] output covariance
    const float* means,          // [n_assets] mean returns
    int n_assets,                // Number of assets
    int n_periods                // Number of time periods
);
```

**Algorithm**:
```
1. For each pair (i, j) of assets:
   Cov[i,j] = (1/(n-1)) * Σ_t (r_it - μ_i)(r_jt - μ_j)

2. Output symmetric matrix
```

**Performance Target**:
- Input: 100 assets x 252 periods (1 year daily)
- Current CPU: ~50ms
- Target GPU: <5ms (10x speedup)

**Use Case**: Portfolio optimization covariance calculation
**File**: `src/applications/financial/mod.rs:193`

---

### 2. Quadratic Programming Gradient Step

**Purpose**: GPU-accelerated gradient descent for mean-variance optimization

**Signature**:
```cuda
__global__ void qp_gradient_step(
    float* weights,              // [n_assets] current weights (in/out)
    const float* gradient,       // [n_assets] gradient direction
    const float* covariance,     // [n_assets x n_assets] covariance matrix
    const float* returns,        // [n_assets] expected returns
    float learning_rate,         // Step size
    float lambda,                // Risk aversion parameter
    int n_assets                 // Number of assets
);
```

**Algorithm**:
```
1. Compute gradient: ∇f = μ - 2λΣw
2. Update weights: w_new = w + η * ∇f
3. Project onto constraints: sum(w) = 1, w_i ∈ [0, 0.4]
```

**Performance Target**:
- 1000 iterations for 100 assets
- Current CPU: ~100ms
- Target GPU: <10ms (10x speedup)

**Use Case**: Portfolio optimization solver
**File**: `src/applications/financial/mod.rs:277`

---

### 3. Batch Transfer Entropy

**Purpose**: Calculate pairwise Transfer Entropy for all asset pairs in parallel

**Signature**:
```cuda
__global__ void batch_transfer_entropy(
    const float* all_series,     // [n_assets x n_periods] all time series
    float* te_matrix,            // [n_assets x n_assets] output TE matrix
    float* p_values,             // [n_assets x n_assets] significance levels
    int n_assets,                // Number of assets
    int n_periods,               // Number of time periods
    int embedding_dim,           // Embedding dimension
    int time_lag                 // Time lag τ
);
```

**Algorithm**:
```
For each pair (i, j):
  TE(i→j) = Σ p(y_{t+τ}, y_t, x_t) log[p(y_{t+τ}|y_t, x_t) / p(y_{t+τ}|y_t)]

Using k-NN estimation or binning method
```

**Performance Target**:
- 100 assets → 10,000 pairwise calculations
- Current CPU: ~30 seconds (sequential)
- Target GPU: <1 second (30x speedup)

**Use Case**: Causal relationship detection for portfolio weighting
**File**: `src/applications/financial/mod.rs:231`

---

## Integration Points

### Kernel Launcher Interface

Worker 4 will use Worker 2's kernel executor:

```rust
use crate::gpu::KernelExecutor;

impl PortfolioOptimizer {
    fn calculate_covariance_matrix_gpu(&self, assets: &[Asset]) -> Result<Array2<f64>> {
        let executor = KernelExecutor::new()?;
        
        // Prepare data
        let returns_flat = self.flatten_returns(assets);
        let means = self.calculate_means(assets);
        
        // Launch kernel
        let covariance = executor.launch_covariance_kernel(
            &returns_flat,
            &means,
            assets.len(),
            returns_flat.len() / assets.len()
        )?;
        
        Ok(covariance)
    }
}
```

### Error Handling

All kernels should return errors via Worker 2's standard error system:
- CUDA errors → `GpuError::CudaError`
- Out of memory → `GpuError::OutOfMemory`
- Invalid input → `GpuError::InvalidInput`

### Memory Management

- Use Worker 2's `GpuMemoryPool` for allocation
- Support both F32 and F64 precision (configurable)
- Auto-cleanup on scope exit

---

## Testing Requirements

### Unit Tests

For each kernel, provide:
1. Small test case (5 assets, 10 periods)
2. Medium test case (50 assets, 100 periods)
3. Large test case (100 assets, 252 periods)
4. Correctness verification vs CPU implementation

### Performance Benchmarks

Measure and report:
- Kernel launch overhead
- Computation time
- Memory bandwidth utilization
- GPU occupancy
- Speedup vs CPU baseline

### Integration Tests

Worker 4 will provide integration tests:
- Full portfolio optimization end-to-end
- Multi-kernel pipeline
- Error handling and recovery

---

## Documentation Needs

Please provide:
1. Kernel implementation details (algorithm, optimizations)
2. Usage examples
3. Performance characteristics (complexity, memory usage)
4. Tuning parameters (block size, grid size)
5. Known limitations

---

## Timeline

**Week 2:**
- Kernel 1 (Covariance Matrix) - Highest priority
- Basic integration testing

**Week 3:**
- Kernel 2 (QP Gradient Step)
- Kernel 3 (Batch Transfer Entropy)
- Performance benchmarking

**Week 4:**
- Integration refinement
- Production deployment

---

## Expected Performance Impact

### Current (CPU)
- Portfolio optimization (100 assets): ~200ms
- Covariance calculation: ~50ms
- Transfer Entropy (100 pairs): ~3 seconds
- **Total**: ~3.25 seconds

### Target (GPU)
- Portfolio optimization (100 assets): ~20ms
- Covariance calculation: ~5ms
- Transfer Entropy (10,000 pairs): ~1 second
- **Total**: ~1.03 seconds

**Expected Speedup**: 3.2x for full pipeline, 30x for TE alone

---

## References

### Financial Module
- Implementation: `03-Source-Code/src/applications/financial/mod.rs`
- Documentation: `03-Source-Code/src/applications/financial/README.md`
- Tests: `03-Source-Code/src/applications/financial/mod.rs:391-471`

### GPU Infrastructure
- Executor: `03-Source-Code/src/gpu/`
- Memory Pool: `03-Source-Code/src/gpu/memory.rs`
- Error Types: `03-Source-Code/src/gpu/error.rs`

---

## Contact

**Worker 4** is ready to:
- Provide test cases and validation data
- Assist with algorithm implementation
- Conduct integration testing
- Profile and optimize performance

Please confirm receipt and estimated timeline for each kernel.

---

**Status**: SUBMITTED
**Date**: 2025-10-12
**Request ID**: W4-GPU-001
