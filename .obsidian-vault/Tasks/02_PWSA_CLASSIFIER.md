# Task 2: Migrate PWSA Active Inference Classifier to GPU

**Status**: PENDING
**Priority**: HIGH (Week 1)
**Dependencies**: Task 1 (gpu_enabled.rs) - COMPLETE

## Objective

Port the PWSA Active Inference Classifier to use GPU kernels for all computations. This is **CRITICAL** infrastructure for threat detection and must achieve real-time performance.

## Current State

**Files**:
- `src/pwsa/active_inference_classifier.rs` - CPU implementation
- `src/pwsa/gpu_classifier.rs` - Partial GPU implementation

**Problems**:
- Variational free energy computed on CPU
- Belief updates use CPU loops
- Policy evaluation is serial
- NO actual GPU kernel execution

## GPU Kernels Required

###1. Variational Free Energy Kernel

```cuda
__global__ void variational_free_energy_kernel(
    float* beliefs, float* observations,
    float* precision, float* free_energy,
    int batch_size, int state_dim
)
```

### 2. Belief Update Kernel

```cuda
__global__ void belief_update_kernel(
    float* prior, float* likelihood,
    float* posterior, float* prediction_error,
    int batch_size, int state_dim
)
```

### 3. KL Divergence Kernel

```cuda
__global__ void kl_divergence_kernel(
    float* p, float* q, float* kl_div,
    int batch_size, int dim
)
```

### 4. Policy Evaluation Kernel

```cuda
__global__ void policy_evaluation_kernel(
    float* beliefs, float* policies,
    float* expected_free_energy, float* preferences,
    int n_policies, int horizon, int state_dim
)
```

## Implementation Steps

1. ✅ Review current implementation
2. ⏳ Create CUDA kernels for core operations
3. ⏳ Implement GPU memory management
4. ⏳ Port belief propagation to GPU
5. ⏳ Port policy evaluation to GPU
6. ⏳ Remove ALL CPU fallback
7. ⏳ Test GPU execution
8. ⏳ Verify performance >100x CPU

## Completion Criteria

- [ ] All computations use GPU kernels
- [ ] NO CPU fallback code remains
- [ ] Compiles with `--features cuda` only
- [ ] Tests verify GPU execution
- [ ] Performance: <1ms inference latency
- [ ] Batch size >1000 supported
- [ ] Passes governance engine checks

## Performance Targets

- **Inference latency**: <1ms per sample
- **Batch throughput**: >10,000 samples/second
- **Memory usage**: <500MB GPU RAM
- **Speedup vs CPU**: >100x

## Testing

```bash
cargo test --release --features cuda pwsa::gpu_classifier --nocapture
```

## Notes

This is **MISSION CRITICAL** infrastructure. PWSA threat detection relies on this module operating at real-time speeds. GPU acceleration is **NON-NEGOTIABLE**.

---

**GPU-ONLY. NO CPU FALLBACK.**