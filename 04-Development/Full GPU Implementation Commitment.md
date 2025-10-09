# Full GPU Policy Evaluation - Implementation Commitment

**Date:** 2025-10-06
**Decision:** Proceed with complete GPU policy evaluation (Option C)
**Commitment:** Full implementation despite complexity
**Expected Effort:** 40-60 hours over 4-6 weeks

---

## Decision Rationale

User has chosen **full GPU implementation** over hybrid approach.

**Why full GPU:**
- True end-to-end GPU acceleration
- Maximum performance potential (231ms → 3-10ms, 23-77x speedup)
- Complete solution, not compromise
- Demonstrates full GPU capability

**Trade-offs accepted:**
- Higher complexity (60 hours vs 27.5 hours)
- More risk (physics simulation validation)
- Longer time to completion (4-6 weeks vs 2-3 weeks)

---

## Implementation Strategy

### Complexity Breakdown

**Level 1: Satellite Evolution (Simple)**
- 6 state variables
- Verlet integration (deterministic)
- Gravitational acceleration
- **GPU Strategy:** Simple arithmetic, 1 thread per policy
- **Effort:** 2 hours

**Level 2: Atmosphere Evolution (Moderate)**
- 50 turbulence modes
- Exponential decorrelation
- Random noise injection (cuRAND)
- **GPU Strategy:** 1 thread per mode, cuRAND state management
- **Effort:** 6 hours

**Level 3: Window Evolution (Complex)**
- 900 window phases
- Langevin dynamics with coupling
- Atmospheric projection
- Multiple substeps (10+)
- **GPU Strategy:** Parallel over windows, chunked for thread limit
- **Effort:** 12 hours

**EFE Computation (Moderate)**
- Observation prediction (100×900 matrix)
- Risk/ambiguity/novelty calculation
- **GPU Strategy:** Use cuBLAS, parallel reductions
- **Effort:** 6 hours

**Integration & Testing (Significant)**
- Rust wrapper
- Integration with PolicySelector
- Validation against CPU
- **Effort:** 20 hours

**Total:** 46 hours (mid-range of 40-60 estimate)

---

## Revised Task Breakdown

### Task 1.1.1: Design Architecture (COMPLETED)
- ✅ GPU data structure design
- ✅ Memory layout planning
- ✅ Complexity assessment
- ✅ Feasibility validation

### Task 1.1.2: CUDA Kernels (12 hours) - NEXT

**1.1.2.1: Satellite Evolution Kernel (2 hours)**
```cuda
__global__ void evolve_satellite_kernel(
    const double* current_state,    // [6] per policy
    double* next_state,             // [6] per policy
    double dt,
    int n_policies
);
```
- Verlet integration
- Gravitational acceleration
- Simple, deterministic

**1.1.2.2: Atmosphere Evolution Kernel (6 hours)**
```cuda
__global__ void evolve_atmosphere_kernel(
    const double* current_modes,    // [50] per policy
    double* next_modes,             // [50] per policy
    double* variances,              // [50] per policy
    curandState* rng_states,        // Random number generation
    double dt,
    double decorrelation_rate,
    double c_n_squared,
    int n_policies,
    int n_modes
);
```
- Exponential decorrelation
- cuRAND for noise injection
- Variance evolution
- RNG state management (complex)

**1.1.2.3: Window Evolution Kernel (12 hours)**
```cuda
__global__ void evolve_windows_kernel(
    const double* current_windows,     // [900] per policy-step
    const double* atmosphere_modes,    // [50] per policy-step
    const double* actions,             // [900] per policy-step
    double* next_windows,              // [900] per policy-step
    double* variances,                 // [900] per policy-step
    curandState* rng_states,           // For diffusion noise
    double dt,
    double damping,
    double diffusion,
    int n_policies,
    int n_windows,
    int substeps
);
```
- Langevin dynamics
- Atmospheric projection
- Control action application
- Multiple substeps
- Most complex kernel

**1.1.2.4: Observation Prediction (2 hours)**
```cuda
// Use cuBLAS instead of custom kernel
cublasDgemv(
    handle,
    CUBLAS_OP_N,
    obs_dim, state_dim,
    &alpha,
    observation_matrix, obs_dim,
    state_vector, 1,
    &beta,
    predicted_obs, 1
);
```

**1.1.2.5: EFE Computation (4 hours)**
```cuda
__global__ void compute_efe_kernel(
    const double* predicted_obs,
    const double* obs_variances,
    const double* preferred_obs,
    const double* future_variances,
    const double* prior_variance,
    double* risk_out,
    double* ambiguity_out,
    double* novelty_out,
    int n_policies,
    int horizon,
    int obs_dim,
    int state_dim
);
```
- Risk: observation error
- Ambiguity: observation uncertainty
- Novelty: entropy difference

**1.1.2.6: RNG Initialization (2 hours)**
```cuda
__global__ void init_rng_states(
    curandState* states,
    unsigned long long seed,
    int n_states
);
```

---

### Task 1.1.3: Rust GPU Wrapper (10 hours)

**File:** `src/active_inference/gpu_policy_eval.rs`

**1.1.3.1: Struct & Initialization (3 hours)**
```rust
pub struct GpuPolicyEvaluator {
    context: Arc<CudaContext>,

    // Kernels
    satellite_kernel: Arc<CudaFunction>,
    atmosphere_kernel: Arc<CudaFunction>,
    windows_kernel: Arc<CudaFunction>,
    efe_kernel: Arc<CudaFunction>,
    rng_init_kernel: Arc<CudaFunction>,

    // cuBLAS handle
    blas_handle: cublasHandle_t,

    // Persistent GPU buffers
    trajectories: GpuTrajectoryBuffers,
    efe_components: GpuEfeBuffers,
    rng_states: CudaSlice<curandState>,

    // Configuration
    n_policies: usize,
    horizon: usize,
    state_dims: StateDimensions,
}

struct StateDimensions {
    satellite: usize,    // 6
    atmosphere: usize,   // 50
    windows: usize,      // 900
    observations: usize, // 100
}
```

**1.1.3.2: Data Upload (2 hours)**
```rust
fn upload_model_state(&mut self, model: &HierarchicalModel) -> Result<()>;
fn upload_policies(&mut self, policies: &[Policy]) -> Result<()>;
fn upload_matrices(&mut self, obs_matrix: &Array2<f64>, preferred: &Array1<f64>) -> Result<()>;
```

**1.1.3.3: Kernel Orchestration (4 hours)**
```rust
pub fn evaluate_policies_gpu(
    &mut self,
    model: &HierarchicalModel,
    policies: &[Policy],
) -> Result<Vec<f64>> {
    // For each policy (parallel)
    //   For each horizon step (sequential within policy)
    //     1. Evolve satellite
    //     2. Evolve atmosphere
    //     3. Evolve windows (with substeps)
    //     4. Predict observations
    //     5. Accumulate EFE components
    //   End
    // End
    // Return EFE values
}
```

**1.1.3.4: Memory Management (1 hour)**
- Allocation
- Deallocation
- Error handling

---

### Task 1.1.4: Integration (6 hours)

**1.1.4.1: Modify PolicySelector (3 hours)**
- Add `gpu_evaluator` field
- Modify `select_policy()` to use GPU path
- Keep CPU fallback

**1.1.4.2: Update Adapter (2 hours)**
- Create `GpuPolicyEvaluator` instance
- Pass shared CUDA context
- Wire to controller

**1.1.4.3: Feature flags (1 hour)**
- `#[cfg(feature = "cuda")]` guards
- Graceful fallback

---

### Task 1.1.5: Testing (8 hours)

**1.1.5.1: Unit Tests (4 hours)**
- Test each physics evolution kernel
- Compare against CPU ground truth
- Validate RNG reproducibility

**1.1.5.2: Integration Tests (2 hours)**
- Full policy evaluation GPU vs CPU
- Tolerance: <1% difference in EFE values
- Edge cases

**1.1.5.3: Performance Profiling (2 hours)**
- nvprof/nsys profiling
- Verify 231ms → <10ms
- Identify any remaining bottlenecks

---

## Updated Timeline

### Week 1-2: CUDA Kernel Development
- **Hours:** 28 hours
- Satellite kernel (2h)
- Atmosphere kernel (6h)
- Window kernel (12h)
- Observation prediction (2h)
- EFE kernel (4h)
- RNG initialization (2h)

### Week 3: Rust Integration
- **Hours:** 10 hours
- GPU wrapper struct (3h)
- Data upload/download (2h)
- Kernel orchestration (4h)
- Memory management (1h)

### Week 4: Integration & Testing
- **Hours:** 14 hours
- Modify PolicySelector (3h)
- Update adapter (2h)
- Feature flags (1h)
- Testing (8h)

### Week 5: Bug Fixes & Optimization
- **Hours:** 8 hours
- Fix issues found in testing
- Optimize kernel performance
- Validate against CPU

**Total: 60 hours over 5 weeks**

---

## Expected Performance

### Conservative Estimate
```
Policy Controller: 231ms → 10ms (23x speedup)
├─ Upload: 2ms (7.3MB)
├─ GPU kernels: 6ms
│  ├─ Satellite: 0.1ms
│  ├─ Atmosphere: 0.5ms
│  ├─ Windows: 4ms (most expensive)
│  └─ EFE: 1ms
└─ Download: 0.01ms

Phase 6 Total: 233ms → 12ms
Pipeline Total: 281ms → 60ms
```

### Optimistic Estimate
```
Policy Controller: 231ms → 5ms (46x speedup)
├─ Upload: 1ms (cached matrices)
├─ GPU kernels: 3ms (optimized)
└─ Download: 0.01ms

Phase 6 Total: 233ms → 7ms
Pipeline Total: 281ms → 55ms
```

**After neuromorphic fix:** 55ms → 15ms

---

## Commitment

✅ **Proceeding with full GPU implementation**

**Next steps:**
1. Task 1.1.2.1 - Implement satellite evolution kernel
2. Task 1.1.2.2 - Implement atmosphere evolution kernel
3. Task 1.1.2.3 - Implement window evolution kernel

**Documents created:**
- `/home/diddy/Desktop/PRISM-AI/docs/gpu_policy_eval_design.md` - Full design
- `/home/diddy/Desktop/PRISM-AI/docs/obsidian-vault/04-Development/Task 1.1.1 Re-evaluation.md` - Re-evaluation

**Ready to begin kernel implementation.**

Should I start with Task 1.1.2.1 (satellite evolution kernel)?