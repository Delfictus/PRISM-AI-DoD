# GPU Kernel Requests - Worker 5 to Worker 2

**Date**: 2025-10-12
**Requesting Worker**: Worker 5 (Thermodynamic & GNN)
**Target Worker**: Worker 2 (GPU Infrastructure)
**Priority**: High
**Week**: Week 2, Task 2.3

---

## Context

Worker 5 has implemented 6 advanced thermodynamic schedule modules for LLM orchestration optimization:
1. Simulated Annealing Schedule (488 lines)
2. Parallel Tempering Schedule (623 lines)
3. Hamiltonian Monte Carlo Schedule (672 lines)
4. Bayesian Optimization Schedule (753 lines)
5. Multi-Objective Schedule (705 lines)
6. Replica Exchange Implementation (652 lines)

These modules currently use CPU-based computations for temperature updates, energy calculations, and statistical sampling. To achieve >80% GPU utilization mandate, we need GPU kernels for core thermodynamic operations.

---

## Required GPU Kernels

### 1. Boltzmann Probability Kernel

**Purpose**: Compute softmax probabilities from energy values using Boltzmann distribution

**Function Signature**:
```rust
pub fn boltzmann_probabilities_gpu(
    energies: &CudaSlice<f32>,          // Input: energy values
    temperature: f32,                    // Temperature parameter
    probabilities: &mut CudaSlice<f32>, // Output: normalized probabilities
) -> Result<()>;
```

**Mathematical Formula**:
```
P_i = exp(-E_i / kT) / sum_j(exp(-E_j / kT))
```

**Used In**:
- `advanced_simulated_annealing.rs` - Boltzmann acceptance
- `advanced_hmc.rs` - Metropolis-Hastings acceptance
- `advanced_bayesian_optimization.rs` - Probability of improvement

**Performance Target**: <0.1ms for 1000 energies
**Batch Support**: Process 100+ energy arrays in parallel

---

### 2. Replica Exchange Swap Kernel

**Purpose**: Compute Metropolis exchange acceptance probabilities for replica pairs

**Function Signature**:
```rust
pub fn replica_swap_acceptance_gpu(
    energies: &CudaSlice<f32>,              // Energy of each replica
    temperatures: &CudaSlice<f32>,          // Temperature of each replica
    swap_pairs: &CudaSlice<i32>,            // Pairs to attempt swapping (indices)
    acceptance_probs: &mut CudaSlice<f32>,  // Output: acceptance probabilities
) -> Result<()>;
```

**Mathematical Formula**:
```
P_accept = min(1, exp((beta_i - beta_j) * (E_j - E_i)))
where beta_i = 1 / (k * T_i)
```

**Used In**:
- `advanced_parallel_tempering.rs` - Geometric ladder exchange
- `advanced_replica_exchange.rs` - Thermodynamic replica swapping

**Performance Target**: <0.5ms for 100 replicas
**Exchange Schedules**: Fixed, Adaptive, Stochastic

---

### 3. Leapfrog Integrator Kernel

**Purpose**: Hamiltonian dynamics integration for HMC sampling

**Function Signature**:
```rust
pub fn leapfrog_integrate_gpu(
    position: &CudaSlice<f32>,              // Current position
    momentum: &CudaSlice<f32>,              // Current momentum
    gradient: &CudaSlice<f32>,              // Gradient of potential energy
    mass_matrix: &CudaSlice<f32>,           // Mass matrix (diagonal)
    step_size: f32,                          // Integration step size
    num_steps: i32,                          // Number of leapfrog steps
    output_position: &mut CudaSlice<f32>,   // Final position
    output_momentum: &mut CudaSlice<f32>,   // Final momentum
) -> Result<()>;
```

**Algorithm**: Symplectic leapfrog integration
```
for step in 0..num_steps:
    momentum += 0.5 * step_size * gradient
    position += step_size * momentum / mass
    gradient = compute_gradient(position)  # User provides
    momentum += 0.5 * step_size * gradient
```

**Used In**:
- `advanced_hmc.rs` - Trajectory generation

**Performance Target**: <1ms for 50 steps
**Dimensions**: Support up to 1000-dimensional spaces

---

### 4. Gaussian Process Covariance Kernel

**Purpose**: Compute covariance matrix for GP surrogate model

**Function Signature**:
```rust
pub fn gp_covariance_gpu(
    X1: &CudaSlice<f32>,                    // First set of points (n1 x d)
    X2: &CudaSlice<f32>,                    // Second set of points (n2 x d)
    kernel_params: &CudaSlice<f32>,         // Kernel hyperparameters
    kernel_type: i32,                        // 0=RBF, 1=Matérn 5/2, 2=RQ
    n1: i32, n2: i32, d: i32,               // Dimensions
    cov_matrix: &mut CudaSlice<f32>,        // Output: n1 x n2 matrix
) -> Result<()>;
```

**Kernel Functions**:

**Squared Exponential (RBF)**:
```
k(x, x') = σ² * exp(-||x - x'||² / (2ℓ²))
params: [length_scale, signal_variance]
```

**Matérn 5/2**:
```
k(x, x') = σ² * (1 + √5*r/ℓ + 5r²/(3ℓ²)) * exp(-√5*r/ℓ)
where r = ||x - x'||
```

**Rational Quadratic**:
```
k(x, x') = σ² * (1 + ||x-x'||²/(2αℓ²))^(-α)
params: [length_scale, signal_variance, alpha]
```

**Used In**:
- `advanced_bayesian_optimization.rs` - GP surrogate modeling

**Performance Target**: <2ms for 100x100 matrix
**Maximum Size**: 1000x1000 covariance matrices

---

### 5. Pareto Dominance Check Kernel

**Purpose**: Check dominance relationships for multi-objective optimization

**Function Signature**:
```rust
pub fn pareto_dominance_gpu(
    objectives: &CudaSlice<f32>,            // n x m matrix (n solutions, m objectives)
    n_solutions: i32,                        // Number of solutions
    n_objectives: i32,                       // Number of objectives
    dominance_matrix: &mut CudaSlice<i32>,  // Output: n x n (-1/0/1)
) -> Result<()>;
```

**Dominance Logic**:
```
Solution i dominates j if:
  - i is better or equal in ALL objectives
  - i is strictly better in at least ONE objective

dominance_matrix[i][j]:
  1  = i dominates j
  -1 = j dominates i
  0  = non-dominated (both on Pareto frontier)
```

**Used In**:
- `advanced_multi_objective.rs` - Pareto frontier tracking

**Performance Target**: <1ms for 100 solutions
**Scalability**: Support up to 1000 solutions, 10 objectives

---

### 6. Batch Temperature Update Kernel

**Purpose**: Update temperatures for multiple replicas/schedules in parallel

**Function Signature**:
```rust
pub fn batch_temperature_update_gpu(
    temperatures: &mut CudaSlice<f32>,      // Current temperatures (in/out)
    cooling_rates: &CudaSlice<f32>,         // Cooling rate per schedule
    cooling_type: &CudaSlice<i32>,          // 0=exponential, 1=log, 2=adaptive
    acceptance_rates: &CudaSlice<f32>,      // For adaptive cooling
    iteration: i32,                          // Current iteration
    n_schedules: i32,                        // Number of schedules
) -> Result<()>;
```

**Cooling Strategies**:

**Exponential**: `T_new = T_old * beta`
**Logarithmic**: `T_new = T_0 / log(1 + iteration)`
**Adaptive**: `T_new = T_old * (1 + adjustment * (target_rate - actual_rate))`

**Used In**:
- All schedule modules for batch processing

**Performance Target**: <0.1ms for 1000 schedules
**Adaptive Support**: Dynamic cooling based on acceptance rates

---

## Integration Plan

### Phase 1: Kernel Implementation (Worker 2)
1. Implement kernels in `src/gpu/thermodynamic_kernels.cu`
2. Add kernel registration to `kernel_executor.rs`
3. Write unit tests with known inputs/outputs
4. Benchmark performance on H200 GPU

### Phase 2: Wrapper Creation (Worker 5)
1. Create GPU wrapper module: `src/orchestration/thermodynamic/gpu_schedule_kernels.rs`
2. Wrap Worker 2's kernels with high-level interfaces
3. Handle memory transfers and error handling
4. Add batch processing utilities

### Phase 3: Integration (Worker 5)
1. Update existing schedule modules to use GPU kernels
2. Add GPU batch schedulers for each schedule type
3. Maintain CPU fallback for testing (but NOT for production)
4. Update tests to verify GPU vs CPU equivalence

### Phase 4: Optimization (Both Workers)
1. Profile end-to-end GPU utilization
2. Identify memory transfer bottlenecks
3. Optimize kernel parameters (block size, grid size)
4. Implement persistent GPU data where possible

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| GPU Utilization | >95% | During thermodynamic operations |
| Latency Reduction | 10-100x | Compared to CPU baseline |
| Batch Throughput | 1000+ schedules/sec | Parallel evaluation |
| Memory Efficiency | <10 transfers/selection | Minimize CPU↔GPU |
| Numerical Accuracy | <1e-6 error | Compared to CPU reference |

---

## Testing Requirements

### Unit Tests (Worker 2)
- Test each kernel with known inputs/outputs
- Verify numerical accuracy against CPU reference
- Test edge cases (zero temperature, negative energies, etc.)
- Test different batch sizes (1, 10, 100, 1000)

### Integration Tests (Worker 5)
- End-to-end schedule execution on GPU
- Compare GPU schedule results vs CPU
- Test all schedule types with GPU kernels
- Verify acceptance rates and convergence

### Performance Tests (Both)
- Benchmark individual kernel execution times
- Profile full thermodynamic consensus pipeline
- Measure GPU utilization during operations
- Compare performance across batch sizes

### Stress Tests
- 1000+ concurrent schedules
- Large dimensional spaces (1000-D for HMC)
- Large covariance matrices (1000x1000)
- Long-running simulations (10000+ iterations)

---

## Dependencies

**Required**:
- `cudarc = "0.9"` (already in Cargo.toml)
- CUDA 12.x runtime
- H200 GPU with 80GB HBM3

**Optional**:
- `curand` for GPU random number generation
- `cublas` for matrix operations (GP covariance)

---

## Timeline

| Week | Phase | Worker | Deliverable |
|------|-------|--------|-------------|
| Week 2 (Current) | Specification | Worker 5 | This document + GitHub issues |
| Week 2-3 | Implementation | Worker 2 | GPU kernels in CUDA |
| Week 3 | Integration | Worker 5 | GPU wrappers + schedule integration |
| Week 3 | Testing | Both | Unit + integration tests |
| Week 3 | Optimization | Both | Performance profiling + tuning |
| Week 4 | Validation | Both | Final benchmarks + documentation |

---

## File Locations

**Worker 2 (to create)**:
- `03-Source-Code/src/gpu/thermodynamic_kernels.cu` - CUDA kernel implementations
- `03-Source-Code/src/gpu/thermodynamic_kernels.rs` - Rust FFI bindings
- Update `03-Source-Code/src/gpu/kernel_executor.rs` - Kernel registration

**Worker 5 (to create)**:
- `03-Source-Code/src/orchestration/thermodynamic/gpu_schedule_kernels.rs` - High-level wrappers
- Update existing `advanced_*.rs` files - Integrate GPU calls

**Shared**:
- Update `03-Source-Code/src/orchestration/thermodynamic/mod.rs` - Export GPU module

---

## References

**Worker 5 Modules** (CPU implementations for reference):
- `03-Source-Code/src/orchestration/thermodynamic/advanced_simulated_annealing.rs`
- `03-Source-Code/src/orchestration/thermodynamic/advanced_parallel_tempering.rs`
- `03-Source-Code/src/orchestration/thermodynamic/advanced_hmc.rs`
- `03-Source-Code/src/orchestration/thermodynamic/advanced_bayesian_optimization.rs`
- `03-Source-Code/src/orchestration/thermodynamic/advanced_multi_objective.rs`
- `03-Source-Code/src/orchestration/thermodynamic/advanced_replica_exchange.rs`

**Existing GPU Infrastructure**:
- `03-Source-Code/src/gpu/kernel_executor.rs` - Kernel registration pattern
- `03-Source-Code/src/gpu/mod.rs` - GPU module structure
- Existing `fused_exp_normalize` kernel - Similar to Boltzmann kernel

---

## Action Items

**For Worker 2**:
1. [ ] Review kernel specifications
2. [ ] Estimate implementation timeline (realistic hours)
3. [ ] Confirm performance targets are achievable on H200
4. [ ] Identify any technical blockers or dependencies
5. [ ] Propose kernel registration approach
6. [ ] Coordinate on testing strategy

**For Worker 5** (me):
1. [x] Document kernel requirements (this file)
2. [ ] Create GitHub issue from this document
3. [ ] Create GPU wrapper module skeleton
4. [ ] Profile current CPU performance baseline
5. [ ] Design integration approach for existing modules
6. [ ] Write integration tests (GPU vs CPU comparison)

**For Coordination**:
1. [ ] Worker 0-Alpha approves kernel request
2. [ ] Workers 2 & 5 agree on interface contracts
3. [ ] Timeline confirmed by both workers
4. [ ] Testing responsibilities divided

---

## Questions for Worker 2

1. **Kernel Library**: Should we use cuBLAS for matrix ops (GP covariance) or custom kernels?
2. **Random Numbers**: cuRAND for sampling or pass RNG state from CPU?
3. **Memory Management**: Use existing GpuMemoryPool or new allocation strategy?
4. **Batch Kernel Design**: Single kernel with batch dimension or launch multiple kernels?
5. **Error Handling**: Return CUDA errors as anyhow::Result or separate error type?
6. **Numerical Precision**: f32 sufficient or need f64 for some kernels?

---

## Priority Order

**High Priority** (Week 2-3):
1. Boltzmann Probability Kernel - Used by 3 modules
2. Batch Temperature Update Kernel - Used by all modules
3. Replica Exchange Swap Kernel - Blocking parallel tempering

**Medium Priority** (Week 3):
4. Gaussian Process Covariance Kernel - Bayesian optimization only
5. Pareto Dominance Check Kernel - Multi-objective only

**Lower Priority** (Week 4):
6. Leapfrog Integrator Kernel - HMC only, can use CPU initially

---

**Status**: Ready for Worker 2 review
**Next Step**: Create GitHub issue and tag Worker 2 + Worker 0-Alpha
**Contact**: Worker 5 for questions or clarifications
