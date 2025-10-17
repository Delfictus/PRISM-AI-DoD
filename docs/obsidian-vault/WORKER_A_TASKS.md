# Worker A - Algorithm Developer Tasks

**Your Focus**: Advanced algorithm implementations (TE, Thermodynamic, Active Inference)
**Your Directories**: `src/orchestration/routing/`, `src/orchestration/thermodynamic/`, `src/active_inference/`
**Your Time**: 125 hours over 3-4 weeks

---

## WEEK 1: Transfer Entropy Router (40 hours)

### Day 1-2: Time-Delay Embedding & k-NN (16 hours)

**Monday Morning** (4h):
```bash
git checkout -b worker-a-algorithms
cd src/orchestration/routing

# Create te_embedding_gpu.rs
touch te_embedding_gpu.rs
```

**Task 1.1.1**: Implement `GpuTimeDelayEmbedding` struct
- [ ] Create struct with executor and context
- [ ] Implement `embed_gpu()` method
- [ ] Use existing `time_delayed_embedding` kernel
- [ ] Add tests with simple time series

**Monday Afternoon** (4h):
**Task 1.1.2**: Handle edge cases
- [ ] Insufficient data handling
- [ ] Boundary conditions
- [ ] Validate embedding dimensions

**Tuesday Morning** (4h):
**Task 1.1.3**: Automatic τ selection
- [ ] Implement autocorrelation on GPU
- [ ] Find optimal lag
- [ ] Add heuristic selection

**Tuesday Afternoon** (4h):
**Task 1.2.1**: Start GPU k-NN
```bash
touch gpu_kdtree.rs
```
- [ ] Implement `GpuNearestNeighbors` struct
- [ ] Create distance computation kernel

### Day 3-4: Complete k-NN (16 hours)

**Task 1.2.2**: Complete k-NN implementation
- [ ] Parallel distance computation
- [ ] Top-k selection (use existing `top_k_sampling` kernel)
- [ ] Optimize memory layout

**Task 1.2.3**: Benchmark
- [ ] Test vs CPU KDTree
- [ ] Find crossover point (when GPU faster)

### Day 5: Full KSG TE - Part 1 (8 hours)

**Task 1.3.1**: Start KSG implementation
```bash
touch ksg_transfer_entropy_gpu.rs
```
- [ ] Create `KSGTransferEntropyGpu` struct
- [ ] Implement joint space concatenation
- [ ] Test embedding pipeline

---

## WEEK 2: Complete TE + Start Thermodynamic (40 hours)

### Day 6-7: KSG TE Completion (14 hours)

**Task 1.3.2**: Marginal neighbor counting
- [ ] Implement counting in marginal spaces
- [ ] GPU kernel for neighbor queries

**Task 1.3.3**: Digamma function
- [ ] GPU approximation of ψ(x)
- [ ] Validate accuracy

**Task 1.3.4**: Full KSG formula
- [ ] Implement: TE = ψ(k) + ⟨ψ(nₓ)⟩ - ⟨ψ(nₓᵧ)⟩ - ψ(N)
- [ ] Test on synthetic data with known TE

### Day 8: TE Advanced Features (6 hours)

**Task 1.3.5**: Statistical significance
- [ ] Permutation test on GPU
- [ ] p-value computation

**Task 1.3.6**: Validation
- [ ] Compare with JIDT
- [ ] Ensure < 5% error

**Task 1.4.1**: Advanced routing
- [ ] Multi-dimensional domains
- [ ] Continuous learning

### Day 9-10: Thermodynamic Energy Model (14 hours)

**Task 2.1.1**: Multi-factor energy
```bash
cd ../thermodynamic
touch advanced_energy.rs
```
- [ ] Cost + quality + latency + uncertainty + context
- [ ] GPU weighted sum kernel

**Task 2.1.2**: Task-specific quality
- [ ] Per-domain quality estimation
- [ ] Historical tracking

**Task 2.1.3**: Bayesian uncertainty
- [ ] Epistemic + aleatoric uncertainty
- [ ] GPU computation

**Task 2.1.4**: Learn weights
- [ ] Gradient descent on GPU
- [ ] Update energy weights from feedback

---

## WEEK 3: Thermodynamic + Active Inference (40 hours)

### Day 11-12: Temperature Schedules (16 hours)

**Task 2.2.1**: Implement 5 schedules
```bash
touch temperature_schedules.rs
```
- [ ] Exponential
- [ ] Logarithmic
- [ ] Adaptive (acceptance-rate based)
- [ ] Fokker-Planck SDE (with cuRAND)
- [ ] Replica exchange

**Task 2.2.2**: Adaptive schedule
- [ ] Track acceptance rate
- [ ] Adjust temperature dynamically
- [ ] Target 23.4% optimal acceptance

**Task 2.2.3**: Fokker-Planck
- [ ] Implement SDE: dT = -γT dt + η√T dW
- [ ] Use cuRAND for stochastic term

### Day 13-15: Replica Exchange (24 hours)

**Task 2.3.1**: Framework
```bash
touch replica_exchange.rs
```
- [ ] Multiple replicas at different temperatures
- [ ] Parallel evolution on GPU

**Task 2.3.2**: Metropolis swaps
- [ ] Swap acceptance: min(1, exp(ΔβΔE))
- [ ] Track swap statistics

**Task 2.3.3**: Convergence diagnostics
- [ ] Gelman-Rubin statistic on GPU
- [ ] Adaptive temperature spacing

---

## WEEK 4: Active Inference + Cleanup (45 hours)

### Day 16-18: Hierarchical Inference (24 hours)

**Task 4.1.1**: Multi-level hierarchy
```bash
cd ../../active_inference
touch hierarchical_inference_gpu.rs
```
- [ ] Create `HierarchicalActiveInferenceGpu`
- [ ] GPU-resident beliefs at each level

**Task 4.1.2**: Precision-weighted errors
- [ ] Implement on GPU
- [ ] Use existing precision matrices

**Task 4.1.3**: Message passing
- [ ] Bottom-up error propagation
- [ ] Top-down predictions
- [ ] All on GPU

### Day 19-20: Policy Search (16 hours)

**Task 4.2.1**: Parallel evaluation
```bash
touch policy_search_gpu.rs
```
- [ ] Evaluate N policies in parallel
- [ ] Use existing kernels

**Task 4.2.2**: Model-based planning
- [ ] Forward simulation on GPU
- [ ] Expected free energy computation

### Day 21: Documentation & Testing (5 hours)

- [ ] Document all new algorithms
- [ ] Integration tests
- [ ] Performance validation

---

## SUCCESS METRICS - WORKER A

**Transfer Entropy**:
- [ ] Actual KSG computation (not proxy)
- [ ] < 5% error vs JIDT
- [ ] < 100ms for 1000 variables

**Thermodynamic**:
- [ ] 5 schedules operational
- [ ] Replica exchange converges faster
- [ ] 40-70% cost savings demonstrated

**Active Inference**:
- [ ] Hierarchical inference working
- [ ] Policy search < 1ms
- [ ] Demonstrates adaptive behavior

---

## TOOLS YOU'LL USE

**GPU Kernels Available** (from Worker B):
- All 43 existing kernels
- Request new kernels via GitHub issues

**Testing**:
```bash
cargo test --lib --features cuda -- [your_test_name]
```

**Benchmarking**:
```bash
cargo bench --features cuda
```

**GPU Monitoring**:
```bash
watch -n 0.5 nvidia-smi
```

---

## GETTING HELP

**Questions About**:
- GPU kernels → Ask Worker B
- Algorithm math → Check papers in `02-Mathematical-Framework/`
- Integration → Coordinate with Worker B

**Resources**:
- JIDT documentation for TE validation
- Statistical mechanics textbooks for thermodynamic
- Friston papers for Active Inference

---

**WORKER A - YOUR MISSION**: Make algorithms sophisticated and production-grade
**YOUR STRENGTH**: Mathematical rigor and algorithmic sophistication
**YOUR DELIVERABLE**: World-class TE, Thermodynamic, and Active Inference implementations