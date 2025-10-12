# 🎯 GPU Module Priority List
## Ranked by Impact & Implementation Complexity

---

## 🔴 CRITICAL PRIORITY (Do First - High Impact, Clear Path)

### 1. **PWSA Active Inference Classifier** ⭐⭐⭐⭐⭐
**File**: `src/pwsa/active_inference_classifier.rs`
- **Current**: CPU-only Linear layers and softmax
- **GPU Ops Needed**: Matrix multiply, softmax, cross-entropy
- **Expected Speedup**: 20-30x
- **Complexity**: Low (well-defined operations)
- **Why First**: Self-contained, immediate benefit, clear operations

### 2. **Transfer Entropy Calculator** ⭐⭐⭐⭐⭐
**File**: `src/cma/transfer_entropy_gpu.rs`
- **Current**: Sequential k-NN and entropy calculations
- **GPU Ops Needed**: Parallel k-NN, histogram, mutual information
- **Expected Speedup**: 50-100x
- **Complexity**: Medium (parallel algorithms needed)
- **Why Priority**: Heavy compute, used frequently

### 3. **GPU Launcher Core** ⭐⭐⭐⭐⭐
**File**: `src/gpu_launcher.rs`
- **Current**: Placeholder implementations
- **GPU Ops Needed**: Memory management, kernel dispatch
- **Expected Speedup**: N/A (infrastructure)
- **Complexity**: Low (fix existing structure)
- **Why Priority**: Required by all other modules

---

## 🟡 HIGH PRIORITY (Significant Performance Gains)

### 4. **Neural Quantum State VMC** ⭐⭐⭐⭐
**File**: `src/cma/neural/neural_quantum.rs`
- **Current**: CPU Monte Carlo sampling
- **GPU Ops Needed**: Parallel sampling, ResNet forward, Metropolis
- **Expected Speedup**: 50-200x
- **Complexity**: High (stochastic algorithms)
- **Why High**: Massive parallelism opportunity

### 5. **Active Inference Policy Evaluation** ⭐⭐⭐⭐
**File**: `src/active_inference/gpu_policy_eval.rs`
- **Current**: Sequential free energy computation
- **GPU Ops Needed**: Belief propagation, expectation
- **Expected Speedup**: 30-50x
- **Complexity**: Medium
- **Why High**: Core to decision making

### 6. **Thermodynamic Evolution** ⭐⭐⭐⭐
**File**: `src/statistical_mechanics/gpu_bindings.rs`
- **Current**: CPU physics simulation
- **GPU Ops Needed**: Hamiltonian evolution, phase space
- **Expected Speedup**: 40-60x
- **Complexity**: Medium
- **Why High**: Time-stepping benefits from GPU

---

## 🟢 MEDIUM PRIORITY (Good Gains, More Complex)

### 7. **Graph Coloring** ⭐⭐⭐
**File**: `src/prct-core/coloring.rs`
- **Current**: Sequential greedy algorithm
- **GPU Ops Needed**: Parallel color assignment
- **Expected Speedup**: 10-20x
- **Complexity**: High (irregular memory)

### 8. **TSP Solver** ⭐⭐⭐
**File**: `src/prct-core/tsp_gpu.rs`
- **Current**: CPU branch-and-bound
- **GPU Ops Needed**: Parallel neighborhood search
- **Expected Speedup**: 15-25x
- **Complexity**: High (complex algorithms)

### 9. **E3-Equivariant GNN** ⭐⭐⭐
**File**: `src/cma/neural/gnn_integration.rs`
- **Current**: Stub implementation
- **GPU Ops Needed**: Message passing, aggregation
- **Expected Speedup**: 20-40x
- **Complexity**: High (graph operations)

### 10. **Consistency Diffusion** ⭐⭐⭐
**File**: `src/cma/neural/diffusion.rs`
- **Current**: Stub implementation
- **GPU Ops Needed**: Noise generation, U-Net
- **Expected Speedup**: 30-50x
- **Complexity**: High (needs full implementation)

---

## 🔵 LOWER PRIORITY (Optimize Later)

### 11. **QUBO Optimization** ⭐⭐
**File**: `src/prct-core/qubo.rs`
- **GPU Ops**: Quadratic form evaluation
- **Expected Speedup**: 10-15x

### 12. **Simulated Annealing** ⭐⭐
**File**: `src/prct-core/simulated_annealing.rs`
- **GPU Ops**: Parallel temperature chains
- **Expected Speedup**: 5-10x

### 13. **PAC-Bayes Bounds** ⭐⭐
**File**: `src/cma/pac_bayes.rs`
- **GPU Ops**: Statistical computations
- **Expected Speedup**: 5-8x

### 14. **Conformal Prediction** ⭐⭐
**File**: `src/cma/conformal_prediction.rs`
- **GPU Ops**: Set operations
- **Expected Speedup**: 3-5x

---

## 📊 Module Dependency Graph

```
gpu_launcher.rs (Infrastructure)
    ├── memory_manager.rs (NEW - Create First)
    └── kernel_launcher.rs (NEW - Create Second)
            │
            ├── active_inference_classifier.rs (Port First)
            ├── transfer_entropy_gpu.rs (Port Second)
            ├── neural_quantum.rs (Port Third)
            └── gpu_policy_eval.rs (Port Fourth)
```

---

## ⚡ Quick Win Implementations

### WEEK 1 TARGETS (Maximum Impact)
1. **Monday**: GPU memory manager + kernel infrastructure
2. **Tuesday**: PWSA classifier GPU acceleration
3. **Wednesday**: Transfer entropy GPU kernel
4. **Thursday**: Test, benchmark, optimize
5. **Friday**: Documentation + plan Week 2

### Expected Week 1 Outcomes:
- ✅ 2 modules fully GPU-accelerated
- ✅ 20-50x speedup demonstrated
- ✅ Infrastructure ready for remaining modules
- ✅ Benchmarks proving GPU advantage

---

## 🔧 Files That Need Creation

### New Infrastructure Files:
```
src/gpu/
├── memory_manager.rs      # GPU memory pool
├── kernel_launcher.rs      # Kernel dispatch
├── tensor_ops.rs          # GPU tensor operations
└── layers/
    ├── linear.rs          # GPU Linear layer
    ├── activation.rs      # GPU activations
    └── normalization.rs   # GPU LayerNorm, etc.

src/kernels/cuda/
├── matrix_ops.cu          # GEMM, transpose
├── reduction_ops.cu       # Sum, mean, max
├── activation_ops.cu      # ReLU, sigmoid, softmax
├── entropy_ops.cu         # Transfer entropy
└── graph_ops.cu          # Graph algorithms
```

---

## 📈 Performance Impact Analysis

| Module | Current Time | GPU Time | Users Impacted | Business Value |
|--------|-------------|----------|----------------|----------------|
| PWSA Classifier | 100ms | 5ms | All inference | ⭐⭐⭐⭐⭐ |
| Transfer Entropy | 500ms | 10ms | Causal analysis | ⭐⭐⭐⭐⭐ |
| Neural Quantum | 1000ms | 20ms | Optimization | ⭐⭐⭐⭐ |
| Policy Eval | 200ms | 10ms | Decision making | ⭐⭐⭐⭐ |
| Graph Coloring | 50ms | 5ms | PRCT core | ⭐⭐⭐ |

---

## ✅ Success Criteria

A module is considered "GPU-accelerated" when:
1. ✅ All compute-intensive operations run on GPU
2. ✅ Achieves >10x speedup over CPU
3. ✅ Passes all existing tests
4. ✅ Has fallback to CPU if GPU unavailable
5. ✅ Memory usage is optimized
6. ✅ No memory leaks verified

---

## 🚀 RECOMMENDED FIRST ACTION

**START WITH**: `src/pwsa/active_inference_classifier.rs`

**Step 1**: Create `src/gpu/layers/linear.rs`
```rust
// Implement GPU-accelerated Linear layer
// This will be reusable across multiple modules
```

**Step 2**: Replace in classifier:
```rust
// Old:
let fc1 = Linear::new(100, 64);

// New:
let fc1 = GpuLinear::new(ctx.clone(), 100, 64)?;
```

**Step 3**: Benchmark:
```bash
cargo bench --features cuda active_inference
```

**Expected Result**: 20-30x speedup on forward pass

---

*This priority list provides clear guidance on which modules to GPU-accelerate first for maximum impact.*