# Architecture Overview

**PRISM-AI System Architecture**

---

## System Layers

```
┌─────────────────────────────────────────────────┐
│           Applications & Demos                  │
│  (Examples, Benchmarks, User Code)             │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│         Public API (src/lib.rs)                 │
│  Mathematics, InfoTheory, ActiveInference, CMA  │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼────────┐ ┌──▼──────────┐
│ Mathematics  │ │ Information│ │   Active    │
│   Module     │ │   Theory   │ │  Inference  │
└───────┬──────┘ └───┬────────┘ └──┬──────────┘
        │             │             │
┌───────▼──────┐ ┌───▼────────┐ ┌──▼──────────┐
│ Statistical  │ │ Integration│ │  Resilience │
│  Mechanics   │ │            │ │             │
└───────┬──────┘ └───┬────────┘ └──┬──────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│      Domain Engines (Workspace Crates)          │
│  Neuromorphic | Quantum | Foundation            │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           GPU Layer (CUDA Kernels)              │
│  23 Custom Kernels + cuBLAS + Candle            │
└─────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. **Mathematics** (`src/mathematics/`)
Mathematical foundations and proofs.

**Sub-modules:**
- `proof_system.rs` - Mathematical theorem proving
- `information_theory.rs` - Entropy, mutual information
- `thermodynamics.rs` - Thermodynamic laws
- `quantum_mechanics.rs` - Quantum properties

**Key Types:**
- `MathematicalStatement` - Theorem definitions
- `ProofResult` - Proof verification results
- `Assumption` - Mathematical assumptions

### 2. **Information Theory** (`src/information_theory/`)
Causal discovery and information flow analysis.

**Files:**
- `transfer_entropy.rs` - Core TE implementation
- `advanced_transfer_entropy.rs` - KSG estimator
- `causal_direction.rs` - Causality detection

**Key Functions:**
- `detect_causal_direction()` - Main API
- Transfer entropy computation
- Statistical significance testing

### 3. **Statistical Mechanics** (`src/statistical_mechanics/`)
Thermodynamically consistent networks.

**Files:**
- `thermodynamic_network.rs` - Oscillator networks
- `entropy_tracking.rs` - Entropy production (dS/dt ≥ 0)

**Features:**
- Langevin dynamics
- Fluctuation-dissipation theorem
- Boltzmann distribution
- GPU acceleration (647x speedup)

### 4. **Active Inference** (`src/active_inference/`)
Hierarchical Bayesian inference framework.

**Sub-modules:**
- `hierarchical_model.rs` - Multi-level models
- `observation_model.rs` - Sensory predictions
- `transition_model.rs` - State dynamics
- `variational_inference.rs` - Free energy minimization
- `policy_selection.rs` - Expected free energy
- `controller.rs` - Main inference loop
- `gpu_inference.rs` - GPU acceleration

**Key Concepts:**
- Free energy minimization
- Predictive coding
- Active sensing
- Policy selection

### 5. **Integration** (`src/integration/`)
Cross-domain coupling mechanisms.

**Components:**
- `cross_domain_bridge.rs` - Neuro-quantum coupling
- `information_channel.rs` - Information flow
- `synchronization.rs` - Phase synchronization
- `unified_platform.rs` - 8-phase pipeline

### 6. **Resilience** (`src/resilience/`)
Enterprise-grade reliability features.

**Components:**
- `circuit_breaker.rs` - Cascading failure prevention
- `fault_tolerance.rs` - Health monitoring
- `checkpoint_manager.rs` - State snapshots

**Guarantees:**
- MTBF > 1000 hours
- Overhead < 5%
- Atomic operations

### 7. **Optimization** (`src/optimization/`)
Performance tuning and GPU optimization.

**Components:**
- `performance_tuner.rs` - Auto-tuning
- `kernel_tuner.rs` - GPU occupancy analysis
- `memory_optimizer.rs` - Triple-buffering

**Achievements:**
- 27-170x speedups demonstrated
- Hardware-aware optimization
- Profile caching

### 8. **CMA Framework** (`src/cma/`)
Phase 6: Causal Manifold Annealing

**Structure:**
```
cma/
├── ensemble_generator.rs      # Ensemble generation
├── causal_discovery.rs        # Manifold discovery
├── quantum_annealer.rs        # Geometric annealing
├── quantum/                   # REAL quantum PIMC
│   └── path_integral.rs
├── neural/                    # Neural enhancements
│   ├── gnn.rs                # E(3)-Equivariant GNN
│   ├── diffusion.rs          # Consistency diffusion
│   └── neural_quantum.rs     # VMC with ResNet
├── guarantees/                # Mathematical guarantees
│   ├── pac_bayes.rs          # PAC-Bayes bounds
│   ├── conformal.rs          # Conformal prediction
│   └── zkp.rs                # Zero-knowledge proofs
├── applications/              # Domain adapters
│   └── mod.rs                # HFT, Materials, Bio
├── gpu_integration.rs         # TSP GPU bridge
├── transfer_entropy_ksg.rs    # KSG estimator
└── cuda/                      # CUDA kernels
```

---

## Workspace Crates

### Internal Crates
Located in `src/`:

1. **neuromorphic-engine** (`src/neuromorphic/`)
   - Reservoir computing
   - STDP learning
   - Pattern detection
   - GPU acceleration

2. **quantum-engine** (`src/quantum/`)
   - Hamiltonian evolution
   - TSP solving
   - Graph coloring
   - Phase resonance

3. **platform-foundation** (`src/foundation/`)
   - Core platform infrastructure
   - Physics coupling
   - Data ingestion

4. **shared-types** (`src/shared-types/`)
   - Zero-dependency data types
   - Graph structures
   - Quantum states
   - Neuro states

5. **prct-core** (`src/prct-core/`)
   - PRCT algorithm
   - Port definitions (interfaces)
   - Domain logic

6. **prct-adapters** (`src/adapters/`)
   - Infrastructure adapters
   - Neuromorphic adapter
   - Quantum adapter
   - Coupling adapter

7. **mathematics** (`src/mathematics/`)
   - Math utilities
   - Numerical methods

---

## GPU Architecture

### CUDA Kernels (23 total)

**Optimization (3 kernels):**
- `optimize_memory_layout`
- `analyze_memory_access`
- `tune_kernel_params`

**Neuromorphic (6 kernels):**
- `reservoir_step`
- `stdp_update`
- `spike_detection`
- Pattern detection kernels

**Quantum (8 kernels):**
- `hamiltonian_evolution`
- `tsp_solver`
- `graph_coloring`
- TSP specific kernels

**Coupling (6 kernels):**
- `phase_sync`
- `transfer_entropy`
- Kuramoto integration

**CMA Phase 6 (13 new kernels):**
- Transfer Entropy KSG (7 kernels)
- Quantum PIMC (6 kernels)

### GPU Libraries Used
- **cudarc** - CUDA bindings
- **candle-core** - Neural network ops
- **cuBLAS** - Linear algebra
- **cuRAND** - Random numbers

---

## Data Flow

### Input → Processing → Output

```
User Input
    ↓
Public API (lib.rs)
    ↓
Module Layer (mathematics, info_theory, etc.)
    ↓
Domain Engines (neuromorphic, quantum)
    ↓
GPU Kernels (CUDA)
    ↓
Hardware (RTX GPU)
    ↓
Results
    ↓
User Output
```

### Cross-Domain Flow

```
Neuromorphic Domain    Quantum Domain
       ↓                    ↓
   SpikePattern        QuantumState
       ↓                    ↓
       └──→ Coupling ←──────┘
              ↓
       PhaseField
       KuramotoState
       TransferEntropy
```

---

## Dependency Graph

```
prism-ai (main)
├── shared-types (no deps)
├── prct-core
│   └── shared-types
├── prct-adapters
│   ├── prct-core
│   ├── shared-types
│   ├── neuromorphic-engine
│   ├── quantum-engine
│   └── platform-foundation
├── neuromorphic-engine
│   └── shared-types
├── quantum-engine
│   └── shared-types
├── platform-foundation
│   ├── neuromorphic-engine
│   ├── quantum-engine
│   └── shared-types
└── mathematics
    └── shared-types
```

**Key:** Zero circular dependencies (DAG structure)

---

## Design Principles

### 1. **Hexagonal Architecture**
- Domain logic in core
- Infrastructure in adapters
- Ports & adapters pattern

### 2. **Information-theoretic Coupling**
- Transfer entropy for causal discovery
- Phase synchronization
- Bidirectional information flow

### 3. **GPU-First Design**
- Custom CUDA kernels
- Zero-copy operations
- Memory pooling
- Batch processing

### 4. **Type Safety**
- Strong typing throughout
- Zero `unsafe` in core logic
- Compile-time guarantees

### 5. **Fault Tolerance**
- Circuit breakers
- Health monitoring
- Checkpoint/restore
- Graceful degradation

---

## Performance Characteristics

### Latency Targets
- Active inference: <2ms
- Recognition model: <100 iterations
- Ensemble generation: <500ms
- Causal discovery: <200ms
- Quantum annealing: <1000ms

### Throughput
- Thermodynamic step: 0.08ms (1024 oscillators)
- TSP solution: 43s (13,509 cities)
- Graph coloring: 938ms (1000 vertices)

### Scalability
- Supports up to 10K oscillators
- TSP instances up to 20K cities
- Parallel GPU execution

---

## Related Documents

- [[Module Reference]] - Detailed module docs
- [[API Documentation]] - Public API reference
- [[Performance Metrics]] - Benchmarks
- [[Development Workflow]] - How to develop

---

*Last Updated: 2025-10-04*
