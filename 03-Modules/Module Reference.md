# Module Reference

**Complete guide to all PRISM-AI modules**

---

## Quick Index

| Module | Purpose | Status | LOC |
|--------|---------|--------|-----|
| [[#Mathematics]] | Math foundations | ✅ Complete | ~550 |
| [[#Information Theory]] | Causal discovery | ✅ Complete | ~3,200 |
| [[#Statistical Mechanics]] | Thermodynamics | ✅ Complete | ~1,800 |
| [[#Active Inference]] | Bayesian inference | ✅ Complete | ~2,700 |
| [[#Integration]] | Cross-domain | ✅ Complete | ~1,800 |
| [[#Resilience]] | Fault tolerance | ✅ Complete | ~2,400 |
| [[#Optimization]] | Performance | ✅ Complete | ~1,200 |
| [[#CMA]] | Phase 6 framework | ✅ Complete | ~6,000 |

---

## Mathematics

**Path:** `src/mathematics/`
**Purpose:** Mathematical foundations and proofs
**Status:** ✅ Complete

### Files
- `mod.rs` - Module exports
- `proof_system.rs` - Theorem proving framework
- `information_theory.rs` - Entropy, mutual information
- `thermodynamics.rs` - Thermodynamic laws
- `quantum_mechanics.rs` - Quantum properties

### Key Types
```rust
pub struct MathematicalStatement {
    pub name: &'static str,
    pub theorem: String,
    pub assumptions: Vec<Assumption>,
}

pub struct ProofResult {
    pub valid: bool,
    pub confidence: f64,
    pub details: String,
}
```

### Usage
```rust
use prism_ai::{MathematicalStatement, ProofResult};

// Verify theorems
let result = verify_all_theorems()?;
```

---

## Information Theory

**Path:** `src/information_theory/`
**Purpose:** Causal discovery via transfer entropy
**Status:** ✅ Complete

### Files
- `mod.rs` - Module exports
- `transfer_entropy.rs` - Core TE implementation
- `advanced_transfer_entropy.rs` - KSG estimator
- `causal_direction.rs` - Causality detection

### Key Functions
```rust
pub fn detect_causal_direction(
    source: &[f64],
    target: &[f64],
    source_embedding: usize,
    target_embedding: usize,
    time_lag: usize,
) -> Result<TransferEntropyResult>
```

### Key Types
```rust
pub struct TransferEntropyResult {
    pub te_xy: f64,  // X → Y
    pub te_yx: f64,  // Y → X
    pub direction: CausalDirection,
    pub confidence: f64,
}

pub enum CausalDirection {
    XToY,    // X causes Y
    YToX,    // Y causes X
    Bidirectional,
    None,
}
```

### Usage
```rust
use prism_ai::{detect_causal_direction, CausalDirection};

let source = vec![1.0, 2.0, 3.0, 4.0];
let target = vec![2.0, 3.0, 4.0, 5.0];

let result = detect_causal_direction(&source, &target, 1, 1, 1)?;
match result.direction {
    CausalDirection::XToY => println!("Source causes target"),
    _ => println!("Other relationship"),
}
```

---

## Statistical Mechanics

**Path:** `src/statistical_mechanics/`
**Purpose:** Thermodynamically consistent networks
**Status:** ✅ Complete (GPU: 647x speedup)

### Files
- `mod.rs` - Module exports
- `thermodynamic_network.rs` - Oscillator networks
- `entropy_tracking.rs` - Entropy production

### Key Types
```rust
pub struct ThermodynamicNetwork {
    config: NetworkConfig,
    state: ThermodynamicState,
    // ... internal fields
}

pub struct NetworkConfig {
    pub num_oscillators: usize,
    pub coupling_strength: f64,
    pub damping: f64,
    pub temperature: f64,
    pub dt: f64,
}
```

### Features
- Langevin dynamics
- Fluctuation-dissipation theorem
- Entropy never decreases (dS/dt ≥ 0)
- Boltzmann distribution at equilibrium
- GPU acceleration

### Usage
```rust
use prism_ai::{ThermodynamicNetwork, NetworkConfig};

let config = NetworkConfig {
    num_oscillators: 100,
    coupling_strength: 0.1,
    damping: 0.01,
    temperature: 300.0,
    dt: 0.001,
};

let mut network = ThermodynamicNetwork::new(config);
let state = network.step()?;
```

---

## Active Inference

**Path:** `src/active_inference/`
**Purpose:** Hierarchical Bayesian inference
**Status:** ✅ Complete

### Files
- `mod.rs` - Module exports
- `hierarchical_model.rs` - Multi-level models
- `observation_model.rs` - Sensory predictions
- `transition_model.rs` - State dynamics
- `variational_inference.rs` - Free energy minimization
- `policy_selection.rs` - Expected free energy
- `controller.rs` - Main control loop
- `gpu_inference.rs` - GPU acceleration

### Key Types
```rust
pub struct HierarchicalModel {
    levels: Vec<StateSpaceLevel>,
    beliefs: Vec<GaussianBelief>,
}

pub struct VariationalInference {
    observation_model: ObservationModel,
    transition_model: TransitionModel,
    // ...
}

pub struct ActiveInferenceController {
    selector: PolicySelector,
    strategy: SensingStrategy,
}
```

### Concepts
- **Free Energy:** F = complexity - accuracy
- **Predictive Coding:** Top-down predictions vs bottom-up sensory data
- **Active Sensing:** Choose observations to minimize expected free energy
- **Policy Selection:** Pick actions that reduce uncertainty

### Usage
```rust
use prism_ai::{HierarchicalModel, ActiveInferenceController};

let model = HierarchicalModel::new();
// Create observation and transition models...
let controller = ActiveInferenceController::new(selector, strategy);
```

---

## Integration

**Path:** `src/integration/`
**Purpose:** Cross-domain coupling
**Status:** ✅ Complete

### Files
- `mod.rs` - Module exports
- `cross_domain_bridge.rs` - Neuro-quantum coupling
- `information_channel.rs` - Information flow
- `synchronization.rs` - Phase sync
- `unified_platform.rs` - 8-phase pipeline

### Key Concepts
- Transfer entropy for coupling discovery
- Phase synchronization (Kuramoto model)
- Bidirectional information flow
- Energy-preserving coupling

### Usage
```rust
use prism_ai::{CrossDomainBridge, InformationChannel};

let bridge = CrossDomainBridge::new();
let coupling = bridge.compute_coupling(&neuro_state, &quantum_state)?;
```

---

## Resilience

**Path:** `src/resilience/`
**Purpose:** Enterprise-grade reliability
**Status:** ✅ Complete

### Files
- `mod.rs` - Module exports
- `fault_tolerance.rs` - Health monitoring
- `circuit_breaker.rs` - Cascading failure prevention
- `checkpoint_manager.rs` - State snapshots

### Key Types
```rust
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<Mutex<CircuitState>>,
}

pub struct HealthMonitor {
    components: DashMap<String, ComponentHealth>,
}

pub struct CheckpointManager<T: Checkpointable> {
    storage: Arc<dyn StorageBackend>,
    // ...
}
```

### Guarantees
- MTBF > 1000 hours
- Overhead < 5%
- Atomic operations

### Usage
```rust
use prism_ai::{CircuitBreaker, CircuitBreakerConfig};

let config = CircuitBreakerConfig::default();
let breaker = CircuitBreaker::new(config);

breaker.execute(|| {
    // Your operation
    Ok(())
})?;
```

---

## Optimization

**Path:** `src/optimization/`
**Purpose:** GPU performance optimization
**Status:** ✅ Complete

### Files
- `mod.rs` - Module exports
- `performance_tuner.rs` - Auto-tuning
- `kernel_tuner.rs` - GPU occupancy
- `memory_optimizer.rs` - Triple-buffering

### Features
- Hardware-aware optimization
- Profile caching
- 27-170x speedups demonstrated
- Automatic parameter search

### Usage
```rust
use prism_ai::{PerformanceTuner, KernelTuner};

let tuner = PerformanceTuner::new();
let optimal_params = tuner.auto_tune(&problem)?;
```

---

## CMA (Phase 6)

**Path:** `src/cma/`
**Purpose:** Causal Manifold Annealing - Precision refinement
**Status:** ✅ Complete (6,000+ LOC)

### Structure
```
cma/
├── ensemble_generator.rs       # Ensemble generation
├── causal_discovery.rs         # Manifold discovery
├── quantum_annealer.rs         # Geometric annealing
├── quantum/                    # REAL quantum PIMC
├── neural/                     # Neural enhancements
├── guarantees/                 # Mathematical guarantees
├── applications/               # Domain adapters
├── gpu_integration.rs          # TSP GPU bridge
├── transfer_entropy_ksg.rs     # KSG estimator
├── transfer_entropy_gpu.rs     # GPU-accelerated TE
└── cuda/                       # CUDA kernels
```

### Sub-modules

#### Quantum (`quantum/`)
- `path_integral.rs` - Real path integral Monte Carlo
- Quantum annealing simulation
- 6 CUDA kernels

#### Neural (`neural/`)
- `gnn.rs` - E(3)-Equivariant GNN
- `diffusion.rs` - Consistency diffusion models
- `neural_quantum.rs` - Variational Monte Carlo with ResNet

#### Guarantees (`guarantees/`)
- `pac_bayes.rs` - PAC-Bayes bounds
- `conformal.rs` - Conformal prediction
- `zkp.rs` - Zero-knowledge proofs

#### Applications (`applications/`)
- High-frequency trading adapter
- Materials discovery adapter
- Biomolecular design adapter

### Key Types
```rust
pub struct CausalManifoldAnnealing {
    ensemble_gen: EnhancedEnsembleGenerator,
    causal_disc: CausalManifoldDiscovery,
    quantum_annealer: GeometricQuantumAnnealer,
    // ...
}

pub trait Problem: Send + Sync {
    fn evaluate(&self, solution: &Solution) -> f64;
}

pub struct Solution {
    pub data: Vec<f64>,
    pub cost: f64,
}
```

### Usage
```rust
use prism_ai::cma::{CausalManifoldAnnealing, Problem, Solution};

// Define problem
struct MyProblem;
impl Problem for MyProblem {
    fn evaluate(&self, sol: &Solution) -> f64 {
        sol.data.iter().map(|x| x.powi(2)).sum()
    }
}

// Solve
let cma = CausalManifoldAnnealing::new(config);
let solution = cma.solve(&problem, &initial)?;
```

---

## Domain Engines

### Neuromorphic Engine

**Path:** `src/neuromorphic/`
**LOC:** ~7,200

**Features:**
- Reservoir computing
- STDP learning
- Pattern detection
- GPU acceleration (100x speedup)

### Quantum Engine

**Path:** `src/quantum/`
**LOC:** ~5,100

**Features:**
- Hamiltonian evolution
- TSP solving (40-180x speedup)
- Graph coloring
- Phase resonance

### Platform Foundation

**Path:** `src/foundation/`
**LOC:** ~8,500

**Features:**
- Core platform infrastructure
- Physics coupling
- Data ingestion

---

## Related Documents

- [[Architecture Overview]] - System architecture
- [[API Documentation]] - Detailed API reference
- [[Getting Started]] - Setup and usage
- [[Performance Metrics]] - Benchmarks

---

*Last Updated: 2025-10-04*
