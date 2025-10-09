//! Prism-AI Platform
//!
//! PRISM: Predictive Reasoning via Information-theoretic Statistical Manifolds
//!
//! GPU-accelerated quantum and neuromorphic computing platform with
//! mathematical guarantees and extreme precision (10^-30 accuracy).
//!
//! Features:
//! - 100x GPU acceleration via CUDA kernels
//! - Double-double (106-bit) precision for mathematical guarantees
//! - Bit-for-bit deterministic reproducibility
//! - Validated against QuTiP reference implementation

pub mod mathematics;
pub mod information_theory;
pub mod statistical_mechanics;
pub mod active_inference;
pub mod integration;
pub mod resilience;
pub mod optimization;
pub mod cma; // Phase 6: Causal Manifold Annealing

// GPU acceleration modules (OBSOLETE - quantum_mlir replaces this)
// #[cfg(feature = "cuda")]
// pub mod cuda_bindings;

// MLIR runtime for JIT compilation
#[cfg(feature = "mlir")]
pub mod mlir_runtime;

// Quantum MLIR Dialect - First-class GPU acceleration with native complex support!
pub mod quantum_mlir;

// GPU-accelerated graph coloring
#[cfg(feature = "cuda")]
pub mod gpu_coloring;

// Re-export key components
pub use mathematics::{
    MathematicalStatement, ProofResult, Assumption,
};

pub use information_theory::{
    TransferEntropy, TransferEntropyResult, CausalDirection,
    detect_causal_direction,
};

pub use statistical_mechanics::{
    ThermodynamicNetwork, ThermodynamicState, NetworkConfig,
    ThermodynamicMetrics, EvolutionResult,
};

pub use active_inference::{
    GenerativeModel, HierarchicalModel, StateSpaceLevel,
    ObservationModel, TransitionModel, VariationalInference,
    PolicySelector, ActiveInferenceController,
};

pub use integration::{
    CrossDomainBridge, DomainState, CouplingStrength,
    InformationChannel, PhaseSynchronizer,
};

pub use resilience::{
    HealthMonitor, ComponentHealth, HealthStatus, SystemState,
    CircuitBreaker, CircuitState, CircuitBreakerConfig, CircuitBreakerError,
    CheckpointManager, Checkpointable, CheckpointMetadata, StorageBackend, CheckpointError,
};

pub use optimization::{
    PerformanceTuner, TuningProfile, SearchAlgorithm, SearchSpace, PerformanceMetrics,
    KernelTuner, GpuProperties, KernelConfig, OccupancyInfo,
};

/// Platform version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Prism-AI";
pub const DESCRIPTION: &str = "PRISM: Predictive Reasoning via Information-theoretic Statistical Manifolds";

