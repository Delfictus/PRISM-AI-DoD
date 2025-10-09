//! Platform Foundation
//!
//! Unified API for the world's first software-based neuromorphic-quantum computing platform

pub mod platform;
pub mod types;
pub mod ingestion;
pub mod adapters;
pub mod coupling_physics;
pub mod adaptive_coupling;
pub mod adp;
pub mod phase_causal_matrix;

// Re-export main components
pub use platform::NeuromorphicQuantumPlatform;
pub use types::*;
pub use ingestion::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, ComponentHealth, DataPoint,
    DataSource, EngineConfig, HealthMetrics, HealthReport, HealthStatus, IngestionConfig,
    IngestionEngine, IngestionError, IngestionStats, RetryConfig, RetryPolicy, SourceConfig,
    SourceInfo,
};
pub use adapters::{AlpacaMarketDataSource, OpticalSensorArray, SyntheticDataSource};
pub use coupling_physics::{
    PhysicsCoupling, NeuroQuantumCoupling, QuantumNeuroCoupling, KuramotoSync,
    InformationMetrics, StabilityAnalysis,
};
pub use adaptive_coupling::{
    AdaptiveCoupling, AdaptiveParameter, PerformanceMetrics, CouplingValues, PerformanceSummary,
};
pub use adp::{
    ReinforcementLearner, RlConfig, RlStats, State, Action,
    AdaptiveDecisionProcessor, Decision, AdpStats,
};
pub use phase_causal_matrix::{PhaseCausalMatrixProcessor, PcmConfig};