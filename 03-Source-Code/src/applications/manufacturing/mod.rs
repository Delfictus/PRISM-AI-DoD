//! Manufacturing Process Optimization Module
//!
//! GPU-accelerated manufacturing optimization and scheduling:
//! - Production scheduling and job shop optimization
//! - Quality control and defect prediction
//! - Predictive maintenance scheduling
//! - Supply chain integration
//! - Multi-objective optimization (cost, quality, throughput)
//! - Active Inference for adaptive manufacturing
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated optimization
//! - Article III: Comprehensive testing required
//! - Article IV: Active Inference for dynamic scheduling

pub mod optimizer;

// Re-export main types
pub use optimizer::{
    ManufacturingOptimizer,
    ProductionLine,
    Machine,
    MachineType,
    Job,
    MaintenanceSchedule,
    QualityMetrics,
    OptimizationResult,
    ManufacturingConfig,
    SchedulingStrategy,
};
