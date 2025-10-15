//! Energy Grid Optimization Module
//!
//! GPU-accelerated power grid management and optimization:
//! - Real-time power flow optimization (AC/DC power flow)
//! - Renewable energy integration (solar/wind forecasting)
//! - Demand response management
//! - Grid stability and frequency control
//! - Multi-objective optimization (cost, reliability, emissions)
//! - Active Inference for adaptive grid control
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated power flow calculations
//! - Article III: Comprehensive testing required
//! - Article IV: Active Inference for dynamic grid management

pub mod optimizer;

// Re-export main types
pub use optimizer::{
    EnergyGridOptimizer,
    PowerGrid,
    Generator,
    GeneratorType,
    Bus,
    BusType,
    TransmissionLine,
    Load,
    OptimizationResult,
    GridConfig,
    OptimizationObjective,
};
