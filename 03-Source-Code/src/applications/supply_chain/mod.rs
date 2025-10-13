//! Supply Chain Optimization Module
//!
//! GPU-accelerated supply chain management and logistics optimization:
//! - Inventory optimization (EOQ, safety stock, reorder points)
//! - Logistics routing (VRP, TSP, multi-depot routing)
//! - Demand forecasting integration
//! - Multi-objective optimization (cost, time, reliability)
//! - Active Inference for adaptive supply chain management
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated optimization algorithms
//! - Article III: Comprehensive testing required
//! - Article IV: Active Inference for dynamic supply chain control

pub mod optimizer;

// Re-export main types
pub use optimizer::{
    SupplyChainOptimizer,
    InventoryPolicy,
    LogisticsNetwork,
    Warehouse,
    Customer,
    Vehicle,
    Route,
    OptimizationResult,
    SupplyChainConfig,
    OptimizationStrategy,
};
