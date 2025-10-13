//! Finance Module - Portfolio Optimization
//!
//! GPU-accelerated portfolio optimization using Modern Portfolio Theory (MPT),
//! Black-Litterman model, and risk parity strategies.
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated covariance computation and optimization
//! - Article III: Comprehensive testing required
//! - Article IV: Active Inference for dynamic rebalancing

pub mod portfolio_optimizer;

// Re-export main types
pub use portfolio_optimizer::{
    PortfolioOptimizer,
    OptimizationStrategy,
    Portfolio,
    Asset,
    PortfolioConfig,
    OptimizationResult,
};
