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
pub mod portfolio_forecaster;
pub mod quantum_voting;
pub mod thermodynamic_consensus;

// Re-export main types
pub use portfolio_optimizer::{
    PortfolioOptimizer,
    OptimizationStrategy,
    Portfolio,
    Asset,
    PortfolioConfig,
    OptimizationResult,
};

pub use portfolio_forecaster::{
    PortfolioForecaster,
    ForecastConfig,
    ForecastedPortfolio,
    RebalanceAction,
};

pub use quantum_voting::{
    QuantumVotingEngine,
    QuantumVotingConfig,
    QuantumVotingResult,
    QuantumPortfolioState,
};

pub use thermodynamic_consensus::{
    ThermodynamicConsensusEngine,
    ThermodynamicConfig,
    ThermodynamicConsensusResult,
    ThermodynamicState,
};
