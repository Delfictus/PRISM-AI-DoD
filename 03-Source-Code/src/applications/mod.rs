//! Applications Domain - Worker 4
//!
//! High-level application implementations that leverage the PRISM-AI platform
//! for specific problem domains.
//!
//! # Modules
//!
//! - `financial`: Portfolio optimization and market analysis tools
//! - `solver`: Universal solver framework for cross-domain problems

pub mod financial;
pub mod solver;

// Re-export key types
pub use financial::{PortfolioOptimizer, Asset, Portfolio};
pub use solver::{UniversalSolver, ProblemType, Solution};
