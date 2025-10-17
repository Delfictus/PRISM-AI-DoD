//! Telecom Network Optimization Module
//!
//! GPU-accelerated network optimization for telecommunications infrastructure:
//! - Traffic routing optimization (shortest path, max flow)
//! - Congestion prediction and proactive load balancing
//! - Network topology analysis and health monitoring
//! - QoS (Quality of Service) management
//! - Active Inference for adaptive routing
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated graph algorithms
//! - Article III: Comprehensive testing required
//! - Article IV: Active Inference for dynamic network control

pub mod network_optimizer;

// Re-export main types
pub use network_optimizer::{
    NetworkOptimizer,
    NetworkTopology,
    NetworkNode,
    NetworkLink,
    NodeType,
    RoutingStrategy,
    OptimizationResult,
    TrafficDemand,
    NetworkConfig,
};
