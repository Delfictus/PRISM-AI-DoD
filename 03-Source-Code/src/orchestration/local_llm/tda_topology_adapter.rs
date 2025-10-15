//! TDA Topology Adapter for Attention Analysis

use anyhow::Result;

/// TDA Topology Adapter trait
pub trait TdaTopologyAdapter {
    fn analyze_topology(&self, data: &[f64]) -> Result<TopologyAnalysis>;
}

#[derive(Debug, Clone)]
pub struct TopologyAnalysis {
    pub betti_numbers: Vec<usize>,
    pub persistence_diagram: Vec<(f64, f64)>,
}
