//! Multi-modal fusion module
//!
//! World-first algorithms for sensor-text fusion

pub mod unified_neuromorphic;

pub use unified_neuromorphic::{
    UnifiedNeuromorphicEncoder,
    BidirectionalCausalFusion,
    JointActiveInference,
    GeometricManifoldFusion,
    QuantumEntangledMultiModal,
};
