pub mod quantum_cache;

pub use quantum_cache::QuantumSemanticCache;

// Type alias for integration compatibility
pub type QuantumApproximateCache = QuantumSemanticCache;
