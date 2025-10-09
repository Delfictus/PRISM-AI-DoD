//! PRCT Infrastructure Adapters
//!
//! Implements port interfaces by wrapping existing neuromorphic and quantum engines.
//! This is the INFRASTRUCTURE layer - connects domain to concrete implementations.

pub mod neuromorphic_adapter;
pub mod quantum_adapter;
pub mod coupling_adapter;

pub use neuromorphic_adapter::NeuromorphicAdapter;
pub use quantum_adapter::QuantumAdapter;
pub use coupling_adapter::CouplingAdapter;
