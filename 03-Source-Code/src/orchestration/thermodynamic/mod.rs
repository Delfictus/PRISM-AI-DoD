//! Thermodynamic module

pub mod hamiltonian;
pub mod quantum_consensus;

pub use hamiltonian::InformationHamiltonian;
pub use quantum_consensus::{QuantumConsensusOptimizer, ConsensusState};
