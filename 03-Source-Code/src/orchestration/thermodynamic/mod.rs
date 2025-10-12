//! Thermodynamic module

pub mod hamiltonian;
pub mod quantum_consensus;
pub mod gpu_thermodynamic_consensus;
pub mod thermodynamic_consensus;

pub use hamiltonian::InformationHamiltonian;
pub use quantum_consensus::{QuantumConsensusOptimizer, ConsensusState};
pub use gpu_thermodynamic_consensus::{GpuThermodynamicConsensus, LLMModel, ThermodynamicState as GpuThermodynamicState};
pub use thermodynamic_consensus::ThermodynamicConsensus;
