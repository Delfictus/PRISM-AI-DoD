//! Thermodynamic module

pub mod hamiltonian;
pub mod quantum_consensus;
pub mod gpu_thermodynamic_consensus;
pub mod thermodynamic_consensus;
pub mod advanced_energy;
pub mod temperature_schedules;
pub mod replica_exchange;

pub use hamiltonian::InformationHamiltonian;
pub use quantum_consensus::{QuantumConsensusOptimizer, ConsensusState};
pub use gpu_thermodynamic_consensus::{GpuThermodynamicConsensus, LLMModel, ThermodynamicState as GpuThermodynamicState};
pub use thermodynamic_consensus::ThermodynamicConsensus;
pub use advanced_energy::{AdvancedEnergyModel, AdvancedLLMModel, TaskType, EnergyWeights};
pub use temperature_schedules::{TemperatureSchedule, ScheduleType, TemperatureConfig, ReplicaExchangeManager, ReplicaExchangeStats};
pub use replica_exchange::{ReplicaExchangeSystem, ReplicaState, ReplicaStats, ReplicaExchangeSystemStats};
