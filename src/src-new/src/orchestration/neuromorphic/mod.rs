//! Neuromorphic consensus module

pub mod spike_consensus;
pub mod unified_neuromorphic;
pub mod gpu_neuromorphic;

pub use spike_consensus::NeuromorphicSpikeConsensus;
pub use unified_neuromorphic::{UnifiedNeuromorphicProcessor, ProcessingResult, NeuromorphicConsensus};
pub use gpu_neuromorphic::{GpuNeuromorphicProcessor, NetworkState, SimulationResult};
