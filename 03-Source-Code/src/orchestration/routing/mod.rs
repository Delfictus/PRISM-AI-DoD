//! Routing module
//!
//! Intelligent LLM routing algorithms

pub mod thermodynamic_balancer;
pub mod transfer_entropy_router;
pub mod gpu_transfer_entropy_router;
pub mod te_embedding_gpu;
pub mod gpu_kdtree;
pub mod ksg_transfer_entropy_gpu;
pub mod te_validation;

pub use thermodynamic_balancer::{ThermodynamicLoadBalancer, QuantumVotingConsensus};
pub use transfer_entropy_router::{TransferEntropyPromptRouter, PIDSynergyDetector};
pub use gpu_transfer_entropy_router::{GpuTransferEntropyRouter, QueryDomain, RoutingDecision};
pub use te_embedding_gpu::GpuTimeDelayEmbedding;
pub use gpu_kdtree::{GpuNearestNeighbors, DistanceMetric};
pub use ksg_transfer_entropy_gpu::{KSGTransferEntropyGpu, KSGConfig};
pub use te_validation::{TEValidator, ValidationResult, SyntheticDataGenerator};

// Type alias for integration compatibility
pub type TransferEntropyRouter = TransferEntropyPromptRouter;
