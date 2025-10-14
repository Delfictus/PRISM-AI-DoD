// Information Theory Module
// Constitution: Phase 1 Task 1.2

pub mod transfer_entropy;
pub mod advanced_transfer_entropy;

// Phase 1 Enhancements: High-Accuracy TE Estimation
pub mod kdtree;
pub mod ksg_estimator;
pub mod conditional_te;
pub mod bootstrap_ci;
pub mod transfer_entropy_gpu;

// Phase 2 Enhancements: Performance Optimizations
pub mod incremental_te;
pub mod memory_efficient;
pub mod adaptive_embedding;
pub mod symbolic_te;

// Phase 3 Enhancements: Research Extensions
pub mod pid;
pub mod multiple_testing;

pub use transfer_entropy::{
    TransferEntropy,
    TransferEntropyResult,
    CausalDirection,
    detect_causal_direction,
};

pub use advanced_transfer_entropy::{
    AdvancedTransferEntropy,
    KozachenkoLeonenkoEstimator,
    SymbolicTransferEntropy,
    RenyiTransferEntropy,
    ConditionalTransferEntropy as AdvancedConditionalTe,
    LocalTransferEntropy,
    SurrogateDataGenerator,
    PartialInformationDecomposition,
    SurrogateMethod,
};

// Phase 1 exports
pub use kdtree::{KdTree, Neighbor};
pub use ksg_estimator::KsgEstimator;
pub use conditional_te::ConditionalTe;
pub use bootstrap_ci::{BootstrapResampler, BootstrapCi, BootstrapMethod};
pub use transfer_entropy_gpu::TransferEntropyGpu;

// Phase 2 exports
pub use incremental_te::IncrementalTe;
pub use memory_efficient::{SparseHistogram, CountMinSketch, CompressedKey, CompressedHistogram};
pub use adaptive_embedding::{AdaptiveEmbedding, EmbeddingParams};
pub use symbolic_te::SymbolicTe;

// Phase 3 exports
pub use pid::{PartialInfoDecomp, PidResult, PidMethod};
pub use multiple_testing::{MultipleTestingCorrection, CorrectedPValues, CorrectionMethod};

// GPU-accelerated transfer entropy
pub mod gpu_transfer_entropy;
pub use gpu_transfer_entropy::{GpuTransferEntropy, TransferEntropyGpuExt};

// GPU-accelerated entropy and divergence (Phase 3)
pub mod gpu_entropy;
pub use gpu_entropy::{GpuEntropyCalculator, GpuKLDivergence};

// Worker 4 Enhancements: Advanced Information-Theoretic Estimators
pub mod kdtree;
pub mod ksg_estimator;
pub mod mutual_information;

pub use kdtree::{KdTree, Point};
pub use ksg_estimator::{KsgEstimator, KsgConfig, KsgResult, digamma};
pub use mutual_information::{
    MutualInformationEstimator,
    MutualInformationResult,
    MiMethod,
};

/// Information-theoretic measures for the Active Inference Platform
pub trait InformationMeasure {
    /// Calculate entropy H(X)
    fn entropy(&self) -> f64;

    /// Calculate mutual information I(X;Y)
    fn mutual_information(&self, other: &Self) -> f64;

    /// Calculate conditional entropy H(X|Y)
    fn conditional_entropy(&self, condition: &Self) -> f64;
}

/// Verify information-theoretic inequalities
pub fn verify_information_bounds(entropy: f64, mutual_info: f64) -> bool {
    // H(X) >= 0 (non-negativity of entropy)
    if entropy < 0.0 {
        return false;
    }

    // I(X;Y) >= 0 (non-negativity of mutual information)
    if mutual_info < 0.0 {
        return false;
    }

    // I(X;Y) <= min(H(X), H(Y)) (mutual information bound)
    // This would need both entropies to verify fully

    true
}