// Information Theory Module
// Constitution: Phase 1 Task 1.2
// Worker 5 Advanced TE Extensions

pub mod transfer_entropy;
pub mod advanced_transfer_entropy;

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
    ConditionalTransferEntropy,
    LocalTransferEntropy,
    SurrogateDataGenerator,
    PartialInformationDecomposition,
    SurrogateMethod,
};

// GPU-accelerated transfer entropy
pub mod gpu_transfer_entropy;
pub use gpu_transfer_entropy::{GpuTransferEntropy, TransferEntropyGpuExt};

// === Worker 5: Advanced Transfer Entropy Modules ===

/// Conditional Transfer Entropy: TE(X → Y | Z)
/// Distinguish direct vs. mediated causal influences
pub mod conditional_te;
pub use conditional_te::{
    ConditionalTE,
    ConditionalTEResult,
    detect_mediation,
};

/// Multivariate Transfer Entropy: TE(X₁, X₂, ..., Xₙ → Y)
/// Analyze synergistic and redundant information transfer
pub mod multivariate_te;
pub use multivariate_te::{
    MultivariateTE,
    MultivariateTEResult,
    analyze_synergy,
    SynergyType,
    SynergyAnalysis,
    pairwise_redundancy_matrix,
};

/// Time-Delayed Transfer Entropy: max_τ TE(X(t-τ) → Y(t))
/// Optimal lag detection and multi-scale temporal analysis
pub mod time_delayed_te;
pub use time_delayed_te::{
    TimeDelayedTE,
    TimeDelayedTEResult,
    MultiScaleTEResult,
    detect_lead_lag,
    LeadLagResult,
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