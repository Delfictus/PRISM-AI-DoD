//! GPU-accelerated KSG Transfer Entropy Estimator
//!
//! Implements the Kraskov-Stögbauer-Grassberger (KSG) algorithm for transfer entropy
//! estimation using GPU acceleration.
//!
//! Transfer Entropy: TE(X→Y) measures information flow from X to Y
//! TE(X→Y) = I(Y_future ; X_past | Y_past)
//!
//! KSG Algorithm:
//! TE = ψ(k) + ⟨ψ(n_x)⟩ - ⟨ψ(n_xy)⟩ - ⟨ψ(n_y)⟩
//!
//! where:
//! - k: number of nearest neighbors
//! - ψ: digamma function
//! - n_x, n_xy, n_y: neighbor counts in marginal spaces

use anyhow::{Result, Context as AnyhowContext};
use ndarray::{Array1, Array2, Axis};

use super::te_embedding_gpu::GpuTimeDelayEmbedding;
use super::gpu_kdtree::{GpuNearestNeighbors, DistanceMetric};

/// Configuration for KSG Transfer Entropy estimation
#[derive(Debug, Clone)]
pub struct KSGConfig {
    /// Number of nearest neighbors (typical: 3-10)
    pub k: usize,
    /// Source embedding dimension
    pub source_embedding_dim: usize,
    /// Target embedding dimension
    pub target_embedding_dim: usize,
    /// Source time delay
    pub source_tau: usize,
    /// Target time delay
    pub target_tau: usize,
    /// Prediction horizon (how far ahead to predict)
    pub prediction_horizon: usize,
}

impl Default for KSGConfig {
    fn default() -> Self {
        Self {
            k: 5,
            source_embedding_dim: 2,
            target_embedding_dim: 2,
            source_tau: 1,
            target_tau: 1,
            prediction_horizon: 1,
        }
    }
}

/// GPU-accelerated KSG Transfer Entropy estimator
///
/// # Example
/// ```no_run
/// use prism_ai::orchestration::routing::ksg_transfer_entropy_gpu::{
///     KSGTransferEntropyGpu, KSGConfig
/// };
///
/// let ksg = KSGTransferEntropyGpu::new()?;
/// let config = KSGConfig::default();
///
/// // Source and target time series
/// let source = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let target = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
///
/// let te = ksg.compute_transfer_entropy(&source, &target, &config)?;
/// println!("Transfer Entropy: {}", te);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct KSGTransferEntropyGpu {
    embedder: GpuTimeDelayEmbedding,
    knn: GpuNearestNeighbors,
}

impl KSGTransferEntropyGpu {
    /// Create new KSG Transfer Entropy estimator
    pub fn new() -> Result<Self> {
        Ok(Self {
            embedder: GpuTimeDelayEmbedding::new()?,
            knn: GpuNearestNeighbors::new()?,
        })
    }

    /// Compute transfer entropy from source to target: TE(X→Y)
    ///
    /// # Arguments
    /// * `source` - Source time series (X)
    /// * `target` - Target time series (Y)
    /// * `config` - KSG configuration
    ///
    /// # Returns
    /// Transfer entropy value in nats (natural logarithm units)
    pub fn compute_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        config: &KSGConfig,
    ) -> Result<f64> {
        anyhow::ensure!(
            source.len() == target.len(),
            "Source and target must have same length"
        );

        anyhow::ensure!(
            source.len() >= 20,
            "Time series too short for reliable TE estimation (need at least 20 samples)"
        );

        // Step 1: Create embeddings
        let (joint_space, marginal_spaces) = self.create_embedding_spaces(source, target, config)?;

        // Step 2: For each point, find k-th nearest neighbor distance in joint space
        let n_points = joint_space.nrows();
        let mut te_sum = 0.0;

        for i in 0..n_points {
            let query_joint = joint_space.row(i).to_owned();

            // Find k-th nearest neighbor distance in joint space (using max norm for KSG)
            let epsilon = self.knn.find_kth_distance(
                &joint_space,
                &query_joint,
                config.k + 1, // +1 to exclude the point itself
                DistanceMetric::MaxNorm,
            )?;

            // Count neighbors within epsilon in each marginal space
            let query_x = marginal_spaces.x.row(i).to_owned();
            let query_y = marginal_spaces.y.row(i).to_owned();
            let query_xy = marginal_spaces.xy.row(i).to_owned();

            let n_x = self.knn.count_within_radius(
                &marginal_spaces.x,
                &query_x,
                epsilon,
                DistanceMetric::MaxNorm,
            )? - 1; // -1 to exclude the point itself

            let n_y = self.knn.count_within_radius(
                &marginal_spaces.y,
                &query_y,
                epsilon,
                DistanceMetric::MaxNorm,
            )? - 1;

            let n_xy = self.knn.count_within_radius(
                &marginal_spaces.xy,
                &query_xy,
                epsilon,
                DistanceMetric::MaxNorm,
            )? - 1;

            // KSG formula: TE_i = ψ(k) + ψ(n_x) - ψ(n_xy) - ψ(n_y)
            let te_i = Self::digamma(config.k as f64)
                + Self::digamma(n_x.max(1) as f64)
                - Self::digamma(n_xy.max(1) as f64)
                - Self::digamma(n_y.max(1) as f64);

            te_sum += te_i;
        }

        // Average over all points
        let te = te_sum / (n_points as f64);

        Ok(te.max(0.0)) // TE cannot be negative
    }

    /// Create joint and marginal embedding spaces for KSG algorithm
    ///
    /// Returns:
    /// - Joint space: [Y_future, Y_past, X_past]
    /// - Marginal spaces: X (X_past), Y (Y_past), XY (Y_past, X_past)
    fn create_embedding_spaces(
        &self,
        source: &[f64],
        target: &[f64],
        config: &KSGConfig,
    ) -> Result<(Array2<f64>, MarginalSpaces)> {
        // Embed source (X) and target (Y)
        let x_embedded = self.embedder.embed_gpu(
            source,
            config.source_embedding_dim,
            config.source_tau,
        )?;

        let y_embedded = self.embedder.embed_gpu(
            target,
            config.target_embedding_dim,
            config.target_tau,
        )?;

        // Align embeddings considering prediction horizon
        let n_samples = x_embedded.nrows().min(y_embedded.nrows()) - config.prediction_horizon;

        // Y_future: target at time t+h (where h is prediction_horizon)
        let y_future = y_embedded.slice(ndarray::s![config.prediction_horizon..n_samples + config.prediction_horizon, ..]);

        // Y_past: target history at time t
        let y_past = y_embedded.slice(ndarray::s![0..n_samples, ..]);

        // X_past: source history at time t
        let x_past = x_embedded.slice(ndarray::s![0..n_samples, ..]);

        // Create joint space: [Y_future, Y_past, X_past]
        let joint_space = ndarray::concatenate(
            Axis(1),
            &[
                y_future.view(),
                y_past.view(),
                x_past.view(),
            ],
        )?;

        // Create marginal spaces
        let marginal_x = x_past.to_owned();
        let marginal_y = y_past.to_owned();
        let marginal_xy = ndarray::concatenate(
            Axis(1),
            &[y_past.view(), x_past.view()],
        )?;

        let marginal_spaces = MarginalSpaces {
            x: marginal_x,
            y: marginal_y,
            xy: marginal_xy,
        };

        Ok((joint_space, marginal_spaces))
    }

    /// Digamma function approximation (ψ(x))
    ///
    /// Uses asymptotic expansion for computational efficiency:
    /// ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴)
    ///
    /// Accurate for x > 0.5, which is typical in KSG applications
    fn digamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }

        if x < 0.5 {
            // Use recurrence relation: ψ(x+1) = ψ(x) + 1/x
            return Self::digamma(x + 1.0) - 1.0 / x;
        }

        // Asymptotic expansion for x >= 0.5
        let x_inv = 1.0 / x;
        let x_inv_sq = x_inv * x_inv;

        x.ln() - 0.5 * x_inv - x_inv_sq / 12.0 + x_inv_sq * x_inv_sq / 120.0
    }

    /// Compute transfer entropy with automatic parameter selection
    ///
    /// Automatically selects embedding dimensions and delays using heuristics
    pub fn compute_transfer_entropy_auto(
        &self,
        source: &[f64],
        target: &[f64],
        k: usize,
    ) -> Result<f64> {
        // Auto-select tau for both series
        let source_tau = self.embedder.select_tau_autocorrelation(source, None)?;
        let target_tau = self.embedder.select_tau_autocorrelation(target, None)?;

        // Use default embedding dimension (3 is good for most cases)
        let config = KSGConfig {
            k,
            source_embedding_dim: 3,
            target_embedding_dim: 3,
            source_tau,
            target_tau,
            prediction_horizon: 1,
        };

        self.compute_transfer_entropy(source, target, &config)
    }

    /// Compute bidirectional transfer entropy
    ///
    /// Returns (TE(X→Y), TE(Y→X))
    pub fn compute_bidirectional_te(
        &self,
        series_x: &[f64],
        series_y: &[f64],
        config: &KSGConfig,
    ) -> Result<(f64, f64)> {
        let te_xy = self.compute_transfer_entropy(series_x, series_y, config)?;
        let te_yx = self.compute_transfer_entropy(series_y, series_x, config)?;
        Ok((te_xy, te_yx))
    }

    /// Compute net information flow: TE(X→Y) - TE(Y→X)
    ///
    /// Positive values indicate X drives Y, negative indicates Y drives X
    pub fn compute_net_flow(
        &self,
        series_x: &[f64],
        series_y: &[f64],
        config: &KSGConfig,
    ) -> Result<f64> {
        let (te_xy, te_yx) = self.compute_bidirectional_te(series_x, series_y, config)?;
        Ok(te_xy - te_yx)
    }
}

/// Marginal embedding spaces for KSG algorithm
struct MarginalSpaces {
    /// X (source past)
    x: Array2<f64>,
    /// Y (target past)
    y: Array2<f64>,
    /// XY (target past + source past)
    xy: Array2<f64>,
}

impl Default for KSGTransferEntropyGpu {
    fn default() -> Self {
        Self::new().expect("Failed to create KSGTransferEntropyGpu")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma_function() {
        let ksg = KSGTransferEntropyGpu::new().unwrap();

        // Test known values (approximate)
        // ψ(1) ≈ -0.5772 (Euler-Mascheroni constant)
        let psi_1 = KSGTransferEntropyGpu::digamma(1.0);
        assert!((psi_1 + 0.5772).abs() < 0.01);

        // ψ(2) ≈ 0.4228
        let psi_2 = KSGTransferEntropyGpu::digamma(2.0);
        assert!((psi_2 - 0.4228).abs() < 0.01);

        // ψ(5) ≈ 1.506
        let psi_5 = KSGTransferEntropyGpu::digamma(5.0);
        assert!((psi_5 - 1.506).abs() < 0.01);
    }

    #[test]
    fn test_ksg_te_simple_coupling() -> Result<()> {
        let ksg = KSGTransferEntropyGpu::new()?;

        // Create simple coupled system: Y(t+1) = X(t) + noise
        let n = 100;
        let source: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut target: Vec<f64> = vec![0.0; n];

        for i in 1..n {
            target[i] = source[i - 1] + 0.1 * ((i as f64) * 0.05).cos();
        }

        let config = KSGConfig {
            k: 5,
            source_embedding_dim: 2,
            target_embedding_dim: 2,
            source_tau: 1,
            target_tau: 1,
            prediction_horizon: 1,
        };

        let te = ksg.compute_transfer_entropy(&source, &target, &config)?;

        // Should detect positive transfer entropy (X influences Y)
        assert!(te > 0.0, "TE should be positive for coupled system: {}", te);

        Ok(())
    }

    #[test]
    fn test_ksg_te_independent() -> Result<()> {
        let ksg = KSGTransferEntropyGpu::new()?;

        // Independent random-like series
        let source: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let target: Vec<f64> = (0..100).map(|i| (i as f64 * 0.17).cos()).collect();

        let config = KSGConfig {
            k: 5,
            source_embedding_dim: 2,
            target_embedding_dim: 2,
            source_tau: 1,
            target_tau: 1,
            prediction_horizon: 1,
        };

        let te = ksg.compute_transfer_entropy(&source, &target, &config)?;

        // Should have low transfer entropy for independent systems
        assert!(te < 0.5, "TE should be low for independent systems: {}", te);

        Ok(())
    }

    #[test]
    fn test_ksg_te_bidirectional() -> Result<()> {
        let ksg = KSGTransferEntropyGpu::new()?;

        // Create asymmetric coupling: X → Y (strong), Y → X (weak)
        let n = 100;
        let mut source: Vec<f64> = vec![0.0; n];
        let mut target: Vec<f64> = vec![0.0; n];

        source[0] = 1.0;
        target[0] = 0.5;

        for i in 1..n {
            source[i] = 0.9 * source[i - 1] + 0.1 * (i as f64 * 0.1).sin();
            target[i] = 0.8 * target[i - 1] + 0.5 * source[i - 1] + 0.1 * (i as f64 * 0.15).cos();
        }

        let config = KSGConfig::default();

        let (te_xy, te_yx) = ksg.compute_bidirectional_te(&source, &target, &config)?;

        // X → Y should be stronger than Y → X
        assert!(te_xy > te_yx, "TE(X→Y) = {} should be > TE(Y→X) = {}", te_xy, te_yx);

        Ok(())
    }

    #[test]
    fn test_ksg_te_auto() -> Result<()> {
        let ksg = KSGTransferEntropyGpu::new()?;

        let source: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut target: Vec<f64> = vec![0.0; 100];

        for i in 1..100 {
            target[i] = 0.7 * source[i - 1] + 0.3 * (i as f64 * 0.05).cos();
        }

        let te = ksg.compute_transfer_entropy_auto(&source, &target, 5)?;

        // Should detect coupling
        assert!(te > 0.0, "Auto TE should be positive: {}", te);

        Ok(())
    }

    #[test]
    fn test_net_information_flow() -> Result<()> {
        let ksg = KSGTransferEntropyGpu::new()?;

        // X drives Y
        let n = 100;
        let source: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut target: Vec<f64> = vec![0.0; n];

        for i in 1..n {
            target[i] = 0.8 * source[i - 1] + 0.2 * (i as f64 * 0.05).cos();
        }

        let config = KSGConfig::default();
        let net_flow = ksg.compute_net_flow(&source, &target, &config)?;

        // Net flow should be positive (X → Y dominant)
        assert!(net_flow > 0.0, "Net flow should be positive: {}", net_flow);

        Ok(())
    }

    #[test]
    fn test_insufficient_data() {
        let ksg = KSGTransferEntropyGpu::new().unwrap();

        let source = vec![1.0, 2.0, 3.0]; // Too short
        let target = vec![2.0, 3.0, 4.0];

        let config = KSGConfig::default();
        let result = ksg.compute_transfer_entropy(&source, &target, &config);

        assert!(result.is_err());
    }
}
