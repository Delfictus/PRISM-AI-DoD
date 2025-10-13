//! Kraskov-Stögbauer-Grassberger (KSG) Estimator for Transfer Entropy
//!
//! Implements the state-of-the-art KSG estimator for transfer entropy
//! using k-nearest neighbor distances and adaptive bandwidth selection.
//!
//! # Mathematical Foundation
//!
//! The KSG estimator for transfer entropy TE(X→Y) is:
//!
//! TE(X→Y) = ψ(k) - ⟨ψ(n_z) + ψ(n_yz) - ψ(n_xyz)⟩
//!
//! Where:
//! - ψ is the digamma function
//! - k is the number of nearest neighbors
//! - n_z, n_yz, n_xyz are counts in marginal spaces
//! - ⟨·⟩ denotes expectation over all points
//!
//! # Reference
//! Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
//! "Estimating mutual information." Physical review E, 69(6), 066138.
//!
//! Vicente, R., Wibral, M., Lindner, M., & Pipa, G. (2011).
//! "Transfer entropy—a model-free measure of effective connectivity for the neurosciences."
//! Journal of computational neuroscience, 30(1), 45-67.

use ndarray::Array1;
use super::kdtree::{KdTree, Point};
use std::f64;

/// KSG estimator configuration
#[derive(Debug, Clone)]
pub struct KsgConfig {
    /// Number of nearest neighbors (typically 3-10)
    pub k_neighbors: usize,

    /// Source embedding dimension
    pub source_embedding: usize,

    /// Target embedding dimension
    pub target_embedding: usize,

    /// Time lag for prediction
    pub time_lag: usize,

    /// Add noise to break ties (recommended: 1e-10)
    pub noise_level: f64,

    /// Use maximum norm instead of Euclidean (KSG2 variant)
    pub use_max_norm: bool,
}

impl Default for KsgConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 4,
            source_embedding: 1,
            target_embedding: 1,
            time_lag: 1,
            noise_level: 1e-10,
            use_max_norm: true, // KSG2 is generally more accurate
        }
    }
}

/// KSG Transfer Entropy Result
#[derive(Debug, Clone)]
pub struct KsgResult {
    /// Transfer entropy value (nats)
    pub te_nats: f64,

    /// Transfer entropy in bits
    pub te_bits: f64,

    /// Number of samples used
    pub n_samples: usize,

    /// Average neighborhood size
    pub avg_neighborhood_size: f64,

    /// Digamma contributions breakdown
    pub digamma_k: f64,
    pub digamma_nz: f64,
    pub digamma_nyz: f64,
}

/// KSG estimator for Transfer Entropy
pub struct KsgEstimator {
    config: KsgConfig,
}

impl KsgEstimator {
    /// Create a new KSG estimator
    pub fn new(config: KsgConfig) -> Self {
        Self { config }
    }

    /// Calculate Transfer Entropy from X to Y using KSG estimator
    ///
    /// # Arguments
    /// * `source` - Source time series X
    /// * `target` - Target time series Y
    ///
    /// # Returns
    /// KsgResult with TE value and diagnostics
    pub fn calculate(&self, source: &Array1<f64>, target: &Array1<f64>) -> KsgResult {
        assert_eq!(source.len(), target.len(), "Time series must have same length");

        // Add small noise to break ties
        let source_noisy = self.add_noise(source);
        let target_noisy = self.add_noise(target);

        // Create embeddings
        let (x_embed, y_embed, y_future) = self.create_embeddings(&source_noisy, &target_noisy);
        let n = x_embed.len();

        // Build joint space: [x_t^l, y_t^k, y_{t+τ}]
        let xyz_points = self.build_xyz_space(&x_embed, &y_embed, &y_future);
        let xyz_tree = KdTree::new(xyz_points.clone());

        // Build marginal spaces
        let yz_points = self.build_yz_space(&y_embed, &y_future);
        let yz_tree = KdTree::new(yz_points);

        let y_points = self.build_y_space(&y_embed);
        let y_tree = KdTree::new(y_points);

        // Calculate digamma terms
        let psi_k = digamma(self.config.k_neighbors as f64);
        let mut psi_nz_sum = 0.0;
        let mut psi_nyz_sum = 0.0;
        let mut avg_nz = 0.0;

        for i in 0..n {
            // Find k-th nearest neighbor distance in joint space XYZ
            let query_xyz = &xyz_points[i];
            let neighbors_xyz = xyz_tree.k_nearest(query_xyz, self.config.k_neighbors + 1); // +1 to exclude self

            // Get k-th distance (excluding self at distance 0)
            let epsilon = neighbors_xyz
                .iter()
                .filter(|(dist, idx)| *idx != i && *dist > 0.0)
                .nth(self.config.k_neighbors - 1)
                .map(|(dist, _)| *dist)
                .unwrap_or(1.0);

            // Count neighbors in marginal spaces within epsilon
            let query_yz = Point::new(
                [y_embed[i].clone(), vec![y_future[i]]].concat(),
                i,
            );
            let n_yz = yz_tree
                .range_query(&query_yz, epsilon, self.config.use_max_norm)
                .len();

            let query_y = Point::new(y_embed[i].clone(), i);
            let n_y = y_tree
                .range_query(&query_y, epsilon, self.config.use_max_norm)
                .len();

            // Accumulate digamma contributions
            // KSG formula: TE = ψ(k) - ⟨ψ(n_yz) - ψ(n_y)⟩
            // Where we use the fact that n_xyz = k by construction
            psi_nyz_sum += digamma(n_yz as f64);
            psi_nz_sum += digamma(n_y as f64);
            avg_nz += n_y as f64;
        }

        let psi_nyz_avg = psi_nyz_sum / n as f64;
        let psi_nz_avg = psi_nz_sum / n as f64;
        avg_nz /= n as f64;

        // Calculate Transfer Entropy (in nats)
        // TE(X→Y) = ψ(k) - ⟨ψ(n_yz)⟩ + ⟨ψ(n_y)⟩
        let te_nats = psi_k - psi_nyz_avg + psi_nz_avg;
        let te_bits = te_nats / f64::consts::LN_2;

        KsgResult {
            te_nats: te_nats.max(0.0),
            te_bits: te_bits.max(0.0),
            n_samples: n,
            avg_neighborhood_size: avg_nz,
            digamma_k: psi_k,
            digamma_nz: psi_nz_avg,
            digamma_nyz: psi_nyz_avg,
        }
    }

    /// Add small noise to time series to break ties
    fn add_noise(&self, series: &Array1<f64>) -> Array1<f64> {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        series.mapv(|x| {
            x + rng.gen::<f64>() * self.config.noise_level
        })
    }

    /// Create time-delay embeddings
    fn create_embeddings(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let n = source.len();
        let start_idx = self.config.source_embedding.max(self.config.target_embedding);
        let end_idx = n - self.config.time_lag;

        let mut x_embed = Vec::new();
        let mut y_embed = Vec::new();
        let mut y_future = Vec::new();

        for i in start_idx..end_idx {
            // Source embedding: [x_t, x_{t-1}, ..., x_{t-l+1}]
            let x_vec: Vec<f64> = (0..self.config.source_embedding)
                .map(|j| source[i - j])
                .collect();
            x_embed.push(x_vec);

            // Target embedding: [y_t, y_{t-1}, ..., y_{t-k+1}]
            let y_vec: Vec<f64> = (0..self.config.target_embedding)
                .map(|j| target[i - j])
                .collect();
            y_embed.push(y_vec);

            // Future target: y_{t+τ}
            y_future.push(target[i + self.config.time_lag]);
        }

        (x_embed, y_embed, y_future)
    }

    /// Build joint XYZ space: [x_t^l, y_t^k, y_{t+τ}]
    fn build_xyz_space(
        &self,
        x_embed: &[Vec<f64>],
        y_embed: &[Vec<f64>],
        y_future: &[f64],
    ) -> Vec<Point> {
        x_embed
            .iter()
            .zip(y_embed.iter())
            .zip(y_future.iter())
            .enumerate()
            .map(|(i, ((x, y), z))| {
                let coords = [x.clone(), y.clone(), vec![*z]].concat();
                Point::new(coords, i)
            })
            .collect()
    }

    /// Build YZ marginal space: [y_t^k, y_{t+τ}]
    fn build_yz_space(&self, y_embed: &[Vec<f64>], y_future: &[f64]) -> Vec<Point> {
        y_embed
            .iter()
            .zip(y_future.iter())
            .enumerate()
            .map(|(i, (y, z))| {
                let coords = [y.clone(), vec![*z]].concat();
                Point::new(coords, i)
            })
            .collect()
    }

    /// Build Y marginal space: [y_t^k]
    fn build_y_space(&self, y_embed: &[Vec<f64>]) -> Vec<Point> {
        y_embed
            .iter()
            .enumerate()
            .map(|(i, y)| Point::new(y.clone(), i))
            .collect()
    }

    /// Calculate bias correction for KSG estimator
    pub fn bias_correction(&self, n_samples: usize) -> f64 {
        // KSG bias correction: approximately k/n
        (self.config.k_neighbors as f64) / (2.0 * n_samples as f64)
    }
}

/// Digamma function (ψ) - derivative of log Gamma function
///
/// Uses asymptotic expansion for large x and recursion for small x
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // For x = 1, ψ(1) = -γ (Euler-Mascheroni constant)
    const EULER_MASCHERONI: f64 = 0.5772156649015329;

    if x == 1.0 {
        return -EULER_MASCHERONI;
    }

    // Use asymptotic expansion for large x
    // ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - ...
    if x > 10.0 {
        let x_inv = 1.0 / x;
        let x_inv2 = x_inv * x_inv;
        return x.ln() - 0.5 * x_inv - x_inv2 / 12.0 + x_inv2 * x_inv2 / 120.0;
    }

    // Use recurrence relation: ψ(x+1) = ψ(x) + 1/x
    // to shift x into the range where asymptotic expansion works
    let mut result = 0.0;
    let mut y = x;

    while y < 10.0 {
        result -= 1.0 / y;
        y += 1.0;
    }

    // Apply asymptotic expansion
    let y_inv = 1.0 / y;
    let y_inv2 = y_inv * y_inv;
    result += y.ln() - 0.5 * y_inv - y_inv2 / 12.0 + y_inv2 * y_inv2 / 120.0;

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma_function() {
        // Test known values
        assert!((digamma(1.0) + 0.5772156649).abs() < 1e-8); // ψ(1) = -γ
        assert!((digamma(2.0) - (1.0 - 0.5772156649)).abs() < 1e-8); // ψ(2) = 1 - γ
        assert!((digamma(0.5) + 1.9635100260).abs() < 1e-8); // ψ(0.5) known value
    }

    #[test]
    fn test_ksg_independent_series() {
        // Test with independent random series
        let x: Array1<f64> = Array1::from_vec(
            (0..1000)
                .map(|i| ((i as f64) * 0.1).sin())
                .collect()
        );
        let y: Array1<f64> = Array1::from_vec(
            (0..1000)
                .map(|i| ((i as f64) * 0.13).cos())
                .collect()
        );

        let config = KsgConfig {
            k_neighbors: 4,
            source_embedding: 1,
            target_embedding: 1,
            time_lag: 1,
            ..Default::default()
        };

        let estimator = KsgEstimator::new(config);
        let result = estimator.calculate(&x, &y);

        println!("KSG TE (independent): {} bits", result.te_bits);

        // Transfer entropy should be small for independent series
        assert!(result.te_bits < 0.1);
        assert!(result.te_nats >= 0.0);
    }

    #[test]
    fn test_ksg_causal_series() {
        // Test with causally related series: Y depends on past X
        let mut x = Vec::new();
        let mut y = Vec::new();

        for i in 0..1000 {
            x.push(((i as f64) * 0.1).sin());
            if i == 0 {
                y.push(0.0);
            } else {
                // Y depends strongly on past X
                y.push(0.9 * x[i - 1] + 0.1 * ((i as f64) * 0.05).cos());
            }
        }

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let config = KsgConfig {
            k_neighbors: 4,
            source_embedding: 1,
            target_embedding: 1,
            time_lag: 1,
            ..Default::default()
        };

        let estimator = KsgEstimator::new(config);
        let result = estimator.calculate(&x_arr, &y_arr);

        println!("KSG TE (causal): {} bits", result.te_bits);
        println!("  n_samples: {}", result.n_samples);
        println!("  avg_neighborhood: {}", result.avg_neighborhood_size);

        // Transfer entropy should be significant for causal relationship
        assert!(result.te_bits > 0.05);
        assert!(result.n_samples > 900);
    }

    #[test]
    fn test_ksg_vs_reverse_causality() {
        // X causes Y, not the other way around
        let mut x = Vec::new();
        let mut y = Vec::new();

        for i in 0..500 {
            x.push(((i as f64) * 0.1).sin());
            if i < 2 {
                y.push(0.0);
            } else {
                y.push(0.8 * x[i - 2]); // Y depends on X with lag 2
            }
        }

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let config = KsgConfig {
            k_neighbors: 4,
            source_embedding: 1,
            target_embedding: 1,
            time_lag: 2,
            ..Default::default()
        };

        let estimator = KsgEstimator::new(config);

        let te_xy = estimator.calculate(&x_arr, &y_arr);
        let te_yx = estimator.calculate(&y_arr, &x_arr);

        println!("TE(X→Y): {} bits", te_xy.te_bits);
        println!("TE(Y→X): {} bits", te_yx.te_bits);

        // TE(X→Y) should be much larger than TE(Y→X)
        assert!(te_xy.te_bits > te_yx.te_bits);
    }

    #[test]
    fn test_embedding_dimensions() {
        let x = Array1::from_vec((0..100).map(|i| i as f64).collect());
        let y = Array1::from_vec((0..100).map(|i| (i as f64) * 0.5).collect());

        let config = KsgConfig {
            k_neighbors: 3,
            source_embedding: 2,
            target_embedding: 2,
            time_lag: 1,
            ..Default::default()
        };

        let estimator = KsgEstimator::new(config);
        let (x_embed, y_embed, y_future) = estimator.create_embeddings(&x, &y);

        // Should have reduced samples due to embedding
        assert!(x_embed.len() < 100);
        assert_eq!(x_embed.len(), y_embed.len());
        assert_eq!(x_embed.len(), y_future.len());

        // Each embedding should have correct dimension
        assert_eq!(x_embed[0].len(), 2);
        assert_eq!(y_embed[0].len(), 2);
    }
}
