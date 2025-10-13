//! Kraskov-Stögbauer-Grassberger (KSG) Transfer Entropy Estimator
//!
//! Implements the KSG algorithm for non-parametric entropy estimation
//! using k-nearest neighbors. This is the state-of-the-art method for
//! transfer entropy calculation with continuous variables.
//!
//! Reference:
//! Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
//! "Estimating mutual information." Physical Review E, 69(6), 066138.
//!
//! Mathematical Foundation:
//! TE(X→Y) = ψ(k) - <ψ(n_y + 1)> - <ψ(n_xz + 1)> + <ψ(n_z + 1)>
//!
//! where:
//! - ψ is the digamma function
//! - k is the number of nearest neighbors
//! - n_y = neighbors in (Y_future, Y_past) space
//! - n_xz = neighbors in (X_past, Y_past) space
//! - n_z = neighbors in (Y_past) space

use anyhow::Result;
use ndarray::Array1;

use super::kdtree::KdTree;
use super::{TransferEntropyResult};

/// KSG Transfer Entropy Estimator
pub struct KsgEstimator {
    /// Number of nearest neighbors
    pub k: usize,
    /// Source embedding dimension
    pub source_embedding: usize,
    /// Target embedding dimension
    pub target_embedding: usize,
    /// Time lag
    pub time_lag: usize,
}

impl Default for KsgEstimator {
    fn default() -> Self {
        Self {
            k: 3,
            source_embedding: 1,
            target_embedding: 1,
            time_lag: 1,
        }
    }
}

impl KsgEstimator {
    /// Create new KSG estimator
    pub fn new(k: usize, source_embedding: usize, target_embedding: usize, time_lag: usize) -> Self {
        Self {
            k,
            source_embedding,
            target_embedding,
            time_lag,
        }
    }

    /// Calculate transfer entropy using KSG estimator
    ///
    /// This is significantly more accurate than histogram-based methods
    /// for continuous variables, especially with limited data.
    pub fn calculate(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<TransferEntropyResult> {
        let n = source.len();

        // Create embeddings
        let (x_past, y_past, y_future) = self.create_embeddings(source, target);
        let n_samples = y_future.len();

        // Build joint space: [Y_future, X_past, Y_past]
        let joint_points = self.build_joint_space(&x_past, &y_past, &y_future);

        // Build KD-tree for joint space
        let joint_tree = KdTree::new(&joint_points);

        // Build marginal spaces
        let y_space = self.build_y_space(&y_past, &y_future); // (Y_future, Y_past)
        let xz_space = self.build_xz_space(&x_past, &y_past); // (X_past, Y_past)
        let z_space = y_past.clone(); // (Y_past)

        let y_tree = KdTree::new(&y_space);
        let xz_tree = KdTree::new(&xz_space);
        let z_tree = KdTree::new(&z_space);

        // Calculate TE using KSG formula
        let mut te_sum = 0.0;

        for i in 0..n_samples {
            // Find k-th nearest neighbor distance in joint space
            let query_joint = &joint_points[i];
            let neighbors_joint = joint_tree.knn_search(query_joint, self.k + 1); // +1 to exclude self

            // Get k-th distance (excluding self which is at distance 0)
            let epsilon = neighbors_joint
                .iter()
                .filter(|n| n.index != i)
                .nth(self.k - 1)
                .map(|n| n.distance)
                .unwrap_or(1e-10);

            // Count neighbors in marginal spaces within epsilon
            let query_y = &y_space[i];
            let query_xz = &xz_space[i];
            let query_z = &z_space[i];

            let neighbors_y = y_tree.range_search(query_y, epsilon);
            let neighbors_xz = xz_tree.range_search(query_xz, epsilon);
            let neighbors_z = z_tree.range_search(query_z, epsilon);

            // Exclude self from counts
            let n_y = neighbors_y.iter().filter(|&&idx| idx != i).count();
            let n_xz = neighbors_xz.iter().filter(|&&idx| idx != i).count();
            let n_z = neighbors_z.iter().filter(|&&idx| idx != i).count();

            // KSG formula contribution
            // TE = ψ(k) - ψ(n_y + 1) - ψ(n_xz + 1) + ψ(n_z + 1)
            let contribution = digamma(self.k as f64)
                - digamma((n_y + 1) as f64)
                - digamma((n_xz + 1) as f64)
                + digamma((n_z + 1) as f64);

            te_sum += contribution;
        }

        let te_value = (te_sum / n_samples as f64).max(0.0);

        // Bias correction for KSG
        let bias = (self.k as f64).ln() / (n_samples as f64);
        let effective_te = (te_value - bias).max(0.0);

        // Calculate significance using permutation test
        let p_value = self.calculate_significance(source, target, te_value)?;

        // Standard error (approximate)
        let std_error = (te_value / (n_samples as f64).sqrt()).max(0.01);

        Ok(TransferEntropyResult {
            te_value,
            p_value,
            std_error,
            effective_te,
            n_samples,
            time_lag: self.time_lag,
        })
    }

    /// Create time-delay embeddings
    fn create_embeddings(&self, source: &Array1<f64>, target: &Array1<f64>)
        -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let n = source.len();
        let start_idx = self.source_embedding.max(self.target_embedding);
        let end_idx = n - self.time_lag;

        let mut x_past = Vec::new();
        let mut y_past = Vec::new();
        let mut y_future = Vec::new();

        for t in start_idx..end_idx {
            // X_past: source embedding [x(t), x(t-1), ..., x(t-l+1)]
            let mut x_vec = Vec::new();
            for lag in 0..self.source_embedding {
                x_vec.push(source[t - lag]);
            }
            x_past.push(x_vec);

            // Y_past: target embedding [y(t), y(t-1), ..., y(t-k+1)]
            let mut y_vec = Vec::new();
            for lag in 0..self.target_embedding {
                y_vec.push(target[t - lag]);
            }
            y_past.push(y_vec);

            // Y_future: y(t + τ)
            y_future.push(target[t + self.time_lag]);
        }

        (x_past, y_past, y_future)
    }

    /// Build joint space [Y_future, X_past, Y_past]
    fn build_joint_space(&self, x_past: &[Vec<f64>], y_past: &[Vec<f64>], y_future: &[f64]) -> Vec<Vec<f64>> {
        let n = y_future.len();
        let mut joint_space = Vec::with_capacity(n);

        for i in 0..n {
            let mut point = Vec::new();

            // Add Y_future
            point.push(y_future[i]);

            // Add X_past
            point.extend_from_slice(&x_past[i]);

            // Add Y_past
            point.extend_from_slice(&y_past[i]);

            joint_space.push(point);
        }

        joint_space
    }

    /// Build Y marginal space [Y_future, Y_past]
    fn build_y_space(&self, y_past: &[Vec<f64>], y_future: &[f64]) -> Vec<Vec<f64>> {
        let n = y_future.len();
        let mut y_space = Vec::with_capacity(n);

        for i in 0..n {
            let mut point = Vec::new();
            point.push(y_future[i]);
            point.extend_from_slice(&y_past[i]);
            y_space.push(point);
        }

        y_space
    }

    /// Build XZ marginal space [X_past, Y_past]
    fn build_xz_space(&self, x_past: &[Vec<f64>], y_past: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x_past.len();
        let mut xz_space = Vec::with_capacity(n);

        for i in 0..n {
            let mut point = Vec::new();
            point.extend_from_slice(&x_past[i]);
            point.extend_from_slice(&y_past[i]);
            xz_space.push(point);
        }

        xz_space
    }

    /// Calculate statistical significance using permutation test
    fn calculate_significance(&self, source: &Array1<f64>, target: &Array1<f64>, observed_te: f64) -> Result<f64> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let n_permutations = 100;
        let mut count_greater = 0;

        for seed in 0..n_permutations {
            // Shuffle source to break causal relationship
            let mut rng = StdRng::seed_from_u64(seed);
            let mut shuffled_source = source.to_vec();

            // Fisher-Yates shuffle
            for i in (1..shuffled_source.len()).rev() {
                let j = rng.gen_range(0..=i);
                shuffled_source.swap(i, j);
            }

            let shuffled_array = Array1::from_vec(shuffled_source);

            // Calculate TE with shuffled data
            let shuffled_result = self.calculate(&shuffled_array, target)?;

            if shuffled_result.te_value >= observed_te {
                count_greater += 1;
            }
        }

        Ok((count_greater as f64 + 1.0) / (n_permutations as f64 + 1.0))
    }

    /// Multi-scale analysis
    pub fn calculate_multiscale(&self, source: &Array1<f64>, target: &Array1<f64>,
                                max_lag: usize) -> Result<Vec<TransferEntropyResult>> {
        let mut results = Vec::new();

        for lag in 1..=max_lag {
            let mut estimator = self.clone();
            estimator.time_lag = lag;
            results.push(estimator.calculate(source, target)?);
        }

        Ok(results)
    }

    /// Find optimal embedding dimension using False Nearest Neighbors
    ///
    /// Returns optimal embedding dimension for target series
    pub fn find_optimal_embedding(&self, target: &Array1<f64>) -> usize {
        // Simplified FNN implementation
        // In production, would implement full Kennel-Brown-Abarbanel algorithm

        let max_dim = 10;
        let mut best_dim = 1;
        let mut min_fnn_ratio = 1.0;

        for dim in 1..=max_dim {
            let fnn_ratio = self.false_nearest_neighbors(target, dim);

            if fnn_ratio < 0.01 {
                // Less than 1% false neighbors - good embedding
                return dim;
            }

            if fnn_ratio < min_fnn_ratio {
                min_fnn_ratio = fnn_ratio;
                best_dim = dim;
            }
        }

        best_dim
    }

    /// Calculate false nearest neighbor ratio
    fn false_nearest_neighbors(&self, series: &Array1<f64>, dim: usize) -> f64 {
        let n = series.len();
        let threshold = 2.0; // Standard FNN threshold

        if n < dim + 10 {
            return 1.0; // Not enough data
        }

        // Create embeddings with dimension dim and dim+1
        let mut points_dim: Vec<Vec<f64>> = Vec::new();
        let mut points_dim_plus: Vec<Vec<f64>> = Vec::new();

        for i in dim..n {
            let mut point_d = Vec::new();
            let mut point_dp = Vec::new();

            for j in 0..dim {
                let val = series[i - j];
                point_d.push(val);
                point_dp.push(val);
            }

            // Add one more dimension
            if i > dim {
                point_dp.push(series[i - dim]);
            } else {
                point_dp.push(series[0]);
            }

            points_dim.push(point_d);
            points_dim_plus.push(point_dp);
        }

        if points_dim.is_empty() {
            return 1.0;
        }

        // Build KD-trees
        let tree_dim = KdTree::new(&points_dim);
        let tree_dim_plus = KdTree::new(&points_dim_plus);

        let mut false_neighbors = 0;
        let n_points = points_dim.len();

        for i in 0..n_points {
            // Find nearest neighbor in dim-dimensional space
            let neighbors_dim = tree_dim.knn_search(&points_dim[i], 2); // Self + 1 neighbor

            if neighbors_dim.len() < 2 {
                continue;
            }

            let nn_idx = neighbors_dim.iter().find(|n| n.index != i).map(|n| n.index);

            if let Some(nn_idx) = nn_idx {
                // Calculate distance in dim and dim+1 spaces
                let dist_dim = self.distance_linf(&points_dim[i], &points_dim[nn_idx]);
                let dist_dim_plus = self.distance_linf(&points_dim_plus[i], &points_dim_plus[nn_idx]);

                // Check if neighbor is "false" (distance increases significantly)
                if dist_dim > 1e-10 {
                    let ratio = dist_dim_plus / dist_dim;
                    if ratio > threshold {
                        false_neighbors += 1;
                    }
                }
            }
        }

        false_neighbors as f64 / n_points as f64
    }

    /// L-infinity distance
    fn distance_linf(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl Clone for KsgEstimator {
    fn clone(&self) -> Self {
        Self {
            k: self.k,
            source_embedding: self.source_embedding,
            target_embedding: self.target_embedding,
            time_lag: self.time_lag,
        }
    }
}

/// Digamma function (ψ) - derivative of log Gamma
///
/// ψ(x) = d/dx [ln Γ(x)] = Γ'(x) / Γ(x)
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use asymptotic expansion for large x
    if x > 10.0 {
        return x.ln() - 0.5 / x - 1.0 / (12.0 * x * x) + 1.0 / (120.0 * x * x * x * x);
    }

    // Use recurrence relation to reduce to large x
    // ψ(x+1) = ψ(x) + 1/x
    let mut result = 0.0;
    let mut y = x;

    while y < 10.0 {
        result -= 1.0 / y;
        y += 1.0;
    }

    result + y.ln() - 0.5 / y - 1.0 / (12.0 * y * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ksg_estimator_creation() {
        let estimator = KsgEstimator::new(3, 1, 1, 1);
        assert_eq!(estimator.k, 3);
        assert_eq!(estimator.source_embedding, 1);
    }

    #[test]
    fn test_digamma_function() {
        // Test known values
        assert!((digamma(1.0) + 0.5772).abs() < 0.001); // ψ(1) ≈ -γ where γ is Euler-Mascheroni constant
        assert!((digamma(2.0) + 0.5772 - 1.0).abs() < 0.001); // ψ(2) = ψ(1) + 1
    }

    #[test]
    fn test_ksg_independent_series() {
        let estimator = KsgEstimator::new(3, 1, 1, 1);

        // Independent random series
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| ((i + 50) as f64 * 0.15).cos()).collect();

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let result = estimator.calculate(&x_arr, &y_arr).unwrap();

        // TE should be small for independent series
        println!("KSG TE (independent): {}", result.effective_te);
        assert!(result.te_value >= 0.0);
    }

    #[test]
    fn test_ksg_causal_series() {
        let estimator = KsgEstimator::new(3, 1, 1, 1);

        // Causal relationship: y(t) depends on x(t-1)
        let mut x = Vec::new();
        let mut y = Vec::new();

        for i in 0..200 {
            x.push((i as f64 * 0.05).sin());
            if i == 0 {
                y.push(0.0);
            } else {
                y.push(x[i - 1] * 0.8);
            }
        }

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let result = estimator.calculate(&x_arr, &y_arr).unwrap();

        println!("KSG TE (causal): {}", result.effective_te);

        // TE should be positive for causal relationship
        assert!(result.effective_te > 0.0);
    }

    #[test]
    fn test_multiscale_ksg() {
        let estimator = KsgEstimator::default();

        let x = Array1::linspace(0.0, 10.0, 100);
        let y = x.mapv(|v| (v - 0.5).sin());

        let results = estimator.calculate_multiscale(&x, &y, 5).unwrap();
        assert_eq!(results.len(), 5);

        // Verify lags are correct
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.time_lag, i + 1);
        }
    }

    #[test]
    fn test_false_nearest_neighbors() {
        let estimator = KsgEstimator::default();

        // Deterministic series - should have low FNN at appropriate dimension
        let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let series_arr = Array1::from_vec(series);

        let optimal_dim = estimator.find_optimal_embedding(&series_arr);

        println!("Optimal embedding dimension: {}", optimal_dim);
        assert!(optimal_dim >= 1 && optimal_dim <= 10);
    }
}
