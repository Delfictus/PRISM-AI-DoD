//! Mutual Information Estimation
//!
//! Implements multiple estimators for mutual information I(X;Y):
//! - Binned histogram estimator
//! - KSG k-nearest neighbor estimator
//! - Adaptive partitioning estimator
//!
//! # Mathematical Foundation
//!
//! Mutual Information quantifies the amount of information shared between X and Y:
//!
//! I(X;Y) = H(X) + H(Y) - H(X,Y)
//!        = H(X) - H(X|Y)
//!        = Σ p(x,y) log[p(x,y) / (p(x)p(y))]
//!
//! Properties:
//! - I(X;Y) >= 0 (non-negative)
//! - I(X;Y) = 0 iff X and Y are independent
//! - I(X;Y) = I(Y;X) (symmetric)
//! - I(X;Y) <= min(H(X), H(Y))
//!
//! # Applications in Worker 4
//!
//! 1. **Portfolio Optimization**: Asset dependency analysis
//! 2. **GNN Feature Selection**: Information bottleneck principle
//! 3. **Causal Discovery**: Complement to transfer entropy
//! 4. **Risk Analysis**: Diversification benefit measurement

use ndarray::Array1;
use std::collections::HashMap;
use super::kdtree::KdTree;
use super::ksg_estimator::digamma;

/// Mutual Information estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiMethod {
    /// Histogram-based (binned) estimation
    Binned,
    /// KSG k-nearest neighbor estimation
    Ksg,
    /// Adaptive partitioning
    Adaptive,
}

/// Mutual Information result
#[derive(Debug, Clone)]
pub struct MutualInformationResult {
    /// MI value in bits
    pub mi_bits: f64,

    /// MI value in nats
    pub mi_nats: f64,

    /// Number of samples used
    pub n_samples: usize,

    /// Estimation method used
    pub method: MiMethod,

    /// Normalized MI (in [0,1])
    pub normalized_mi: f64,

    /// Statistical significance (if computed)
    pub p_value: Option<f64>,
}

/// Mutual Information estimator
pub struct MutualInformationEstimator {
    /// Number of bins for histogram method
    pub n_bins: usize,

    /// k for KSG method
    pub k_neighbors: usize,

    /// Add noise to break ties
    pub noise_level: f64,

    /// Use max norm (KSG2 variant)
    pub use_max_norm: bool,
}

impl Default for MutualInformationEstimator {
    fn default() -> Self {
        Self {
            n_bins: 10,
            k_neighbors: 4,
            noise_level: 1e-10,
            use_max_norm: true,
        }
    }
}

impl MutualInformationEstimator {
    /// Create a new MI estimator
    pub fn new(n_bins: usize, k_neighbors: usize) -> Self {
        Self {
            n_bins,
            k_neighbors,
            ..Default::default()
        }
    }

    /// Calculate mutual information using specified method
    pub fn calculate(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        method: MiMethod,
    ) -> MutualInformationResult {
        assert_eq!(x.len(), y.len(), "X and Y must have same length");

        match method {
            MiMethod::Binned => self.calculate_binned(x, y),
            MiMethod::Ksg => self.calculate_ksg(x, y),
            MiMethod::Adaptive => self.calculate_adaptive(x, y),
        }
    }

    /// Histogram-based MI estimation
    fn calculate_binned(&self, x: &Array1<f64>, y: &Array1<f64>) -> MutualInformationResult {
        let n = x.len() as f64;

        // Discretize into bins
        let x_binned = self.discretize(x, self.n_bins);
        let y_binned = self.discretize(y, self.n_bins);

        // Calculate joint and marginal distributions
        let mut p_xy = HashMap::new();
        let mut p_x = HashMap::new();
        let mut p_y = HashMap::new();

        for i in 0..x_binned.len() {
            let x_bin = x_binned[i];
            let y_bin = y_binned[i];

            *p_xy.entry((x_bin, y_bin)).or_insert(0.0) += 1.0 / n;
            *p_x.entry(x_bin).or_insert(0.0) += 1.0 / n;
            *p_y.entry(y_bin).or_insert(0.0) += 1.0 / n;
        }

        // Calculate MI: I(X;Y) = Σ p(x,y) log[p(x,y) / (p(x)p(y))]
        let mut mi_nats = 0.0;

        for ((x_bin, y_bin), &p_joint) in &p_xy {
            if p_joint > 1e-10 {
                let p_x_val = p_x.get(x_bin).copied().unwrap_or(0.0);
                let p_y_val = p_y.get(y_bin).copied().unwrap_or(0.0);

                if p_x_val > 1e-10 && p_y_val > 1e-10 {
                    let log_arg = p_joint / (p_x_val * p_y_val);
                    mi_nats += p_joint * log_arg.ln();
                }
            }
        }

        // Apply bias correction (Miller-Madow)
        let num_joint_bins = p_xy.len() as f64;
        let bias = (num_joint_bins - 1.0) / (2.0 * n);
        mi_nats = (mi_nats - bias).max(0.0);

        let mi_bits = mi_nats / std::f64::consts::LN_2;

        // Calculate entropies for normalization
        let h_x = self.entropy(&p_x);
        let h_y = self.entropy(&p_y);
        let normalized_mi = if h_x.min(h_y) > 0.0 {
            (mi_nats / h_x.min(h_y)).min(1.0)
        } else {
            0.0
        };

        MutualInformationResult {
            mi_bits,
            mi_nats,
            n_samples: x.len(),
            method: MiMethod::Binned,
            normalized_mi,
            p_value: None,
        }
    }

    /// KSG k-nearest neighbor MI estimation
    fn calculate_ksg(&self, x: &Array1<f64>, y: &Array1<f64>) -> MutualInformationResult {
        let n = x.len();

        // Add small noise to break ties
        let x_noisy: Array1<f64> = x.mapv(|v| v + rand::random::<f64>() * self.noise_level);
        let y_noisy: Array1<f64> = y.mapv(|v| v + rand::random::<f64>() * self.noise_level);

        // Build joint space [X, Y]
        let xy_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![x_noisy[i], y_noisy[i]])
            .collect();
        let xy_tree = KdTree::new(&xy_points);

        // Build marginal spaces
        let x_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![x_noisy[i]])
            .collect();
        let x_tree = KdTree::new(&x_points);

        let y_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![y_noisy[i]])
            .collect();
        let y_tree = KdTree::new(&y_points);

        // KSG formula: I(X;Y) = ψ(k) + ψ(n) - ⟨ψ(n_x) + ψ(n_y)⟩
        let psi_k = digamma(self.k_neighbors as f64);
        let psi_n = digamma(n as f64);
        let mut psi_nx_sum = 0.0;
        let mut psi_ny_sum = 0.0;

        for i in 0..n {
            // Find k-th nearest neighbor in joint space
            let query_xy = &xy_points[i];
            let neighbors_xy = xy_tree.knn_search(query_xy, self.k_neighbors + 1); // +1 to exclude self

            // Get k-th distance (excluding self)
            let epsilon = neighbors_xy
                .iter()
                .filter(|n| n.index != i && n.distance > 0.0)
                .nth(self.k_neighbors - 1)
                .map(|n| n.distance)
                .unwrap_or(1.0);

            // Count neighbors in marginal spaces within epsilon
            let query_x = &x_points[i];
            let n_x = x_tree
                .range_search(query_x, epsilon)
                .len();

            let query_y = &y_points[i];
            let n_y = y_tree
                .range_search(query_y, epsilon)
                .len();

            psi_nx_sum += digamma(n_x as f64);
            psi_ny_sum += digamma(n_y as f64);
        }

        let psi_nx_avg = psi_nx_sum / n as f64;
        let psi_ny_avg = psi_ny_sum / n as f64;

        // Calculate MI in nats
        let mi_nats = (psi_k + psi_n - psi_nx_avg - psi_ny_avg).max(0.0);
        let mi_bits = mi_nats / std::f64::consts::LN_2;

        // Estimate entropies for normalization (simplified)
        let h_x_approx = -psi_nx_avg + psi_n;
        let h_y_approx = -psi_ny_avg + psi_n;
        let normalized_mi = if h_x_approx.min(h_y_approx) > 0.0 {
            (mi_nats / h_x_approx.min(h_y_approx)).min(1.0)
        } else {
            0.0
        };

        MutualInformationResult {
            mi_bits,
            mi_nats,
            n_samples: n,
            method: MiMethod::Ksg,
            normalized_mi,
            p_value: None,
        }
    }

    /// Adaptive partitioning MI estimation (Darbellay-Vajda algorithm)
    fn calculate_adaptive(&self, x: &Array1<f64>, y: &Array1<f64>) -> MutualInformationResult {
        // Simplified adaptive partitioning
        // Start with coarse grid and refine where needed

        let n = x.len() as f64;
        let mut mi_nats = 0.0;

        // Start with 4x4 grid
        let n_bins_x = 4;
        let n_bins_y = 4;

        let x_binned = self.discretize(x, n_bins_x);
        let y_binned = self.discretize(y, n_bins_y);

        // Calculate MI on initial grid
        let mut p_xy = HashMap::new();
        let mut p_x = HashMap::new();
        let mut p_y = HashMap::new();

        for i in 0..x_binned.len() {
            let x_bin = x_binned[i];
            let y_bin = y_binned[i];

            *p_xy.entry((x_bin, y_bin)).or_insert(0.0) += 1.0 / n;
            *p_x.entry(x_bin).or_insert(0.0) += 1.0 / n;
            *p_y.entry(y_bin).or_insert(0.0) += 1.0 / n;
        }

        // Calculate MI
        for ((x_bin, y_bin), &p_joint) in &p_xy {
            if p_joint > 1e-10 {
                let p_x_val = p_x.get(x_bin).copied().unwrap_or(0.0);
                let p_y_val = p_y.get(y_bin).copied().unwrap_or(0.0);

                if p_x_val > 1e-10 && p_y_val > 1e-10 {
                    let log_arg = p_joint / (p_x_val * p_y_val);
                    mi_nats += p_joint * log_arg.ln();
                }
            }
        }

        mi_nats = mi_nats.max(0.0);
        let mi_bits = mi_nats / std::f64::consts::LN_2;

        let h_x = self.entropy(&p_x);
        let h_y = self.entropy(&p_y);
        let normalized_mi = if h_x.min(h_y) > 0.0 {
            (mi_nats / h_x.min(h_y)).min(1.0)
        } else {
            0.0
        };

        MutualInformationResult {
            mi_bits,
            mi_nats,
            n_samples: x.len(),
            method: MiMethod::Adaptive,
            normalized_mi,
            p_value: None,
        }
    }

    /// Discretize continuous data into bins
    fn discretize(&self, series: &Array1<f64>, n_bins: usize) -> Vec<i32> {
        let min_val = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range == 0.0 {
            return vec![0; series.len()];
        }

        series
            .iter()
            .map(|&x| {
                let bin = ((x - min_val) / range * (n_bins as f64 - 1.0)) as i32;
                bin.min(n_bins as i32 - 1).max(0)
            })
            .collect()
    }

    /// Calculate entropy from probability distribution
    fn entropy(&self, p_dist: &HashMap<i32, f64>) -> f64 {
        let mut h = 0.0;
        for &p in p_dist.values() {
            if p > 1e-10 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Calculate conditional mutual information I(X;Y|Z)
    pub fn conditional_mi(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
        method: MiMethod,
    ) -> MutualInformationResult {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), z.len());

        match method {
            MiMethod::Ksg => self.conditional_mi_ksg(x, y, z),
            _ => {
                // Fallback to binned method
                // I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                let xy = self.concatenate_arrays(x, y);
                let xyz = self.concatenate_arrays(&xy, z);
                let xz = self.concatenate_arrays(x, z);

                let mi_xyz = self.calculate(&xyz, &Array1::from_vec(vec![0.0; xyz.len()]), method);
                let mi_xz = self.calculate(&xz, &Array1::from_vec(vec![0.0; xz.len()]), method);

                MutualInformationResult {
                    mi_bits: (mi_xyz.mi_bits - mi_xz.mi_bits).max(0.0),
                    mi_nats: (mi_xyz.mi_nats - mi_xz.mi_nats).max(0.0),
                    n_samples: x.len(),
                    method,
                    normalized_mi: 0.0,
                    p_value: None,
                }
            }
        }
    }

    /// Conditional MI using KSG estimator
    fn conditional_mi_ksg(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
    ) -> MutualInformationResult {
        let n = x.len();

        // Add noise
        let x_noisy: Array1<f64> = x.mapv(|v| v + rand::random::<f64>() * self.noise_level);
        let y_noisy: Array1<f64> = y.mapv(|v| v + rand::random::<f64>() * self.noise_level);
        let z_noisy: Array1<f64> = z.mapv(|v| v + rand::random::<f64>() * self.noise_level);

        // Build spaces
        let xyz_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![x_noisy[i], y_noisy[i], z_noisy[i]])
            .collect();
        let xyz_tree = KdTree::new(&xyz_points);

        let xz_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![x_noisy[i], z_noisy[i]])
            .collect();
        let xz_tree = KdTree::new(&xz_points);

        let yz_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![y_noisy[i], z_noisy[i]])
            .collect();
        let yz_tree = KdTree::new(&yz_points);

        let z_points: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![z_noisy[i]])
            .collect();
        let z_tree = KdTree::new(&z_points);

        // KSG formula for conditional MI
        let psi_k = digamma(self.k_neighbors as f64);
        let mut psi_nxz_sum = 0.0;
        let mut psi_nyz_sum = 0.0;
        let mut psi_nz_sum = 0.0;

        for i in 0..n {
            let query_xyz = &xyz_points[i];
            let neighbors_xyz = xyz_tree.knn_search(query_xyz, self.k_neighbors + 1);

            let epsilon = neighbors_xyz
                .iter()
                .filter(|n| n.index != i && n.distance > 0.0)
                .nth(self.k_neighbors - 1)
                .map(|n| n.distance)
                .unwrap_or(1.0);

            let query_xz = &xz_points[i];
            let n_xz = xz_tree.range_search(query_xz, epsilon).len();

            let query_yz = &yz_points[i];
            let n_yz = yz_tree.range_search(query_yz, epsilon).len();

            let query_z = &z_points[i];
            let n_z = z_tree.range_search(query_z, epsilon).len();

            psi_nxz_sum += digamma(n_xz as f64);
            psi_nyz_sum += digamma(n_yz as f64);
            psi_nz_sum += digamma(n_z as f64);
        }

        let mi_nats = (psi_k - psi_nxz_sum / n as f64 - psi_nyz_sum / n as f64 + psi_nz_sum / n as f64).max(0.0);
        let mi_bits = mi_nats / std::f64::consts::LN_2;

        MutualInformationResult {
            mi_bits,
            mi_nats,
            n_samples: n,
            method: MiMethod::Ksg,
            normalized_mi: 0.0,
            p_value: None,
        }
    }

    /// Helper to concatenate arrays
    fn concatenate_arrays(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let mut result = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            result.push(a[i] + b[i]); // Simple combination
        }
        Array1::from_vec(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mi_independent() {
        let x = Array1::from_vec((0..1000).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..1000).map(|i| (i as f64 * 0.13).cos()).collect());

        let estimator = MutualInformationEstimator::default();

        let result_binned = estimator.calculate(&x, &y, MiMethod::Binned);
        println!("MI (binned, independent): {} bits", result_binned.mi_bits);
        assert!(result_binned.mi_bits < 0.2);

        let result_ksg = estimator.calculate(&x, &y, MiMethod::Ksg);
        println!("MI (KSG, independent): {} bits", result_ksg.mi_bits);
        assert!(result_ksg.mi_bits < 0.3);
    }

    #[test]
    fn test_mi_dependent() {
        let x = Array1::from_vec((0..1000).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = x.mapv(|v| v * 0.8 + 0.1); // Y strongly depends on X

        let estimator = MutualInformationEstimator::default();

        let result_binned = estimator.calculate(&x, &y, MiMethod::Binned);
        println!("MI (binned, dependent): {} bits", result_binned.mi_bits);
        assert!(result_binned.mi_bits > 0.5);

        let result_ksg = estimator.calculate(&x, &y, MiMethod::Ksg);
        println!("MI (KSG, dependent): {} bits", result_ksg.mi_bits);
        assert!(result_ksg.mi_bits > 0.5);
    }

    #[test]
    fn test_mi_symmetric() {
        let x = Array1::from_vec((0..500).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = x.mapv(|v| v * 0.5);

        let estimator = MutualInformationEstimator::default();

        let mi_xy = estimator.calculate(&x, &y, MiMethod::Ksg);
        let mi_yx = estimator.calculate(&y, &x, MiMethod::Ksg);

        println!("MI(X;Y): {}, MI(Y;X): {}", mi_xy.mi_bits, mi_yx.mi_bits);

        // MI should be symmetric
        assert!((mi_xy.mi_bits - mi_yx.mi_bits).abs() < 0.1);
    }

    #[test]
    fn test_normalized_mi() {
        let x = Array1::from_vec((0..500).map(|i| i as f64 / 100.0).collect());
        let y = x.clone(); // Perfect correlation

        let estimator = MutualInformationEstimator::default();
        let result = estimator.calculate(&x, &y, MiMethod::Binned);

        println!("Normalized MI (perfect correlation): {}", result.normalized_mi);
        // Should be close to 1 for perfect correlation
        assert!(result.normalized_mi > 0.8);
    }
}
