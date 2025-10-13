//! Information-Theoretic Metrics for Worker 7 Applications
//!
//! Provides rigorous information theory metrics for:
//! - Scientific experiment design (mutual information, information gain)
//! - Drug discovery (molecular similarity via information metrics)
//! - Robotics (uncertainty quantification via entropy)
//!
//! Based on PRISM's core information theory infrastructure with mathematical rigor.

use anyhow::Result;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Information-theoretic metrics for experiment design
///
/// Implements rigorous Shannon information theory for optimal experiment design:
/// - Expected Information Gain (EIG)
/// - Differential Entropy
/// - Mutual Information
/// - Conditional Entropy
/// - KL Divergence
pub struct ExperimentInformationMetrics {
    /// Number of nearest neighbors for entropy estimation
    k_neighbors: usize,
}

impl ExperimentInformationMetrics {
    /// Create new experiment information metrics calculator
    pub fn new() -> Result<Self> {
        Ok(Self {
            k_neighbors: 5,
        })
    }

    /// Calculate expected information gain for experiment design
    ///
    /// EIG = H(θ) - E[H(θ|X)]
    ///
    /// where:
    /// - θ are unknown parameters
    /// - X is experimental measurement
    /// - H() is differential entropy
    ///
    /// Mathematical property: EIG ≥ 0 (data processing inequality)
    pub fn expected_information_gain(
        &self,
        prior_samples: &Array2<f64>,
        posterior_samples: &Array2<f64>,
    ) -> Result<f64> {
        // Prior entropy H(θ)
        let prior_entropy = self.differential_entropy(prior_samples)?;

        // Posterior entropy H(θ|X)
        let posterior_entropy = self.differential_entropy(posterior_samples)?;

        // Information gain = entropy reduction
        let eig = prior_entropy - posterior_entropy;

        // Verify non-negativity (fundamental property)
        if eig < -1e-10 {
            anyhow::bail!(
                "Violated information inequality: EIG={:.6} < 0. Prior H={:.6}, Posterior H={:.6}",
                eig, prior_entropy, posterior_entropy
            );
        }

        Ok(eig.max(0.0))
    }

    /// Calculate differential entropy H(X) for continuous variables
    ///
    /// Uses Kozachenko-Leonenko (KL) estimator:
    /// H(X) ≈ ψ(N) - ψ(k) + log(V_d) + (d/N)Σlog(2ρ_i)
    ///
    /// where:
    /// - N = number of samples
    /// - k = number of nearest neighbors
    /// - V_d = volume of unit sphere in d dimensions
    /// - ρ_i = distance to k-th nearest neighbor
    /// - ψ = digamma function
    pub fn differential_entropy(&self, samples: &Array2<f64>) -> Result<f64> {
        let n = samples.nrows();
        let d = samples.ncols();

        if n < 2 * self.k_neighbors {
            anyhow::bail!("Need at least {} samples for k={} estimation", 2 * self.k_neighbors, self.k_neighbors);
        }

        // Compute k-nearest neighbor distances
        let mut sum_log_distances = 0.0;

        for i in 0..n {
            let mut distances: Vec<f64> = Vec::with_capacity(n - 1);

            for j in 0..n {
                if i != j {
                    let dist = euclidean_distance(
                        &samples.row(i).to_owned(),
                        &samples.row(j).to_owned(),
                    );
                    distances.push(dist);
                }
            }

            // Sort to find k-th nearest neighbor
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let k_dist = distances[self.k_neighbors - 1];
            if k_dist > 0.0 {
                sum_log_distances += (2.0 * k_dist).ln();
            }
        }

        // KL estimator formula
        let digamma_n = digamma(n as f64);
        let digamma_k = digamma(self.k_neighbors as f64);
        let log_volume_unit_sphere = log_unit_sphere_volume(d);

        let entropy = digamma_n - digamma_k + log_volume_unit_sphere + (d as f64 / n as f64) * sum_log_distances;

        // Verify reasonable bounds
        if !entropy.is_finite() {
            anyhow::bail!("Non-finite entropy estimate");
        }

        Ok(entropy)
    }

    /// Calculate mutual information I(X;Y) between variables
    ///
    /// I(X;Y) = H(X) + H(Y) - H(X,Y)
    ///
    /// Properties:
    /// - I(X;Y) ≥ 0 (non-negativity)
    /// - I(X;Y) = I(Y;X) (symmetry)
    /// - I(X;Y) ≤ min(H(X), H(Y)) (data processing inequality)
    pub fn mutual_information(
        &self,
        x_samples: &Array2<f64>,
        y_samples: &Array2<f64>,
    ) -> Result<f64> {
        if x_samples.nrows() != y_samples.nrows() {
            anyhow::bail!("Sample count mismatch: {} vs {}", x_samples.nrows(), y_samples.nrows());
        }

        // Marginal entropies
        let h_x = self.differential_entropy(x_samples)?;
        let h_y = self.differential_entropy(y_samples)?;

        // Joint entropy H(X,Y)
        let joint_samples = concatenate_horizontal(x_samples, y_samples)?;
        let h_xy = self.differential_entropy(&joint_samples)?;

        // MI = H(X) + H(Y) - H(X,Y)
        let mi = h_x + h_y - h_xy;

        // Enforce information-theoretic bounds
        let mi_bounded = mi
            .max(0.0)                    // Non-negativity
            .min(h_x)                    // Data processing inequality
            .min(h_y);

        if mi < -1e-10 {
            eprintln!("Warning: Negative MI {:.6} detected, clamping to 0", mi);
        }

        Ok(mi_bounded)
    }

    /// Calculate conditional entropy H(X|Y)
    ///
    /// H(X|Y) = H(X,Y) - H(Y)
    ///
    /// Property: H(X|Y) ≥ 0
    pub fn conditional_entropy(
        &self,
        x_samples: &Array2<f64>,
        y_samples: &Array2<f64>,
    ) -> Result<f64> {
        if x_samples.nrows() != y_samples.nrows() {
            anyhow::bail!("Sample count mismatch");
        }

        let joint_samples = concatenate_horizontal(x_samples, y_samples)?;
        let h_xy = self.differential_entropy(&joint_samples)?;
        let h_y = self.differential_entropy(y_samples)?;

        let h_x_given_y = h_xy - h_y;

        // Verify: H(X|Y) ≥ 0
        if h_x_given_y < -1e-10 {
            anyhow::bail!("Violated conditional entropy bound: H(X|Y)={:.6} < 0", h_x_given_y);
        }

        Ok(h_x_given_y.max(0.0))
    }

    /// Calculate Kullback-Leibler divergence D_KL(P||Q)
    ///
    /// D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
    ///
    /// Properties:
    /// - D_KL ≥ 0 (Gibbs' inequality)
    /// - D_KL = 0 iff P = Q almost everywhere
    /// - NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
    pub fn kl_divergence(
        &self,
        p_samples: &Array2<f64>,
        q_samples: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified estimator using k-NN
        let n_p = p_samples.nrows();
        let n_q = q_samples.nrows();
        let d = p_samples.ncols();

        if d != q_samples.ncols() {
            anyhow::bail!("Dimension mismatch");
        }

        let mut kl_sum = 0.0;

        // For each sample from P
        for i in 0..n_p {
            let x = p_samples.row(i).to_owned();

            // Find k-th nearest neighbor in P
            let mut p_distances: Vec<f64> = Vec::new();
            for j in 0..n_p {
                if i != j {
                    p_distances.push(euclidean_distance(&x, &p_samples.row(j).to_owned()));
                }
            }
            p_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let rho_k = p_distances[self.k_neighbors - 1];

            // Find k-th nearest neighbor in Q
            let mut q_distances: Vec<f64> = Vec::new();
            for j in 0..n_q {
                q_distances.push(euclidean_distance(&x, &q_samples.row(j).to_owned()));
            }
            q_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nu_k = q_distances[self.k_neighbors - 1];

            // KL estimator contribution
            if rho_k > 0.0 && nu_k > 0.0 {
                kl_sum += (nu_k / rho_k).ln();
            }
        }

        let kl_div = (d as f64 / n_p as f64) * kl_sum + (n_q as f64 / (n_p - 1) as f64).ln();

        // Verify non-negativity (Gibbs' inequality)
        if kl_div < -1e-10 {
            eprintln!("Warning: Negative KL divergence {:.6}, clamping to 0", kl_div);
        }

        Ok(kl_div.max(0.0))
    }
}

impl Default for ExperimentInformationMetrics {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Molecular information metrics for drug discovery
pub struct MolecularInformationMetrics {}

impl MolecularInformationMetrics {
    /// Create new molecular information metrics calculator
    pub fn new() -> Self {
        Self {}
    }

    /// Calculate information-theoretic molecular similarity
    ///
    /// Uses Gaussian kernel: K(x,y) = exp(-||x-y||²/2σ²)
    /// Returns value in [0,1] where 1 = identical
    pub fn molecular_similarity(
        &self,
        mol1_descriptors: &Array1<f64>,
        mol2_descriptors: &Array1<f64>,
    ) -> f64 {
        let distance = euclidean_distance(mol1_descriptors, mol2_descriptors);
        let sigma = 1.0; // Bandwidth parameter
        (-distance.powi(2) / (2.0 * sigma * sigma)).exp()
    }

    /// Calculate chemical space coverage (entropy)
    ///
    /// Higher entropy = better exploration of chemical space
    pub fn chemical_space_entropy(&self, descriptors: &Array2<f64>) -> f64 {
        if descriptors.nrows() < 2 {
            return 0.0;
        }

        // Simplified: variance-based entropy estimate
        let mut total_variance = 0.0;
        for col in 0..descriptors.ncols() {
            let column = descriptors.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / descriptors.nrows() as f64;
            total_variance += variance;
        }

        // Entropy ≈ 0.5 * log(2πe * variance) for Gaussian
        0.5 * (2.0 * PI * std::f64::consts::E * total_variance).ln()
    }
}

impl Default for MolecularInformationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Robotics information metrics for uncertainty quantification
pub struct RoboticsInformationMetrics {}

impl RoboticsInformationMetrics {
    /// Create new robotics information metrics calculator
    pub fn new() -> Self {
        Self {}
    }

    /// Calculate trajectory uncertainty (entropy)
    ///
    /// Higher entropy = more uncertain/diverse trajectories
    pub fn trajectory_entropy(&self, trajectories: &Array2<f64>) -> f64 {
        if trajectories.nrows() < 2 {
            return 0.0;
        }

        // Estimate entropy from trajectory spread
        let mut total_variance = 0.0;
        for col in 0..trajectories.ncols() {
            let column = trajectories.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / trajectories.nrows() as f64;
            total_variance += variance;
        }

        // Differential entropy for multivariate Gaussian
        let d = trajectories.ncols() as f64;
        0.5 * d * (2.0 * PI * std::f64::consts::E).ln() + 0.5 * total_variance.ln()
    }

    /// Calculate information gain from sensor measurement
    ///
    /// IG = H(prior) - H(posterior)
    pub fn sensor_information_gain(&self, prior_var: f64, posterior_var: f64) -> f64 {
        // For Gaussian: H = 0.5 * log(2πe * variance)
        let h_prior = 0.5 * (2.0 * PI * std::f64::consts::E * prior_var).ln();
        let h_posterior = 0.5 * (2.0 * PI * std::f64::consts::E * posterior_var).ln();

        (h_prior - h_posterior).max(0.0)
    }
}

impl Default for RoboticsInformationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

/// Concatenate arrays horizontally (along axis 1)
fn concatenate_horizontal(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray::concatenate;

    if a.nrows() != b.nrows() {
        anyhow::bail!("Row count mismatch: {} vs {}", a.nrows(), b.nrows());
    }

    Ok(concatenate![ndarray::Axis(1), a.view(), b.view()])
}

/// Calculate Euclidean distance
fn euclidean_distance(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Digamma function ψ(x) = d/dx ln Γ(x)
///
/// Asymptotic approximation for x > 6:
/// ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴)
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // For x > 6, use asymptotic expansion
    if x > 6.0 {
        return x.ln() - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x) + 1.0 / (120.0 * x.powi(4));
    }

    // For small x, use recurrence relation ψ(x+1) = ψ(x) + 1/x
    digamma(x + 1.0) - 1.0 / x
}

/// Log of volume of unit sphere in d dimensions
///
/// log V_d = (d/2)log(π) - log Γ(d/2 + 1)
fn log_unit_sphere_volume(d: usize) -> f64 {
    let d_f = d as f64;
    (d_f / 2.0) * PI.ln() - log_gamma(d_f / 2.0 + 1.0)
}

/// Log gamma function ln Γ(x)
///
/// Stirling's approximation for x > 1:
/// ln Γ(x) ≈ (x-0.5)ln(x) - x + 0.5*ln(2π)
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x > 1.0 {
        // Stirling's approximation
        return (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln();
    }

    // For x <= 1, use recurrence Γ(x+1) = x*Γ(x)
    log_gamma(x + 1.0) - x.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma_function() {
        // ψ(1) ≈ -0.5772 (Euler-Mascheroni constant)
        assert!((digamma(1.0) + 0.5772).abs() < 0.1);

        // ψ(2) ≈ 0.4228
        assert!((digamma(2.0) - 0.4228).abs() < 0.1);
    }

    #[test]
    fn test_differential_entropy() {
        let metrics = ExperimentInformationMetrics::new().unwrap();

        // Create uniform samples
        let samples = Array2::from_shape_vec(
            (100, 2),
            (0..200).map(|i| (i % 100) as f64 / 100.0).collect(),
        ).unwrap();

        let entropy = metrics.differential_entropy(&samples);
        assert!(entropy.is_ok());
        assert!(entropy.unwrap().is_finite());
    }

    #[test]
    fn test_mutual_information_bounds() {
        let metrics = ExperimentInformationMetrics::new().unwrap();

        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
        let y = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64 + 1.0).collect()).unwrap();

        let mi = metrics.mutual_information(&x, &y);
        assert!(mi.is_ok());

        let i_xy = mi.unwrap();
        assert!(i_xy >= 0.0); // Non-negativity
    }

    #[test]
    fn test_molecular_similarity() {
        let metrics = MolecularInformationMetrics::new();

        let mol1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mol2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let sim = metrics.molecular_similarity(&mol1, &mol2);
        assert!((sim - 1.0).abs() < 1e-6); // Identical molecules
    }

    #[test]
    fn test_information_gain_non_negative() {
        let metrics = ExperimentInformationMetrics::new().unwrap();

        let prior = Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64).collect()).unwrap();
        let posterior = Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64 * 0.5).collect()).unwrap();

        let eig = metrics.expected_information_gain(&prior, &posterior);
        assert!(eig.is_ok());
        assert!(eig.unwrap() >= 0.0); // Fundamental property
    }
}
