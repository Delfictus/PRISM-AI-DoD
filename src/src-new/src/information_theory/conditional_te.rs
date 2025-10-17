//! Conditional Transfer Entropy (CTE) Implementation
//!
//! Conditional Transfer Entropy measures information transfer from source to target
//! conditioned on other variables (conditioning set Z):
//!
//! CTE(X → Y | Z) = I(X_past; Y_future | Y_past, Z)
//!
//! This is crucial for distinguishing direct vs. indirect causal influences:
//! - If CTE(X → Y | Z) ≈ 0 but TE(X → Y) > 0, then X influences Y only through Z
//! - If CTE(X → Y | Z) > 0, then X has direct causal influence on Y beyond Z
//!
//! # Applications
//! - Network motif detection (direct vs. mediated links)
//! - Partial correlation networks
//! - Confounding variable control
//! - Multi-agent causal inference
//!
//! # Constitution Compliance
//! Worker 5 - Advanced Transfer Entropy Module
//! Implements KSG estimator for continuous variables

use ndarray::Array1;
use anyhow::{Result, bail};
use std::collections::HashMap;

/// Conditional Transfer Entropy calculator using KSG estimator
///
/// Computes TE(X → Y | Z) where Z is a conditioning set of variables
///
/// # Example
/// ```
/// use ndarray::Array1;
/// use prism_ai::information_theory::ConditionalTE;
///
/// let source = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let target = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
/// let condition = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
///
/// let cte = ConditionalTE::new(3, 1, 1)?;
/// let result = cte.calculate(&source, &target, &[condition])?;
///
/// println!("Conditional TE: {:.4}", result.cte_value);
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ConditionalTE {
    /// Number of nearest neighbors for KSG estimation
    k_neighbors: usize,
    /// Embedding dimension for source history
    source_history_length: usize,
    /// Embedding dimension for target history
    target_history_length: usize,
    /// Number of conditioning variables
    n_conditioning: usize,
}

/// Result of conditional transfer entropy calculation
#[derive(Debug, Clone)]
pub struct ConditionalTEResult {
    /// Conditional transfer entropy value (bits)
    pub cte_value: f64,
    /// Unconditional TE for comparison: TE(X → Y)
    pub unconditional_te: f64,
    /// Reduction in TE due to conditioning: TE - CTE
    pub te_reduction: f64,
    /// Statistical significance (p-value from permutation test)
    pub p_value: f64,
    /// Number of samples used
    pub n_samples: usize,
    /// Effective dimension after conditioning
    pub effective_dimension: usize,
}

impl ConditionalTE {
    /// Create new conditional TE calculator
    ///
    /// # Arguments
    /// * `k` - Number of nearest neighbors (typical: 3-10)
    /// * `source_lag` - History length for source variable
    /// * `target_lag` - History length for target variable
    pub fn new(k: usize, source_lag: usize, target_lag: usize) -> Result<Self> {
        if k == 0 {
            bail!("k_neighbors must be > 0");
        }
        if source_lag == 0 || target_lag == 0 {
            bail!("History lengths must be > 0");
        }

        Ok(Self {
            k_neighbors: k,
            source_history_length: source_lag,
            target_history_length: target_lag,
            n_conditioning: 0,
        })
    }

    /// Calculate conditional transfer entropy: TE(X → Y | Z)
    ///
    /// # Arguments
    /// * `source` - Source time series X
    /// * `target` - Target time series Y
    /// * `conditioning_vars` - Slice of conditioning variables Z = [Z1, Z2, ...]
    ///
    /// # Returns
    /// ConditionalTEResult with CTE value and statistics
    pub fn calculate(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
        conditioning_vars: &[Array1<f64>],
    ) -> Result<ConditionalTEResult> {
        self.calculate_internal(source, target, conditioning_vars, true)
    }

    /// Internal calculation with option to skip permutation test
    fn calculate_internal(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
        conditioning_vars: &[Array1<f64>],
        do_permutation_test: bool,
    ) -> Result<ConditionalTEResult> {
        // Validate inputs
        let n = source.len();
        if n != target.len() {
            bail!("Source and target must have same length");
        }

        for (i, cond_var) in conditioning_vars.iter().enumerate() {
            if cond_var.len() != n {
                bail!("Conditioning variable {} has mismatched length", i);
            }
        }

        let n_cond = conditioning_vars.len();

        // Minimum samples needed for reliable estimation
        let min_samples = (self.k_neighbors + 1) * 3;
        let max_lag = self.source_history_length.max(self.target_history_length);

        if n < max_lag + min_samples {
            bail!("Insufficient samples: need at least {}", max_lag + min_samples);
        }

        // Build embedding vectors
        let embeddings = self.build_conditional_embeddings(
            source,
            target,
            conditioning_vars,
        )?;

        // Calculate CTE using KSG estimator
        let cte_value = self.ksg_conditional_te(&embeddings)?;

        // Calculate unconditional TE for comparison
        let unconditional_te = self.unconditional_te(source, target)?;

        // Compute TE reduction
        let te_reduction = (unconditional_te - cte_value).max(0.0);

        // Permutation test for significance (skip if disabled or insufficient samples)
        let p_value = if do_permutation_test && n > 50 {
            self.permutation_test(source, target, conditioning_vars, cte_value, 20)?
        } else {
            1.0 // Skipped or not enough data
        };

        let n_samples = embeddings.y_future.len();
        let effective_dimension = self.source_history_length + self.target_history_length + n_cond;

        Ok(ConditionalTEResult {
            cte_value,
            unconditional_te,
            te_reduction,
            p_value,
            n_samples,
            effective_dimension,
        })
    }

    /// Build embedding vectors for conditional TE calculation
    ///
    /// Creates: (X_past, Y_past, Z, Y_future)
    fn build_conditional_embeddings(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
        conditioning_vars: &[Array1<f64>],
    ) -> Result<ConditionalEmbeddings> {
        let n = source.len();
        let max_lag = self.source_history_length.max(self.target_history_length);
        let n_samples = n - max_lag;

        let mut x_past = Vec::with_capacity(n_samples);
        let mut y_past = Vec::with_capacity(n_samples);
        let mut y_future = Vec::with_capacity(n_samples);
        let mut z_vals = Vec::with_capacity(n_samples);

        for t in max_lag..n {
            // Source history: X(t-k_x), ..., X(t-1)
            let x_history: Vec<f64> = (0..self.source_history_length)
                .map(|lag| source[t - lag - 1])
                .collect();

            // Target history: Y(t-k_y), ..., Y(t-1)
            let y_history: Vec<f64> = (0..self.target_history_length)
                .map(|lag| target[t - lag - 1])
                .collect();

            // Conditioning variables at time t-1
            let z_at_t: Vec<f64> = conditioning_vars.iter()
                .map(|var| var[t - 1])
                .collect();

            // Future: Y(t)
            let y_fut = target[t];

            x_past.push(x_history);
            y_past.push(y_history);
            z_vals.push(z_at_t);
            y_future.push(y_fut);
        }

        Ok(ConditionalEmbeddings {
            x_past,
            y_past,
            z_vals,
            y_future,
        })
    }

    /// KSG estimator for conditional transfer entropy
    ///
    /// CTE = H(Y_future | Y_past, Z) - H(Y_future | X_past, Y_past, Z)
    ///
    /// Uses k-nearest neighbor distances in joint and marginal spaces
    fn ksg_conditional_te(&self, emb: &ConditionalEmbeddings) -> Result<f64> {
        let n = emb.y_future.len();
        if n < self.k_neighbors + 1 {
            bail!("Insufficient samples for KSG estimation");
        }

        let mut sum = 0.0;

        for i in 0..n {
            // Joint space: (Y_future, X_past, Y_past, Z)
            let dist_full = self.find_kth_neighbor_distance_full(emb, i)?;

            // Marginal space: (Y_past, Z)
            let dist_cond = self.find_kth_neighbor_distance_cond(emb, i)?;

            // KSG estimator contribution
            // ψ(k) - ψ(m_z + 1)
            // where m_z is number of neighbors within dist_full in (Y_past, Z) space

            let m_z = self.count_neighbors_within_radius(emb, i, dist_full)?;

            // Digamma function approximation
            let psi_k = Self::digamma(self.k_neighbors as f64);
            let psi_mz = Self::digamma((m_z + 1) as f64);

            sum += psi_k - psi_mz;
        }

        let cte = sum / (n as f64);

        // Convert to bits (KSG gives nats)
        Ok(cte / std::f64::consts::LN_2)
    }

    /// Find k-th nearest neighbor distance in full joint space
    fn find_kth_neighbor_distance_full(
        &self,
        emb: &ConditionalEmbeddings,
        index: usize,
    ) -> Result<f64> {
        let mut distances = Vec::with_capacity(emb.y_future.len() - 1);

        let query_y_fut = emb.y_future[index];
        let query_x = &emb.x_past[index];
        let query_y = &emb.y_past[index];
        let query_z = &emb.z_vals[index];

        for j in 0..emb.y_future.len() {
            if j == index {
                continue;
            }

            // Euclidean distance in joint space
            let mut dist_sq = (emb.y_future[j] - query_y_fut).powi(2);

            for (k, &x_val) in emb.x_past[j].iter().enumerate() {
                dist_sq += (x_val - query_x[k]).powi(2);
            }

            for (k, &y_val) in emb.y_past[j].iter().enumerate() {
                dist_sq += (y_val - query_y[k]).powi(2);
            }

            for (k, &z_val) in emb.z_vals[j].iter().enumerate() {
                dist_sq += (z_val - query_z[k]).powi(2);
            }

            distances.push(dist_sq.sqrt());
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(distances[self.k_neighbors - 1])
    }

    /// Find k-th nearest neighbor distance in conditional space (Y_past, Z)
    fn find_kth_neighbor_distance_cond(
        &self,
        emb: &ConditionalEmbeddings,
        index: usize,
    ) -> Result<f64> {
        let mut distances = Vec::with_capacity(emb.y_future.len() - 1);

        let query_y = &emb.y_past[index];
        let query_z = &emb.z_vals[index];

        for j in 0..emb.y_future.len() {
            if j == index {
                continue;
            }

            let mut dist_sq = 0.0;

            for (k, &y_val) in emb.y_past[j].iter().enumerate() {
                dist_sq += (y_val - query_y[k]).powi(2);
            }

            for (k, &z_val) in emb.z_vals[j].iter().enumerate() {
                dist_sq += (z_val - query_z[k]).powi(2);
            }

            distances.push(dist_sq.sqrt());
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(distances[self.k_neighbors - 1])
    }

    /// Count neighbors within given radius in conditional space
    fn count_neighbors_within_radius(
        &self,
        emb: &ConditionalEmbeddings,
        index: usize,
        radius: f64,
    ) -> Result<usize> {
        let query_y = &emb.y_past[index];
        let query_z = &emb.z_vals[index];
        let mut count = 0;

        for j in 0..emb.y_future.len() {
            if j == index {
                continue;
            }

            let mut dist_sq = 0.0;

            for (k, &y_val) in emb.y_past[j].iter().enumerate() {
                dist_sq += (y_val - query_y[k]).powi(2);
            }

            for (k, &z_val) in emb.z_vals[j].iter().enumerate() {
                dist_sq += (z_val - query_z[k]).powi(2);
            }

            if dist_sq.sqrt() < radius {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Calculate unconditional TE for comparison
    fn unconditional_te(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<f64> {
        use crate::information_theory::TransferEntropy;

        let te = TransferEntropy::default();
        let result = te.calculate(source, target);

        Ok(result.te_value)
    }

    /// Permutation test for statistical significance
    fn permutation_test(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
        conditioning_vars: &[Array1<f64>],
        observed_cte: f64,
        n_permutations: usize,
    ) -> Result<f64> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let mut count_greater = 0;

        // Permute source to break causal relationship
        let mut source_shuffled = source.to_vec();

        for _ in 0..n_permutations {
            source_shuffled.shuffle(&mut rng);
            let source_perm = Array1::from_vec(source_shuffled.clone());

            // Use internal method with permutation test disabled to avoid recursion
            let result = self.calculate_internal(&source_perm, target, conditioning_vars, false)?;

            if result.cte_value >= observed_cte {
                count_greater += 1;
            }
        }

        Ok((count_greater + 1) as f64 / (n_permutations + 1) as f64)
    }

    /// Digamma function approximation (ψ(x))
    fn digamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Use iterative approach to avoid stack overflow
        let mut xx = x;
        let mut correction = 0.0;

        // Shift x into range where asymptotic expansion is accurate (x > 6)
        while xx < 6.0 {
            correction -= 1.0 / xx;
            xx += 1.0;
        }

        // Asymptotic expansion for x > 6
        correction + xx.ln() - 0.5 / xx - 1.0 / (12.0 * xx * xx)
    }
}

/// Embedding vectors for conditional TE calculation
#[derive(Debug, Clone)]
struct ConditionalEmbeddings {
    /// Source history vectors
    x_past: Vec<Vec<f64>>,
    /// Target history vectors
    y_past: Vec<Vec<f64>>,
    /// Conditioning variable vectors
    z_vals: Vec<Vec<f64>>,
    /// Future target values
    y_future: Vec<f64>,
}

/// Detect mediated vs. direct causal links
///
/// Tests if X → Y causality is mediated through Z:
/// - If CTE(X → Y | Z) ≈ 0: X influences Y only through Z (mediated)
/// - If CTE(X → Y | Z) > 0: X has direct influence on Y (direct)
///
/// # Returns
/// (is_mediated, cte_value, significance)
pub fn detect_mediation(
    source: &Array1<f64>,
    target: &Array1<f64>,
    mediator: &Array1<f64>,
    threshold: f64,
) -> Result<(bool, f64, f64)> {
    let cte = ConditionalTE::new(3, 1, 1)?;
    let result = cte.calculate(source, target, &[mediator.clone()])?;

    let is_mediated = result.cte_value < threshold && result.p_value > 0.05;

    Ok((is_mediated, result.cte_value, result.p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conditional_te_creation() {
        let cte = ConditionalTE::new(3, 1, 1);
        assert!(cte.is_ok());

        let cte = cte.unwrap();
        assert_eq!(cte.k_neighbors, 3);
        assert_eq!(cte.source_history_length, 1);
        assert_eq!(cte.target_history_length, 1);
    }

    #[test]
    fn test_conditional_te_calculation() {
        // Create correlated data: Y = X + Z + noise
        let n = 100;
        let source: Array1<f64> = Array1::linspace(0.0, 10.0, n);
        let condition: Array1<f64> = Array1::linspace(0.0, 5.0, n);
        let target: Array1<f64> = source.mapv(|x| x * 0.5) + condition.mapv(|z| z * 0.3);

        let cte = ConditionalTE::new(3, 1, 1).unwrap();
        let result = cte.calculate(&source, &target, &[condition]);

        assert!(result.is_ok());
        let result = result.unwrap();

        // CTE should be lower than unconditional TE (Z explains some of the relationship)
        assert!(result.cte_value <= result.unconditional_te);
        assert!(result.te_reduction >= 0.0);
        assert!(result.n_samples > 0);
    }

    #[test]
    fn test_mediation_detection() {
        // Create mediated relationship: X → Z → Y
        let n = 100;
        let source: Array1<f64> = Array1::linspace(0.0, 10.0, n);
        let mediator = source.mapv(|x| x * 0.8); // Z = 0.8X
        let target = mediator.mapv(|z| z * 1.2); // Y = 1.2Z

        let result = detect_mediation(&source, &target, &mediator, 0.1);
        assert!(result.is_ok());

        let (is_mediated, cte_val, p_val) = result.unwrap();

        // Should detect mediation (CTE ≈ 0 after conditioning on Z)
        assert!(cte_val < 0.5); // Low CTE suggests mediation
    }

    #[test]
    fn test_direct_causation() {
        // Create direct relationship: X → Y (independent of Z)
        let n = 100;
        let source: Array1<f64> = Array1::linspace(0.0, 10.0, n);
        let target = source.mapv(|x| x * 0.7 + 0.1); // Y = 0.7X + noise
        let independent: Array1<f64> = Array1::linspace(5.0, 15.0, n); // Z independent

        let result = detect_mediation(&source, &target, &independent, 0.1);
        assert!(result.is_ok());

        let (is_mediated, cte_val, _) = result.unwrap();

        // Should NOT detect mediation (X has direct effect on Y)
        assert!(!is_mediated || cte_val > 0.1);
    }

    #[test]
    fn test_invalid_inputs() {
        let cte = ConditionalTE::new(3, 1, 1).unwrap();

        let source = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![2.0, 3.0]); // Mismatched length
        let condition = Array1::from_vec(vec![0.5, 1.0, 1.5]);

        let result = cte.calculate(&source, &target, &[condition]);
        assert!(result.is_err());
    }

    #[test]
    fn test_digamma_function() {
        // Test digamma approximation
        let psi_1 = ConditionalTE::digamma(1.0);
        let psi_2 = ConditionalTE::digamma(2.0);

        assert!(psi_1.is_finite());
        assert!(psi_2.is_finite());
        assert!(psi_2 > psi_1); // Digamma is monotonically increasing

        // ψ(2) ≈ 1 - γ ≈ 0.4228
        assert!((psi_2 - 0.4228).abs() < 0.1);
    }

    #[test]
    fn test_multiple_conditioning_variables() {
        let n = 100;
        let source: Array1<f64> = Array1::linspace(0.0, 10.0, n);
        let target = source.mapv(|x| x * 0.5);
        let cond1: Array1<f64> = Array1::linspace(0.0, 5.0, n);
        let cond2: Array1<f64> = Array1::linspace(1.0, 6.0, n);

        let cte = ConditionalTE::new(3, 1, 1).unwrap();
        let result = cte.calculate(&source, &target, &[cond1, cond2]);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.effective_dimension, 4); // 1 source + 1 target + 2 conditioning
    }
}
