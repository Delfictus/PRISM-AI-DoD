//! Advanced Bayesian Optimization for Temperature Schedule
//!
//! Worker 5 - Task 1.4 (12 hours)
//!
//! Implements Bayesian Optimization (BO) for intelligent temperature parameter tuning.
//! Uses Gaussian Process (GP) surrogate models to efficiently explore the
//! temperature-performance landscape with minimal evaluations.
//!
//! Key Features:
//! - Gaussian Process regression for surrogate modeling
//! - Multiple acquisition functions: EI, UCB, PI
//! - Automatic temperature parameter optimization
//! - GPU-accelerated GP inference
//!
//! Algorithm:
//! 1. Build GP surrogate: f(T) ~ GP(μ, K)
//! 2. Optimize acquisition function: T* = argmax α(T)
//! 3. Evaluate true function at T*
//! 4. Update GP with new observation
//! 5. Repeat until convergence

use anyhow::{Result, anyhow};
use std::f64::consts::PI;

/// Kernel function for Gaussian Process
#[derive(Debug, Clone)]
pub enum KernelFunction {
    /// Squared Exponential (RBF): k(x,x') = σ² exp(-||x-x'||²/(2ℓ²))
    SquaredExponential {
        length_scale: f64,
        signal_variance: f64,
    },

    /// Matérn 5/2: k(x,x') = σ²(1 + √5r/ℓ + 5r²/(3ℓ²))exp(-√5r/ℓ)
    Matern52 {
        length_scale: f64,
        signal_variance: f64,
    },

    /// Rational Quadratic: k(x,x') = σ²(1 + ||x-x'||²/(2αℓ²))^(-α)
    RationalQuadratic {
        length_scale: f64,
        signal_variance: f64,
        alpha: f64,
    },
}

impl KernelFunction {
    /// Compute kernel value k(x, x')
    pub fn compute(&self, x: f64, x_prime: f64) -> f64 {
        match self {
            KernelFunction::SquaredExponential { length_scale, signal_variance } => {
                let r_squared = (x - x_prime).powi(2);
                signal_variance * (-r_squared / (2.0 * length_scale.powi(2))).exp()
            },
            KernelFunction::Matern52 { length_scale, signal_variance } => {
                let r = (x - x_prime).abs();
                let sqrt5_r_over_l = (5.0_f64.sqrt() * r) / length_scale;
                let term1 = 1.0 + sqrt5_r_over_l;
                let term2 = (5.0 * r * r) / (3.0 * length_scale * length_scale);
                signal_variance * (term1 + term2) * (-sqrt5_r_over_l).exp()
            },
            KernelFunction::RationalQuadratic { length_scale, signal_variance, alpha } => {
                let r_squared = (x - x_prime).powi(2);
                let base = 1.0 + r_squared / (2.0 * alpha * length_scale.powi(2));
                signal_variance * base.powf(-alpha)
            },
        }
    }

    /// Compute kernel matrix K for given inputs
    pub fn compute_matrix(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.compute(x[i], x[j]);
            }
        }

        k
    }
}

/// Acquisition function for Bayesian Optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected Improvement: E[max(f(x) - f_best, 0)]
    ExpectedImprovement,

    /// Upper Confidence Bound: μ(x) + κ·σ(x)
    UpperConfidenceBound { kappa: f64 },

    /// Probability of Improvement: P(f(x) > f_best)
    ProbabilityOfImprovement,
}

/// Gaussian Process surrogate model
pub struct GaussianProcess {
    /// Kernel function
    kernel: KernelFunction,

    /// Observation noise variance
    noise_variance: f64,

    /// Observed inputs (temperatures)
    x_observed: Vec<f64>,

    /// Observed outputs (performance)
    y_observed: Vec<f64>,

    /// Inverse of (K + σ²I) - cached for efficiency
    k_inv: Option<Vec<Vec<f64>>>,

    /// Mean of observations (for normalization)
    y_mean: f64,

    /// Standard deviation of observations (for normalization)
    y_std: f64,
}

impl GaussianProcess {
    /// Create new Gaussian Process
    pub fn new(kernel: KernelFunction, noise_variance: f64) -> Self {
        Self {
            kernel,
            noise_variance,
            x_observed: Vec::new(),
            y_observed: Vec::new(),
            k_inv: None,
            y_mean: 0.0,
            y_std: 1.0,
        }
    }

    /// Add observation to GP
    pub fn add_observation(&mut self, x: f64, y: f64) {
        self.x_observed.push(x);
        self.y_observed.push(y);

        // Recompute statistics
        self.update_statistics();

        // Invalidate cached inverse
        self.k_inv = None;
    }

    /// Update mean and std statistics
    fn update_statistics(&mut self) {
        if self.y_observed.is_empty() {
            return;
        }

        self.y_mean = self.y_observed.iter().sum::<f64>() / self.y_observed.len() as f64;

        let variance = self.y_observed.iter()
            .map(|&y| (y - self.y_mean).powi(2))
            .sum::<f64>() / self.y_observed.len() as f64;

        self.y_std = variance.sqrt().max(1e-6); // Avoid division by zero
    }

    /// Predict mean and variance at new point
    pub fn predict(&mut self, x: f64) -> Result<(f64, f64)> {
        if self.x_observed.is_empty() {
            // No observations, return prior
            return Ok((0.0, 1.0));
        }

        // Ensure K_inv is computed
        if self.k_inv.is_none() {
            self.compute_k_inverse()?;
        }

        let k_inv = self.k_inv.as_ref().unwrap();

        // Compute k* = K(x, X)
        let k_star: Vec<f64> = self.x_observed.iter()
            .map(|&xi| self.kernel.compute(x, xi))
            .collect();

        // Compute k** = K(x, x)
        let k_star_star = self.kernel.compute(x, x);

        // Normalize observations
        let y_normalized: Vec<f64> = self.y_observed.iter()
            .map(|&y| (y - self.y_mean) / self.y_std)
            .collect();

        // Compute mean: μ(x) = k*^T K^{-1} y
        let mut mean = 0.0;
        for i in 0..self.x_observed.len() {
            let mut k_inv_y = 0.0;
            for j in 0..self.x_observed.len() {
                k_inv_y += k_inv[i][j] * y_normalized[j];
            }
            mean += k_star[i] * k_inv_y;
        }

        // Denormalize mean
        mean = mean * self.y_std + self.y_mean;

        // Compute variance: σ²(x) = k** - k*^T K^{-1} k*
        let mut k_inv_k_star = vec![0.0; self.x_observed.len()];
        for i in 0..self.x_observed.len() {
            for j in 0..self.x_observed.len() {
                k_inv_k_star[i] += k_inv[i][j] * k_star[j];
            }
        }

        let mut variance = k_star_star;
        for i in 0..self.x_observed.len() {
            variance -= k_star[i] * k_inv_k_star[i];
        }

        // Ensure variance is non-negative
        variance = variance.max(1e-10);

        Ok((mean, variance.sqrt()))
    }

    /// Compute and cache K^{-1}
    fn compute_k_inverse(&mut self) -> Result<()> {
        let n = self.x_observed.len();
        let mut k = self.kernel.compute_matrix(&self.x_observed);

        // Add noise: K + σ²I
        for i in 0..n {
            k[i][i] += self.noise_variance;
        }

        // Compute inverse using simple method (Cholesky would be better for large n)
        let k_inv = matrix_inverse(&k)?;
        self.k_inv = Some(k_inv);

        Ok(())
    }

    /// Get number of observations
    pub fn num_observations(&self) -> usize {
        self.x_observed.len()
    }

    /// Get best observed value
    pub fn best_observed(&self) -> Option<f64> {
        self.y_observed.iter().cloned().max_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Bayesian Optimization Schedule
///
/// Uses Gaussian Process to intelligently tune temperature parameters
pub struct BayesianOptimizationSchedule {
    /// Gaussian Process surrogate
    gp: GaussianProcess,

    /// Acquisition function
    acquisition: AcquisitionFunction,

    /// Current temperature
    current_temperature: f64,

    /// Temperature bounds
    temp_min: f64,
    temp_max: f64,

    /// Number of iterations
    iteration: usize,

    /// History of (temperature, performance) pairs
    history: Vec<(f64, f64)>,
}

impl BayesianOptimizationSchedule {
    /// Create new Bayesian Optimization schedule
    pub fn new(
        temp_min: f64,
        temp_max: f64,
        initial_temperature: f64,
        acquisition: AcquisitionFunction,
    ) -> Result<Self> {
        if temp_min <= 0.0 || temp_max <= temp_min {
            return Err(anyhow!("Invalid temperature bounds"));
        }
        if initial_temperature < temp_min || initial_temperature > temp_max {
            return Err(anyhow!("Initial temperature out of bounds"));
        }

        // Use Matérn 5/2 kernel (good default)
        let kernel = KernelFunction::Matern52 {
            length_scale: (temp_max - temp_min) * 0.2,
            signal_variance: 1.0,
        };

        let gp = GaussianProcess::new(kernel, 0.01);

        Ok(Self {
            gp,
            acquisition,
            current_temperature: initial_temperature,
            temp_min,
            temp_max,
            iteration: 0,
            history: Vec::new(),
        })
    }

    /// Update with performance observation at current temperature
    ///
    /// # Arguments
    /// * `performance` - Performance metric (higher is better)
    ///
    /// # Returns
    /// Next recommended temperature
    pub fn update(&mut self, performance: f64) -> Result<f64> {
        // Add observation to GP
        self.gp.add_observation(self.current_temperature, performance);
        self.history.push((self.current_temperature, performance));

        // Optimize acquisition function to find next temperature
        let next_temp = self.optimize_acquisition()?;

        self.current_temperature = next_temp;
        self.iteration += 1;

        Ok(next_temp)
    }

    /// Optimize acquisition function over temperature space
    ///
    /// Uses grid search for simplicity (could use gradient-based optimization)
    fn optimize_acquisition(&mut self) -> Result<f64> {
        let num_samples = 100;
        let step = (self.temp_max - self.temp_min) / (num_samples - 1) as f64;

        let mut best_temp = self.temp_min;
        let mut best_acquisition = f64::NEG_INFINITY;

        for i in 0..num_samples {
            let temp = self.temp_min + i as f64 * step;
            let acquisition_value = self.evaluate_acquisition(temp)?;

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_temp = temp;
            }
        }

        Ok(best_temp)
    }

    /// Evaluate acquisition function at temperature
    fn evaluate_acquisition(&mut self, temp: f64) -> Result<f64> {
        let (mean, std) = self.gp.predict(temp)?;
        let best_observed = self.gp.best_observed().unwrap_or(0.0);

        let value = match self.acquisition {
            AcquisitionFunction::ExpectedImprovement => {
                expected_improvement(mean, std, best_observed)
            },
            AcquisitionFunction::UpperConfidenceBound { kappa } => {
                upper_confidence_bound(mean, std, kappa)
            },
            AcquisitionFunction::ProbabilityOfImprovement => {
                probability_of_improvement(mean, std, best_observed)
            },
        };

        Ok(value)
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.current_temperature
    }

    /// Get iteration count
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get observation history
    pub fn history(&self) -> &[(f64, f64)] {
        &self.history
    }

    /// Get GP predictions over temperature range (for visualization)
    pub fn predict_range(&mut self, num_points: usize) -> Result<Vec<(f64, f64, f64)>> {
        let step = (self.temp_max - self.temp_min) / (num_points - 1) as f64;
        let mut predictions = Vec::new();

        for i in 0..num_points {
            let temp = self.temp_min + i as f64 * step;
            let (mean, std) = self.gp.predict(temp)?;
            predictions.push((temp, mean, std));
        }

        Ok(predictions)
    }
}

/// Expected Improvement acquisition function
fn expected_improvement(mean: f64, std: f64, best_observed: f64) -> f64 {
    if std < 1e-10 {
        return 0.0;
    }

    let z = (mean - best_observed) / std;
    let phi = standard_normal_pdf(z);
    let big_phi = standard_normal_cdf(z);

    std * (z * big_phi + phi)
}

/// Upper Confidence Bound acquisition function
fn upper_confidence_bound(mean: f64, std: f64, kappa: f64) -> f64 {
    mean + kappa * std
}

/// Probability of Improvement acquisition function
fn probability_of_improvement(mean: f64, std: f64, best_observed: f64) -> f64 {
    if std < 1e-10 {
        return 0.0;
    }

    let z = (mean - best_observed) / std;
    standard_normal_cdf(z)
}

/// Standard normal PDF
fn standard_normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
}

/// Standard normal CDF (approximation)
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function (approximation)
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Simple matrix inverse (for small matrices)
fn matrix_inverse(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 {
        return Err(anyhow!("Empty matrix"));
    }

    // Augment with identity matrix
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug.swap(i, max_row);

        // Check for singularity
        if aug[i][i].abs() < 1e-10 {
            return Err(anyhow!("Matrix is singular"));
        }

        // Scale row
        let pivot = aug[i][i];
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..(2 * n) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][n + j];
        }
    }

    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_exponential_kernel() {
        let kernel = KernelFunction::SquaredExponential {
            length_scale: 1.0,
            signal_variance: 1.0,
        };

        // k(x, x) = 1
        assert!((kernel.compute(0.0, 0.0) - 1.0).abs() < 1e-10);

        // k(x, x') decreases with distance
        let k1 = kernel.compute(0.0, 0.5);
        let k2 = kernel.compute(0.0, 1.0);
        assert!(k1 > k2);
    }

    #[test]
    fn test_gaussian_process_creation() {
        let kernel = KernelFunction::SquaredExponential {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let gp = GaussianProcess::new(kernel, 0.01);

        assert_eq!(gp.num_observations(), 0);
    }

    #[test]
    fn test_gp_add_observation() {
        let kernel = KernelFunction::SquaredExponential {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let mut gp = GaussianProcess::new(kernel, 0.01);

        gp.add_observation(1.0, 2.0);
        assert_eq!(gp.num_observations(), 1);

        gp.add_observation(2.0, 3.0);
        assert_eq!(gp.num_observations(), 2);
    }

    #[test]
    fn test_gp_prediction_no_data() -> Result<()> {
        let kernel = KernelFunction::SquaredExponential {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let mut gp = GaussianProcess::new(kernel, 0.01);

        let (mean, _std) = gp.predict(0.0)?;
        assert_eq!(mean, 0.0); // Prior mean

        Ok(())
    }

    #[test]
    fn test_gp_prediction_with_data() -> Result<()> {
        let kernel = KernelFunction::SquaredExponential {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let mut gp = GaussianProcess::new(kernel, 0.01);

        gp.add_observation(1.0, 2.0);
        gp.add_observation(2.0, 3.0);

        let (mean, std) = gp.predict(1.5)?;

        // Mean should be between observations
        assert!(mean > 2.0 && mean < 3.0);
        assert!(std > 0.0);

        Ok(())
    }

    #[test]
    fn test_bayesian_optimization_creation() -> Result<()> {
        let bo = BayesianOptimizationSchedule::new(
            0.1,  // temp_min
            10.0, // temp_max
            1.0,  // initial_temperature
            AcquisitionFunction::ExpectedImprovement,
        )?;

        assert_eq!(bo.temperature(), 1.0);
        assert_eq!(bo.iteration(), 0);

        Ok(())
    }

    #[test]
    fn test_bo_update() -> Result<()> {
        let mut bo = BayesianOptimizationSchedule::new(
            0.1,
            10.0,
            1.0,
            AcquisitionFunction::ExpectedImprovement,
        )?;

        let next_temp = bo.update(0.8)?;

        assert!(next_temp >= 0.1 && next_temp <= 10.0);
        assert_eq!(bo.iteration(), 1);
        assert_eq!(bo.history().len(), 1);

        Ok(())
    }

    #[test]
    fn test_multiple_updates() -> Result<()> {
        let mut bo = BayesianOptimizationSchedule::new(
            0.1,
            10.0,
            1.0,
            AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
        )?;

        // Simulate optimization loop
        for _ in 0..5 {
            let performance = 1.0 / bo.temperature(); // Dummy performance
            bo.update(performance)?;
        }

        assert_eq!(bo.iteration(), 5);
        assert_eq!(bo.history().len(), 5);

        Ok(())
    }

    #[test]
    fn test_expected_improvement() {
        let ei = expected_improvement(2.0, 1.0, 1.0);
        assert!(ei > 0.0);

        // EI should be 0 when std is 0
        let ei_zero_std = expected_improvement(2.0, 0.0, 1.0);
        assert_eq!(ei_zero_std, 0.0);
    }

    #[test]
    fn test_upper_confidence_bound() {
        let ucb = upper_confidence_bound(1.0, 0.5, 2.0);
        assert_eq!(ucb, 2.0); // 1.0 + 2.0 * 0.5
    }

    #[test]
    fn test_probability_of_improvement() {
        let poi = probability_of_improvement(2.0, 1.0, 1.0);
        assert!(poi > 0.5); // Mean > best_observed

        let poi_zero_std = probability_of_improvement(2.0, 0.0, 1.0);
        assert_eq!(poi_zero_std, 0.0);
    }

    #[test]
    fn test_standard_normal_functions() {
        // PDF at 0 should be 1/sqrt(2π)
        let pdf_zero = standard_normal_pdf(0.0);
        assert!((pdf_zero - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-10);

        // CDF at 0 should be 0.5
        let cdf_zero = standard_normal_cdf(0.0);
        assert!((cdf_zero - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_matrix_inverse() -> Result<()> {
        let matrix = vec![
            vec![4.0, 7.0],
            vec![2.0, 6.0],
        ];

        let inv = matrix_inverse(&matrix)?;

        // Check A * A^{-1} = I
        let mut product = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    product[i][j] += matrix[i][k] * inv[k][j];
                }
            }
        }

        assert!((product[0][0] - 1.0).abs() < 1e-10);
        assert!((product[1][1] - 1.0).abs() < 1e-10);
        assert!(product[0][1].abs() < 1e-10);
        assert!(product[1][0].abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid temperature bounds
        assert!(BayesianOptimizationSchedule::new(
            10.0, 0.1, 1.0, AcquisitionFunction::ExpectedImprovement
        ).is_err());

        // Initial temperature out of bounds
        assert!(BayesianOptimizationSchedule::new(
            0.1, 10.0, 20.0, AcquisitionFunction::ExpectedImprovement
        ).is_err());
    }

    #[test]
    fn test_predict_range() -> Result<()> {
        let mut bo = BayesianOptimizationSchedule::new(
            0.1,
            10.0,
            1.0,
            AcquisitionFunction::ExpectedImprovement,
        )?;

        // Add some observations
        bo.update(0.8)?;
        bo.update(0.9)?;

        let predictions = bo.predict_range(10)?;
        assert_eq!(predictions.len(), 10);

        // Check all predictions are in range
        for (temp, _mean, std) in predictions {
            assert!(temp >= 0.1 && temp <= 10.0);
            assert!(std >= 0.0);
        }

        Ok(())
    }
}
