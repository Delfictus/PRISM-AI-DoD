//! Bootstrap Confidence Intervals for Transfer Entropy
//!
//! Implements bootstrap resampling methods for uncertainty quantification
//! in transfer entropy estimation. Includes:
//!
//! 1. **Standard Bootstrap**: Basic percentile method
//! 2. **BCa Bootstrap**: Bias-corrected and accelerated intervals
//! 3. **Block Bootstrap**: Preserves temporal structure
//!
//! The BCa method provides more accurate coverage than standard bootstrap,
//! especially for skewed sampling distributions.
//!
//! Reference:
//! Efron, B., & Tibshirani, R. J. (1994).
//! "An introduction to the bootstrap." CRC press.

use anyhow::Result;
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;


/// Bootstrap confidence interval result
#[derive(Debug, Clone)]
pub struct BootstrapCi {
    /// Point estimate (original TE value)
    pub estimate: f64,
    /// Lower bound of CI
    pub lower: f64,
    /// Upper bound of CI
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
    /// Method used
    pub method: BootstrapMethod,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
}

/// Bootstrap methods
#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapMethod {
    /// Standard percentile method
    Percentile,
    /// Bias-corrected and accelerated (BCa)
    BCa,
    /// Block bootstrap for time series
    Block,
}

/// Bootstrap resampler for Transfer Entropy
pub struct BootstrapResampler {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Confidence level (0.95 for 95% CI)
    pub confidence_level: f64,
    /// Block size for block bootstrap
    pub block_size: usize,
    /// Bootstrap method
    pub method: BootstrapMethod,
}

impl Default for BootstrapResampler {
    fn default() -> Self {
        Self {
            n_bootstrap: 1000,
            confidence_level: 0.95,
            block_size: 10,
            method: BootstrapMethod::BCa,
        }
    }
}

impl BootstrapResampler {
    /// Create new bootstrap resampler
    pub fn new(n_bootstrap: usize, confidence_level: f64) -> Self {
        Self {
            n_bootstrap,
            confidence_level,
            ..Default::default()
        }
    }

    /// Calculate confidence interval for TE using specified method
    ///
    /// # Arguments
    /// * `te_calculator` - Closure that calculates TE given source and target
    /// * `source` - Source time series
    /// * `target` - Target time series
    /// * `observed_te` - Original TE estimate
    ///
    /// # Returns
    /// BootstrapCi with lower and upper confidence bounds
    pub fn calculate_ci<F>(
        &self,
        te_calculator: F,
        source: &Array1<f64>,
        target: &Array1<f64>,
        observed_te: f64,
    ) -> Result<BootstrapCi>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
    {
        match self.method {
            BootstrapMethod::Percentile => {
                self.percentile_ci(te_calculator, source, target, observed_te)
            }
            BootstrapMethod::BCa => {
                self.bca_ci(te_calculator, source, target, observed_te)
            }
            BootstrapMethod::Block => {
                self.block_bootstrap_ci(te_calculator, source, target, observed_te)
            }
        }
    }

    /// Standard percentile bootstrap
    fn percentile_ci<F>(
        &self,
        te_calculator: F,
        source: &Array1<f64>,
        target: &Array1<f64>,
        observed_te: f64,
    ) -> Result<BootstrapCi>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
    {
        let n = source.len();
        let mut bootstrap_estimates = Vec::with_capacity(self.n_bootstrap);

        // Generate bootstrap samples
        for seed in 0..self.n_bootstrap {
            let mut rng = StdRng::seed_from_u64(seed as u64);

            // Resample with replacement
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let source_boot = Self::resample(source, &indices);
            let target_boot = Self::resample(target, &indices);

            // Calculate TE on bootstrap sample
            if let Ok(te_boot) = te_calculator(&source_boot, &target_boot) {
                bootstrap_estimates.push(te_boot);
            }
        }

        // Sort estimates
        bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentile bounds
        let alpha = 1.0 - self.confidence_level;
        let lower_idx = (bootstrap_estimates.len() as f64 * alpha / 2.0) as usize;
        let upper_idx = (bootstrap_estimates.len() as f64 * (1.0 - alpha / 2.0)) as usize;

        let lower = bootstrap_estimates[lower_idx.min(bootstrap_estimates.len() - 1)];
        let upper = bootstrap_estimates[upper_idx.min(bootstrap_estimates.len() - 1)];

        Ok(BootstrapCi {
            estimate: observed_te,
            lower,
            upper,
            confidence_level: self.confidence_level,
            method: BootstrapMethod::Percentile,
            n_bootstrap: self.n_bootstrap,
        })
    }

    /// Bias-corrected and accelerated (BCa) bootstrap
    ///
    /// Provides better coverage than percentile method, especially for
    /// skewed distributions and small samples.
    fn bca_ci<F>(
        &self,
        te_calculator: F,
        source: &Array1<f64>,
        target: &Array1<f64>,
        observed_te: f64,
    ) -> Result<BootstrapCi>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
    {
        let n = source.len();
        let mut bootstrap_estimates = Vec::with_capacity(self.n_bootstrap);

        // Generate bootstrap samples
        for seed in 0..self.n_bootstrap {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let source_boot = Self::resample(source, &indices);
            let target_boot = Self::resample(target, &indices);

            if let Ok(te_boot) = te_calculator(&source_boot, &target_boot) {
                bootstrap_estimates.push(te_boot);
            }
        }

        bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate bias-correction factor (z0)
        let count_less = bootstrap_estimates.iter().filter(|&&x| x < observed_te).count();
        let proportion = count_less as f64 / bootstrap_estimates.len() as f64;

        let z0 = if proportion > 0.0 && proportion < 1.0 {
            Self::inverse_normal_cdf(proportion)
        } else {
            0.0
        };

        // Calculate acceleration factor (a) using jackknife
        let acceleration = self.calculate_acceleration(&te_calculator, source, target, observed_te)?;

        // Calculate adjusted percentiles
        let alpha = 1.0 - self.confidence_level;
        let z_alpha_lower = Self::inverse_normal_cdf(alpha / 2.0);
        let z_alpha_upper = Self::inverse_normal_cdf(1.0 - alpha / 2.0);

        // BCa adjusted percentiles
        let p_lower = Self::normal_cdf(z0 + (z0 + z_alpha_lower) / (1.0 - acceleration * (z0 + z_alpha_lower)));
        let p_upper = Self::normal_cdf(z0 + (z0 + z_alpha_upper) / (1.0 - acceleration * (z0 + z_alpha_upper)));

        // Clamp to [0, 1]
        let p_lower = p_lower.max(0.0).min(1.0);
        let p_upper = p_upper.max(0.0).min(1.0);

        let lower_idx = (bootstrap_estimates.len() as f64 * p_lower) as usize;
        let upper_idx = (bootstrap_estimates.len() as f64 * p_upper) as usize;

        let lower = bootstrap_estimates[lower_idx.min(bootstrap_estimates.len() - 1)];
        let upper = bootstrap_estimates[upper_idx.min(bootstrap_estimates.len() - 1)];

        Ok(BootstrapCi {
            estimate: observed_te,
            lower,
            upper,
            confidence_level: self.confidence_level,
            method: BootstrapMethod::BCa,
            n_bootstrap: self.n_bootstrap,
        })
    }

    /// Block bootstrap for time series
    ///
    /// Preserves temporal structure by resampling blocks instead of individual points
    fn block_bootstrap_ci<F>(
        &self,
        te_calculator: F,
        source: &Array1<f64>,
        target: &Array1<f64>,
        observed_te: f64,
    ) -> Result<BootstrapCi>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
    {
        let n = source.len();
        let n_blocks = n / self.block_size;
        let mut bootstrap_estimates = Vec::with_capacity(self.n_bootstrap);

        // Generate block bootstrap samples
        for seed in 0..self.n_bootstrap {
            let mut rng = StdRng::seed_from_u64(seed as u64);

            // Resample blocks
            let mut source_boot = Vec::new();
            let mut target_boot = Vec::new();

            for _ in 0..n_blocks {
                let block_start = rng.gen_range(0..=(n - self.block_size));

                for offset in 0..self.block_size {
                    source_boot.push(source[block_start + offset]);
                    target_boot.push(target[block_start + offset]);
                }
            }

            let source_array = Array1::from_vec(source_boot);
            let target_array = Array1::from_vec(target_boot);

            if let Ok(te_boot) = te_calculator(&source_array, &target_array) {
                bootstrap_estimates.push(te_boot);
            }
        }

        bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Percentile bounds
        let alpha = 1.0 - self.confidence_level;
        let lower_idx = (bootstrap_estimates.len() as f64 * alpha / 2.0) as usize;
        let upper_idx = (bootstrap_estimates.len() as f64 * (1.0 - alpha / 2.0)) as usize;

        let lower = bootstrap_estimates[lower_idx.min(bootstrap_estimates.len() - 1)];
        let upper = bootstrap_estimates[upper_idx.min(bootstrap_estimates.len() - 1)];

        Ok(BootstrapCi {
            estimate: observed_te,
            lower,
            upper,
            confidence_level: self.confidence_level,
            method: BootstrapMethod::Block,
            n_bootstrap: self.n_bootstrap,
        })
    }

    /// Calculate acceleration factor using jackknife
    fn calculate_acceleration<F>(
        &self,
        te_calculator: &F,
        source: &Array1<f64>,
        target: &Array1<f64>,
        _observed_te: f64,
    ) -> Result<f64>
    where
        F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
    {
        let n = source.len();
        let mut jackknife_estimates = Vec::with_capacity(n);

        // Leave-one-out estimates
        for i in 0..n {
            let indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();

            let source_jack = Self::resample(source, &indices);
            let target_jack = Self::resample(target, &indices);

            if let Ok(te_jack) = te_calculator(&source_jack, &target_jack) {
                jackknife_estimates.push(te_jack);
            }
        }

        if jackknife_estimates.is_empty() {
            return Ok(0.0);
        }

        // Mean of jackknife estimates
        let mean_jack = jackknife_estimates.iter().sum::<f64>() / jackknife_estimates.len() as f64;

        // Calculate acceleration
        let numerator: f64 = jackknife_estimates
            .iter()
            .map(|&jack| (mean_jack - jack).powi(3))
            .sum();

        let denominator: f64 = jackknife_estimates
            .iter()
            .map(|&jack| (mean_jack - jack).powi(2))
            .sum();

        let acceleration = if denominator > 1e-10 {
            numerator / (6.0 * denominator.powf(1.5))
        } else {
            0.0
        };

        Ok(acceleration)
    }

    /// Resample array with given indices
    fn resample(data: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        let resampled: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
        Array1::from_vec(resampled)
    }

    /// Inverse normal CDF (probit function)
    ///
    /// Approximation for standard normal quantile function
    fn inverse_normal_cdf(p: f64) -> f64 {
        // Beasley-Springer-Moro algorithm
        const A: [f64; 4] = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
        const B: [f64; 4] = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833];
        const C: [f64; 9] = [
            0.3374754822726147,
            0.9761690190917186,
            0.1607979714918209,
            0.0276438810333863,
            0.0038405729373609,
            0.0003951896511919,
            0.0000321767881768,
            0.0000002888167364,
            0.0000003960315187,
        ];

        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        let y = p - 0.5;

        if y.abs() < 0.42 {
            // Central region
            let r = y * y;
            let num = ((A[3] * r + A[2]) * r + A[1]) * r + A[0];
            let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
            return y * num / den;
        }

        // Tail region
        let r = if y < 0.0 { p } else { 1.0 - p };
        let s = (-r.ln()).sqrt();
        let t = s - C[4];

        let num = (((((((C[8] * t + C[7]) * t + C[6]) * t + C[5]) * t + C[4]) * t + C[3]) * t + C[2]) * t + C[1]) * t + C[0];

        if y < 0.0 { -num } else { num }
    }

    /// Normal CDF
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-x * x).exp();

        sign * y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_percentile() {
        let resampler = BootstrapResampler {
            n_bootstrap: 100,
            confidence_level: 0.95,
            method: BootstrapMethod::Percentile,
            ..Default::default()
        };

        // Simple TE calculator (mock)
        let te_calc = |_source: &Array1<f64>, _target: &Array1<f64>| -> Result<f64> {
            Ok(0.5) // Mock TE value
        };

        let source = Array1::linspace(0.0, 10.0, 100);
        let target = source.mapv(|x: f64| x.sin());

        let ci = resampler.calculate_ci(te_calc, &source, &target, 0.5).unwrap();

        assert!(ci.lower <= ci.estimate);
        assert!(ci.estimate <= ci.upper);
        assert_eq!(ci.confidence_level, 0.95);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Test known values
        let z_025 = BootstrapResampler::inverse_normal_cdf(0.025);
        let z_975 = BootstrapResampler::inverse_normal_cdf(0.975);

        // z_0.025 ≈ -1.96, z_0.975 ≈ 1.96
        assert!((z_025 + 1.96).abs() < 0.01);
        assert!((z_975 - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        // Φ(0) = 0.5
        assert!((BootstrapResampler::normal_cdf(0.0) - 0.5).abs() < 0.001);

        // Φ(1.96) ≈ 0.975
        assert!((BootstrapResampler::normal_cdf(1.96) - 0.975).abs() < 0.01);
    }

    #[test]
    fn test_block_bootstrap() {
        let resampler = BootstrapResampler {
            n_bootstrap: 50,
            confidence_level: 0.95,
            method: BootstrapMethod::Block,
            block_size: 5,
        };

        let te_calc = |_source: &Array1<f64>, _target: &Array1<f64>| -> Result<f64> {
            Ok(0.3)
        };

        let source = Array1::linspace(0.0, 10.0, 100);
        let target = source.mapv(|x: f64| x.sin());

        let ci = resampler.calculate_ci(te_calc, &source, &target, 0.3).unwrap();

        assert!(ci.lower <= ci.estimate);
        assert!(ci.estimate <= ci.upper);
        assert_eq!(ci.method, BootstrapMethod::Block);
    }

    #[test]
    fn test_bca_bootstrap() {
        let resampler = BootstrapResampler {
            n_bootstrap: 50,
            confidence_level: 0.95,
            method: BootstrapMethod::BCa,
            ..Default::default()
        };

        let te_calc = |source: &Array1<f64>, _target: &Array1<f64>| -> Result<f64> {
            // Simple statistic for testing
            Ok(source.mean().unwrap())
        };

        let source = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let target = Array1::from_vec((0..50).map(|i| (i as f64).sin()).collect());

        let ci = resampler.calculate_ci(te_calc, &source, &target, 24.5).unwrap();

        println!("BCa CI: [{}, {}]", ci.lower, ci.upper);

        assert!(ci.lower <= ci.estimate);
        assert!(ci.estimate <= ci.upper);
        assert_eq!(ci.method, BootstrapMethod::BCa);
    }
}
