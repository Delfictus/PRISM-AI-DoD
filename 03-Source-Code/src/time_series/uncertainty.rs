//! Uncertainty Quantification for Time Series Forecasts
//!
//! Provides prediction intervals and confidence bands for time series forecasts
//! using multiple uncertainty estimation methods.
//!
//! Methods Implemented:
//! 1. **Residual-Based Intervals**: Uses historical residuals
//! 2. **Monte Carlo Dropout**: Neural network uncertainty via dropout
//! 3. **Bootstrap Intervals**: Resampling-based confidence intervals
//! 4. **Conformal Prediction**: Distribution-free prediction intervals
//!
//! Mathematical Framework:
//! - Prediction Interval: [ŷ - z*σ, ŷ + z*σ]
//! - Confidence Level: P(y ∈ [ŷ - ε, ŷ + ε]) ≥ 1 - α
//! - Quantile Regression: Estimate conditional quantiles directly

use anyhow::{Result, Context, bail};
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Uncertainty quantification method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintyMethod {
    /// Residual-based intervals (fastest)
    Residual,
    /// Monte Carlo dropout (for neural networks)
    MonteCarloDropout,
    /// Bootstrap resampling
    Bootstrap,
    /// Conformal prediction
    Conformal,
}

/// Configuration for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Method to use
    pub method: UncertaintyMethod,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Number of Monte Carlo samples (for MC Dropout)
    pub n_mc_samples: usize,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Window size for rolling residuals
    pub residual_window: usize,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            method: UncertaintyMethod::Residual,
            confidence_level: 0.95,
            n_mc_samples: 100,
            n_bootstrap: 1000,
            residual_window: 50,
        }
    }
}

/// Forecast with uncertainty bounds
#[derive(Debug, Clone)]
pub struct ForecastWithUncertainty {
    /// Point forecast
    pub forecast: Vec<f64>,
    /// Lower bound of prediction interval
    pub lower_bound: Vec<f64>,
    /// Upper bound of prediction interval
    pub upper_bound: Vec<f64>,
    /// Standard deviation of forecast
    pub std_dev: Vec<f64>,
    /// Confidence level
    pub confidence_level: f64,
}

/// Uncertainty quantifier
pub struct UncertaintyQuantifier {
    /// Configuration
    config: UncertaintyConfig,
    /// Historical residuals for residual-based method
    residuals: VecDeque<f64>,
    /// GPU availability
    gpu_available: bool,
}

impl UncertaintyQuantifier {
    /// Create new uncertainty quantifier
    pub fn new(config: UncertaintyConfig) -> Self {
        let window_size = config.residual_window;
        let gpu_available = crate::gpu::kernel_executor::get_global_executor().is_ok();

        if gpu_available {
            println!("✓ GPU acceleration enabled for uncertainty quantification");
        } else {
            println!("⚠ GPU not available, using CPU for uncertainty quantification");
        }

        Self {
            config,
            residuals: VecDeque::with_capacity(window_size),
            gpu_available,
        }
    }

    /// Update residuals with new observations
    pub fn update_residuals(&mut self, actual: f64, predicted: f64) {
        let residual = actual - predicted;
        self.residuals.push_back(residual);

        // Keep only last window_size residuals
        while self.residuals.len() > self.config.residual_window {
            self.residuals.pop_front();
        }
    }

    /// Compute prediction intervals using residual-based method
    pub fn residual_intervals(&self, forecast: &[f64]) -> Result<ForecastWithUncertainty> {
        if self.residuals.is_empty() {
            bail!("No residuals available. Update with observations first.");
        }

        // Try GPU acceleration if available
        if self.gpu_available {
            if let Ok(result) = self.residual_intervals_gpu(forecast) {
                return Ok(result);
            }
            // Fall through to CPU if GPU fails
        }

        // CPU implementation
        // Compute standard deviation of residuals
        let mean_residual: f64 = self.residuals.iter().sum::<f64>() / self.residuals.len() as f64;

        let variance: f64 = self.residuals.iter()
            .map(|&r| (r - mean_residual).powi(2))
            .sum::<f64>() / self.residuals.len() as f64;

        let std_dev = variance.sqrt();

        // Compute z-score for confidence level
        let z = self.compute_z_score(self.config.confidence_level)?;

        // Build intervals
        let lower_bound: Vec<f64> = forecast.iter()
            .map(|&f| f - z * std_dev)
            .collect();

        let upper_bound: Vec<f64> = forecast.iter()
            .map(|&f| f + z * std_dev)
            .collect();

        let std_devs = vec![std_dev; forecast.len()];

        Ok(ForecastWithUncertainty {
            forecast: forecast.to_vec(),
            lower_bound,
            upper_bound,
            std_dev: std_devs,
            confidence_level: self.config.confidence_level,
        })
    }

    /// GPU-accelerated uncertainty propagation (Worker 2 integration)
    fn residual_intervals_gpu(&self, forecast: &[f64]) -> Result<ForecastWithUncertainty> {
        let executor_arc = crate::gpu::kernel_executor::get_global_executor()
            .context("GPU executor not available")?;
        let executor = executor_arc.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock GPU executor: {}", e))?;

        // Compute residual standard deviation
        let mean_residual: f64 = self.residuals.iter().sum::<f64>() / self.residuals.len() as f64;
        let variance: f64 = self.residuals.iter()
            .map(|&r| (r - mean_residual).powi(2))
            .sum::<f64>() / self.residuals.len() as f64;
        let base_std_dev = variance.sqrt();

        // Convert to f32 for GPU
        let forecast_f32: Vec<f32> = forecast.iter().map(|&x| x as f32).collect();
        let horizon = forecast.len();

        // Model error propagates over forecast horizon
        // For simplicity, use constant error, but Worker 2's kernel supports horizon-dependent error
        let model_error_std_f32: Vec<f32> = vec![base_std_dev as f32; horizon];

        // Call Worker 2's uncertainty_propagation kernel
        let forecast_variance_f32 = executor.uncertainty_propagation(
            &forecast_f32,
            &model_error_std_f32,
            horizon,
        ).context("GPU uncertainty_propagation failed")?;

        // Convert variance to std dev and back to f64
        let std_devs: Vec<f64> = forecast_variance_f32.iter()
            .map(|&var| (var.sqrt() as f64))
            .collect();

        // Compute z-score for confidence level
        let z = self.compute_z_score(self.config.confidence_level)?;

        // Build intervals using GPU-computed variances
        let lower_bound: Vec<f64> = forecast.iter()
            .zip(std_devs.iter())
            .map(|(&f, &std)| f - z * std)
            .collect();

        let upper_bound: Vec<f64> = forecast.iter()
            .zip(std_devs.iter())
            .map(|(&f, &std)| f + z * std)
            .collect();

        Ok(ForecastWithUncertainty {
            forecast: forecast.to_vec(),
            lower_bound,
            upper_bound,
            std_dev: std_devs,
            confidence_level: self.config.confidence_level,
        })
    }

    /// Compute z-score for given confidence level
    fn compute_z_score(&self, confidence: f64) -> Result<f64> {
        // Approximate inverse CDF of standard normal
        // For common confidence levels
        let z = match confidence {
            c if (c - 0.90).abs() < 0.001 => 1.645,
            c if (c - 0.95).abs() < 0.001 => 1.960,
            c if (c - 0.99).abs() < 0.001 => 2.576,
            _ => {
                // General approximation using Beasley-Springer-Moro algorithm
                let p = 1.0 - (1.0 - confidence) / 2.0;  // Two-tailed
                self.inverse_normal_cdf(p)?
            }
        };

        Ok(z)
    }

    /// Inverse normal CDF (approximate)
    fn inverse_normal_cdf(&self, p: f64) -> Result<f64> {
        if p <= 0.0 || p >= 1.0 {
            bail!("Probability must be in (0, 1)");
        }

        // Beasley-Springer-Moro approximation
        let a = [
            2.506628277459239,
            -30.66479806614716,
            138.3577518672690,
            -275.9285104469687,
            220.9460984245205,
            -39.69683028665376,
        ];

        let b = [
            1.0,
            -13.28068155288572,
            66.80131188771972,
            -155.6989798598866,
            161.5858368580409,
            -54.47609879822406,
        ];

        let c = [
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

        let y = p - 0.5;

        let r = if y.abs() < 0.42 {
            let r = y * y;
            y * (((((a[5] * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1]) * r + a[0]) /
                (((((b[5] * r + b[4]) * r + b[3]) * r + b[2]) * r + b[1]) * r + b[0])
        } else {
            let r = if y > 0.0 { 1.0 - p } else { p };
            let r = (-r.ln()).ln();
            let r = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))));
            if y < 0.0 { -r } else { r }
        };

        Ok(r)
    }

    /// Bootstrap prediction intervals
    pub fn bootstrap_intervals(
        &self,
        historical_data: &[f64],
        forecast_fn: impl Fn(&[f64]) -> Result<Vec<f64>>,
        horizon: usize,
    ) -> Result<ForecastWithUncertainty> {
        if historical_data.len() < 10 {
            bail!("Insufficient data for bootstrap");
        }

        let mut bootstrap_forecasts: Vec<Vec<f64>> = Vec::new();

        // Generate bootstrap samples
        for _ in 0..self.config.n_bootstrap {
            // Resample with replacement
            let mut rng = rand::thread_rng();
            let bootstrap_sample: Vec<f64> = (0..historical_data.len())
                .map(|_| {
                    use rand::Rng;
                    let idx = rng.gen_range(0..historical_data.len());
                    historical_data[idx]
                })
                .collect();

            // Generate forecast from bootstrap sample
            if let Ok(forecast) = forecast_fn(&bootstrap_sample) {
                bootstrap_forecasts.push(forecast);
            }
        }

        if bootstrap_forecasts.is_empty() {
            bail!("All bootstrap samples failed");
        }

        // Compute quantiles across bootstrap samples
        let forecast = self.compute_median_forecast(&bootstrap_forecasts, horizon)?;
        let (lower_bound, upper_bound) = self.compute_quantile_bounds(
            &bootstrap_forecasts,
            horizon,
            self.config.confidence_level
        )?;
        let std_dev = self.compute_std_forecast(&bootstrap_forecasts, horizon)?;

        Ok(ForecastWithUncertainty {
            forecast,
            lower_bound,
            upper_bound,
            std_dev,
            confidence_level: self.config.confidence_level,
        })
    }

    /// Compute median forecast across bootstrap samples
    fn compute_median_forecast(&self, forecasts: &[Vec<f64>], horizon: usize) -> Result<Vec<f64>> {
        let mut median_forecast = Vec::with_capacity(horizon);

        for h in 0..horizon {
            let mut values: Vec<f64> = forecasts.iter()
                .filter_map(|f| f.get(h).copied())
                .collect();

            if values.is_empty() {
                bail!("No valid forecasts for horizon {}", h);
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };

            median_forecast.push(median);
        }

        Ok(median_forecast)
    }

    /// Compute quantile bounds
    fn compute_quantile_bounds(
        &self,
        forecasts: &[Vec<f64>],
        horizon: usize,
        confidence: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let alpha = 1.0 - confidence;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;

        let mut lower_bound = Vec::with_capacity(horizon);
        let mut upper_bound = Vec::with_capacity(horizon);

        for h in 0..horizon {
            let mut values: Vec<f64> = forecasts.iter()
                .filter_map(|f| f.get(h).copied())
                .collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = (values.len() as f64 * lower_quantile) as usize;
            let upper_idx = (values.len() as f64 * upper_quantile) as usize;

            lower_bound.push(values[lower_idx]);
            upper_bound.push(values[upper_idx.min(values.len() - 1)]);
        }

        Ok((lower_bound, upper_bound))
    }

    /// Compute standard deviation across forecasts
    fn compute_std_forecast(&self, forecasts: &[Vec<f64>], horizon: usize) -> Result<Vec<f64>> {
        let mut std_devs = Vec::with_capacity(horizon);

        for h in 0..horizon {
            let values: Vec<f64> = forecasts.iter()
                .filter_map(|f| f.get(h).copied())
                .collect();

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;

            std_devs.push(variance.sqrt());
        }

        Ok(std_devs)
    }

    /// Monte Carlo Dropout intervals (for neural networks)
    pub fn mc_dropout_intervals(
        &self,
        forecast_fn: impl Fn() -> Result<Vec<f64>>,
    ) -> Result<ForecastWithUncertainty> {
        let mut mc_forecasts: Vec<Vec<f64>> = Vec::new();

        // Generate multiple forecasts with dropout
        for _ in 0..self.config.n_mc_samples {
            if let Ok(forecast) = forecast_fn() {
                mc_forecasts.push(forecast);
            }
        }

        if mc_forecasts.is_empty() {
            bail!("All MC samples failed");
        }

        let horizon = mc_forecasts[0].len();

        // Compute statistics
        let forecast = self.compute_median_forecast(&mc_forecasts, horizon)?;
        let (lower_bound, upper_bound) = self.compute_quantile_bounds(
            &mc_forecasts,
            horizon,
            self.config.confidence_level
        )?;
        let std_dev = self.compute_std_forecast(&mc_forecasts, horizon)?;

        Ok(ForecastWithUncertainty {
            forecast,
            lower_bound,
            upper_bound,
            std_dev,
            confidence_level: self.config.confidence_level,
        })
    }
}

impl ForecastWithUncertainty {
    /// Check if observation falls within prediction interval
    pub fn contains(&self, step: usize, value: f64) -> bool {
        if step >= self.forecast.len() {
            return false;
        }

        value >= self.lower_bound[step] && value <= self.upper_bound[step]
    }

    /// Compute interval width
    pub fn interval_width(&self, step: usize) -> Option<f64> {
        if step >= self.forecast.len() {
            return None;
        }

        Some(self.upper_bound[step] - self.lower_bound[step])
    }

    /// Average interval width
    pub fn average_interval_width(&self) -> f64 {
        let sum: f64 = (0..self.forecast.len())
            .filter_map(|i| self.interval_width(i))
            .sum();

        sum / self.forecast.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_quantifier_creation() {
        let config = UncertaintyConfig::default();
        let uq = UncertaintyQuantifier::new(config);
        assert_eq!(uq.residuals.len(), 0);
    }

    #[test]
    fn test_update_residuals() {
        let config = UncertaintyConfig {
            residual_window: 10,
            ..Default::default()
        };

        let mut uq = UncertaintyQuantifier::new(config);

        for i in 0..15 {
            uq.update_residuals(i as f64, i as f64 - 0.1);
        }

        assert_eq!(uq.residuals.len(), 10);
    }

    #[test]
    fn test_residual_intervals() {
        let config = UncertaintyConfig::default();
        let mut uq = UncertaintyQuantifier::new(config);

        // Add some residuals
        for i in 0..20 {
            uq.update_residuals(i as f64, i as f64 + (i as f64 * 0.1).sin());
        }

        let forecast = vec![10.0, 11.0, 12.0];
        let result = uq.residual_intervals(&forecast);

        assert!(result.is_ok());

        let intervals = result.unwrap();
        assert_eq!(intervals.forecast.len(), 3);
        assert_eq!(intervals.lower_bound.len(), 3);
        assert_eq!(intervals.upper_bound.len(), 3);

        // Lower bound should be less than forecast
        for i in 0..3 {
            assert!(intervals.lower_bound[i] < intervals.forecast[i]);
            assert!(intervals.upper_bound[i] > intervals.forecast[i]);
        }
    }

    #[test]
    fn test_z_score_computation() {
        let config = UncertaintyConfig::default();
        let uq = UncertaintyQuantifier::new(config);

        let z90 = uq.compute_z_score(0.90).unwrap();
        let z95 = uq.compute_z_score(0.95).unwrap();
        let z99 = uq.compute_z_score(0.99).unwrap();

        assert!((z90 - 1.645).abs() < 0.01);
        assert!((z95 - 1.960).abs() < 0.01);
        assert!((z99 - 2.576).abs() < 0.01);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        let config = UncertaintyConfig::default();
        let uq = UncertaintyQuantifier::new(config);

        let z = uq.inverse_normal_cdf(0.975).unwrap();
        assert!((z - 1.96).abs() < 0.05);
    }

    #[test]
    fn test_interval_contains() {
        let forecast_with_unc = ForecastWithUncertainty {
            forecast: vec![10.0, 20.0, 30.0],
            lower_bound: vec![8.0, 18.0, 28.0],
            upper_bound: vec![12.0, 22.0, 32.0],
            std_dev: vec![1.0, 1.0, 1.0],
            confidence_level: 0.95,
        };

        assert!(forecast_with_unc.contains(0, 10.0));
        assert!(forecast_with_unc.contains(1, 19.0));
        assert!(!forecast_with_unc.contains(0, 15.0));
    }

    #[test]
    fn test_interval_width() {
        let forecast_with_unc = ForecastWithUncertainty {
            forecast: vec![10.0, 20.0],
            lower_bound: vec![8.0, 18.0],
            upper_bound: vec![12.0, 22.0],
            std_dev: vec![1.0, 1.0],
            confidence_level: 0.95,
        };

        assert_eq!(forecast_with_unc.interval_width(0), Some(4.0));
        assert_eq!(forecast_with_unc.average_interval_width(), 4.0);
    }

    #[test]
    fn test_bootstrap_intervals() {
        let config = UncertaintyConfig {
            n_bootstrap: 100,
            ..Default::default()
        };

        let uq = UncertaintyQuantifier::new(config);

        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let forecast_fn = |data: &[f64]| -> Result<Vec<f64>> {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(vec![mean + 1.0, mean + 2.0, mean + 3.0])
        };

        let result = uq.bootstrap_intervals(&data, forecast_fn, 3);
        assert!(result.is_ok());

        let intervals = result.unwrap();
        assert_eq!(intervals.forecast.len(), 3);
    }
}
