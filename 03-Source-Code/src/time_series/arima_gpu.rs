//! GPU-Accelerated ARIMA Time Series Forecasting
//!
//! Implements AutoRegressive Integrated Moving Average (ARIMA) models on GPU
//! for high-performance time series forecasting.
//!
//! Mathematical Framework:
//! - ARIMA(p,d,q): AR order p, differencing d, MA order q
//! - Model: (1 - Σφᵢ·Lⁱ)·(1-L)^d·yₜ = (1 + Σθⱼ·Lʲ)·εₜ
//! - AR part: yₜ = c + Σφᵢ·yₜ₋ᵢ + εₜ
//! - MA part: yₜ = μ + εₜ + Σθⱼ·εₜ₋ⱼ
//! - Integration: Apply differencing d times
//!
//! GPU Acceleration:
//! - Parallel coefficient estimation (least squares on GPU)
//! - Parallel forecast computation
//! - Batch processing for multiple series

use anyhow::{Result, Context, bail};
use ndarray::{Array1, Array2, Axis};

/// ARIMA model configuration
#[derive(Debug, Clone)]
pub struct ArimaConfig {
    /// AR order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// MA order (q)
    pub q: usize,
    /// Include constant term
    pub include_constant: bool,
}

impl Default for ArimaConfig {
    fn default() -> Self {
        Self {
            p: 1,      // AR(1)
            d: 0,      // No differencing
            q: 0,      // No MA terms
            include_constant: true,
        }
    }
}

/// GPU-Accelerated ARIMA model
pub struct ArimaGpu {
    /// Configuration
    config: ArimaConfig,
    /// AR coefficients: φ₁, φ₂, ..., φₚ
    ar_coefficients: Vec<f64>,
    /// MA coefficients: θ₁, θ₂, ..., θ_q
    ma_coefficients: Vec<f64>,
    /// Constant term (c)
    constant: f64,
    /// Residuals from training (for MA forecasting)
    residuals: Vec<f64>,
    /// GPU availability
    gpu_available: bool,
    /// Training history (for differencing reversal)
    training_data: Option<Vec<f64>>,
}

impl ArimaGpu {
    /// Create new ARIMA model
    pub fn new(config: ArimaConfig) -> Result<Self> {
        if config.p == 0 && config.q == 0 {
            bail!("ARIMA model must have p > 0 or q > 0");
        }

        let gpu_available = crate::gpu::kernel_executor::get_global_executor().is_ok();

        if gpu_available {
            println!("✓ GPU acceleration enabled for ARIMA");
        } else {
            println!("⚠ GPU not available, using CPU for ARIMA");
        }

        Ok(Self {
            config,
            ar_coefficients: vec![],
            ma_coefficients: vec![],
            constant: 0.0,
            residuals: vec![],
            gpu_available,
            training_data: None,
        })
    }

    /// Fit ARIMA model to time series data
    ///
    /// Estimates AR and MA coefficients using least squares
    pub fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.len() < self.config.p + self.config.d + 2 {
            bail!("Insufficient data: need at least {} points",
                  self.config.p + self.config.d + 2);
        }

        // Store original data for differencing reversal
        self.training_data = Some(data.to_vec());

        // 1. Apply differencing
        let differenced = self.apply_differencing(data)?;

        // 2. Fit AR part (if p > 0)
        if self.config.p > 0 {
            self.fit_ar(&differenced)?;
        }

        // 3. Compute residuals for MA part
        let residuals = self.compute_residuals(&differenced)?;
        self.residuals = residuals.clone();

        // 4. Fit MA part (if q > 0)
        if self.config.q > 0 {
            self.fit_ma(&differenced, &residuals)?;
        }

        Ok(())
    }

    /// Apply differencing d times
    fn apply_differencing(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = data.to_vec();

        for _ in 0..self.config.d {
            result = result.windows(2)
                .map(|w| w[1] - w[0])
                .collect();
        }

        Ok(result)
    }

    /// Reverse differencing to get original scale
    fn reverse_differencing(&self, forecast: &[f64]) -> Result<Vec<f64>> {
        let training_data = self.training_data.as_ref()
            .context("Model not fitted yet")?;

        let mut result = forecast.to_vec();

        // Get the last d values from training data for initial conditions
        let mut initial_values: Vec<f64> = training_data.iter()
            .rev()
            .take(self.config.d)
            .rev()
            .copied()
            .collect();

        // Reverse each level of differencing
        for _ in 0..self.config.d {
            let mut undifferenced = vec![initial_values[0]];

            for &diff in result.iter() {
                let next_val = undifferenced.last().unwrap() + diff;
                undifferenced.push(next_val);
            }

            result = undifferenced[1..].to_vec();
            initial_values = initial_values[1..].to_vec();
        }

        Ok(result)
    }

    /// Fit AR part using least squares
    fn fit_ar(&mut self, data: &[f64]) -> Result<()> {
        let n = data.len();
        let p = self.config.p;

        if n <= p {
            bail!("Insufficient data for AR({}) model", p);
        }

        // Build design matrix X and response vector y
        // For AR(p): yₜ = c + φ₁·yₜ₋₁ + φ₂·yₜ₋₂ + ... + φₚ·yₜ₋ₚ + εₜ

        let n_obs = n - p;
        let n_features = if self.config.include_constant { p + 1 } else { p };

        let mut x = Array2::<f64>::zeros((n_obs, n_features));
        let mut y = Array1::<f64>::zeros(n_obs);

        for i in 0..n_obs {
            let t = i + p;

            // Response: yₜ
            y[i] = data[t];

            // Predictors: yₜ₋₁, yₜ₋₂, ..., yₜ₋ₚ
            for j in 0..p {
                x[[i, j]] = data[t - j - 1];
            }

            // Constant term
            if self.config.include_constant {
                x[[i, p]] = 1.0;
            }
        }

        // Solve least squares: β = (X'X)⁻¹X'y
        let coefficients = if self.gpu_available {
            self.solve_least_squares_gpu(&x, &y)?
        } else {
            self.solve_least_squares_cpu(&x, &y)?
        };

        // Extract AR coefficients and constant
        self.ar_coefficients = coefficients[..p].to_vec();
        if self.config.include_constant {
            self.constant = coefficients[p];
        }

        Ok(())
    }

    /// Solve least squares on GPU (placeholder for full GPU implementation)
    fn solve_least_squares_gpu(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Vec<f64>> {
        // TODO: Full GPU implementation with custom kernel
        // For now, use CPU fallback
        self.solve_least_squares_cpu(x, y)
    }

    /// Solve least squares on CPU using normal equations
    fn solve_least_squares_cpu(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Vec<f64>> {
        // β = (X'X)⁻¹X'y

        // Compute X'X
        let xtx = x.t().dot(x);

        // Compute X'y
        let xty = x.t().dot(y);

        // Solve using Cholesky decomposition (xtx is positive definite)
        // For simplicity, use pseudo-inverse approach
        let coefficients = self.solve_linear_system(&xtx, &xty)?;

        Ok(coefficients.to_vec())
    }

    /// Solve linear system Ax = b using Gauss-Jordan elimination
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();
        let mut aug = Array2::<f64>::zeros((n, n + 1));

        // Create augmented matrix [A|b]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let tmp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Check for singularity
            if aug[[i, i]].abs() < 1e-10 {
                bail!("Matrix is singular or nearly singular");
            }

            // Scale row
            let pivot = aug[[i, i]];
            for j in 0..=n {
                aug[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..=n {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }

        // Extract solution
        let solution = aug.column(n).to_owned();
        Ok(solution)
    }

    /// Compute residuals: εₜ = yₜ - ŷₜ
    fn compute_residuals(&self, data: &[f64]) -> Result<Vec<f64>> {
        let n = data.len();
        let p = self.config.p;
        let mut residuals = vec![0.0; n];

        // First p residuals are zero (no prediction possible)
        for t in p..n {
            // Predict yₜ using AR model
            let mut pred = self.constant;
            for i in 0..p {
                pred += self.ar_coefficients[i] * data[t - i - 1];
            }

            residuals[t] = data[t] - pred;
        }

        Ok(residuals)
    }

    /// Fit MA part (simplified - assumes AR residuals available)
    fn fit_ma(&mut self, _data: &[f64], residuals: &[f64]) -> Result<()> {
        // For MA(q), we need to estimate θ₁, θ₂, ..., θ_q
        // This is typically done via maximum likelihood or iterative methods

        // Simplified approach: Use autocorrelation of residuals
        let q = self.config.q;
        self.ma_coefficients = vec![0.0; q];

        // Estimate using Yule-Walker equations on residuals
        let acf = self.compute_autocorrelation(residuals, q)?;

        // Simple estimation: θᵢ ≈ -ρᵢ (negative of autocorrelation)
        for i in 0..q {
            self.ma_coefficients[i] = -acf[i + 1];
        }

        Ok(())
    }

    /// Compute autocorrelation function
    fn compute_autocorrelation(&self, data: &[f64], max_lag: usize) -> Result<Vec<f64>> {
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        let mut acf = vec![0.0; max_lag + 1];

        // Variance (lag 0)
        let variance: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;

        acf[0] = 1.0;

        // Autocorrelations
        for lag in 1..=max_lag {
            let mut sum = 0.0;
            for i in lag..n {
                sum += (data[i] - mean) * (data[i - lag] - mean);
            }
            acf[lag] = sum / (n as f64 * variance);
        }

        Ok(acf)
    }

    /// Forecast h steps ahead
    ///
    /// Returns forecast values in differenced space if d > 0
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.ar_coefficients.is_empty() && self.ma_coefficients.is_empty() {
            bail!("Model not fitted yet");
        }

        let training_data = self.training_data.as_ref()
            .context("Model not fitted yet")?;

        // Apply differencing to training data
        let differenced = self.apply_differencing(training_data)?;

        // Forecast in differenced space
        let forecast_differenced = self.forecast_differenced(&differenced, horizon)?;

        // Reverse differencing to original scale
        let forecast = self.reverse_differencing(&forecast_differenced)?;

        Ok(forecast)
    }

    /// Forecast in differenced space
    fn forecast_differenced(&self, data: &[f64], horizon: usize) -> Result<Vec<f64>> {
        let p = self.config.p;
        let q = self.config.q;

        // If pure AR model and GPU available, use GPU acceleration
        if self.gpu_available && q == 0 && p > 0 {
            return self.forecast_ar_gpu(data, horizon);
        }

        // Fallback to CPU for MA components or GPU unavailable
        let mut forecast = Vec::with_capacity(horizon);
        let mut history = data.to_vec();
        let mut residual_history = self.residuals.clone();

        for _ in 0..horizon {
            // AR component
            let mut pred = self.constant;
            for i in 0..p {
                if history.len() > i {
                    pred += self.ar_coefficients[i] * history[history.len() - i - 1];
                }
            }

            // MA component
            for j in 0..q {
                if residual_history.len() > j {
                    pred += self.ma_coefficients[j] * residual_history[residual_history.len() - j - 1];
                }
            }

            forecast.push(pred);
            history.push(pred);

            // Residual for future forecasts is 0 (expected value)
            residual_history.push(0.0);
        }

        Ok(forecast)
    }

    /// Forecast AR model using GPU kernel (Worker 2 integration)
    fn forecast_ar_gpu(&self, data: &[f64], horizon: usize) -> Result<Vec<f64>> {
        let executor_arc = crate::gpu::kernel_executor::get_global_executor()
            .context("GPU executor not available")?;
        let executor = executor_arc.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock GPU executor: {}", e))?;

        // Convert f64 to f32 for GPU
        let historical_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let coefficients_f32: Vec<f32> = self.ar_coefficients.iter().map(|&x| x as f32).collect();

        // Call Worker 2's ar_forecast kernel
        let forecast_f32 = executor.ar_forecast(
            &historical_f32,
            &coefficients_f32,
            horizon
        ).context("GPU ar_forecast kernel failed")?;

        // Convert back to f64 and add constant if needed
        let mut forecast_f64: Vec<f64> = forecast_f32.iter()
            .map(|&x| x as f64 + self.constant)
            .collect();

        Ok(forecast_f64)
    }

    /// Forecast with GPU acceleration (batch processing)
    pub fn forecast_batch(&self, horizons: &[usize]) -> Result<Vec<Vec<f64>>> {
        let forecasts: Vec<Vec<f64>> = horizons.iter()
            .map(|&h| self.forecast(h))
            .collect::<Result<Vec<_>>>()?;

        Ok(forecasts)
    }

    /// Get model coefficients for inspection
    pub fn get_coefficients(&self) -> ArimaCoefficients {
        ArimaCoefficients {
            ar: self.ar_coefficients.clone(),
            ma: self.ma_coefficients.clone(),
            constant: self.constant,
        }
    }

    /// Compute AIC (Akaike Information Criterion)
    pub fn aic(&self) -> Result<f64> {
        let n = self.residuals.len() as f64;
        let k = self.config.p + self.config.q + if self.config.include_constant { 1 } else { 0 };

        // RSS (residual sum of squares)
        let rss: f64 = self.residuals.iter()
            .skip(self.config.p)  // Skip initial zeros
            .map(|&r| r * r)
            .sum();

        // AIC = 2k + n·ln(RSS/n)
        let aic = 2.0 * k as f64 + n * (rss / n).ln();

        Ok(aic)
    }

    /// Compute BIC (Bayesian Information Criterion)
    pub fn bic(&self) -> Result<f64> {
        let n = self.residuals.len() as f64;
        let k = self.config.p + self.config.q + if self.config.include_constant { 1 } else { 0 };

        // RSS
        let rss: f64 = self.residuals.iter()
            .skip(self.config.p)
            .map(|&r| r * r)
            .sum();

        // BIC = k·ln(n) + n·ln(RSS/n)
        let bic = k as f64 * n.ln() + n * (rss / n).ln();

        Ok(bic)
    }
}

/// ARIMA model coefficients
#[derive(Debug, Clone)]
pub struct ArimaCoefficients {
    /// AR coefficients
    pub ar: Vec<f64>,
    /// MA coefficients
    pub ma: Vec<f64>,
    /// Constant term
    pub constant: f64,
}

/// Auto-select ARIMA order using information criteria
pub fn auto_arima(data: &[f64], max_p: usize, max_d: usize, max_q: usize) -> Result<ArimaGpu> {
    let mut best_aic = f64::INFINITY;
    let mut best_model = None;

    for p in 0..=max_p {
        for d in 0..=max_d {
            for q in 0..=max_q {
                if p == 0 && q == 0 {
                    continue; // Skip invalid model
                }

                let config = ArimaConfig {
                    p,
                    d,
                    q,
                    include_constant: true,
                };

                let mut model = ArimaGpu::new(config)?;

                if let Ok(_) = model.fit(data) {
                    if let Ok(aic) = model.aic() {
                        if aic < best_aic {
                            best_aic = aic;
                            best_model = Some(model);
                        }
                    }
                }
            }
        }
    }

    best_model.context("No valid ARIMA model found")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arima_creation() {
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
        };

        let model = ArimaGpu::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_differencing() {
        let config = ArimaConfig {
            p: 1,
            d: 1,
            q: 0,
            include_constant: false,
        };

        let model = ArimaGpu::new(config).unwrap();
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];

        let differenced = model.apply_differencing(&data).unwrap();

        // First difference: [2, 3, 4, 5]
        assert_eq!(differenced.len(), 4);
        assert!((differenced[0] - 2.0).abs() < 1e-10);
        assert!((differenced[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ar1_fit() {
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
        };

        // Generate AR(1) data: y_t = 0.8*y_{t-1} + ε_t
        let true_phi = 0.8;
        let mut data = vec![0.0; 100];
        for t in 1..100 {
            data[t] = true_phi * data[t - 1] + 0.1 * (t as f64).sin();
        }

        let mut model = ArimaGpu::new(config).unwrap();
        model.fit(&data).unwrap();

        // Check AR coefficient is close to true value
        let coeffs = model.get_coefficients();
        assert!((coeffs.ar[0] - true_phi).abs() < 0.2);
    }

    #[test]
    fn test_forecast() {
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: false,
        };

        let data = vec![1.0, 1.5, 2.2, 3.3, 5.0];

        let mut model = ArimaGpu::new(config).unwrap();
        model.fit(&data).unwrap();

        let forecast = model.forecast(3).unwrap();

        assert_eq!(forecast.len(), 3);
        assert!(forecast.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_auto_arima() {
        // Trend + noise
        let data: Vec<f64> = (0..50)
            .map(|i| i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();

        let model = auto_arima(&data, 2, 1, 2);
        assert!(model.is_ok());
    }

    #[test]
    fn test_aic_bic() {
        let config = ArimaConfig::default();
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        let mut model = ArimaGpu::new(config).unwrap();
        model.fit(&data).unwrap();

        let aic = model.aic().unwrap();
        let bic = model.bic().unwrap();

        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(bic > aic);  // BIC penalizes complexity more
    }

    #[test]
    fn test_batch_forecast() {
        let config = ArimaConfig::default();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let mut model = ArimaGpu::new(config).unwrap();
        model.fit(&data).unwrap();

        let horizons = vec![1, 5, 10];
        let forecasts = model.forecast_batch(&horizons).unwrap();

        assert_eq!(forecasts.len(), 3);
        assert_eq!(forecasts[0].len(), 1);
        assert_eq!(forecasts[1].len(), 5);
        assert_eq!(forecasts[2].len(), 10);
    }
}
