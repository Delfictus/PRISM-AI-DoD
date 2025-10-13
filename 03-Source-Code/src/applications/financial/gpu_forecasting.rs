// GPU-Accelerated Time Series Forecasting for Financial Markets
// Integrates with Worker 2's time series GPU kernels
// Constitution: Financial Application + Full GPU Optimization

use anyhow::{Result, Context};
use ndarray::Array1;

#[cfg(feature = "cuda")]
use crate::gpu::kernel_executor::get_global_executor;

/// GPU-accelerated time series forecaster
///
/// Leverages Worker 2's time series GPU kernels:
/// - AR (Autoregressive) forecasting
/// - LSTM cell forward pass
/// - GRU cell forward pass
/// - Kalman filter step
/// - Uncertainty propagation
pub struct GpuTimeSeriesForecaster {
    /// Use GPU acceleration
    pub use_gpu: bool,

    /// Forecast method
    pub method: ForecastMethod,

    /// Forecast horizon
    pub horizon: usize,
}

/// Forecasting method
#[derive(Debug, Clone, Copy)]
pub enum ForecastMethod {
    /// Autoregressive (fast, simple patterns)
    AR { order: usize },

    /// LSTM (complex non-linear patterns)
    LSTM { hidden_size: usize },

    /// GRU (balanced between AR and LSTM)
    GRU { hidden_size: usize },

    /// Kalman Filter (optimal for linear systems with noise)
    Kalman,
}

/// Forecast result with uncertainty
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Forecasted values
    pub forecast: Vec<f64>,

    /// Uncertainty bounds (standard deviation)
    pub uncertainty: Vec<f64>,

    /// Lower confidence bound (95%)
    pub lower_bound: Vec<f64>,

    /// Upper confidence bound (95%)
    pub upper_bound: Vec<f64>,

    /// GPU execution time (ms)
    pub gpu_time_ms: f64,

    /// Whether GPU was used
    pub used_gpu: bool,
}

impl Default for GpuTimeSeriesForecaster {
    fn default() -> Self {
        Self {
            use_gpu: true,
            method: ForecastMethod::AR { order: 3 },
            horizon: 5,
        }
    }
}

impl GpuTimeSeriesForecaster {
    /// Create new GPU forecaster
    pub fn new(method: ForecastMethod, horizon: usize) -> Self {
        Self {
            use_gpu: true,
            method,
            horizon,
        }
    }

    /// Create AR forecaster
    pub fn ar(order: usize, horizon: usize) -> Self {
        Self::new(ForecastMethod::AR { order }, horizon)
    }

    /// Create LSTM forecaster
    pub fn lstm(hidden_size: usize, horizon: usize) -> Self {
        Self::new(ForecastMethod::LSTM { hidden_size }, horizon)
    }

    /// Create GRU forecaster
    pub fn gru(hidden_size: usize, horizon: usize) -> Self {
        Self::new(ForecastMethod::GRU { hidden_size }, horizon)
    }

    /// Forecast future values using GPU acceleration
    ///
    /// # Arguments
    /// * `historical` - Historical time series data
    ///
    /// # Returns
    /// Forecast with uncertainty bounds
    pub fn forecast(&self, historical: &Array1<f64>) -> Result<ForecastResult> {
        use std::time::Instant;

        let start = Instant::now();

        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                // Try GPU acceleration
                if let Ok(result) = self.forecast_gpu(historical) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        self.forecast_cpu(historical, start.elapsed().as_secs_f64() * 1000.0)
    }

    /// GPU implementation using Worker 2's time series kernels
    #[cfg(feature = "cuda")]
    fn forecast_gpu(&self, historical: &Array1<f64>) -> Result<ForecastResult> {
        use std::time::Instant;

        let start = Instant::now();

        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU
        let historical_f32: Vec<f32> = historical.iter().map(|&x| x as f32).collect();

        let forecast_f32 = match self.method {
            ForecastMethod::AR { order } => {
                // Estimate AR coefficients using Yule-Walker
                let coefficients = self.estimate_ar_coefficients(&historical_f32, order);

                // Use Worker 2's AR forecast kernel
                executor.ar_forecast(&historical_f32, &coefficients, self.horizon)?
            }

            ForecastMethod::LSTM { hidden_size } => {
                // Initialize LSTM state
                let mut hidden = vec![0.0f32; hidden_size];
                let mut cell = vec![0.0f32; hidden_size];

                // LSTM weights (in production, these would be trained)
                let input_size = 1;
                let weights_ih = vec![0.1f32; 4 * hidden_size * input_size];
                let weights_hh = vec![0.1f32; 4 * hidden_size * hidden_size];
                let bias = vec![0.0f32; 4 * hidden_size];

                let mut forecasts = Vec::new();

                // Warm up LSTM with historical data
                for &value in historical_f32.iter().rev().take(20).rev() {
                    let input = vec![value];

                    let (new_hidden, new_cell) = executor.lstm_cell_forward(
                        &input,
                        &hidden,
                        &cell,
                        &weights_ih,
                        &weights_hh,
                        &bias,
                        input_size,
                        hidden_size,
                    )?;

                    hidden = new_hidden;
                    cell = new_cell;
                }

                // Generate forecast
                let mut last_value = *historical_f32.last().unwrap();
                for _ in 0..self.horizon {
                    let input = vec![last_value];

                    let (new_hidden, new_cell) = executor.lstm_cell_forward(
                        &input,
                        &hidden,
                        &cell,
                        &weights_ih,
                        &weights_hh,
                        &bias,
                        input_size,
                        hidden_size,
                    )?;

                    // Use first hidden state element as forecast
                    last_value = new_hidden[0];
                    forecasts.push(last_value);

                    hidden = new_hidden;
                    cell = new_cell;
                }

                forecasts
            }

            ForecastMethod::GRU { hidden_size } => {
                // Initialize GRU state
                let mut hidden = vec![0.0f32; hidden_size];

                // GRU weights (in production, these would be trained)
                let input_size = 1;
                let weights_ih = vec![0.1f32; 3 * hidden_size * input_size];
                let weights_hh = vec![0.1f32; 3 * hidden_size * hidden_size];
                let bias = vec![0.0f32; 3 * hidden_size];

                let mut forecasts = Vec::new();

                // Warm up GRU with historical data
                for &value in historical_f32.iter().rev().take(20).rev() {
                    let input = vec![value];

                    hidden = executor.gru_cell_forward(
                        &input,
                        &hidden,
                        &weights_ih,
                        &weights_hh,
                        &bias,
                        input_size,
                        hidden_size,
                    )?;
                }

                // Generate forecast
                let mut last_value = *historical_f32.last().unwrap();
                for _ in 0..self.horizon {
                    let input = vec![last_value];

                    hidden = executor.gru_cell_forward(
                        &input,
                        &hidden,
                        &weights_ih,
                        &weights_hh,
                        &bias,
                        input_size,
                        hidden_size,
                    )?;

                    // Use first hidden state element as forecast
                    last_value = hidden[0];
                    forecasts.push(last_value);
                }

                forecasts
            }

            ForecastMethod::Kalman => {
                // Use Worker 2's kalman_filter_step kernel for GPU acceleration
                self.forecast_kalman_gpu(&historical_f32, self.horizon, &executor)?
            }
        };

        let elapsed = start.elapsed();

        // Convert back to f64
        let forecast: Vec<f64> = forecast_f32.iter().map(|&x| x as f64).collect();

        // Estimate uncertainty (simplified)
        let historical_std = self.calculate_std(&historical.to_vec());
        let uncertainty = vec![historical_std; self.horizon];

        let lower_bound: Vec<f64> = forecast.iter()
            .zip(&uncertainty)
            .map(|(f, u)| f - 1.96 * u)
            .collect();

        let upper_bound: Vec<f64> = forecast.iter()
            .zip(&uncertainty)
            .map(|(f, u)| f + 1.96 * u)
            .collect();

        Ok(ForecastResult {
            forecast,
            uncertainty,
            lower_bound,
            upper_bound,
            gpu_time_ms: elapsed.as_secs_f64() * 1000.0,
            used_gpu: true,
        })
    }

    /// Estimate AR coefficients using Yule-Walker equations
    fn estimate_ar_coefficients(&self, data: &[f32], order: usize) -> Vec<f32> {
        if data.len() < order + 1 {
            return vec![0.1f32; order]; // Fallback
        }

        // Simplified: use autocorrelation-based estimation
        let mut coefficients = vec![0.0f32; order];

        for lag in 0..order {
            let mut sum = 0.0;
            let mut count = 0;

            for i in lag+1..data.len() {
                sum += data[i] * data[i - lag - 1];
                count += 1;
            }

            if count > 0 {
                coefficients[lag] = sum / count as f32;
            }
        }

        // Normalize
        let sum: f32 = coefficients.iter().sum();
        if sum.abs() > 1e-6 {
            for coef in &mut coefficients {
                *coef /= sum;
            }
        }

        coefficients
    }

    /// GPU Kalman filter implementation using Worker 2's kalman_filter_step kernel
    #[cfg(feature = "cuda")]
    fn forecast_kalman_gpu(&self, data: &[f32], horizon: usize, executor: &std::sync::MutexGuard<crate::gpu::kernel_executor::GpuKernelExecutor>) -> Result<Vec<f32>> {
        // Univariate Kalman filter for time series forecasting
        // State: x_t (current value)
        // Measurement: y_t (observed value)

        let state_dim = 1;

        // Initialize state and covariance
        let mut state = vec![data.last().copied().unwrap_or(0.0)];
        let mut covariance = vec![0.1f32]; // Initial uncertainty

        // Transition matrix (random walk: x_{t+1} = x_t)
        let transition = vec![1.0f32];

        // Measurement matrix (direct observation: y_t = x_t)
        let measurement_matrix = vec![1.0f32];

        // Process noise (how much state changes)
        let process_noise = vec![0.01f32];

        // Measurement noise (observation uncertainty)
        let measurement_noise = vec![0.05f32];

        // Filter historical data to update state estimate
        for &observation in data.iter().rev().take(20).rev() {
            let measurement = vec![observation];

            // Use Worker 2's kalman_filter_step kernel
            let (new_state, new_cov) = executor.kalman_filter_step(
                &state,
                &covariance,
                &measurement,
                &transition,
                &measurement_matrix,
                &process_noise,
                &measurement_noise,
                state_dim,
            ).context("GPU kalman_filter_step failed")?;

            state = new_state;
            covariance = new_cov;
        }

        // Generate forecast
        let mut forecasts = Vec::new();
        for _ in 0..horizon {
            // Predict next state (for random walk, stays same)
            forecasts.push(state[0]);

            // Update state covariance (uncertainty grows over time)
            let measurement = vec![state[0]]; // Use prediction as pseudo-measurement

            let (new_state, new_cov) = executor.kalman_filter_step(
                &state,
                &covariance,
                &measurement,
                &transition,
                &measurement_matrix,
                &process_noise,
                &measurement_noise,
                state_dim,
            ).context("GPU kalman_filter_step failed")?;

            state = new_state;
            covariance = new_cov;
        }

        Ok(forecasts)
    }

    /// Kalman filter CPU implementation
    fn forecast_kalman_cpu(&self, data: &[f32], horizon: usize) -> Vec<f32> {
        // Simple random walk + noise model
        let mut forecasts = Vec::new();
        let last_value = *data.last().unwrap_or(&0.0);

        // For Kalman, forecast is just the last value (random walk assumption)
        for _ in 0..horizon {
            forecasts.push(last_value);
        }

        forecasts
    }

    /// CPU fallback implementation
    fn forecast_cpu(&self, historical: &Array1<f64>, _elapsed: f64) -> Result<ForecastResult> {
        let historical_vec = historical.to_vec();

        // Simple AR-based forecast
        let forecast = match self.method {
            ForecastMethod::AR { order } => {
                self.ar_forecast_cpu(&historical_vec, order, self.horizon)
            }
            _ => {
                // Fallback: naive forecast (last value repeated)
                vec![*historical.last().unwrap(); self.horizon]
            }
        };

        let historical_std = self.calculate_std(&historical_vec);
        let uncertainty = vec![historical_std; self.horizon];

        let lower_bound: Vec<f64> = forecast.iter()
            .zip(&uncertainty)
            .map(|(f, u)| f - 1.96 * u)
            .collect();

        let upper_bound: Vec<f64> = forecast.iter()
            .zip(&uncertainty)
            .map(|(f, u)| f + 1.96 * u)
            .collect();

        Ok(ForecastResult {
            forecast,
            uncertainty,
            lower_bound,
            upper_bound,
            gpu_time_ms: 0.0,
            used_gpu: false,
        })
    }

    /// Simple AR forecast (CPU)
    fn ar_forecast_cpu(&self, data: &[f64], order: usize, horizon: usize) -> Vec<f64> {
        let mut forecasts = Vec::new();
        let mut extended_data = data.to_vec();

        for _ in 0..horizon {
            let n = extended_data.len();
            if n < order {
                forecasts.push(*extended_data.last().unwrap());
                continue;
            }

            // Simple AR: weighted average of last `order` values
            let mut forecast = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..order {
                let weight = 1.0 / (i as f64 + 1.0);
                forecast += weight * extended_data[n - 1 - i];
                weight_sum += weight;
            }

            forecast /= weight_sum;
            forecasts.push(forecast);
            extended_data.push(forecast);
        }

        forecasts
    }

    /// Calculate standard deviation
    fn calculate_std(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_ar_forecast() {
        let data = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());

        let forecaster = GpuTimeSeriesForecaster::ar(3, 5);
        let result = forecaster.forecast(&data).unwrap();

        assert_eq!(result.forecast.len(), 5);
        assert_eq!(result.uncertainty.len(), 5);
        assert_eq!(result.lower_bound.len(), 5);
        assert_eq!(result.upper_bound.len(), 5);

        // Check bounds are reasonable
        for i in 0..5 {
            assert!(result.lower_bound[i] < result.forecast[i]);
            assert!(result.upper_bound[i] > result.forecast[i]);
        }
    }

    #[test]
    fn test_lstm_forecast() {
        let data = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());

        let forecaster = GpuTimeSeriesForecaster::lstm(32, 5);
        let result = forecaster.forecast(&data).unwrap();

        assert_eq!(result.forecast.len(), 5);
    }

    #[test]
    fn test_gru_forecast() {
        let data = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());

        let forecaster = GpuTimeSeriesForecaster::gru(32, 5);
        let result = forecaster.forecast(&data).unwrap();

        assert_eq!(result.forecast.len(), 5);
    }

    #[test]
    fn test_forecast_uncertainty() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let forecaster = GpuTimeSeriesForecaster::ar(2, 3);
        let result = forecaster.forecast(&data).unwrap();

        // All uncertainty values should be positive
        for u in &result.uncertainty {
            assert!(*u >= 0.0);
        }
    }

    #[test]
    fn test_ar_coefficient_estimation() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let forecaster = GpuTimeSeriesForecaster::ar(2, 5);
        let coeffs = forecaster.estimate_ar_coefficients(&data, 2);

        assert_eq!(coeffs.len(), 2);

        // Coefficients should sum to approximately 1.0 (normalized)
        let sum: f32 = coeffs.iter().sum();
        assert!((sum - 1.0).abs() < 0.1 || sum == 0.0);
    }
}
