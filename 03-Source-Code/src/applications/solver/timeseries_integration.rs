//! Time Series Integration for Universal Solver
//!
//! This module provides integration points for Worker 1's time series forecasting.
//! It defines the interface expected from Worker 1 and provides mock implementations
//! for testing until Worker 1 delivers the actual forecaster.
//!
//! # Integration Plan
//!
//! Week 3-4: Worker 1 delivers time series forecaster
//! Week 5: Worker 4 integrates with financial module
//!
//! # Expected API from Worker 1
//!
//! ```rust,ignore
//! // Worker 1 will provide:
//! pub struct TimeSeriesForecaster {
//!     model_type: ModelType,  // ARIMA, LSTM, GRU
//!     horizon: usize,
//! }
//!
//! impl TimeSeriesForecaster {
//!     pub fn new(model_type: ModelType) -> Self;
//!     pub async fn forecast(&self, historical: &[f64], horizon: usize) -> Result<Forecast>;
//!     pub fn forecast_with_uncertainty(&self, historical: &[f64]) -> Result<ForecastWithBounds>;
//! }
//! ```

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Time series forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forecast {
    /// Predicted values
    pub predictions: Vec<f64>,

    /// Forecast horizon (number of steps ahead)
    pub horizon: usize,

    /// Model confidence (0-1)
    pub confidence: f64,

    /// Model type used
    pub model_type: String,

    /// Computation time in milliseconds
    pub computation_time_ms: f64,
}

/// Forecast with uncertainty bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastWithUncertainty {
    /// Point forecast
    pub forecast: Forecast,

    /// Lower bound (e.g., 5th percentile)
    pub lower_bound: Vec<f64>,

    /// Upper bound (e.g., 95th percentile)
    pub upper_bound: Vec<f64>,

    /// Standard error at each step
    pub std_errors: Vec<f64>,
}

/// Model type for time series forecasting
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Autoregressive Integrated Moving Average
    ARIMA,

    /// Long Short-Term Memory network
    LSTM,

    /// Gated Recurrent Unit
    GRU,

    /// Simple exponential smoothing (fallback)
    ExponentialSmoothing,
}

/// Time series forecaster interface
///
/// This is a stub implementation that will be replaced when Worker 1
/// delivers the actual GPU-accelerated forecaster.
pub struct TimeSeriesForecaster {
    model_type: ModelType,
    horizon: usize,
}

impl TimeSeriesForecaster {
    /// Create a new time series forecaster
    ///
    /// # Arguments
    /// * `model_type` - Type of forecasting model to use
    /// * `horizon` - Number of steps ahead to forecast
    pub fn new(model_type: ModelType, horizon: usize) -> Self {
        Self {
            model_type,
            horizon,
        }
    }

    /// Forecast future values
    ///
    /// **MOCK IMPLEMENTATION**: Returns simple moving average projection
    /// Will be replaced with Worker 1's GPU-accelerated forecaster
    ///
    /// # Arguments
    /// * `historical` - Historical time series data
    ///
    /// # Returns
    /// Forecast with predicted values
    pub async fn forecast(&self, historical: &[f64]) -> Result<Forecast> {
        use std::time::Instant;
        let start = Instant::now();

        // MOCK: Simple moving average forecast
        let window_size = 5.min(historical.len());
        let recent_avg: f64 = historical
            .iter()
            .rev()
            .take(window_size)
            .sum::<f64>() / window_size as f64;

        // Predict constant value (naive forecast)
        let predictions = vec![recent_avg; self.horizon];

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Forecast {
            predictions,
            horizon: self.horizon,
            confidence: 0.5, // Low confidence for mock
            model_type: format!("{:?} (MOCK)", self.model_type),
            computation_time_ms: elapsed,
        })
    }

    /// Forecast with uncertainty quantification
    ///
    /// **MOCK IMPLEMENTATION**: Returns ±10% bounds
    /// Will be replaced with Worker 1's proper uncertainty estimation
    pub async fn forecast_with_uncertainty(
        &self,
        historical: &[f64],
    ) -> Result<ForecastWithUncertainty> {
        let forecast = self.forecast(historical).await?;

        // MOCK: Simple ±10% bounds
        let std_dev = historical.iter()
            .map(|&x| {
                let mean = historical.iter().sum::<f64>() / historical.len() as f64;
                (x - mean).powi(2)
            })
            .sum::<f64>() / historical.len() as f64;
        let std_dev = std_dev.sqrt();

        let lower_bound = forecast.predictions.iter()
            .map(|&x| x - 1.96 * std_dev)
            .collect();

        let upper_bound = forecast.predictions.iter()
            .map(|&x| x + 1.96 * std_dev)
            .collect();

        let std_errors = vec![std_dev; forecast.horizon];

        Ok(ForecastWithUncertainty {
            forecast,
            lower_bound,
            upper_bound,
            std_errors,
        })
    }
}

/// Integration point for Worker 1's time series module
///
/// This function will be updated to use Worker 1's actual implementation
/// when it becomes available (Week 3-4).
///
/// # TODO: Worker 1 Integration
/// - [ ] Replace mock forecaster with `worker1::time_series::TimeSeriesForecaster`
/// - [ ] Add GPU kernel integration
/// - [ ] Enable ARIMA/LSTM model selection
/// - [ ] Add proper uncertainty quantification
pub async fn create_forecaster(model_type: ModelType, horizon: usize) -> Result<TimeSeriesForecaster> {
    // TODO: When Worker 1 ready:
    // use worker1::time_series::TimeSeriesForecaster;
    // TimeSeriesForecaster::new_gpu_accelerated(model_type, horizon)

    // For now, use mock
    Ok(TimeSeriesForecaster::new(model_type, horizon))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_forecaster() {
        let forecaster = TimeSeriesForecaster::new(ModelType::ARIMA, 5);
        let historical = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];

        let result = forecaster.forecast(&historical).await;
        assert!(result.is_ok());

        let forecast = result.unwrap();
        assert_eq!(forecast.horizon, 5);
        assert_eq!(forecast.predictions.len(), 5);
        assert!(forecast.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_forecast_with_uncertainty() {
        let forecaster = TimeSeriesForecaster::new(ModelType::LSTM, 3);
        let historical = vec![10.0, 11.0, 12.0, 11.5, 13.0];

        let result = forecaster.forecast_with_uncertainty(&historical).await;
        assert!(result.is_ok());

        let forecast_unc = result.unwrap();
        assert_eq!(forecast_unc.lower_bound.len(), 3);
        assert_eq!(forecast_unc.upper_bound.len(), 3);

        // Lower bound should be less than predictions
        for (lb, pred) in forecast_unc.lower_bound.iter()
            .zip(forecast_unc.forecast.predictions.iter())
        {
            assert!(lb < pred);
        }
    }

    #[tokio::test]
    async fn test_create_forecaster() {
        let result = create_forecaster(ModelType::GRU, 10).await;
        assert!(result.is_ok());

        let forecaster = result.unwrap();
        assert_eq!(forecaster.horizon, 10);
        assert_eq!(forecaster.model_type, ModelType::GRU);
    }
}
