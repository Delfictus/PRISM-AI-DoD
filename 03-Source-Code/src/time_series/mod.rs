//! Time Series Forecasting Module
//!
//! GPU-accelerated time series forecasting with multiple methods:
//! - ARIMA: Classical statistical forecasting
//! - LSTM/GRU: Deep learning for complex patterns
//! - Uncertainty Quantification: Prediction intervals
//!
//! Integration Points:
//! - PWSA: Trajectory prediction for missile intercept
//! - Finance: Price/volatility forecasting
//! - Telecom: Traffic prediction for proactive routing
//! - LLM: Cost forecasting for budget optimization

pub mod arima_gpu;
pub mod lstm_forecaster;
pub mod uncertainty;

// GPU-Optimized modules (Phase 2: Full GPU utilization)
pub mod arima_gpu_optimized;
pub mod lstm_gpu_optimized;
pub mod uncertainty_gpu_optimized;

pub use arima_gpu::{ArimaGpu, ArimaConfig, ArimaCoefficients, auto_arima};
pub use lstm_forecaster::{LstmForecaster, LstmConfig, CellType};
pub use uncertainty::{
    UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod,
    ForecastWithUncertainty
};

// Re-export GPU-optimized modules
pub use arima_gpu_optimized::ArimaGpuOptimized;
pub use lstm_gpu_optimized::LstmGpuOptimized;
pub use uncertainty_gpu_optimized::UncertaintyGpuOptimized;

use anyhow::Result;

/// Unified time series forecasting interface
pub struct TimeSeriesForecaster {
    /// ARIMA model
    arima: Option<ArimaGpu>,
    /// LSTM model
    lstm: Option<LstmForecaster>,
    /// Uncertainty quantifier
    uncertainty: UncertaintyQuantifier,
}

impl TimeSeriesForecaster {
    /// Create new forecaster with default configuration
    pub fn new() -> Self {
        Self {
            arima: None,
            lstm: None,
            uncertainty: UncertaintyQuantifier::new(UncertaintyConfig::default()),
        }
    }

    /// Fit ARIMA model
    pub fn fit_arima(&mut self, data: &[f64], config: ArimaConfig) -> Result<()> {
        let mut model = ArimaGpu::new(config)?;
        model.fit(data)?;
        self.arima = Some(model);
        Ok(())
    }

    /// Fit LSTM model
    pub fn fit_lstm(&mut self, data: &[f64], config: LstmConfig) -> Result<()> {
        let mut model = LstmForecaster::new(config)?;
        model.fit(data)?;
        self.lstm = Some(model);
        Ok(())
    }

    /// Forecast using ARIMA
    pub fn forecast_arima(&self, horizon: usize) -> Result<Vec<f64>> {
        let model = self.arima.as_ref()
            .ok_or_else(|| anyhow::anyhow!("ARIMA model not fitted"))?;
        model.forecast(horizon)
    }

    /// Forecast using LSTM
    pub fn forecast_lstm(&mut self, data: &[f64], horizon: usize) -> Result<Vec<f64>> {
        let model = self.lstm.as_mut()
            .ok_or_else(|| anyhow::anyhow!("LSTM model not fitted"))?;
        model.forecast(data, horizon)
    }

    /// Forecast with uncertainty quantification
    pub fn forecast_with_uncertainty(
        &self,
        horizon: usize,
    ) -> Result<ForecastWithUncertainty> {
        let forecast = self.forecast_arima(horizon)?;
        self.uncertainty.residual_intervals(&forecast)
    }

    /// Auto-select best model
    pub fn auto_forecast(&mut self, data: &[f64], horizon: usize) -> Result<Vec<f64>> {
        // Try ARIMA first (faster)
        if let Ok(model) = auto_arima(data, 2, 1, 2) {
            return model.forecast(horizon);
        }

        // Fallback to LSTM
        let config = LstmConfig {
            sequence_length: 10,
            epochs: 50,
            ..Default::default()
        };

        self.fit_lstm(data, config)?;
        self.forecast_lstm(data, horizon)
    }
}

impl Default for TimeSeriesForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecaster_creation() {
        let forecaster = TimeSeriesForecaster::new();
        assert!(forecaster.arima.is_none());
        assert!(forecaster.lstm.is_none());
    }

    #[test]
    fn test_arima_workflow() {
        let mut forecaster = TimeSeriesForecaster::new();

        let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();

        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
        };

        forecaster.fit_arima(&data, config).unwrap();

        let forecast = forecaster.forecast_arima(5).unwrap();
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_lstm_workflow() {
        let mut forecaster = TimeSeriesForecaster::new();

        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin()).collect();

        let config = LstmConfig {
            hidden_size: 10,
            sequence_length: 5,
            epochs: 10,
            ..Default::default()
        };

        forecaster.fit_lstm(&data, config).unwrap();

        let forecast = forecaster.forecast_lstm(&data, 3).unwrap();
        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_auto_forecast() {
        let mut forecaster = TimeSeriesForecaster::new();

        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let forecast = forecaster.auto_forecast(&data, 5).unwrap();
        assert_eq!(forecast.len(), 5);
    }
}
