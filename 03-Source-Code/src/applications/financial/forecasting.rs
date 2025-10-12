//! Financial Forecasting Module
//!
//! Integrates time series forecasting with portfolio optimization.
//! Uses Worker 1's time series forecaster to predict future returns and volatility.
//!
//! # Integration with Worker 1
//!
//! This module provides the interface between Worker 4's financial optimizer
//! and Worker 1's time series forecaster. When Worker 1 delivers their
//! GPU-accelerated forecaster (Week 3-4), it will be integrated here.

use anyhow::Result;
use ndarray::Array1;

use super::{Asset, Portfolio};
use crate::applications::solver::timeseries_integration::{
    TimeSeriesForecaster, ModelType, ForecastWithUncertainty,
};

/// Portfolio forecaster that predicts future returns and risks
pub struct PortfolioForecaster {
    /// Model type for return forecasting
    return_model: ModelType,

    /// Model type for volatility forecasting
    volatility_model: ModelType,

    /// Forecast horizon (days)
    horizon: usize,
}

impl PortfolioForecaster {
    /// Create a new portfolio forecaster
    ///
    /// # Arguments
    /// * `horizon` - Number of days to forecast ahead
    pub fn new(horizon: usize) -> Self {
        Self {
            return_model: ModelType::LSTM,  // LSTM for returns
            volatility_model: ModelType::ARIMA,  // ARIMA for volatility
            horizon,
        }
    }

    /// Forecast future returns for assets
    ///
    /// Uses time series forecasting to predict expected returns
    /// over the specified horizon.
    ///
    /// # Arguments
    /// * `assets` - Assets with historical returns
    ///
    /// # Returns
    /// Forecasted returns for each asset
    pub async fn forecast_returns(&self, assets: &[Asset]) -> Result<Vec<f64>> {
        let forecaster = TimeSeriesForecaster::new(self.return_model, self.horizon);

        let mut forecasted_returns = Vec::with_capacity(assets.len());

        for asset in assets {
            if asset.historical_returns.is_empty() {
                forecasted_returns.push(0.0);
                continue;
            }

            // Forecast returns
            let forecast = forecaster.forecast(&asset.historical_returns).await?;

            // Use mean of forecasted values as expected return
            let expected_return = forecast.predictions.iter().sum::<f64>()
                / forecast.predictions.len() as f64;

            forecasted_returns.push(expected_return);
        }

        Ok(forecasted_returns)
    }

    /// Forecast future volatility for assets
    ///
    /// Predicts future volatility (risk) using time series forecasting
    /// of squared returns (GARCH-style approach).
    ///
    /// # Arguments
    /// * `assets` - Assets with historical returns
    ///
    /// # Returns
    /// Forecasted volatility (standard deviation) for each asset
    pub async fn forecast_volatility(&self, assets: &[Asset]) -> Result<Vec<f64>> {
        let forecaster = TimeSeriesForecaster::new(self.volatility_model, self.horizon);

        let mut forecasted_volatilities = Vec::with_capacity(assets.len());

        for asset in assets {
            if asset.historical_returns.is_empty() {
                forecasted_volatilities.push(0.0);
                continue;
            }

            // Compute squared returns (volatility proxy)
            let squared_returns: Vec<f64> = asset.historical_returns.iter()
                .map(|&r| r * r)
                .collect();

            // Forecast squared returns
            let forecast = forecaster.forecast(&squared_returns).await?;

            // Volatility = sqrt(mean of forecasted squared returns)
            let volatility = (forecast.predictions.iter().sum::<f64>()
                / forecast.predictions.len() as f64)
                .sqrt();

            forecasted_volatilities.push(volatility);
        }

        Ok(forecasted_volatilities)
    }

    /// Forecast returns with uncertainty bounds
    ///
    /// Provides point forecast plus confidence intervals,
    /// useful for risk-aware portfolio construction.
    ///
    /// # Arguments
    /// * `assets` - Assets with historical returns
    ///
    /// # Returns
    /// Forecasted returns with lower and upper bounds for each asset
    pub async fn forecast_returns_with_uncertainty(
        &self,
        assets: &[Asset],
    ) -> Result<Vec<ForecastWithUncertainty>> {
        let forecaster = TimeSeriesForecaster::new(self.return_model, self.horizon);

        let mut forecasts = Vec::with_capacity(assets.len());

        for asset in assets {
            if asset.historical_returns.is_empty() {
                continue;
            }

            let forecast = forecaster
                .forecast_with_uncertainty(&asset.historical_returns)
                .await?;

            forecasts.push(forecast);
        }

        Ok(forecasts)
    }

    /// Optimize portfolio using forecasted returns
    ///
    /// Combines time series forecasting with portfolio optimization.
    /// This is the main integration point between Worker 1 and Worker 4.
    ///
    /// # Arguments
    /// * `assets` - Assets to optimize over
    /// * `optimizer` - Portfolio optimizer
    ///
    /// # Returns
    /// Optimized portfolio using forecasted returns
    ///
    /// # Example
    /// ```rust,ignore
    /// let forecaster = PortfolioForecaster::new(30);  // 30-day horizon
    /// let optimized = forecaster.optimize_with_forecast(
    ///     &assets,
    ///     &mut optimizer
    /// ).await?;
    /// ```
    pub async fn optimize_with_forecast(
        &self,
        assets: &[Asset],
        optimizer: &mut super::PortfolioOptimizer,
    ) -> Result<Portfolio> {
        // Step 1: Forecast future returns
        let forecasted_returns = self.forecast_returns(assets).await?;

        // Step 2: Update assets with forecasted returns
        let mut forecast_assets = assets.to_vec();
        for (asset, &forecasted_return) in forecast_assets.iter_mut().zip(forecasted_returns.iter()) {
            // Replace historical mean with forecast
            // (In production, would use more sophisticated approach)
            if !asset.historical_returns.is_empty() {
                asset.historical_returns.push(forecasted_return);
            }
        }

        // Step 3: Optimize using forecasted returns
        let portfolio = optimizer.optimize(forecast_assets)?;

        Ok(portfolio)
    }
}

/// Forecast-aware optimization configuration
pub struct ForecastOptimizationConfig {
    /// Forecast horizon in days
    pub horizon: usize,

    /// Model type for returns
    pub return_model: ModelType,

    /// Model type for volatility
    pub volatility_model: ModelType,

    /// Whether to use uncertainty in optimization
    pub use_uncertainty: bool,

    /// Confidence level for bounds (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl Default for ForecastOptimizationConfig {
    fn default() -> Self {
        Self {
            horizon: 30,
            return_model: ModelType::LSTM,
            volatility_model: ModelType::ARIMA,
            use_uncertainty: true,
            confidence_level: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets() -> Vec<Asset> {
        vec![
            Asset {
                symbol: "ASSET1".to_string(),
                name: "Test Asset 1".to_string(),
                current_price: 100.0,
                historical_returns: vec![0.01, 0.02, 0.015, 0.02, 0.018],
            },
            Asset {
                symbol: "ASSET2".to_string(),
                name: "Test Asset 2".to_string(),
                current_price: 50.0,
                historical_returns: vec![0.005, 0.008, 0.006, 0.007, 0.006],
            },
        ]
    }

    #[tokio::test]
    async fn test_forecast_returns() {
        let forecaster = PortfolioForecaster::new(5);
        let assets = create_test_assets();

        let result = forecaster.forecast_returns(&assets).await;
        assert!(result.is_ok());

        let forecasted = result.unwrap();
        assert_eq!(forecasted.len(), 2);
        assert!(forecasted[0] > 0.0);
        assert!(forecasted[1] > 0.0);
    }

    #[tokio::test]
    async fn test_forecast_volatility() {
        let forecaster = PortfolioForecaster::new(5);
        let assets = create_test_assets();

        let result = forecaster.forecast_volatility(&assets).await;
        assert!(result.is_ok());

        let volatilities = result.unwrap();
        assert_eq!(volatilities.len(), 2);
        assert!(volatilities[0] > 0.0);
        assert!(volatilities[1] > 0.0);
    }

    #[tokio::test]
    async fn test_forecast_with_uncertainty() {
        let forecaster = PortfolioForecaster::new(3);
        let assets = create_test_assets();

        let result = forecaster.forecast_returns_with_uncertainty(&assets).await;
        assert!(result.is_ok());

        let forecasts = result.unwrap();
        assert_eq!(forecasts.len(), 2);

        // Check bounds exist
        for forecast in &forecasts {
            assert!(!forecast.lower_bound.is_empty());
            assert!(!forecast.upper_bound.is_empty());
        }
    }

    #[tokio::test]
    async fn test_optimize_with_forecast() {
        use super::super::{PortfolioOptimizer, OptimizationConfig};

        let forecaster = PortfolioForecaster::new(10);
        let assets = create_test_assets();
        let mut optimizer = PortfolioOptimizer::new(OptimizationConfig::default());

        let result = forecaster.optimize_with_forecast(&assets, &mut optimizer).await;
        assert!(result.is_ok());

        let portfolio = result.unwrap();
        assert_eq!(portfolio.weights.len(), 2);

        // Weights should sum to 1
        let total: f64 = portfolio.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
