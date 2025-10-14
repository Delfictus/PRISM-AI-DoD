//! Portfolio Forecasting with Time Series Integration
//!
//! Enhances portfolio optimization with temporal forecasting:
//! - Price/return forecasting using ARIMA/LSTM
//! - Volatility forecasting with uncertainty quantification
//! - Dynamic portfolio rebalancing based on forecasts
//! - Risk trajectory prediction

use ndarray::Array1;
use anyhow::{Result, Context};
use crate::time_series::{TimeSeriesForecaster, ArimaConfig, LstmConfig, UncertaintyConfig};
use crate::finance::portfolio_optimizer::{Asset, Portfolio, PortfolioConfig, PortfolioOptimizer, OptimizationStrategy};

/// Portfolio with forecasting capabilities
pub struct PortfolioForecaster {
    /// Base portfolio optimizer
    optimizer: PortfolioOptimizer,
    /// Time series forecaster
    forecaster: TimeSeriesForecaster,
    /// Configuration
    config: ForecastConfig,
}

/// Forecasting configuration
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Forecast horizon (number of periods)
    pub horizon: usize,
    /// Use ARIMA (true) or LSTM (false)
    pub use_arima: bool,
    /// ARIMA configuration
    pub arima_config: ArimaConfig,
    /// LSTM configuration
    pub lstm_config: LstmConfig,
    /// Uncertainty quantification config
    pub uncertainty_config: UncertaintyConfig,
    /// Rebalancing trigger threshold (portfolio drift)
    pub rebalance_threshold: f64,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            horizon: 20,  // 20 trading days (~1 month)
            use_arima: true,  // ARIMA is faster for returns
            arima_config: ArimaConfig {
                p: 2,
                d: 1,
                q: 1,
                include_constant: true,
            },
            lstm_config: LstmConfig {
                hidden_size: 20,
                sequence_length: 10,
                epochs: 50,
                ..Default::default()
            },
            uncertainty_config: UncertaintyConfig {
                confidence_level: 0.95,
                ..Default::default()
            },
            rebalance_threshold: 0.05,  // 5% drift triggers rebalancing
        }
    }
}

/// Forecasted portfolio with uncertainty
#[derive(Debug, Clone)]
pub struct ForecastedPortfolio {
    /// Current optimal portfolio
    pub current: Portfolio,
    /// Forecasted returns for each asset
    pub forecasted_returns: Vec<Vec<f64>>,
    /// Forecasted volatilities
    pub forecasted_volatilities: Vec<Vec<f64>>,
    /// Forecast uncertainty (95% confidence intervals)
    pub return_lower_bounds: Vec<Vec<f64>>,
    pub return_upper_bounds: Vec<Vec<f64>>,
    /// Recommended rebalancing action
    pub rebalance_recommendation: RebalanceAction,
}

/// Rebalancing recommendation
#[derive(Debug, Clone, PartialEq)]
pub enum RebalanceAction {
    /// No rebalancing needed
    Hold,
    /// Rebalance to new weights
    Rebalance(Array1<f64>),
    /// Reduce risk exposure
    ReduceRisk,
    /// Increase risk exposure
    IncreaseRisk,
}

impl PortfolioForecaster {
    /// Create new portfolio forecaster
    pub fn new(portfolio_config: PortfolioConfig, forecast_config: ForecastConfig) -> Result<Self> {
        let optimizer = PortfolioOptimizer::new(portfolio_config)?;
        let forecaster = TimeSeriesForecaster::new();

        Ok(Self {
            optimizer,
            forecaster,
            config: forecast_config,
        })
    }

    /// Forecast portfolio performance and generate recommendations
    pub fn forecast_and_optimize(&mut self, assets: &[Asset], strategy: OptimizationStrategy) -> Result<ForecastedPortfolio> {
        // Step 1: Forecast returns for each asset
        let mut forecasted_returns = Vec::new();
        let mut forecasted_volatilities = Vec::new();
        let mut return_lower_bounds = Vec::new();
        let mut return_upper_bounds = Vec::new();

        for asset in assets {
            let (returns_forecast, volatility_forecast, lower, upper) =
                self.forecast_asset_returns(&asset.prices)?;

            forecasted_returns.push(returns_forecast);
            forecasted_volatilities.push(volatility_forecast);
            return_lower_bounds.push(lower);
            return_upper_bounds.push(upper);
        }

        // Step 2: Compute forecasted expected returns (mean of forecast)
        let expected_returns: Vec<f64> = forecasted_returns.iter()
            .map(|forecast| forecast.iter().sum::<f64>() / forecast.len() as f64)
            .collect();

        // Step 3: Create updated assets with forecasted returns
        let mut updated_assets: Vec<Asset> = assets.iter()
            .zip(expected_returns.iter())
            .map(|(asset, &forecasted_return)| {
                let mut updated = asset.clone();
                updated.expected_return = forecasted_return;
                updated
            })
            .collect();

        // Step 4: Optimize portfolio with forecasted parameters
        let result = self.optimizer.optimize(&updated_assets, strategy)?;
        let current_portfolio = result.portfolio;

        // Step 5: Determine rebalancing recommendation
        let rebalance_recommendation = self.assess_rebalancing_need(
            assets,
            &current_portfolio,
            &forecasted_returns,
            &forecasted_volatilities,
        )?;

        Ok(ForecastedPortfolio {
            current: current_portfolio,
            forecasted_returns,
            forecasted_volatilities,
            return_lower_bounds,
            return_upper_bounds,
            rebalance_recommendation,
        })
    }

    /// Forecast returns for a single asset
    fn forecast_asset_returns(&mut self, prices: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        // Convert prices to returns
        let mut returns: Vec<f64> = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            returns.push((prices[i] / prices[i-1]) - 1.0);
        }

        if returns.len() < 10 {
            anyhow::bail!("Insufficient price history for forecasting (need at least 10 periods)");
        }

        // Forecast using ARIMA or LSTM
        let forecast = if self.config.use_arima {
            self.forecaster.fit_arima(&returns, self.config.arima_config.clone())?;
            self.forecaster.forecast_arima(self.config.horizon)?
        } else {
            self.forecaster.fit_lstm(&returns, self.config.lstm_config.clone())?;
            self.forecaster.forecast_lstm(&returns, self.config.horizon)?
        };

        // Compute uncertainty intervals
        // Note: uncertainty quantifier needs historical residuals, so we compute them from returns
        let forecast_with_uncertainty = if returns.len() >= self.config.horizon * 2 {
            // Enough data for uncertainty quantification
            self.forecaster.forecast_with_uncertainty(self.config.horizon).unwrap_or_else(|_| {
                // Fallback: use simple std dev bands
                let std_dev = {
                    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                    let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                    var.sqrt()
                };
                let lower = forecast.iter().map(|&f| f - 1.96 * std_dev).collect();
                let upper = forecast.iter().map(|&f| f + 1.96 * std_dev).collect();
                crate::time_series::ForecastWithUncertainty {
                    forecast: forecast.clone(),
                    lower_bound: lower,
                    upper_bound: upper,
                    std_dev: vec![std_dev; forecast.len()],
                    confidence_level: 0.95,
                }
            })
        } else {
            // Not enough data - use simple std dev
            let std_dev = {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                var.sqrt()
            };
            let lower = forecast.iter().map(|&f| f - 1.96 * std_dev).collect();
            let upper = forecast.iter().map(|&f| f + 1.96 * std_dev).collect();
            crate::time_series::ForecastWithUncertainty {
                forecast: forecast.clone(),
                lower_bound: lower,
                upper_bound: upper,
                std_dev: vec![std_dev; forecast.len()],
                confidence_level: 0.95,
            }
        };

        // Estimate volatility (rolling std dev of forecasted returns)
        let volatility_forecast = self.compute_rolling_volatility(&forecast)?;

        Ok((
            forecast,
            volatility_forecast,
            forecast_with_uncertainty.lower_bound,
            forecast_with_uncertainty.upper_bound,
        ))
    }

    /// Compute rolling volatility from forecasted returns
    fn compute_rolling_volatility(&self, forecast: &[f64]) -> Result<Vec<f64>> {
        let window = 5.min(forecast.len());
        let mut volatilities = Vec::with_capacity(forecast.len());

        for i in 0..forecast.len() {
            let start = if i >= window { i - window } else { 0 };
            let window_data = &forecast[start..=i];

            // Compute std dev
            let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
            let variance = window_data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window_data.len() as f64;

            volatilities.push(variance.sqrt());
        }

        Ok(volatilities)
    }

    /// Assess if rebalancing is needed
    fn assess_rebalancing_need(
        &self,
        _assets: &[Asset],
        current_portfolio: &Portfolio,
        forecasted_returns: &[Vec<f64>],
        forecasted_volatilities: &[Vec<f64>],
    ) -> Result<RebalanceAction> {
        // Compute forecasted portfolio metrics
        let n_assets = forecasted_returns.len();
        let horizon = forecasted_returns[0].len();

        // Compute expected portfolio return over forecast horizon
        let mut total_expected_return = 0.0;
        for i in 0..n_assets {
            let asset_return: f64 = forecasted_returns[i].iter().sum::<f64>() / horizon as f64;
            total_expected_return += current_portfolio.weights[i] * asset_return;
        }

        // Compute expected portfolio risk
        let mut total_expected_risk = 0.0;
        for i in 0..n_assets {
            let asset_vol: f64 = forecasted_volatilities[i].iter().sum::<f64>() / horizon as f64;
            total_expected_risk += current_portfolio.weights[i] * asset_vol;
        }

        // Decision logic:
        // 1. If forecasted return < risk-free rate: Reduce risk
        // 2. If forecasted volatility > 2x current volatility: Reduce risk
        // 3. If forecasted return > 1.5x current return AND vol stable: Increase risk
        // 4. Otherwise: Hold

        let risk_free_rate = 0.02;  // 2% annual

        if total_expected_return < risk_free_rate {
            Ok(RebalanceAction::ReduceRisk)
        } else if total_expected_risk > 2.0 * current_portfolio.volatility {
            Ok(RebalanceAction::ReduceRisk)
        } else if total_expected_return > 1.5 * current_portfolio.expected_return
               && total_expected_risk < 1.2 * current_portfolio.volatility {
            Ok(RebalanceAction::IncreaseRisk)
        } else {
            // Check for drift - if forecasted optimal weights differ significantly
            let drift = (total_expected_return - current_portfolio.expected_return).abs();
            if drift > self.config.rebalance_threshold {
                Ok(RebalanceAction::Rebalance(current_portfolio.weights.clone()))
            } else {
                Ok(RebalanceAction::Hold)
            }
        }
    }

    /// Generate multi-period rebalancing strategy
    pub fn generate_rebalancing_schedule(
        &mut self,
        assets: &[Asset],
        strategy: OptimizationStrategy,
        periods: usize,
    ) -> Result<Vec<Portfolio>> {
        let mut rebalancing_schedule = Vec::with_capacity(periods);
        let mut current_assets = assets.to_vec();

        for period in 0..periods {
            println!("Planning rebalancing for period {}...", period + 1);

            // Forecast and optimize
            let forecasted = self.forecast_and_optimize(&current_assets, strategy)?;
            rebalancing_schedule.push(forecasted.current.clone());

            // Update assets with forecasted prices (simulate forward in time)
            // Use first forecasted return to update prices
            for (i, asset) in current_assets.iter_mut().enumerate() {
                if !forecasted.forecasted_returns[i].is_empty() {
                    let next_return = forecasted.forecasted_returns[i][0];
                    let last_price = asset.prices.last().unwrap();
                    let next_price = last_price * (1.0 + next_return);
                    asset.prices.push(next_price);

                    // Update expected return with forecast
                    asset.expected_return = forecasted.forecasted_returns[i]
                        .iter().sum::<f64>() / forecasted.forecasted_returns[i].len() as f64;
                }
            }
        }

        Ok(rebalancing_schedule)
    }
}

impl ForecastedPortfolio {
    /// Print forecast summary
    pub fn print_summary(&self) {
        println!("\n=== Portfolio Forecast Summary ===");
        println!("Current Portfolio:");
        println!("  Expected Return: {:.2}%", self.current.expected_return * 100.0);
        println!("  Volatility: {:.2}%", self.current.volatility * 100.0);
        println!("  Sharpe Ratio: {:.2}", self.current.sharpe_ratio);

        println!("\nAsset Allocations:");
        for (i, ticker) in self.current.assets.iter().enumerate() {
            println!("  {}: {:.1}%", ticker, self.current.weights[i] * 100.0);
        }

        println!("\nForecasted Returns (mean ± 95% CI):");
        for (i, ticker) in self.current.assets.iter().enumerate() {
            let mean = self.forecasted_returns[i].iter().sum::<f64>() / self.forecasted_returns[i].len() as f64;
            let lower = self.return_lower_bounds[i].iter().sum::<f64>() / self.return_lower_bounds[i].len() as f64;
            let upper = self.return_upper_bounds[i].iter().sum::<f64>() / self.return_upper_bounds[i].len() as f64;
            println!("  {}: {:.2}% [{:.2}%, {:.2}%]",
                     ticker, mean * 100.0, lower * 100.0, upper * 100.0);
        }

        println!("\nRecommendation: {:?}", self.rebalance_recommendation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets_with_history() -> Vec<Asset> {
        // Generate 60 days of price history
        let mut aapl_prices = vec![100.0];
        let mut googl_prices = vec![200.0];
        let mut msft_prices = vec![150.0];

        for i in 1..60 {
            let t = i as f64 * 0.1;
            aapl_prices.push(100.0 + 5.0 * t.sin() + t * 0.5);
            googl_prices.push(200.0 + 10.0 * (t * 1.2).sin() + t * 0.8);
            msft_prices.push(150.0 + 3.0 * (t * 0.8).sin() + t * 0.3);
        }

        vec![
            Asset {
                ticker: "AAPL".to_string(),
                expected_return: 0.12,
                prices: aapl_prices,
                min_weight: 0.0,
                max_weight: 1.0,
            },
            Asset {
                ticker: "GOOGL".to_string(),
                expected_return: 0.15,
                prices: googl_prices,
                min_weight: 0.0,
                max_weight: 1.0,
            },
            Asset {
                ticker: "MSFT".to_string(),
                expected_return: 0.10,
                prices: msft_prices,
                min_weight: 0.0,
                max_weight: 1.0,
            },
        ]
    }

    #[test]
    fn test_portfolio_forecaster_creation() {
        let portfolio_config = PortfolioConfig::default();
        let forecast_config = ForecastConfig::default();

        let forecaster = PortfolioForecaster::new(portfolio_config, forecast_config);
        assert!(forecaster.is_ok());
    }

    #[test]
    fn test_forecast_and_optimize() {
        let assets = create_test_assets_with_history();
        let portfolio_config = PortfolioConfig::default();
        let mut forecast_config = ForecastConfig::default();
        forecast_config.horizon = 5;  // Short horizon for testing

        let mut forecaster = PortfolioForecaster::new(portfolio_config, forecast_config).unwrap();

        let result = forecaster.forecast_and_optimize(&assets, OptimizationStrategy::MaxSharpe);
        assert!(result.is_ok());

        let forecasted = result.unwrap();
        assert_eq!(forecasted.forecasted_returns.len(), 3);
        assert_eq!(forecasted.forecasted_returns[0].len(), 5);
    }

    #[test]
    fn test_rebalancing_schedule() {
        let assets = create_test_assets_with_history();
        let portfolio_config = PortfolioConfig::default();
        let mut forecast_config = ForecastConfig::default();
        forecast_config.horizon = 3;

        let mut forecaster = PortfolioForecaster::new(portfolio_config, forecast_config).unwrap();

        let schedule = forecaster.generate_rebalancing_schedule(&assets, OptimizationStrategy::MaxSharpe, 2);
        assert!(schedule.is_ok());

        let portfolios = schedule.unwrap();
        assert_eq!(portfolios.len(), 2);
    }
}
