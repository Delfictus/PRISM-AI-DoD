//! Portfolio Backtesting Framework - Worker 4
//!
//! Historical performance simulation and analysis:
//! - Walk-forward backtesting
//! - Performance metrics (Sharpe, Sortino, Calmar)
//! - Drawdown analysis
//! - Rolling window analysis
//! - Comparison with benchmarks
//!
//! # Metrics
//!
//! **Sharpe Ratio**: (R_p - R_f) / σ_p
//! **Sortino Ratio**: (R_p - R_f) / σ_downside
//! **Calmar Ratio**: R_p / Max Drawdown
//! **Maximum Drawdown**: max_t [(Peak_t - Trough_t) / Peak_t]

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{Asset, Portfolio, PortfolioOptimizer, OptimizationConfig};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial portfolio value
    pub initial_value: f64,

    /// Risk-free rate for Sharpe/Sortino calculations
    pub risk_free_rate: f64,

    /// Rebalancing frequency (in periods)
    pub rebalance_frequency: usize,

    /// Whether to compound returns
    pub compound_returns: bool,

    /// Window size for rolling metrics
    pub rolling_window: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_value: 1_000_000.0,
            risk_free_rate: 0.02,
            rebalance_frequency: 21, // Monthly (assuming daily data)
            compound_returns: true,
            rolling_window: 252, // One year of daily data
        }
    }
}

/// Performance metrics for a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return (as fraction)
    pub total_return: f64,

    /// Annualized return
    pub annualized_return: f64,

    /// Annualized volatility
    pub annualized_volatility: f64,

    /// Sharpe ratio
    pub sharpe_ratio: f64,

    /// Sortino ratio (downside risk only)
    pub sortino_ratio: f64,

    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,

    /// Maximum drawdown (as fraction)
    pub max_drawdown: f64,

    /// Win rate (fraction of positive periods)
    pub win_rate: f64,

    /// Number of periods
    pub num_periods: usize,

    /// Number of rebalances
    pub num_rebalances: usize,
}

/// Drawdown information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownInfo {
    /// Start period of drawdown
    pub start_period: usize,

    /// Bottom period of drawdown
    pub bottom_period: usize,

    /// Recovery period (None if not recovered)
    pub recovery_period: Option<usize>,

    /// Drawdown magnitude (as fraction)
    pub magnitude: f64,

    /// Duration in periods
    pub duration: usize,
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Portfolio values over time
    pub portfolio_values: Vec<f64>,

    /// Portfolio returns for each period
    pub returns: Vec<f64>,

    /// Cumulative returns
    pub cumulative_returns: Vec<f64>,

    /// Performance metrics
    pub metrics: PerformanceMetrics,

    /// Drawdown history
    pub drawdowns: Vec<DrawdownInfo>,

    /// Rolling Sharpe ratios (if calculated)
    pub rolling_sharpe: Vec<f64>,

    /// Configuration used
    pub config: BacktestConfig,
}

impl BacktestResult {
    /// Get final portfolio value
    pub fn final_value(&self) -> f64 {
        *self.portfolio_values.last().unwrap_or(&0.0)
    }

    /// Get largest drawdown
    pub fn worst_drawdown(&self) -> Option<&DrawdownInfo> {
        self.drawdowns
            .iter()
            .max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap())
    }

    /// Get average drawdown magnitude
    pub fn average_drawdown(&self) -> f64 {
        if self.drawdowns.is_empty() {
            0.0
        } else {
            self.drawdowns.iter().map(|d| d.magnitude).sum::<f64>() / self.drawdowns.len() as f64
        }
    }

    /// Get summary report
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n\
             Total Return: {:.2}%\n\
             Annualized Return: {:.2}%\n\
             Annualized Volatility: {:.2}%\n\
             Sharpe Ratio: {:.3}\n\
             Sortino Ratio: {:.3}\n\
             Calmar Ratio: {:.3}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.1}%\n\
             Periods: {}\n\
             Rebalances: {}",
            self.metrics.total_return * 100.0,
            self.metrics.annualized_return * 100.0,
            self.metrics.annualized_volatility * 100.0,
            self.metrics.sharpe_ratio,
            self.metrics.sortino_ratio,
            self.metrics.calmar_ratio,
            self.metrics.max_drawdown * 100.0,
            self.metrics.win_rate * 100.0,
            self.metrics.num_periods,
            self.metrics.num_rebalances,
        )
    }
}

/// Portfolio backtester
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on historical data
    pub fn run(
        &self,
        assets: &[Asset],
        target_weights: &Array1<f64>,
    ) -> Result<BacktestResult> {
        if assets.is_empty() {
            anyhow::bail!("Cannot backtest with no assets");
        }

        if assets.len() != target_weights.len() {
            anyhow::bail!("Asset count mismatch with target weights");
        }

        // Find minimum historical data length
        let num_periods = assets
            .iter()
            .map(|a| a.historical_returns.len())
            .min()
            .unwrap_or(0);

        if num_periods == 0 {
            anyhow::bail!("No historical data available");
        }

        let mut portfolio_values = vec![self.config.initial_value];
        let mut returns = Vec::new();
        let mut num_rebalances = 0;

        // Simulate portfolio over time
        for t in 0..num_periods {
            // Calculate period return
            let mut period_return = 0.0;
            for (i, asset) in assets.iter().enumerate() {
                period_return += target_weights[i] * asset.historical_returns[t];
            }

            returns.push(period_return);

            // Update portfolio value
            let prev_value = portfolio_values[t];
            let new_value = if self.config.compound_returns {
                prev_value * (1.0 + period_return)
            } else {
                prev_value + prev_value * period_return
            };
            portfolio_values.push(new_value);

            // Check if rebalancing is needed
            if (t + 1) % self.config.rebalance_frequency == 0 {
                num_rebalances += 1;
                // In a real backtest, we would simulate rebalancing here
                // For now, we assume weights stay constant
            }
        }

        // Calculate cumulative returns
        let cumulative_returns = self.calculate_cumulative_returns(&returns);

        // Calculate performance metrics
        let metrics = self.calculate_metrics(&returns, num_rebalances)?;

        // Find drawdowns
        let drawdowns = self.find_drawdowns(&portfolio_values);

        // Calculate rolling Sharpe ratios
        let rolling_sharpe = self.calculate_rolling_sharpe(&returns);

        Ok(BacktestResult {
            portfolio_values,
            returns,
            cumulative_returns,
            metrics,
            drawdowns,
            rolling_sharpe,
            config: self.config.clone(),
        })
    }

    /// Calculate cumulative returns
    fn calculate_cumulative_returns(&self, returns: &[f64]) -> Vec<f64> {
        let mut cumulative = Vec::with_capacity(returns.len());
        let mut cum_return = 0.0;

        for &r in returns {
            if self.config.compound_returns {
                cum_return = (1.0 + cum_return) * (1.0 + r) - 1.0;
            } else {
                cum_return += r;
            }
            cumulative.push(cum_return);
        }

        cumulative
    }

    /// Calculate all performance metrics
    fn calculate_metrics(
        &self,
        returns: &[f64],
        num_rebalances: usize,
    ) -> Result<PerformanceMetrics> {
        let num_periods = returns.len();

        if num_periods == 0 {
            anyhow::bail!("Cannot calculate metrics with no returns");
        }

        // Total return
        let total_return = if self.config.compound_returns {
            returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0
        } else {
            returns.iter().sum::<f64>()
        };

        // Annualized return (assuming 252 trading days per year)
        let periods_per_year = 252.0;
        let years = num_periods as f64 / periods_per_year;
        let annualized_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            total_return
        };

        // Volatility
        let mean_return = returns.iter().sum::<f64>() / num_periods as f64;
        let variance = returns
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / num_periods as f64;
        let volatility = variance.sqrt();
        let annualized_volatility = volatility * periods_per_year.sqrt();

        // Sharpe ratio
        let excess_return = annualized_return - self.config.risk_free_rate;
        let sharpe_ratio = if annualized_volatility > 0.0 {
            excess_return / annualized_volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns
            .iter()
            .map(|&r| if r < 0.0 { r } else { 0.0 })
            .collect();
        let downside_variance = downside_returns
            .iter()
            .map(|&r| r.powi(2))
            .sum::<f64>()
            / num_periods as f64;
        let downside_deviation = downside_variance.sqrt() * periods_per_year.sqrt();
        let sortino_ratio = if downside_deviation > 0.0 {
            excess_return / downside_deviation
        } else {
            0.0
        };

        // Maximum drawdown
        let cumulative_returns = self.calculate_cumulative_returns(returns);
        let max_drawdown = self.calculate_max_drawdown(&cumulative_returns);

        // Calmar ratio
        let calmar_ratio = if max_drawdown.abs() > 0.0 {
            annualized_return / max_drawdown.abs()
        } else {
            0.0
        };

        // Win rate
        let winning_periods = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = winning_periods as f64 / num_periods as f64;

        Ok(PerformanceMetrics {
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            win_rate,
            num_periods,
            num_rebalances,
        })
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, cumulative_returns: &[f64]) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = 1.0; // Start at 100% (no return)

        for &cum_return in cumulative_returns {
            let value = 1.0 + cum_return;
            peak = f64::max(peak, value);
            let drawdown = (peak - value) / peak;
            max_drawdown = f64::max(max_drawdown, drawdown);
        }

        max_drawdown
    }

    /// Find all drawdown periods
    fn find_drawdowns(&self, portfolio_values: &[f64]) -> Vec<DrawdownInfo> {
        let mut drawdowns = Vec::new();
        let mut peak = portfolio_values[0];
        let mut peak_period = 0;
        let mut in_drawdown = false;
        let mut drawdown_start = 0;
        let mut drawdown_bottom = 0;
        let mut drawdown_magnitude = 0.0;

        for (t, &value) in portfolio_values.iter().enumerate() {
            if value >= peak {
                // New peak
                if in_drawdown {
                    // Drawdown recovered
                    drawdowns.push(DrawdownInfo {
                        start_period: drawdown_start,
                        bottom_period: drawdown_bottom,
                        recovery_period: Some(t),
                        magnitude: drawdown_magnitude,
                        duration: t - drawdown_start,
                    });
                    in_drawdown = false;
                }
                peak = value;
                peak_period = t;
            } else {
                // Below peak
                let current_drawdown = (peak - value) / peak;

                if !in_drawdown {
                    // Start new drawdown
                    in_drawdown = true;
                    drawdown_start = peak_period;
                    drawdown_bottom = t;
                    drawdown_magnitude = current_drawdown;
                } else if current_drawdown > drawdown_magnitude {
                    // Deeper drawdown
                    drawdown_bottom = t;
                    drawdown_magnitude = current_drawdown;
                }
            }
        }

        // If still in drawdown at end, add it
        if in_drawdown {
            drawdowns.push(DrawdownInfo {
                start_period: drawdown_start,
                bottom_period: drawdown_bottom,
                recovery_period: None,
                magnitude: drawdown_magnitude,
                duration: portfolio_values.len() - drawdown_start,
            });
        }

        drawdowns
    }

    /// Calculate rolling Sharpe ratio
    fn calculate_rolling_sharpe(&self, returns: &[f64]) -> Vec<f64> {
        let window = self.config.rolling_window;
        let mut rolling_sharpe = Vec::new();

        if returns.len() < window {
            return rolling_sharpe;
        }

        for i in window..=returns.len() {
            let window_returns = &returns[i - window..i];
            let mean_return = window_returns.iter().sum::<f64>() / window as f64;
            let variance = window_returns
                .iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f64>()
                / window as f64;
            let volatility = variance.sqrt();

            let sharpe = if volatility > 0.0 {
                (mean_return - self.config.risk_free_rate / 252.0) / volatility * (252.0_f64).sqrt()
            } else {
                0.0
            };

            rolling_sharpe.push(sharpe);
        }

        rolling_sharpe
    }

    /// Compare portfolio against benchmark
    pub fn compare_to_benchmark(
        &self,
        portfolio_result: &BacktestResult,
        benchmark_returns: &[f64],
    ) -> Result<ComparisonResult> {
        if portfolio_result.returns.len() != benchmark_returns.len() {
            anyhow::bail!("Portfolio and benchmark must have same length");
        }

        // Calculate benchmark metrics
        let benchmark_config = self.config.clone();
        let benchmark_metrics = self.calculate_metrics(benchmark_returns, 0)?;

        // Calculate tracking error
        let tracking_error = self.calculate_tracking_error(
            &portfolio_result.returns,
            benchmark_returns,
        );

        // Calculate information ratio
        let excess_returns: Vec<f64> = portfolio_result
            .returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();

        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let information_ratio = if tracking_error > 0.0 {
            (mean_excess * 252.0) / tracking_error
        } else {
            0.0
        };

        // Calculate beta
        let beta = self.calculate_beta(&portfolio_result.returns, benchmark_returns);

        // Calculate alpha
        let alpha = portfolio_result.metrics.annualized_return
            - (self.config.risk_free_rate + beta * (benchmark_metrics.annualized_return - self.config.risk_free_rate));

        Ok(ComparisonResult {
            portfolio_metrics: portfolio_result.metrics.clone(),
            benchmark_metrics,
            tracking_error,
            information_ratio,
            beta,
            alpha,
        })
    }

    /// Calculate tracking error (annualized)
    fn calculate_tracking_error(&self, portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();

        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let variance = excess_returns
            .iter()
            .map(|&r| (r - mean_excess).powi(2))
            .sum::<f64>()
            / excess_returns.len() as f64;

        variance.sqrt() * (252.0_f64).sqrt()
    }

    /// Calculate portfolio beta relative to benchmark
    fn calculate_beta(&self, portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        let n = portfolio_returns.len() as f64;

        let mean_portfolio = portfolio_returns.iter().sum::<f64>() / n;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / n;

        let covariance: f64 = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| (p - mean_portfolio) * (b - mean_benchmark))
            .sum::<f64>()
            / n;

        let benchmark_variance: f64 = benchmark_returns
            .iter()
            .map(|&b| (b - mean_benchmark).powi(2))
            .sum::<f64>()
            / n;

        if benchmark_variance > 0.0 {
            covariance / benchmark_variance
        } else {
            1.0
        }
    }
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Portfolio metrics
    pub portfolio_metrics: PerformanceMetrics,

    /// Benchmark metrics
    pub benchmark_metrics: PerformanceMetrics,

    /// Tracking error (annualized)
    pub tracking_error: f64,

    /// Information ratio
    pub information_ratio: f64,

    /// Beta (systematic risk)
    pub beta: f64,

    /// Alpha (excess return)
    pub alpha: f64,
}

impl ComparisonResult {
    pub fn summary(&self) -> String {
        format!(
            "Portfolio vs Benchmark:\n\
             Portfolio Return: {:.2}% | Benchmark: {:.2}%\n\
             Portfolio Sharpe: {:.3} | Benchmark: {:.3}\n\
             Tracking Error: {:.2}%\n\
             Information Ratio: {:.3}\n\
             Beta: {:.3}\n\
             Alpha: {:.2}%",
            self.portfolio_metrics.annualized_return * 100.0,
            self.benchmark_metrics.annualized_return * 100.0,
            self.portfolio_metrics.sharpe_ratio,
            self.benchmark_metrics.sharpe_ratio,
            self.tracking_error * 100.0,
            self.information_ratio,
            self.beta,
            self.alpha * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn create_test_assets() -> Vec<Asset> {
        vec![
            Asset {
                symbol: "AAPL".to_string(),
                name: "Apple Inc.".to_string(),
                current_price: 150.0,
                historical_returns: vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, 0.03],
            },
            Asset {
                symbol: "GOOGL".to_string(),
                name: "Alphabet Inc.".to_string(),
                current_price: 2800.0,
                historical_returns: vec![0.02, 0.01, -0.01, 0.02, 0.01, 0.01, 0.02, 0.01],
            },
        ]
    }

    #[test]
    fn test_backtest_execution() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);
        let assets = create_test_assets();
        let weights = arr1(&[0.5, 0.5]);

        let result = backtester.run(&assets, &weights);
        assert!(result.is_ok());

        let backtest = result.unwrap();
        assert_eq!(backtest.returns.len(), 8);
        assert!(backtest.metrics.num_periods > 0);
    }

    #[test]
    fn test_performance_metrics() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);
        let assets = create_test_assets();
        let weights = arr1(&[0.6, 0.4]);

        let backtest = backtester.run(&assets, &weights).unwrap();
        let metrics = &backtest.metrics;

        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.sortino_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_drawdown_detection() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);
        let assets = create_test_assets();
        let weights = arr1(&[0.5, 0.5]);

        let backtest = backtester.run(&assets, &weights).unwrap();

        // Should detect at least one drawdown period
        assert!(!backtest.drawdowns.is_empty() || backtest.metrics.max_drawdown == 0.0);
    }

    #[test]
    fn test_rolling_sharpe() {
        let config = BacktestConfig {
            rolling_window: 4,
            ..Default::default()
        };
        let backtester = Backtester::new(config);
        let assets = create_test_assets();
        let weights = arr1(&[0.5, 0.5]);

        let backtest = backtester.run(&assets, &weights).unwrap();

        // With 8 periods and window=4, should have 5 rolling Sharpe values
        assert_eq!(backtest.rolling_sharpe.len(), 5);
    }

    #[test]
    fn test_benchmark_comparison() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);
        let assets = create_test_assets();
        let weights = arr1(&[0.5, 0.5]);

        let backtest = backtester.run(&assets, &weights).unwrap();

        // Benchmark returns (e.g., market index)
        let benchmark_returns = vec![0.015, 0.005, 0.01, 0.015, 0.005, 0.015, 0.015, 0.02];

        let comparison = backtester.compare_to_benchmark(&backtest, &benchmark_returns);
        assert!(comparison.is_ok());

        let result = comparison.unwrap();
        assert!(result.beta.is_finite());
        assert!(result.alpha.is_finite());
        assert!(result.tracking_error >= 0.0);
    }
}
