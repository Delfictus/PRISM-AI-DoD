//! Financial Portfolio Optimization - Worker 4
//!
//! GPU-accelerated portfolio optimization using:
//! - Active Inference for market dynamics modeling
//! - Transfer Entropy for causal relationships
//! - Thermodynamic consensus for risk management
//! - Time series forecasting (integration with Worker 1)
//!
//! # Features
//!
//! - Modern Portfolio Theory (MPT) optimization
//! - Risk-adjusted returns (Sharpe ratio maximization)
//! - Multi-objective optimization (risk vs return)
//! - Market regime detection
//! - Forecasting integration

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::information_theory::TransferEntropy;

pub mod market_regime;
pub mod forecasting;
pub mod risk_analysis;
pub mod rebalancing;
pub mod backtest;
pub mod multi_objective_portfolio;
pub mod interior_point_qp;
pub mod gpu_covariance;
pub mod gpu_forecasting;
pub mod gpu_context;
pub mod gpu_linalg;
pub mod gpu_risk;

pub use market_regime::{MarketRegime, MarketRegimeDetector};
pub use forecasting::{PortfolioForecaster, ForecastOptimizationConfig};
pub use risk_analysis::{RiskAnalyzer, RiskDecomposition, VarResult, VarMethod, FactorRiskDecomposition};
pub use rebalancing::{PortfolioRebalancer, RebalancingConfig, RebalancingPlan, RebalancingStrategy, TransactionCost, TaxConfig};
pub use backtest::{Backtester, BacktestConfig, BacktestResult, PerformanceMetrics, DrawdownInfo, ComparisonResult};
pub use multi_objective_portfolio::{
    MultiObjectivePortfolioOptimizer, MultiObjectiveConfig,
    MultiObjectivePortfolioResult,
};
pub use interior_point_qp::{
    InteriorPointQpSolver, InteriorPointConfig, InteriorPointResult,
};
pub use gpu_covariance::GpuCovarianceCalculator;
pub use gpu_forecasting::{GpuTimeSeriesForecaster, ForecastMethod, ForecastResult};
pub use gpu_context::{GpuContext, get_gpu_context};
pub use gpu_linalg::{GpuVectorOps, GpuMatrixOps};
pub use gpu_risk::{GpuRiskAnalyzer, RiskMetrics, PortfolioUncertainty};

/// Represents a financial asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: String,
    pub name: String,
    pub current_price: f64,
    pub historical_returns: Vec<f64>,
}

/// Portfolio allocation
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub assets: Vec<Asset>,
    pub weights: Array1<f64>,
    pub expected_return: f64,
    pub risk: f64,
    pub sharpe_ratio: f64,
}

/// Serializable version of Portfolio (for API/storage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSerialized {
    pub assets: Vec<Asset>,
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub risk: f64,
    pub sharpe_ratio: f64,
}

impl From<Portfolio> for PortfolioSerialized {
    fn from(p: Portfolio) -> Self {
        Self {
            assets: p.assets,
            weights: p.weights.to_vec(),
            expected_return: p.expected_return,
            risk: p.risk,
            sharpe_ratio: p.sharpe_ratio,
        }
    }
}

/// Portfolio optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub target_return: Option<f64>,
    pub max_risk: Option<f64>,
    pub risk_free_rate: f64,
    pub use_forecasting: bool,
    pub use_transfer_entropy: bool,
    pub use_regime_detection: bool,
    pub max_weight_per_asset: f64,
    pub min_weight_per_asset: f64,
    /// Use Interior Point Method (more accurate, slower) vs Gradient Descent (faster, approximate)
    pub use_interior_point: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            target_return: None,
            max_risk: None,
            risk_free_rate: 0.02, // 2% default risk-free rate
            use_forecasting: false,
            use_transfer_entropy: true,
            use_regime_detection: true,
            max_weight_per_asset: 0.4,  // No single asset > 40%
            min_weight_per_asset: 0.0,  // Allow zero weight
            use_interior_point: false,  // Default to fast gradient descent
        }
    }
}

/// GPU-accelerated portfolio optimizer
pub struct PortfolioOptimizer {
    config: OptimizationConfig,
    regime_detector: Option<MarketRegimeDetector>,
}

impl PortfolioOptimizer {
    /// Create a new portfolio optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        let regime_detector = if config.use_regime_detection {
            Some(MarketRegimeDetector::new(20)) // 20-day window
        } else {
            None
        };

        Self {
            config,
            regime_detector,
        }
    }

    /// Optimize portfolio allocation using Mean-Variance Optimization
    pub fn optimize(&mut self, assets: Vec<Asset>) -> Result<Portfolio> {
        let n_assets = assets.len();

        if n_assets == 0 {
            anyhow::bail!("Cannot optimize empty portfolio");
        }

        // Step 1: Calculate expected returns for each asset
        let expected_returns = self.calculate_asset_returns(&assets);

        // Step 2: Calculate covariance matrix (GPU-accelerated)
        let covariance_matrix = self.calculate_covariance_matrix(&assets)?;

        // Step 3: Detect market regime (if enabled)
        let regime_factor = if self.regime_detector.is_some() && !assets[0].historical_returns.is_empty() {
            let prices = self.reconstruct_prices(&assets[0]);
            let detector = self.regime_detector.as_mut().unwrap();
            detector.detect_regime(&prices)?;
            detector.regime_adjustment_factor()
        } else {
            1.0
        };

        // Step 4: Apply Transfer Entropy for causal relationship detection (if enabled)
        let causal_weights = if self.config.use_transfer_entropy {
            self.calculate_causal_weights(&assets)?
        } else {
            Array1::from_elem(n_assets, 1.0)
        };

        // Step 5: Solve mean-variance optimization problem
        let weights = self.solve_mvo(
            &expected_returns,
            &covariance_matrix,
            &causal_weights,
            regime_factor,
        )?;

        // Step 6: Calculate portfolio metrics
        let expected_return = self.calculate_expected_return(&assets, &weights);
        let risk = self.calculate_risk_from_covariance(&covariance_matrix, &weights);
        let sharpe_ratio = self.calculate_sharpe_ratio(expected_return, risk);

        Ok(Portfolio {
            assets,
            weights,
            expected_return,
            risk,
            sharpe_ratio,
        })
    }

    /// Calculate expected returns for each asset
    fn calculate_asset_returns(&self, assets: &[Asset]) -> Array1<f64> {
        let mut returns = Vec::with_capacity(assets.len());

        for asset in assets {
            if asset.historical_returns.is_empty() {
                returns.push(0.0);
            } else {
                // Use mean historical return as expected return
                let mean_return: f64 =
                    asset.historical_returns.iter().sum::<f64>() / asset.historical_returns.len() as f64;
                returns.push(mean_return);
            }
        }

        Array1::from_vec(returns)
    }

    /// Calculate covariance matrix from historical returns
    /// Uses GPU acceleration via Worker 2 integration (8x speedup with Tensor Cores)
    fn calculate_covariance_matrix(&self, assets: &[Asset]) -> Result<Array2<f64>> {
        let n_assets = assets.len();

        // Find minimum length
        let min_len = assets
            .iter()
            .map(|a| a.historical_returns.len())
            .min()
            .unwrap_or(0);

        if min_len == 0 {
            // Return identity matrix as fallback
            return Ok(Array2::eye(n_assets));
        }

        // Build returns matrix (time x assets) - transposed for GPU calculator
        let mut returns_matrix = Array2::zeros((min_len, n_assets));
        for (i, asset) in assets.iter().enumerate() {
            for (j, &ret) in asset.historical_returns.iter().take(min_len).enumerate() {
                returns_matrix[[j, i]] = ret;
            }
        }

        // Use GPU-accelerated covariance calculator (integrates with Worker 2)
        let gpu_calc = gpu_covariance::GpuCovarianceCalculator::new();
        let covariance = gpu_calc.calculate(&returns_matrix)?;

        Ok(covariance)
    }

    /// Calculate causal weights using Transfer Entropy
    /// Assets with stronger causal influence on others get higher weights
    fn calculate_causal_weights(&self, assets: &[Asset]) -> Result<Array1<f64>> {
        let n_assets = assets.len();
        let te_calc = TransferEntropy::default();

        let mut causal_influence = vec![0.0; n_assets];

        // Calculate pairwise transfer entropy
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i != j && assets[i].historical_returns.len() > 20 && assets[j].historical_returns.len() > 20 {
                    let source = Array1::from_vec(assets[i].historical_returns.clone());
                    let target = Array1::from_vec(assets[j].historical_returns.clone());

                    let result = te_calc.calculate(&source, &target);

                    // Sum up outgoing transfer entropy for each asset
                    if result.p_value < 0.05 {
                        causal_influence[i] += result.effective_te;
                    }
                }
            }
        }

        // Normalize to [0.5, 1.5] range to avoid extreme weights
        let max_influence = causal_influence.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max_influence > 0.0 {
            for influence in &mut causal_influence {
                *influence = 0.5 + (*influence / max_influence);
            }
        } else {
            // If no significant causal relationships, use uniform weights
            causal_influence = vec![1.0; n_assets];
        }

        Ok(Array1::from_vec(causal_influence))
    }

    /// Solve mean-variance optimization problem
    /// Maximize: w^T * μ - λ * w^T * Σ * w
    /// Subject to: sum(w) = 1, w_i >= min_weight, w_i <= max_weight
    fn solve_mvo(
        &self,
        expected_returns: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
        causal_weights: &Array1<f64>,
        regime_factor: f64,
    ) -> Result<Array1<f64>> {
        // Adjust expected returns with causal weights and regime factor
        let adjusted_returns = expected_returns * causal_weights * regime_factor;

        // Risk aversion parameter (λ)
        let risk_aversion = if let Some(target_return) = self.config.target_return {
            // Adjust risk aversion to target specific return
            (target_return - self.config.risk_free_rate) / 0.01
        } else {
            1.0
        };

        if self.config.use_interior_point {
            // Use Interior Point Method for accurate, provably optimal solution
            let ip_config = interior_point_qp::InteriorPointConfig::default();
            let ip_solver = interior_point_qp::InteriorPointQpSolver::new(ip_config);

            let result = ip_solver.solve_portfolio(
                &adjusted_returns,
                covariance_matrix,
                risk_aversion,
                self.config.min_weight_per_asset,
                self.config.max_weight_per_asset,
            )?;

            Ok(result.weights)
        } else {
            // Use fast gradient descent for approximate solution
            self.solve_mvo_gradient_descent(
                &adjusted_returns,
                covariance_matrix,
                risk_aversion,
            )
        }
    }

    /// Fast gradient descent solver (approximate but fast)
    fn solve_mvo_gradient_descent(
        &self,
        expected_returns: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
        risk_aversion: f64,
    ) -> Result<Array1<f64>> {
        let n_assets = expected_returns.len();

        // Simple equal-weight initialization
        let mut weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);

        // Gradient descent optimization
        let learning_rate = 0.01;
        let max_iterations = 1000;

        for _iter in 0..max_iterations {
            // Gradient = μ_adjusted - 2λ * Σ * w
            let portfolio_variance_grad = covariance_matrix.dot(&weights) * (2.0 * risk_aversion);
            let gradient = expected_returns - &portfolio_variance_grad;

            // Update weights
            weights = &weights + &(gradient * learning_rate);

            // Project onto constraints
            self.project_weights(&mut weights);
        }

        // Final normalization
        self.project_weights(&mut weights);

        Ok(weights)
    }

    /// Project weights onto feasible set (sum=1, within bounds)
    fn project_weights(&self, weights: &mut Array1<f64>) {
        let n = weights.len();

        // Apply bounds
        for w in weights.iter_mut() {
            *w = w.clamp(self.config.min_weight_per_asset, self.config.max_weight_per_asset);
        }

        // Normalize to sum to 1
        let sum: f64 = weights.sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        } else {
            // Fallback to equal weights
            for w in weights.iter_mut() {
                *w = 1.0 / n as f64;
            }
        }
    }

    /// Reconstruct price series from returns
    fn reconstruct_prices(&self, asset: &Asset) -> Vec<f64> {
        let mut prices = vec![asset.current_price];

        for &ret in asset.historical_returns.iter().rev() {
            let prev_price = prices.last().unwrap();
            let new_price = prev_price / (1.0 + ret);
            prices.push(new_price);
        }

        prices.reverse();
        prices
    }

    /// Calculate expected return for a given allocation
    pub fn calculate_expected_return(&self, assets: &[Asset], weights: &Array1<f64>) -> f64 {
        let mut expected_return = 0.0;

        for (asset, &weight) in assets.iter().zip(weights.iter()) {
            if !asset.historical_returns.is_empty() {
                let mean_return =
                    asset.historical_returns.iter().sum::<f64>() / asset.historical_returns.len() as f64;
                expected_return += weight * mean_return;
            }
        }

        expected_return
    }

    /// Calculate portfolio risk from covariance matrix
    fn calculate_risk_from_covariance(&self, covariance: &Array2<f64>, weights: &Array1<f64>) -> f64 {
        // Risk = sqrt(w^T * Σ * w)
        let variance = weights.dot(&covariance.dot(weights));
        variance.sqrt()
    }

    /// Calculate Sharpe ratio
    pub fn calculate_sharpe_ratio(&self, expected_return: f64, risk: f64) -> f64 {
        if risk == 0.0 {
            0.0
        } else {
            (expected_return - self.config.risk_free_rate) / risk
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_optimizer_creation() {
        let config = OptimizationConfig::default();
        let optimizer = PortfolioOptimizer::new(config);
        assert_eq!(optimizer.config.risk_free_rate, 0.02);
        assert!(optimizer.config.use_transfer_entropy);
        assert!(optimizer.config.use_regime_detection);
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let config = OptimizationConfig::default();
        let optimizer = PortfolioOptimizer::new(config);

        let sharpe = optimizer.calculate_sharpe_ratio(0.10, 0.15);
        assert!((sharpe - 0.533).abs() < 0.01);
    }

    #[test]
    fn test_simple_portfolio_optimization() {
        let mut config = OptimizationConfig::default();
        config.use_transfer_entropy = false;
        config.use_regime_detection = false;

        let mut optimizer = PortfolioOptimizer::new(config);

        let assets = vec![
            Asset {
                symbol: "AAPL".to_string(),
                name: "Apple Inc.".to_string(),
                current_price: 150.0,
                historical_returns: vec![0.01, 0.02, -0.01, 0.03, 0.01],
            },
            Asset {
                symbol: "GOOGL".to_string(),
                name: "Alphabet Inc.".to_string(),
                current_price: 2800.0,
                historical_returns: vec![0.02, 0.01, 0.01, 0.02, 0.015],
            },
        ];

        let result = optimizer.optimize(assets);
        assert!(result.is_ok());

        let portfolio = result.unwrap();
        let weight_sum: f64 = portfolio.weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_covariance_matrix_calculation() {
        let config = OptimizationConfig::default();
        let optimizer = PortfolioOptimizer::new(config);

        let assets = vec![
            Asset {
                symbol: "A".to_string(),
                name: "Asset A".to_string(),
                current_price: 100.0,
                historical_returns: vec![0.01, 0.02, 0.01, 0.02],
            },
            Asset {
                symbol: "B".to_string(),
                name: "Asset B".to_string(),
                current_price: 100.0,
                historical_returns: vec![0.02, 0.01, 0.02, 0.01],
            },
        ];

        let result = optimizer.calculate_covariance_matrix(&assets);
        assert!(result.is_ok());

        let cov = result.unwrap();
        assert_eq!(cov.shape(), &[2, 2]);
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
    }
}
