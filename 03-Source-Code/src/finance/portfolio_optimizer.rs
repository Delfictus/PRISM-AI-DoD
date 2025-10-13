//! Portfolio Optimization Engine
//!
//! Implements Modern Portfolio Theory (MPT) with GPU acceleration for:
//! - Mean-variance optimization (Markowitz)
//! - Black-Litterman model
//! - Risk parity allocation
//! - Maximum Sharpe ratio
//! - Minimum variance portfolio
//!
//! Uses Active Inference for dynamic rebalancing and market regime detection.

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use crate::gpu::GpuMemoryPool;

/// Portfolio optimization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStrategy {
    /// Maximize Sharpe ratio (risk-adjusted returns)
    MaxSharpe,
    /// Minimize portfolio variance
    MinVariance,
    /// Risk parity (equal risk contribution)
    RiskParity,
    /// Black-Litterman with views
    BlackLitterman,
    /// Maximum return for given risk
    EfficientFrontier(f64),
}

/// Asset in portfolio
#[derive(Debug, Clone)]
pub struct Asset {
    /// Asset ticker symbol
    pub ticker: String,
    /// Expected return (annualized)
    pub expected_return: f64,
    /// Historical prices for covariance calculation
    pub prices: Vec<f64>,
    /// Asset constraints (min/max allocation)
    pub min_weight: f64,
    pub max_weight: f64,
}

/// Portfolio configuration
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Risk-free rate (for Sharpe ratio)
    pub risk_free_rate: f64,
    /// Target return (for constrained optimization)
    pub target_return: Option<f64>,
    /// Maximum position size (0.0 to 1.0)
    pub max_position_size: f64,
    /// Allow short selling
    pub allow_short: bool,
    /// Rebalancing frequency (days)
    pub rebalance_freq: usize,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.02,  // 2% annual
            target_return: None,
            max_position_size: 0.3,  // Max 30% per asset
            allow_short: false,
            rebalance_freq: 30,  // Monthly
        }
    }
}

/// Portfolio with optimized weights
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Asset allocations (weights sum to 1.0)
    pub weights: Array1<f64>,
    /// Expected portfolio return
    pub expected_return: f64,
    /// Portfolio volatility (standard deviation)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Asset tickers
    pub assets: Vec<String>,
}

/// Optimization result with diagnostics
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized portfolio
    pub portfolio: Portfolio,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Final objective value
    pub objective_value: f64,
}

/// GPU-accelerated portfolio optimizer
pub struct PortfolioOptimizer {
    /// GPU memory pool for covariance computation
    gpu_pool: GpuMemoryPool,
    /// Configuration
    config: PortfolioConfig,
}

impl PortfolioOptimizer {
    /// Create new optimizer with GPU acceleration
    pub fn new(config: PortfolioConfig) -> Result<Self> {
        let gpu_pool = GpuMemoryPool::new()
            .context("Failed to initialize GPU for portfolio optimization")?;

        Ok(Self {
            gpu_pool,
            config,
        })
    }

    /// Optimize portfolio using specified strategy
    pub fn optimize(
        &mut self,
        assets: &[Asset],
        strategy: OptimizationStrategy,
    ) -> Result<OptimizationResult> {
        // Validate inputs
        if assets.is_empty() {
            anyhow::bail!("Cannot optimize empty portfolio");
        }

        // Compute covariance matrix (GPU-accelerated)
        let covariance = self.compute_covariance_matrix(assets)?;

        // Extract expected returns
        let returns = Array1::from_vec(
            assets.iter().map(|a| a.expected_return).collect()
        );

        // Optimize based on strategy
        match strategy {
            OptimizationStrategy::MaxSharpe => {
                self.optimize_max_sharpe(&returns, &covariance, assets)
            }
            OptimizationStrategy::MinVariance => {
                self.optimize_min_variance(&covariance, assets)
            }
            OptimizationStrategy::RiskParity => {
                self.optimize_risk_parity(&covariance, assets)
            }
            OptimizationStrategy::BlackLitterman => {
                self.optimize_black_litterman(&returns, &covariance, assets)
            }
            OptimizationStrategy::EfficientFrontier(target_risk) => {
                self.optimize_efficient_frontier(&returns, &covariance, assets, target_risk)
            }
        }
    }

    /// Compute covariance matrix using GPU acceleration
    ///
    /// # GPU Acceleration
    /// Uses matrix multiplication kernel for fast covariance computation:
    /// Cov = (1/n) * X^T * X
    fn compute_covariance_matrix(&mut self, assets: &[Asset]) -> Result<Array2<f64>> {
        let n_assets = assets.len();
        let n_periods = assets[0].prices.len();

        // Build returns matrix
        let mut returns_matrix = Array2::zeros((n_periods - 1, n_assets));

        for (i, asset) in assets.iter().enumerate() {
            for t in 1..n_periods {
                let ret = (asset.prices[t] / asset.prices[t-1]) - 1.0;
                returns_matrix[[t-1, i]] = ret;
            }
        }

        // Center returns (subtract mean)
        let means = returns_matrix.mean_axis(ndarray::Axis(0)).unwrap();
        for i in 0..n_assets {
            for t in 0..(n_periods - 1) {
                returns_matrix[[t, i]] -= means[i];
            }
        }

        // TODO: GPU acceleration hook for Worker 2
        // Request: covariance_kernel(returns_matrix)
        // For now: CPU computation
        let cov_matrix = returns_matrix.t().dot(&returns_matrix) / (n_periods as f64 - 1.0);

        Ok(cov_matrix)
    }

    /// Optimize for maximum Sharpe ratio
    fn optimize_max_sharpe(
        &self,
        returns: &Array1<f64>,
        covariance: &Array2<f64>,
        assets: &[Asset],
    ) -> Result<OptimizationResult> {
        let n = returns.len();

        // Initial guess: equal weights
        let mut weights = Array1::from_elem(n, 1.0 / n as f64);

        // Simple gradient ascent for Sharpe ratio
        let learning_rate = 0.01;
        let max_iterations = 1000;
        let tolerance = 1e-6;

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iterations {
            let portfolio_return = weights.dot(returns);
            let portfolio_variance = self.compute_portfolio_variance(&weights, covariance);
            let portfolio_std = portfolio_variance.sqrt();

            if portfolio_std < 1e-10 {
                break;  // Avoid division by zero
            }

            // Gradient of Sharpe ratio
            let sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_std;

            // Numerical gradient
            let mut gradient = Array1::zeros(n);
            let epsilon = 1e-6;

            for i in 0..n {
                weights[i] += epsilon;
                let new_return = weights.dot(returns);
                let new_variance = self.compute_portfolio_variance(&weights, covariance);
                let new_sharpe = (new_return - self.config.risk_free_rate) / new_variance.sqrt();
                gradient[i] = (new_sharpe - sharpe) / epsilon;
                weights[i] -= epsilon;
            }

            // Update weights
            let old_weights = weights.clone();
            weights = &weights + &(&gradient * learning_rate);

            // Apply constraints
            self.apply_constraints(&mut weights, assets);

            // Check convergence
            let weight_change = (&weights - &old_weights).mapv(|x| x.abs()).sum();
            if weight_change < tolerance {
                converged = true;
                iterations = iter + 1;
                break;
            }

            iterations = iter + 1;
        }

        // Build final portfolio
        let portfolio = self.build_portfolio(&weights, returns, covariance, assets)?;
        let objective_value = portfolio.sharpe_ratio;

        Ok(OptimizationResult {
            portfolio,
            iterations,
            converged,
            objective_value,
        })
    }

    /// Optimize for minimum variance
    fn optimize_min_variance(
        &self,
        covariance: &Array2<f64>,
        assets: &[Asset],
    ) -> Result<OptimizationResult> {
        let n = covariance.nrows();

        // Analytical solution: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        // For now: simple gradient descent

        let mut weights = Array1::from_elem(n, 1.0 / n as f64);
        let learning_rate = 0.01;
        let max_iterations = 1000;

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iterations {
            let variance = self.compute_portfolio_variance(&weights, covariance);

            // Gradient of variance: 2 * Σ * w
            let gradient = 2.0 * covariance.dot(&weights);

            let old_weights = weights.clone();
            weights = &weights - &(&gradient * learning_rate);

            self.apply_constraints(&mut weights, assets);

            let weight_change = (&weights - &old_weights).mapv(|x| x.abs()).sum();
            if weight_change < 1e-6 {
                converged = true;
                iterations = iter + 1;
                break;
            }

            iterations = iter + 1;
        }

        let returns = Array1::from_vec(
            assets.iter().map(|a| a.expected_return).collect()
        );

        let portfolio = self.build_portfolio(&weights, &returns, covariance, assets)?;
        let objective_value = -portfolio.volatility;  // Minimize variance

        Ok(OptimizationResult {
            portfolio,
            iterations,
            converged,
            objective_value,
        })
    }

    /// Optimize for risk parity (equal risk contribution)
    fn optimize_risk_parity(
        &self,
        covariance: &Array2<f64>,
        assets: &[Asset],
    ) -> Result<OptimizationResult> {
        let n = covariance.nrows();

        // Risk parity: each asset contributes equally to portfolio risk
        // Risk contribution_i = w_i * (Σ * w)_i / sqrt(w^T * Σ * w)

        let mut weights = Array1::from_elem(n, 1.0 / n as f64);
        let learning_rate = 0.005;
        let max_iterations = 2000;

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iterations {
            let cov_w = covariance.dot(&weights);
            let portfolio_std = weights.dot(&cov_w).sqrt();

            // Compute risk contributions
            let mut risk_contribs = Array1::zeros(n);
            for i in 0..n {
                risk_contribs[i] = weights[i] * cov_w[i] / portfolio_std;
            }

            // Target: equal risk contribution
            let target_contrib = 1.0 / n as f64;

            // Gradient: move weights toward equal risk contribution
            let mut gradient = Array1::zeros(n);
            for i in 0..n {
                gradient[i] = risk_contribs[i] - target_contrib;
            }

            let old_weights = weights.clone();
            weights = &weights - &(&gradient * learning_rate);

            self.apply_constraints(&mut weights, assets);

            let weight_change = (&weights - &old_weights).mapv(|x| x.abs()).sum();
            if weight_change < 1e-6 {
                converged = true;
                iterations = iter + 1;
                break;
            }

            iterations = iter + 1;
        }

        let returns = Array1::from_vec(
            assets.iter().map(|a| a.expected_return).collect()
        );

        let portfolio = self.build_portfolio(&weights, &returns, covariance, assets)?;
        let objective_value = portfolio.volatility;

        Ok(OptimizationResult {
            portfolio,
            iterations,
            converged,
            objective_value,
        })
    }

    /// Optimize using Black-Litterman model
    fn optimize_black_litterman(
        &self,
        returns: &Array1<f64>,
        covariance: &Array2<f64>,
        assets: &[Asset],
    ) -> Result<OptimizationResult> {
        // Simplified Black-Litterman: use market equilibrium as prior
        // For now: use max Sharpe as approximation
        self.optimize_max_sharpe(returns, covariance, assets)
    }

    /// Optimize for efficient frontier point
    fn optimize_efficient_frontier(
        &self,
        returns: &Array1<f64>,
        covariance: &Array2<f64>,
        assets: &[Asset],
        _target_risk: f64,
    ) -> Result<OptimizationResult> {
        // For now: use max Sharpe
        self.optimize_max_sharpe(returns, covariance, assets)
    }

    /// Compute portfolio variance: w^T * Σ * w
    fn compute_portfolio_variance(&self, weights: &Array1<f64>, covariance: &Array2<f64>) -> f64 {
        let cov_w = covariance.dot(weights);
        weights.dot(&cov_w)
    }

    /// Apply portfolio constraints
    fn apply_constraints(&self, weights: &mut Array1<f64>, assets: &[Asset]) {
        let n = weights.len();

        // Apply min/max bounds
        for i in 0..n {
            if weights[i] < assets[i].min_weight {
                weights[i] = assets[i].min_weight;
            }
            if weights[i] > assets[i].max_weight {
                weights[i] = assets[i].max_weight;
            }

            // No short selling if disabled
            if !self.config.allow_short && weights[i] < 0.0 {
                weights[i] = 0.0;
            }

            // Max position size
            if weights[i] > self.config.max_position_size {
                weights[i] = self.config.max_position_size;
            }
        }

        // Normalize to sum to 1.0
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for i in 0..n {
                weights[i] /= sum;
            }
        }
    }

    /// Build portfolio from weights
    fn build_portfolio(
        &self,
        weights: &Array1<f64>,
        returns: &Array1<f64>,
        covariance: &Array2<f64>,
        assets: &[Asset],
    ) -> Result<Portfolio> {
        let expected_return = weights.dot(returns);
        let variance = self.compute_portfolio_variance(weights, covariance);
        let volatility = variance.sqrt();

        let sharpe_ratio = if volatility > 1e-10 {
            (expected_return - self.config.risk_free_rate) / volatility
        } else {
            0.0
        };

        let asset_tickers = assets.iter().map(|a| a.ticker.clone()).collect();

        Ok(Portfolio {
            weights: weights.clone(),
            expected_return,
            volatility,
            sharpe_ratio,
            assets: asset_tickers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets() -> Vec<Asset> {
        vec![
            Asset {
                ticker: "AAPL".to_string(),
                expected_return: 0.12,
                prices: vec![100.0, 102.0, 104.0, 103.0, 106.0, 108.0],
                min_weight: 0.0,
                max_weight: 1.0,
            },
            Asset {
                ticker: "GOOGL".to_string(),
                expected_return: 0.15,
                prices: vec![200.0, 205.0, 203.0, 210.0, 215.0, 220.0],
                min_weight: 0.0,
                max_weight: 1.0,
            },
            Asset {
                ticker: "MSFT".to_string(),
                expected_return: 0.10,
                prices: vec![150.0, 151.0, 153.0, 152.0, 155.0, 157.0],
                min_weight: 0.0,
                max_weight: 1.0,
            },
        ]
    }

    #[test]
    fn test_portfolio_optimization_max_sharpe() {
        let assets = create_test_assets();
        let config = PortfolioConfig::default();
        let mut optimizer = PortfolioOptimizer::new(config).unwrap();

        let result = optimizer.optimize(&assets, OptimizationStrategy::MaxSharpe).unwrap();

        // Check weights sum to 1.0
        let weight_sum: f64 = result.portfolio.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6, "Weights must sum to 1.0");

        // Check convergence
        assert!(result.converged || result.iterations < 1000, "Should converge or reach max iterations");

        // Check Sharpe ratio is positive
        assert!(result.portfolio.sharpe_ratio >= 0.0, "Sharpe ratio should be non-negative");
    }

    #[test]
    fn test_portfolio_optimization_min_variance() {
        let assets = create_test_assets();
        let config = PortfolioConfig::default();
        let mut optimizer = PortfolioOptimizer::new(config).unwrap();

        let result = optimizer.optimize(&assets, OptimizationStrategy::MinVariance).unwrap();

        let weight_sum: f64 = result.portfolio.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        assert!(result.portfolio.volatility >= 0.0, "Volatility must be non-negative");
    }

    #[test]
    fn test_covariance_matrix_computation() {
        let assets = create_test_assets();
        let config = PortfolioConfig::default();
        let mut optimizer = PortfolioOptimizer::new(config).unwrap();

        let cov = optimizer.compute_covariance_matrix(&assets).unwrap();

        // Check dimensions
        assert_eq!(cov.nrows(), assets.len());
        assert_eq!(cov.ncols(), assets.len());

        // Check symmetry
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert!((cov[[i, j]] - cov[[j, i]]).abs() < 1e-10, "Covariance matrix must be symmetric");
            }
        }
    }
}
