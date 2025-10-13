//! Portfolio optimization using Modern Portfolio Theory
//!
//! Implements Markowitz mean-variance optimization and related algorithms
//! for optimal asset allocation under risk constraints.

use serde::{Deserialize, Serialize};

/// Portfolio optimizer using quadratic programming
#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    /// Risk-free rate for Sharpe ratio calculation
    risk_free_rate: f64,
}

impl PortfolioOptimizer {
    pub fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    /// Optimize portfolio using Markowitz mean-variance framework
    ///
    /// Solves: minimize w^T Σ w
    ///         subject to: μ^T w >= target_return
    ///                     Σ w_i = 1
    ///                     w_i >= 0 (long-only)
    ///                     w_i <= max_weight
    pub fn optimize_markowitz(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        target_return: Option<f64>,
        max_weight: f64,
    ) -> OptimizationResult {
        let n = expected_returns.len();

        if n == 0 {
            return OptimizationResult::default();
        }

        // If no target return specified, maximize Sharpe ratio
        let weights = if let Some(target) = target_return {
            self.optimize_for_target_return(expected_returns, covariance_matrix, target, max_weight)
        } else {
            self.maximize_sharpe_ratio(expected_returns, covariance_matrix, max_weight)
        };

        // Calculate portfolio metrics
        let portfolio_return = self.calculate_expected_return(&weights, expected_returns);
        let portfolio_risk = self.calculate_portfolio_risk(&weights, covariance_matrix);
        let sharpe_ratio = if portfolio_risk > 1e-10 {
            (portfolio_return - self.risk_free_rate) / portfolio_risk
        } else {
            0.0
        };

        OptimizationResult {
            weights,
            expected_return: portfolio_return,
            expected_risk: portfolio_risk,
            sharpe_ratio,
        }
    }

    /// Maximize Sharpe ratio: (μ^T w - rf) / sqrt(w^T Σ w)
    fn maximize_sharpe_ratio(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        max_weight: f64,
    ) -> Vec<f64> {
        let n = expected_returns.len();

        // Use gradient ascent to maximize Sharpe ratio
        let mut weights = vec![1.0 / n as f64; n];
        let learning_rate = 0.01;
        let iterations = 1000;

        for _ in 0..iterations {
            let portfolio_return = self.calculate_expected_return(&weights, expected_returns);
            let portfolio_risk = self.calculate_portfolio_risk(&weights, covariance_matrix);

            if portfolio_risk < 1e-10 {
                break;
            }

            let sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk;

            // Gradient of Sharpe ratio
            let mut gradient = vec![0.0; n];
            for i in 0..n {
                let return_gradient = expected_returns[i];

                // Risk gradient: d/dw_i sqrt(w^T Σ w) = (Σw)_i / sqrt(w^T Σ w)
                let mut risk_gradient = 0.0;
                for j in 0..n {
                    risk_gradient += covariance_matrix[i][j] * weights[j];
                }
                risk_gradient /= portfolio_risk;

                // Sharpe gradient: d/dw_i [(μ^T w - rf) / σ]
                gradient[i] = (return_gradient * portfolio_risk
                    - (portfolio_return - self.risk_free_rate) * risk_gradient)
                    / (portfolio_risk * portfolio_risk);
            }

            // Update weights with gradient ascent
            for i in 0..n {
                weights[i] += learning_rate * gradient[i];
                weights[i] = weights[i].max(0.0).min(max_weight); // Apply constraints
            }

            // Normalize to sum to 1
            let sum: f64 = weights.iter().sum();
            if sum > 1e-10 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }

        weights
    }

    /// Optimize for specific target return
    fn optimize_for_target_return(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        target_return: f64,
        max_weight: f64,
    ) -> Vec<f64> {
        let n = expected_returns.len();

        // Use Lagrangian method with equality constraint
        // L = w^T Σ w + λ1(target - μ^T w) + λ2(1 - Σw_i)

        let mut weights = vec![1.0 / n as f64; n];
        let learning_rate = 0.01;
        let iterations = 1000;

        for _ in 0..iterations {
            let portfolio_return = self.calculate_expected_return(&weights, expected_returns);
            let weights_sum: f64 = weights.iter().sum();

            // Lagrange multipliers (adaptive)
            let lambda1 = 10.0 * (target_return - portfolio_return);
            let lambda2 = 10.0 * (1.0 - weights_sum);

            // Gradient descent on Lagrangian
            let mut gradient = vec![0.0; n];
            for i in 0..n {
                // Risk gradient
                let mut risk_grad = 0.0;
                for j in 0..n {
                    risk_grad += 2.0 * covariance_matrix[i][j] * weights[j];
                }

                // Lagrangian gradient
                gradient[i] = risk_grad - lambda1 * expected_returns[i] - lambda2;
            }

            // Update weights
            for i in 0..n {
                weights[i] -= learning_rate * gradient[i];
                weights[i] = weights[i].max(0.0).min(max_weight);
            }

            // Normalize
            let sum: f64 = weights.iter().sum();
            if sum > 1e-10 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }

        weights
    }

    /// Calculate efficient frontier points
    pub fn efficient_frontier(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        num_points: usize,
    ) -> Vec<(f64, f64)> {
        // Find min and max possible returns
        let min_return = expected_returns.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_return = expected_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut frontier = Vec::new();

        for i in 0..num_points {
            let target_return = min_return + (max_return - min_return) * (i as f64 / (num_points - 1) as f64);
            let result = self.optimize_markowitz(expected_returns, covariance_matrix, Some(target_return), 1.0);
            frontier.push((result.expected_risk, result.expected_return));
        }

        frontier
    }

    /// Calculate Value at Risk (VaR) for portfolio
    pub fn calculate_var(
        &self,
        weights: &[f64],
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        confidence_level: f64,
        time_horizon_days: u32,
    ) -> f64 {
        let portfolio_return = self.calculate_expected_return(weights, expected_returns);
        let portfolio_std = self.calculate_portfolio_risk(weights, covariance_matrix);

        // Scale to time horizon
        let horizon_factor = (time_horizon_days as f64).sqrt();
        let horizon_return = portfolio_return * time_horizon_days as f64 / 252.0; // 252 trading days/year
        let horizon_std = portfolio_std * horizon_factor;

        // Z-score for confidence level (e.g., 95% = -1.645)
        let z_score = match confidence_level {
            0.90 => -1.282,
            0.95 => -1.645,
            0.99 => -2.326,
            _ => -1.645, // Default to 95%
        };

        // VaR = -(μ + z*σ)
        -(horizon_return + z_score * horizon_std)
    }

    /// Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
    pub fn calculate_cvar(
        &self,
        weights: &[f64],
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        confidence_level: f64,
        time_horizon_days: u32,
    ) -> f64 {
        let var = self.calculate_var(weights, expected_returns, covariance_matrix, confidence_level, time_horizon_days);

        // CVaR is approximately VaR * adjustment_factor
        // For normal distribution: CVaR/VaR ≈ φ(z) / (1 - α)
        // where φ is standard normal PDF, α is confidence level
        let adjustment_factor = match confidence_level {
            0.90 => 1.16,
            0.95 => 1.22,
            0.99 => 1.27,
            _ => 1.22,
        };

        var * adjustment_factor
    }

    /// Calculate maximum drawdown
    pub fn calculate_max_drawdown(
        &self,
        weights: &[f64],
        historical_returns: &[Vec<f64>],
    ) -> f64 {
        if historical_returns.is_empty() {
            return 0.0;
        }

        let mut cumulative_value = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for period_returns in historical_returns {
            // Calculate portfolio return for this period
            let portfolio_return: f64 = weights
                .iter()
                .zip(period_returns.iter())
                .map(|(w, r)| w * r)
                .sum();

            cumulative_value *= 1.0 + portfolio_return;
            peak = peak.max(cumulative_value);

            let drawdown = (peak - cumulative_value) / peak;
            max_dd = max_dd.max(drawdown);
        }

        max_dd
    }

    // Helper functions

    fn calculate_expected_return(&self, weights: &[f64], expected_returns: &[f64]) -> f64 {
        weights
            .iter()
            .zip(expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum()
    }

    fn calculate_portfolio_risk(&self, weights: &[f64], covariance_matrix: &[Vec<f64>]) -> f64 {
        let n = weights.len();
        let mut variance = 0.0;

        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * covariance_matrix[i][j];
            }
        }

        variance.sqrt()
    }
}

impl Default for PortfolioOptimizer {
    fn default() -> Self {
        Self::new(0.02) // 2% risk-free rate
    }
}

/// Result of portfolio optimization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationResult {
    /// Optimal asset weights
    pub weights: Vec<f64>,

    /// Expected portfolio return (annualized)
    pub expected_return: f64,

    /// Expected portfolio risk (annualized standard deviation)
    pub expected_risk: f64,

    /// Sharpe ratio: (return - rf) / risk
    pub sharpe_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weight_portfolio() {
        let optimizer = PortfolioOptimizer::default();
        let returns = vec![0.10, 0.12, 0.08];
        let cov_matrix = vec![
            vec![0.04, 0.01, 0.01],
            vec![0.01, 0.09, 0.01],
            vec![0.01, 0.01, 0.16],
        ];

        let weights = vec![1.0 / 3.0; 3];
        let portfolio_return = optimizer.calculate_expected_return(&weights, &returns);
        let portfolio_risk = optimizer.calculate_portfolio_risk(&weights, &cov_matrix);

        assert!(portfolio_return > 0.0);
        assert!(portfolio_risk > 0.0);
    }

    #[test]
    fn test_sharpe_optimization() {
        let optimizer = PortfolioOptimizer::new(0.02);
        let returns = vec![0.10, 0.12, 0.08];
        let cov_matrix = vec![
            vec![0.04, 0.01, 0.01],
            vec![0.01, 0.09, 0.01],
            vec![0.01, 0.01, 0.16],
        ];

        let result = optimizer.optimize_markowitz(&returns, &cov_matrix, None, 1.0);

        // Check constraints
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // Sum to 1
        assert!(result.weights.iter().all(|&w| w >= 0.0 && w <= 1.0)); // Long-only

        // Sharpe ratio should be positive
        assert!(result.sharpe_ratio > 0.0);
    }

    #[test]
    fn test_var_calculation() {
        let optimizer = PortfolioOptimizer::default();
        let weights = vec![0.6, 0.4];
        let returns = vec![0.10, 0.08];
        let cov_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let var = optimizer.calculate_var(&weights, &returns, &cov_matrix, 0.95, 10);

        // VaR should be positive (represents potential loss)
        assert!(var > 0.0);
    }

    #[test]
    fn test_efficient_frontier() {
        let optimizer = PortfolioOptimizer::default();
        let returns = vec![0.08, 0.12];
        let cov_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let frontier = optimizer.efficient_frontier(&returns, &cov_matrix, 10);

        assert_eq!(frontier.len(), 10);
        // Returns should generally increase with risk on efficient frontier
        assert!(frontier[0].1 <= frontier[frontier.len() - 1].1);
    }
}
