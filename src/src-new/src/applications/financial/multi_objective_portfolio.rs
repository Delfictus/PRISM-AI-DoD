//! Multi-Objective Portfolio Optimization - Worker 4
//!
//! Simultaneously optimize multiple portfolio objectives:
//! 1. **Maximize Return**: Expected portfolio return
//! 2. **Minimize Risk**: Portfolio volatility (standard deviation)
//! 3. **Minimize Turnover**: Reduction in transaction costs
//!
//! # Mathematical Foundation
//!
//! **Objectives**:
//! - f₁(w) = -E[R_p] (maximize return → minimize negative return)
//! - f₂(w) = σ_p = √(w^T Σ w)
//! - f₃(w) = ||w - w_current|| (turnover from current allocation)
//!
//! **Constraints**:
//! - Σ w_i = 1 (full investment)
//! - 0 ≤ w_i ≤ max_weight (position limits)

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::{Asset, Portfolio, PortfolioSerialized, PortfolioOptimizer, OptimizationConfig};
use crate::applications::solver::multi_objective::{
    MultiObjectiveProblem, NsgaIIOptimizer, NsgaIIConfig, ParetoFront, MultiObjectiveSolution,
};
use crate::applications::solver::{ProblemType, Problem};
use crate::applications::solver::problem::{ProblemData, ObjectiveFunction};

/// Multi-objective portfolio optimization configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Assets to optimize
    pub assets: Vec<Asset>,

    /// Current allocation (for turnover calculation)
    pub current_weights: Option<Array1<f64>>,

    /// Risk-free rate
    pub risk_free_rate: f64,

    /// Maximum weight per asset
    pub max_weight_per_asset: f64,

    /// Minimum weight per asset
    pub min_weight_per_asset: f64,

    /// NSGA-II configuration
    pub nsga_config: NsgaIIConfig,

    /// Whether to use forecasting
    pub use_forecasting: bool,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            assets: Vec::new(),
            current_weights: None,
            risk_free_rate: 0.02,
            max_weight_per_asset: 0.4,
            min_weight_per_asset: 0.0,
            nsga_config: NsgaIIConfig::default(),
            use_forecasting: false,
        }
    }
}

/// Result of multi-objective portfolio optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectivePortfolioResult {
    /// Pareto front of portfolios
    pub pareto_front: ParetoFront,

    /// Recommended portfolio (knee point)
    pub recommended_portfolio: PortfolioSerialized,

    /// Portfolio with maximum return
    pub max_return_portfolio: PortfolioSerialized,

    /// Portfolio with minimum risk
    pub min_risk_portfolio: PortfolioSerialized,

    /// Balanced portfolio (best Sharpe ratio on front)
    pub balanced_portfolio: PortfolioSerialized,
}

impl MultiObjectivePortfolioResult {
    /// Get portfolio by index on Pareto front
    pub fn get_portfolio(&self, index: usize, assets: &[Asset]) -> Option<Portfolio> {
        let solution = self.pareto_front.solutions.get(index)?;
        self.solution_to_portfolio(solution, assets)
    }

    /// Convert a multi-objective solution to a portfolio
    fn solution_to_portfolio(&self, solution: &MultiObjectiveSolution, assets: &[Asset]) -> Option<Portfolio> {
        if solution.solution.len() != assets.len() {
            return None;
        }

        let weights = Array1::from_vec(solution.solution.clone());
        let expected_return = -solution.objectives[0]; // Negative because we minimized negative return
        let risk = solution.objectives[1];

        let sharpe_ratio = if risk > 0.0 {
            (expected_return - 0.02) / risk // Using default risk-free rate
        } else {
            0.0
        };

        Some(Portfolio {
            assets: assets.to_vec(),
            weights,
            expected_return,
            risk,
            sharpe_ratio,
        })
    }
}

/// Multi-objective portfolio optimizer
pub struct MultiObjectivePortfolioOptimizer {
    config: MultiObjectiveConfig,
    covariance_matrix: Option<Array2<f64>>,
    expected_returns: Option<Array1<f64>>,
}

impl MultiObjectivePortfolioOptimizer {
    /// Create a new multi-objective portfolio optimizer
    pub fn new(config: MultiObjectiveConfig) -> Self {
        Self {
            config,
            covariance_matrix: None,
            expected_returns: None,
        }
    }

    /// Optimize portfolio with multiple objectives
    pub fn optimize(&mut self) -> Result<MultiObjectivePortfolioResult> {
        if self.config.assets.is_empty() {
            anyhow::bail!("Cannot optimize with no assets");
        }

        // Pre-calculate covariance matrix and expected returns
        self.calculate_statistics()?;

        // Create multi-objective problem
        let problem = self.create_multi_objective_problem()?;

        // Run NSGA-II
        let mut optimizer = NsgaIIOptimizer::new(self.config.nsga_config.clone());
        let pareto_front = optimizer.optimize(&problem)?;

        // Extract key portfolios from Pareto front
        self.extract_portfolios(pareto_front)
    }

    /// Calculate covariance matrix and expected returns
    fn calculate_statistics(&mut self) -> Result<()> {
        let single_obj_config = OptimizationConfig {
            risk_free_rate: self.config.risk_free_rate,
            max_weight_per_asset: self.config.max_weight_per_asset,
            min_weight_per_asset: self.config.min_weight_per_asset,
            use_forecasting: false,
            use_transfer_entropy: false,
            use_regime_detection: false,
            ..Default::default()
        };

        let optimizer = PortfolioOptimizer::new(single_obj_config);

        // Calculate covariance matrix
        let covariance = optimizer.calculate_covariance_matrix(&self.config.assets)?;
        self.covariance_matrix = Some(covariance);

        // Calculate expected returns
        let returns = optimizer.calculate_asset_returns(&self.config.assets);
        self.expected_returns = Some(returns);

        Ok(())
    }

    /// Create multi-objective problem specification
    fn create_multi_objective_problem(&self) -> Result<MultiObjectiveProblem> {
        let n_assets = self.config.assets.len();

        // Clone data for the closure
        let covariance = self.covariance_matrix.as_ref().unwrap().clone();
        let expected_returns = self.expected_returns.as_ref().unwrap().clone();
        let current_weights = self.config.current_weights.clone();

        // Create evaluation function
        let evaluate = move |weights: &Array1<f64>| -> Vec<f64> {
            // Objective 1: Maximize return (minimize negative return)
            let portfolio_return = expected_returns.dot(weights);
            let f1 = -portfolio_return;

            // Objective 2: Minimize risk (volatility)
            let portfolio_variance = weights.dot(&covariance.dot(weights));
            let f2 = portfolio_variance.sqrt();

            // Objective 3: Minimize turnover
            let f3 = if let Some(ref current) = current_weights {
                let diff = weights - current;
                diff.dot(&diff).sqrt()
            } else {
                0.0 // No turnover penalty if no current portfolio
            };

            vec![f1, f2, f3]
        };

        let variables: Vec<String> = self.config.assets
            .iter()
            .map(|a| a.symbol.clone())
            .collect();

        Ok(MultiObjectiveProblem {
            num_objectives: 3,
            objective_names: vec![
                "Negative Return".to_string(),
                "Risk (Volatility)".to_string(),
                "Turnover".to_string(),
            ],
            minimize: vec![true, true, true],
            problem: Problem {
                problem_type: ProblemType::PortfolioOptimization,
                description: "Multi-objective portfolio optimization".to_string(),
                data: ProblemData::Continuous {
                    variables: variables.clone(),
                    bounds: vec![(self.config.min_weight_per_asset, self.config.max_weight_per_asset); n_assets],
                    objective: ObjectiveFunction::Minimize("multi-objective".to_string()),
                },
                constraints: Vec::new(),
                metadata: std::collections::HashMap::new(),
            },
            evaluate_objectives: Box::new(evaluate),
        })
    }

    /// Extract key portfolios from Pareto front
    fn extract_portfolios(&self, pareto_front: ParetoFront) -> Result<MultiObjectivePortfolioResult> {
        if pareto_front.solutions.is_empty() {
            anyhow::bail!("Pareto front is empty");
        }

        let minimize = vec![true, true, true];

        // Find recommended portfolio (knee point)
        let knee_solution = pareto_front.knee_point(&minimize)
            .ok_or_else(|| anyhow::anyhow!("Could not find knee point"))?;
        let recommended_portfolio = self.solution_to_portfolio(knee_solution)?;

        // Find maximum return portfolio (minimum negative return)
        let max_return_solution = pareto_front.best_for_objective(0, true)
            .ok_or_else(|| anyhow::anyhow!("Could not find max return portfolio"))?;
        let max_return_portfolio = self.solution_to_portfolio(max_return_solution)?;

        // Find minimum risk portfolio
        let min_risk_solution = pareto_front.best_for_objective(1, true)
            .ok_or_else(|| anyhow::anyhow!("Could not find min risk portfolio"))?;
        let min_risk_portfolio = self.solution_to_portfolio(min_risk_solution)?;

        // Find balanced portfolio (best Sharpe ratio)
        let balanced_portfolio = self.find_best_sharpe_portfolio(&pareto_front)?;

        Ok(MultiObjectivePortfolioResult {
            pareto_front,
            recommended_portfolio: recommended_portfolio.into(),
            max_return_portfolio: max_return_portfolio.into(),
            min_risk_portfolio: min_risk_portfolio.into(),
            balanced_portfolio: balanced_portfolio.into(),
        })
    }

    /// Convert a multi-objective solution to a portfolio
    fn solution_to_portfolio(&self, solution: &MultiObjectiveSolution) -> Result<Portfolio> {
        let weights = Array1::from_vec(solution.solution.clone());

        // Normalize weights to sum to 1
        let weight_sum = weights.sum();
        let normalized_weights = if weight_sum > 0.0 {
            &weights / weight_sum
        } else {
            Array1::from_elem(weights.len(), 1.0 / weights.len() as f64)
        };

        let expected_return = -solution.objectives[0]; // Negative because we minimized negative return
        let risk = solution.objectives[1];

        let sharpe_ratio = if risk > 0.0 {
            (expected_return - self.config.risk_free_rate) / risk
        } else {
            0.0
        };

        Ok(Portfolio {
            assets: self.config.assets.clone(),
            weights: normalized_weights,
            expected_return,
            risk,
            sharpe_ratio,
        })
    }

    /// Find portfolio with best Sharpe ratio on Pareto front
    fn find_best_sharpe_portfolio(&self, pareto_front: &ParetoFront) -> Result<Portfolio> {
        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_solution = None;

        for solution in &pareto_front.solutions {
            let expected_return = -solution.objectives[0];
            let risk = solution.objectives[1];

            if risk > 0.0 {
                let sharpe = (expected_return - self.config.risk_free_rate) / risk;
                if sharpe > best_sharpe {
                    best_sharpe = sharpe;
                    best_solution = Some(solution);
                }
            }
        }

        let solution = best_solution
            .ok_or_else(|| anyhow::anyhow!("Could not find portfolio with positive risk"))?;

        self.solution_to_portfolio(solution)
    }

    /// Visualize Pareto front (returns data for plotting)
    pub fn get_pareto_front_data(&self, result: &MultiObjectivePortfolioResult) -> Vec<(f64, f64, f64)> {
        result.pareto_front.solutions
            .iter()
            .map(|sol| {
                let return_value = -sol.objectives[0]; // Convert back to positive return
                let risk = sol.objectives[1];
                let turnover = sol.objectives[2];
                (return_value, risk, turnover)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets() -> Vec<Asset> {
        vec![
            Asset {
                symbol: "AAPL".to_string(),
                name: "Apple Inc.".to_string(),
                current_price: 150.0,
                historical_returns: vec![0.01, 0.02, -0.01, 0.03, 0.01, 0.02],
            },
            Asset {
                symbol: "GOOGL".to_string(),
                name: "Alphabet Inc.".to_string(),
                current_price: 2800.0,
                historical_returns: vec![0.02, 0.01, 0.01, 0.02, 0.015, 0.018],
            },
            Asset {
                symbol: "MSFT".to_string(),
                name: "Microsoft Corp.".to_string(),
                current_price: 300.0,
                historical_returns: vec![0.015, 0.01, 0.005, 0.025, 0.012, 0.02],
            },
        ]
    }

    #[test]
    fn test_multi_objective_optimizer_creation() {
        let assets = create_test_assets();
        let config = MultiObjectiveConfig {
            assets,
            ..Default::default()
        };

        let optimizer = MultiObjectivePortfolioOptimizer::new(config);
        assert!(optimizer.covariance_matrix.is_none());
        assert!(optimizer.expected_returns.is_none());
    }

    #[test]
    fn test_multi_objective_optimization() {
        let assets = create_test_assets();
        let mut config = MultiObjectiveConfig {
            assets,
            ..Default::default()
        };

        config.nsga_config.population_size = 20;
        config.nsga_config.num_generations = 5;

        let mut optimizer = MultiObjectivePortfolioOptimizer::new(config);
        let result = optimizer.optimize();

        assert!(result.is_ok());
        let portfolios = result.unwrap();

        assert!(!portfolios.pareto_front.solutions.is_empty());
        assert!(portfolios.recommended_portfolio.weights.sum() > 0.99);
        assert!(portfolios.recommended_portfolio.weights.sum() < 1.01);
    }

    #[test]
    fn test_portfolio_extraction() {
        let assets = create_test_assets();
        let mut config = MultiObjectiveConfig {
            assets,
            ..Default::default()
        };

        config.nsga_config.population_size = 10;
        config.nsga_config.num_generations = 3;

        let mut optimizer = MultiObjectivePortfolioOptimizer::new(config);
        let result = optimizer.optimize().unwrap();

        // Check that different portfolios exist
        let max_return = result.max_return_portfolio.expected_return;
        let min_risk = result.min_risk_portfolio.risk;

        assert!(max_return >= result.min_risk_portfolio.expected_return);
        assert!(min_risk <= result.max_return_portfolio.risk);
    }

    #[test]
    fn test_pareto_front_data() {
        let assets = create_test_assets();
        let mut config = MultiObjectiveConfig {
            assets,
            ..Default::default()
        };

        config.nsga_config.population_size = 10;
        config.nsga_config.num_generations = 3;

        let mut optimizer = MultiObjectivePortfolioOptimizer::new(config);
        let result = optimizer.optimize().unwrap();

        let data = optimizer.get_pareto_front_data(&result);
        assert!(!data.is_empty());

        // Check that all objectives are reasonable
        for (return_val, risk, turnover) in data {
            assert!(return_val.is_finite());
            assert!(risk >= 0.0);
            assert!(turnover >= 0.0);
        }
    }

    #[test]
    fn test_with_current_weights() {
        let assets = create_test_assets();
        let current_weights = Array1::from_vec(vec![0.33, 0.33, 0.34]);

        let mut config = MultiObjectiveConfig {
            assets,
            current_weights: Some(current_weights),
            ..Default::default()
        };

        config.nsga_config.population_size = 10;
        config.nsga_config.num_generations = 3;

        let mut optimizer = MultiObjectivePortfolioOptimizer::new(config);
        let result = optimizer.optimize();

        assert!(result.is_ok());
    }
}
