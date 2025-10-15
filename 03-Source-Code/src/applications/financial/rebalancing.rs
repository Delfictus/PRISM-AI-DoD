//! Portfolio Rebalancing - Worker 4
//!
//! Optimal rebalancing strategies that account for:
//! - Transaction costs (fixed + proportional)
//! - Tax implications (capital gains)
//! - Drift from target allocation
//! - Rebalancing frequency optimization
//! - Threshold-based vs. periodic rebalancing
//!
//! # Mathematical Foundation
//!
//! **Rebalancing Cost**:
//! C = Σ_i (c_fixed × I[Δw_i > 0] + c_prop × |Δw_i| × P_i)
//!
//! **Tax-Aware Rebalancing**:
//! Tax_i = τ × max(0, P_i - B_i) × |sell_i|
//! where τ is tax rate, P_i is current price, B_i is cost basis
//!
//! **Optimal Rebalancing**:
//! Minimize: Cost + Tax - Benefit
//! Subject to: Σ w_i = 1, w_i ≥ 0

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{Asset, Portfolio, PortfolioOptimizer, OptimizationConfig};

/// Rebalancing strategy type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RebalancingStrategy {
    /// Periodic rebalancing (e.g., monthly, quarterly)
    Periodic,
    /// Threshold-based (rebalance when drift exceeds threshold)
    Threshold,
    /// Tax-aware (minimize tax impact)
    TaxAware,
    /// Cost-minimizing (minimize transaction costs)
    CostMinimizing,
}

/// Transaction cost model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCost {
    /// Fixed cost per transaction (in dollars)
    pub fixed_cost: f64,

    /// Proportional cost (as fraction of transaction value)
    pub proportional_cost: f64,

    /// Minimum transaction size (to avoid excessive tiny trades)
    pub min_trade_size: f64,
}

impl Default for TransactionCost {
    fn default() -> Self {
        Self {
            fixed_cost: 0.0,          // $0 commission (modern brokers)
            proportional_cost: 0.0005, // 5 basis points spread
            min_trade_size: 100.0,     // $100 minimum trade
        }
    }
}

/// Tax configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxConfig {
    /// Short-term capital gains tax rate
    pub short_term_rate: f64,

    /// Long-term capital gains tax rate
    pub long_term_rate: f64,

    /// Holding period for long-term status (days)
    pub long_term_threshold_days: u32,

    /// Current cost basis for each asset (purchase price)
    pub cost_basis: HashMap<String, f64>,

    /// Holding period for each asset (days)
    pub holding_periods: HashMap<String, u32>,
}

impl Default for TaxConfig {
    fn default() -> Self {
        Self {
            short_term_rate: 0.37,  // 37% federal short-term rate
            long_term_rate: 0.20,   // 20% federal long-term rate
            long_term_threshold_days: 365,
            cost_basis: HashMap::new(),
            holding_periods: HashMap::new(),
        }
    }
}

/// Rebalancing configuration
#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    /// Rebalancing strategy
    pub strategy: RebalancingStrategy,

    /// Drift threshold for threshold-based rebalancing (as fraction, e.g., 0.05 = 5%)
    pub drift_threshold: f64,

    /// Transaction cost model
    pub transaction_cost: TransactionCost,

    /// Tax configuration
    pub tax_config: Option<TaxConfig>,

    /// Portfolio value (in dollars)
    pub portfolio_value: f64,

    /// Whether to allow partial rebalancing
    pub allow_partial: bool,
}

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            strategy: RebalancingStrategy::Threshold,
            drift_threshold: 0.05, // 5% drift triggers rebalancing
            transaction_cost: TransactionCost::default(),
            tax_config: None,
            portfolio_value: 1_000_000.0, // $1M default
            allow_partial: true,
        }
    }
}

/// Rebalancing action for a single asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingAction {
    /// Asset symbol
    pub symbol: String,

    /// Current weight
    pub current_weight: f64,

    /// Target weight
    pub target_weight: f64,

    /// Weight change
    pub weight_change: f64,

    /// Dollar amount to trade (positive = buy, negative = sell)
    pub trade_amount: f64,

    /// Number of shares to trade
    pub shares: f64,

    /// Transaction cost for this trade
    pub transaction_cost: f64,

    /// Tax impact (if applicable)
    pub tax_impact: f64,
}

/// Complete rebalancing plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingPlan {
    /// Individual rebalancing actions
    pub actions: Vec<RebalancingAction>,

    /// Total transaction cost
    pub total_cost: f64,

    /// Total tax impact
    pub total_tax: f64,

    /// Total cost (transaction + tax)
    pub total_impact: f64,

    /// Expected benefit from rebalancing (improved Sharpe ratio, reduced drift)
    pub expected_benefit: f64,

    /// Net benefit (benefit - cost - tax)
    pub net_benefit: f64,

    /// Whether rebalancing is recommended
    pub is_recommended: bool,
}

impl RebalancingPlan {
    /// Get actions that involve selling
    pub fn sell_actions(&self) -> Vec<&RebalancingAction> {
        self.actions.iter().filter(|a| a.trade_amount < 0.0).collect()
    }

    /// Get actions that involve buying
    pub fn buy_actions(&self) -> Vec<&RebalancingAction> {
        self.actions.iter().filter(|a| a.trade_amount > 0.0).collect()
    }

    /// Get total turnover (as fraction of portfolio value)
    pub fn turnover_rate(&self, portfolio_value: f64) -> f64 {
        let total_traded: f64 = self.actions.iter().map(|a| a.trade_amount.abs()).sum();
        total_traded / (2.0 * portfolio_value) // Divide by 2 because each rebalance involves buy and sell
    }
}

/// Portfolio rebalancer
pub struct PortfolioRebalancer {
    config: RebalancingConfig,
}

impl PortfolioRebalancer {
    /// Create a new rebalancer
    pub fn new(config: RebalancingConfig) -> Self {
        Self { config }
    }

    /// Check if rebalancing is needed based on drift
    pub fn needs_rebalancing(
        &self,
        current_portfolio: &Portfolio,
        target_weights: &Array1<f64>,
    ) -> bool {
        match self.config.strategy {
            RebalancingStrategy::Threshold => {
                self.check_drift_threshold(current_portfolio, target_weights)
            }
            RebalancingStrategy::Periodic => true, // Always rebalance on schedule
            _ => {
                // For tax-aware and cost-minimizing, check if benefit exceeds cost
                let plan = self.create_rebalancing_plan(current_portfolio, target_weights)
                    .unwrap_or_else(|_| RebalancingPlan {
                        actions: Vec::new(),
                        total_cost: 0.0,
                        total_tax: 0.0,
                        total_impact: 0.0,
                        expected_benefit: 0.0,
                        net_benefit: 0.0,
                        is_recommended: false,
                    });
                plan.is_recommended
            }
        }
    }

    /// Check if drift exceeds threshold
    fn check_drift_threshold(
        &self,
        current_portfolio: &Portfolio,
        target_weights: &Array1<f64>,
    ) -> bool {
        let max_drift = current_portfolio
            .weights
            .iter()
            .zip(target_weights.iter())
            .map(|(&current, &target)| (current - target).abs())
            .fold(0.0, f64::max);

        max_drift > self.config.drift_threshold
    }

    /// Create a complete rebalancing plan
    pub fn create_rebalancing_plan(
        &self,
        current_portfolio: &Portfolio,
        target_weights: &Array1<f64>,
    ) -> Result<RebalancingPlan> {
        if current_portfolio.assets.len() != target_weights.len() {
            anyhow::bail!("Asset count mismatch");
        }

        let mut actions = Vec::new();
        let mut total_cost = 0.0;
        let mut total_tax = 0.0;

        // Calculate rebalancing actions for each asset
        for (i, asset) in current_portfolio.assets.iter().enumerate() {
            let current_weight = current_portfolio.weights[i];
            let target_weight = target_weights[i];
            let weight_change = target_weight - current_weight;

            let trade_amount = weight_change * self.config.portfolio_value;

            // Skip if trade is too small
            if trade_amount.abs() < self.config.transaction_cost.min_trade_size {
                continue;
            }

            let shares = trade_amount / asset.current_price;

            // Calculate transaction cost
            let transaction_cost = self.calculate_transaction_cost(trade_amount.abs());

            // Calculate tax impact (only on sales)
            let tax_impact = if trade_amount < 0.0 {
                self.calculate_tax_impact(asset, trade_amount.abs())
            } else {
                0.0
            };

            total_cost += transaction_cost;
            total_tax += tax_impact;

            actions.push(RebalancingAction {
                symbol: asset.symbol.clone(),
                current_weight,
                target_weight,
                weight_change,
                trade_amount,
                shares,
                transaction_cost,
                tax_impact,
            });
        }

        let total_impact = total_cost + total_tax;

        // Estimate benefit from rebalancing
        let expected_benefit = self.estimate_rebalancing_benefit(
            current_portfolio,
            target_weights,
        );

        let net_benefit = expected_benefit - total_impact;

        // Recommend rebalancing if net benefit is positive
        let is_recommended = net_benefit > 0.0;

        Ok(RebalancingPlan {
            actions,
            total_cost,
            total_tax,
            total_impact,
            expected_benefit,
            net_benefit,
            is_recommended,
        })
    }

    /// Execute rebalancing and return new portfolio
    pub fn rebalance(
        &self,
        current_portfolio: &Portfolio,
        target_weights: &Array1<f64>,
    ) -> Result<Portfolio> {
        let plan = self.create_rebalancing_plan(current_portfolio, target_weights)?;

        if !plan.is_recommended && !self.config.allow_partial {
            anyhow::bail!("Rebalancing not recommended due to high costs");
        }

        // Create new portfolio with target weights
        let mut new_portfolio = current_portfolio.clone();
        new_portfolio.weights = target_weights.clone();

        // Recalculate portfolio metrics
        // In production, this would use actual optimizer
        let expected_return = self.calculate_expected_return(&new_portfolio);
        new_portfolio.expected_return = expected_return;

        Ok(new_portfolio)
    }

    /// Calculate transaction cost for a trade
    fn calculate_transaction_cost(&self, trade_value: f64) -> f64 {
        let fixed = if trade_value > 0.0 {
            self.config.transaction_cost.fixed_cost
        } else {
            0.0
        };

        let proportional = trade_value * self.config.transaction_cost.proportional_cost;

        fixed + proportional
    }

    /// Calculate tax impact for selling an asset
    fn calculate_tax_impact(&self, asset: &Asset, sell_value: f64) -> f64 {
        if let Some(tax_config) = &self.config.tax_config {
            // Get cost basis
            let cost_basis = tax_config
                .cost_basis
                .get(&asset.symbol)
                .copied()
                .unwrap_or(asset.current_price);

            // Calculate capital gain
            let capital_gain = (asset.current_price - cost_basis) * (sell_value / asset.current_price);

            if capital_gain <= 0.0 {
                return 0.0; // No tax on losses (simplification)
            }

            // Determine tax rate based on holding period
            let holding_days = tax_config
                .holding_periods
                .get(&asset.symbol)
                .copied()
                .unwrap_or(0);

            let tax_rate = if holding_days >= tax_config.long_term_threshold_days {
                tax_config.long_term_rate
            } else {
                tax_config.short_term_rate
            };

            capital_gain * tax_rate
        } else {
            0.0
        }
    }

    /// Estimate benefit from rebalancing
    fn estimate_rebalancing_benefit(
        &self,
        current_portfolio: &Portfolio,
        target_weights: &Array1<f64>,
    ) -> f64 {
        // Simple heuristic: benefit proportional to drift magnitude
        // In production, would calculate expected improvement in Sharpe ratio

        let total_drift: f64 = current_portfolio
            .weights
            .iter()
            .zip(target_weights.iter())
            .map(|(&current, &target)| (current - target).abs())
            .sum();

        // Assume benefit is 1% of portfolio value per 10% drift
        let benefit_rate = 0.01 * (total_drift / 0.10);

        benefit_rate * self.config.portfolio_value
    }

    /// Calculate expected return for portfolio
    fn calculate_expected_return(&self, portfolio: &Portfolio) -> f64 {
        let mut expected_return = 0.0;

        for (asset, &weight) in portfolio.assets.iter().zip(portfolio.weights.iter()) {
            if !asset.historical_returns.is_empty() {
                let mean_return = asset.historical_returns.iter().sum::<f64>()
                    / asset.historical_returns.len() as f64;
                expected_return += weight * mean_return;
            }
        }

        expected_return
    }

    /// Optimize rebalancing frequency
    pub fn optimize_rebalancing_frequency(
        &self,
        historical_portfolios: &[Portfolio],
        target_weights: &Array1<f64>,
        candidate_frequencies: &[u32], // in days
    ) -> Result<(u32, f64)> {
        let mut best_frequency = candidate_frequencies[0];
        let mut best_net_benefit = f64::NEG_INFINITY;

        for &frequency in candidate_frequencies {
            // Simulate rebalancing at this frequency
            let total_benefit = self.simulate_rebalancing_schedule(
                historical_portfolios,
                target_weights,
                frequency,
            )?;

            if total_benefit > best_net_benefit {
                best_net_benefit = total_benefit;
                best_frequency = frequency;
            }
        }

        Ok((best_frequency, best_net_benefit))
    }

    /// Simulate rebalancing on a schedule
    fn simulate_rebalancing_schedule(
        &self,
        historical_portfolios: &[Portfolio],
        target_weights: &Array1<f64>,
        frequency_days: u32,
    ) -> Result<f64> {
        let mut total_net_benefit = 0.0;
        let mut days_since_rebalance = 0;

        for portfolio in historical_portfolios {
            days_since_rebalance += 1;

            if days_since_rebalance >= frequency_days {
                // Time to rebalance
                let plan = self.create_rebalancing_plan(portfolio, target_weights)?;
                total_net_benefit += plan.net_benefit;
                days_since_rebalance = 0;
            }
        }

        Ok(total_net_benefit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn create_test_portfolio() -> Portfolio {
        let assets = vec![
            Asset {
                symbol: "AAPL".to_string(),
                name: "Apple Inc.".to_string(),
                current_price: 150.0,
                historical_returns: vec![0.01, 0.02, -0.01, 0.03],
            },
            Asset {
                symbol: "GOOGL".to_string(),
                name: "Alphabet Inc.".to_string(),
                current_price: 2800.0,
                historical_returns: vec![0.02, 0.01, 0.01, 0.02],
            },
            Asset {
                symbol: "MSFT".to_string(),
                name: "Microsoft Corp.".to_string(),
                current_price: 300.0,
                historical_returns: vec![0.015, 0.01, 0.005, 0.025],
            },
        ];

        Portfolio {
            assets,
            weights: arr1(&[0.45, 0.35, 0.20]), // Drifted from target
            expected_return: 0.015,
            risk: 0.012,
            sharpe_ratio: 1.25,
        }
    }

    #[test]
    fn test_rebalancing_plan_creation() {
        let config = RebalancingConfig::default();
        let rebalancer = PortfolioRebalancer::new(config);
        let portfolio = create_test_portfolio();
        let target_weights = arr1(&[0.40, 0.30, 0.30]); // Target allocation

        let result = rebalancer.create_rebalancing_plan(&portfolio, &target_weights);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(!plan.actions.is_empty());
        assert!(plan.total_cost >= 0.0);
        assert!(plan.total_tax >= 0.0);
    }

    #[test]
    fn test_drift_detection() {
        let config = RebalancingConfig {
            drift_threshold: 0.05,
            ..Default::default()
        };
        let rebalancer = PortfolioRebalancer::new(config);
        let portfolio = create_test_portfolio();
        let target_weights = arr1(&[0.40, 0.30, 0.30]);

        let needs_rebalancing = rebalancer.needs_rebalancing(&portfolio, &target_weights);
        assert!(needs_rebalancing); // 0.45 vs 0.40 = 5% drift
    }

    #[test]
    fn test_transaction_cost_calculation() {
        let config = RebalancingConfig {
            transaction_cost: TransactionCost {
                fixed_cost: 10.0,
                proportional_cost: 0.001,
                min_trade_size: 100.0,
            },
            ..Default::default()
        };
        let rebalancer = PortfolioRebalancer::new(config);

        let cost = rebalancer.calculate_transaction_cost(10000.0);
        assert_eq!(cost, 10.0 + 10.0); // $10 fixed + $10 proportional (0.1% of $10k)
    }

    #[test]
    fn test_tax_aware_rebalancing() {
        let mut tax_config = TaxConfig::default();
        tax_config.cost_basis.insert("AAPL".to_string(), 100.0);
        tax_config.holding_periods.insert("AAPL".to_string(), 400);

        let config = RebalancingConfig {
            tax_config: Some(tax_config),
            portfolio_value: 100000.0,
            ..Default::default()
        };

        let rebalancer = PortfolioRebalancer::new(config);
        let portfolio = create_test_portfolio();
        let target_weights = arr1(&[0.30, 0.35, 0.35]); // Sell AAPL

        let plan = rebalancer.create_rebalancing_plan(&portfolio, &target_weights).unwrap();

        // Should have tax impact on AAPL sale
        let aapl_action = plan.actions.iter().find(|a| a.symbol == "AAPL");
        assert!(aapl_action.is_some());
        assert!(plan.total_tax >= 0.0);
    }

    #[test]
    fn test_rebalancing_execution() {
        let config = RebalancingConfig::default();
        let rebalancer = PortfolioRebalancer::new(config);
        let portfolio = create_test_portfolio();
        let target_weights = arr1(&[0.40, 0.30, 0.30]);

        let result = rebalancer.rebalance(&portfolio, &target_weights);
        assert!(result.is_ok());

        let new_portfolio = result.unwrap();
        assert_eq!(new_portfolio.weights, target_weights);
    }

    #[test]
    fn test_buy_and_sell_actions() {
        let config = RebalancingConfig::default();
        let rebalancer = PortfolioRebalancer::new(config);
        let portfolio = create_test_portfolio();
        let target_weights = arr1(&[0.30, 0.35, 0.35]);

        let plan = rebalancer.create_rebalancing_plan(&portfolio, &target_weights).unwrap();

        let buy_actions = plan.buy_actions();
        let sell_actions = plan.sell_actions();

        assert!(!buy_actions.is_empty());
        assert!(!sell_actions.is_empty());
    }
}
