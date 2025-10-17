//! Risk Analysis and Decomposition - Worker 4
//!
//! Advanced risk analytics for portfolio management:
//! - Factor risk decomposition
//! - Marginal contribution to risk (MCR)
//! - Value-at-Risk (VaR) - Historical, Parametric, Monte Carlo
//! - Conditional Value-at-Risk (CVaR) / Expected Shortfall
//! - Risk attribution and decomposition
//!
//! # Mathematical Foundation
//!
//! **Portfolio Risk**: σ_p = √(w^T Σ w)
//!
//! **Marginal Contribution to Risk (MCR)**:
//! MCR_i = ∂σ_p/∂w_i = (Σw)_i / σ_p
//!
//! **Component Contribution to Risk (CCR)**:
//! CCR_i = w_i × MCR_i
//!
//! **Value-at-Risk (VaR)**: Loss exceeded with probability α
//! Pr(Loss ≤ VaR_α) = α
//!
//! **Conditional VaR (CVaR)**: Expected loss given loss exceeds VaR
//! CVaR_α = E[Loss | Loss ≤ VaR_α]

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{Asset, Portfolio};

/// VaR calculation method
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VarMethod {
    /// Historical simulation (empirical quantiles)
    Historical,
    /// Parametric (assumes normal distribution)
    Parametric,
    /// Monte Carlo simulation
    MonteCarlo,
}

/// Risk decomposition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDecomposition {
    /// Total portfolio risk (standard deviation)
    pub total_risk: f64,

    /// Marginal contribution to risk for each asset
    pub marginal_contributions: Vec<f64>,

    /// Component contribution to risk for each asset (MCR × weight)
    pub component_contributions: Vec<f64>,

    /// Percentage contribution to risk for each asset
    pub percentage_contributions: Vec<f64>,

    /// Asset symbols
    pub asset_symbols: Vec<String>,
}

impl RiskDecomposition {
    /// Get risk contribution for a specific asset
    pub fn get_asset_contribution(&self, symbol: &str) -> Option<f64> {
        self.asset_symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| self.percentage_contributions[idx])
    }

    /// Get top N risk contributors
    pub fn top_contributors(&self, n: usize) -> Vec<(String, f64)> {
        let mut contributions: Vec<_> = self.asset_symbols
            .iter()
            .zip(self.percentage_contributions.iter())
            .map(|(s, &c)| (s.clone(), c))
            .collect();

        contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        contributions.into_iter().take(n).collect()
    }
}

/// Value-at-Risk (VaR) and CVaR results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarResult {
    /// Confidence level (e.g., 0.95 for 95% VaR)
    pub confidence_level: f64,

    /// Value-at-Risk (loss exceeded with probability 1-α)
    pub var: f64,

    /// Conditional VaR / Expected Shortfall
    pub cvar: f64,

    /// Method used for calculation
    pub method: VarMethod,

    /// Number of scenarios used (for Monte Carlo/Historical)
    pub num_scenarios: usize,
}

impl VarResult {
    /// Get VaR as a percentage of portfolio value
    pub fn var_percentage(&self, portfolio_value: f64) -> f64 {
        (self.var.abs() / portfolio_value) * 100.0
    }

    /// Get CVaR as a percentage of portfolio value
    pub fn cvar_percentage(&self, portfolio_value: f64) -> f64 {
        (self.cvar.abs() / portfolio_value) * 100.0
    }
}

/// Factor risk decomposition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorRiskDecomposition {
    /// Factor names
    pub factor_names: Vec<String>,

    /// Risk contribution from each factor
    pub factor_risks: Vec<f64>,

    /// Percentage contribution from each factor
    pub factor_percentages: Vec<f64>,

    /// Idiosyncratic (asset-specific) risk
    pub idiosyncratic_risk: f64,

    /// Systematic (factor-driven) risk
    pub systematic_risk: f64,
}

/// Risk analyzer for portfolios
pub struct RiskAnalyzer {
    /// Historical returns data for Monte Carlo
    historical_returns: Option<Vec<Array1<f64>>>,

    /// Number of Monte Carlo simulations
    num_simulations: usize,

    /// Random seed for reproducibility
    seed: u64,
}

impl RiskAnalyzer {
    /// Create a new risk analyzer
    pub fn new() -> Self {
        Self {
            historical_returns: None,
            num_simulations: 10000,
            seed: 42,
        }
    }

    /// Set historical returns data for analysis
    pub fn with_historical_returns(mut self, returns: Vec<Array1<f64>>) -> Self {
        self.historical_returns = Some(returns);
        self
    }

    /// Set number of Monte Carlo simulations
    pub fn with_num_simulations(mut self, num: usize) -> Self {
        self.num_simulations = num;
        self
    }

    /// Decompose portfolio risk into marginal and component contributions
    pub fn decompose_risk(
        &self,
        portfolio: &Portfolio,
        covariance_matrix: &Array2<f64>,
    ) -> Result<RiskDecomposition> {
        let n_assets = portfolio.assets.len();

        if covariance_matrix.nrows() != n_assets || covariance_matrix.ncols() != n_assets {
            anyhow::bail!(
                "Covariance matrix dimension mismatch: expected {}x{}, got {}x{}",
                n_assets, n_assets, covariance_matrix.nrows(), covariance_matrix.ncols()
            );
        }

        let weights = &portfolio.weights;
        let total_risk = portfolio.risk;

        // Calculate marginal contribution to risk: MCR_i = (Σw)_i / σ_p
        let sigma_w = covariance_matrix.dot(weights);
        let marginal_contributions: Vec<f64> = if total_risk > 0.0 {
            sigma_w.iter().map(|&x| x / total_risk).collect()
        } else {
            vec![0.0; n_assets]
        };

        // Calculate component contribution: CCR_i = w_i × MCR_i
        let component_contributions: Vec<f64> = weights
            .iter()
            .zip(marginal_contributions.iter())
            .map(|(&w, &mcr)| w * mcr)
            .collect();

        // Calculate percentage contributions
        let total_component = component_contributions.iter().sum::<f64>();
        let percentage_contributions: Vec<f64> = if total_component > 0.0 {
            component_contributions
                .iter()
                .map(|&ccr| (ccr / total_component) * 100.0)
                .collect()
        } else {
            vec![100.0 / n_assets as f64; n_assets]
        };

        let asset_symbols = portfolio.assets.iter().map(|a| a.symbol.clone()).collect();

        Ok(RiskDecomposition {
            total_risk,
            marginal_contributions,
            component_contributions,
            percentage_contributions,
            asset_symbols,
        })
    }

    /// Calculate Value-at-Risk (VaR) using specified method
    pub fn calculate_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
        method: VarMethod,
        covariance_matrix: Option<&Array2<f64>>,
    ) -> Result<VarResult> {
        if !(0.0..1.0).contains(&confidence_level) {
            anyhow::bail!("Confidence level must be between 0 and 1");
        }

        match method {
            VarMethod::Historical => self.calculate_historical_var(portfolio, confidence_level),
            VarMethod::Parametric => {
                let cov = covariance_matrix
                    .ok_or_else(|| anyhow::anyhow!("Covariance matrix required for parametric VaR"))?;
                self.calculate_parametric_var(portfolio, confidence_level, cov)
            }
            VarMethod::MonteCarlo => {
                let cov = covariance_matrix
                    .ok_or_else(|| anyhow::anyhow!("Covariance matrix required for Monte Carlo VaR"))?;
                self.calculate_monte_carlo_var(portfolio, confidence_level, cov)
            }
        }
    }

    /// Calculate VaR using historical simulation
    fn calculate_historical_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<VarResult> {
        // Collect historical portfolio returns
        let mut portfolio_returns = Vec::new();

        // Find minimum length across all assets
        let min_len = portfolio.assets
            .iter()
            .map(|a| a.historical_returns.len())
            .min()
            .unwrap_or(0);

        if min_len == 0 {
            anyhow::bail!("No historical returns available for VaR calculation");
        }

        // Calculate portfolio return for each historical period
        for t in 0..min_len {
            let mut portfolio_return = 0.0;
            for (asset, &weight) in portfolio.assets.iter().zip(portfolio.weights.iter()) {
                portfolio_return += weight * asset.historical_returns[t];
            }
            portfolio_returns.push(portfolio_return);
        }

        // Sort returns (ascending, so losses are at the beginning)
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // VaR is the α-quantile (loss side)
        let alpha = 1.0 - confidence_level;
        let var_index = (alpha * portfolio_returns.len() as f64).floor() as usize;
        let var = portfolio_returns[var_index];

        // CVaR is the average of losses beyond VaR
        let cvar = if var_index > 0 {
            portfolio_returns[0..=var_index].iter().sum::<f64>() / (var_index + 1) as f64
        } else {
            var
        };

        Ok(VarResult {
            confidence_level,
            var,
            cvar,
            method: VarMethod::Historical,
            num_scenarios: portfolio_returns.len(),
        })
    }

    /// Calculate VaR using parametric method (assumes normal distribution)
    fn calculate_parametric_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
        covariance_matrix: &Array2<f64>,
    ) -> Result<VarResult> {
        // Expected return
        let expected_return = portfolio.expected_return;

        // Portfolio volatility (standard deviation)
        let volatility = portfolio.risk;

        // For normal distribution: VaR = μ - z_α × σ
        // where z_α is the α-quantile of standard normal
        let alpha = 1.0 - confidence_level;
        let z_score = self.inverse_normal_cdf(alpha);

        let var = expected_return + z_score * volatility;

        // CVaR for normal distribution: CVaR = μ - σ × φ(z_α) / α
        // where φ is the standard normal PDF
        let pdf_value = self.normal_pdf(z_score);
        let cvar = expected_return - volatility * (pdf_value / alpha);

        Ok(VarResult {
            confidence_level,
            var,
            cvar,
            method: VarMethod::Parametric,
            num_scenarios: 0,
        })
    }

    /// Calculate VaR using Monte Carlo simulation
    fn calculate_monte_carlo_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
        covariance_matrix: &Array2<f64>,
    ) -> Result<VarResult> {
        // Generate simulated portfolio returns
        let expected_return = portfolio.expected_return;
        let n_assets = portfolio.assets.len();

        // Use simple random sampling (in production, use proper multivariate normal)
        let mut simulated_returns = Vec::with_capacity(self.num_simulations);

        // Simple Monte Carlo: sample from normal distribution
        use std::f64::consts::PI;
        let seed_base = self.seed;

        for i in 0..self.num_simulations {
            let mut portfolio_return = expected_return;

            // Add random noise based on portfolio volatility
            let u1 = ((seed_base + i as u64) as f64 / u64::MAX as f64).clamp(1e-10, 1.0 - 1e-10);
            let u2 = ((seed_base + i as u64 + 1000) as f64 / u64::MAX as f64).clamp(1e-10, 1.0 - 1e-10);

            // Box-Muller transform
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

            portfolio_return += z * portfolio.risk;
            simulated_returns.push(portfolio_return);
        }

        // Sort returns
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate VaR and CVaR
        let alpha = 1.0 - confidence_level;
        let var_index = (alpha * simulated_returns.len() as f64).floor() as usize;
        let var = simulated_returns[var_index];

        let cvar = if var_index > 0 {
            simulated_returns[0..=var_index].iter().sum::<f64>() / (var_index + 1) as f64
        } else {
            var
        };

        Ok(VarResult {
            confidence_level,
            var,
            cvar,
            method: VarMethod::MonteCarlo,
            num_scenarios: self.num_simulations,
        })
    }

    /// Approximate inverse normal CDF (for parametric VaR)
    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Approximation for standard normal quantile function
        // Good enough for risk calculations
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        // Rational approximation (Beasley-Springer-Moro algorithm)
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];

        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];

        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];

        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let q = if p < 0.5 { p } else { 1.0 - p };

        let r = if q > 0.02425 {
            q - 0.5
        } else {
            (q * 2.0 * std::f64::consts::PI).sqrt()
        };

        let mut num = 0.0;
        let mut den = 1.0;

        if q > 0.02425 {
            for i in 0..6 {
                num += a[i] * r.powi(i as i32);
            }
            for i in 0..5 {
                den += b[i] * r.powi((i + 1) as i32);
            }
        } else {
            for i in 0..6 {
                num += c[i] * r.powi(i as i32);
            }
            for i in 0..4 {
                den += d[i] * r.powi((i + 1) as i32);
            }
        }

        let result = num / den;

        if p < 0.5 {
            -result
        } else {
            result
        }
    }

    /// Standard normal PDF
    fn normal_pdf(&self, x: f64) -> f64 {
        use std::f64::consts::PI;
        (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
    }

    /// Calculate risk attribution by factor
    pub fn factor_decomposition(
        &self,
        portfolio: &Portfolio,
        factor_loadings: &Array2<f64>,
        factor_covariance: &Array2<f64>,
    ) -> Result<FactorRiskDecomposition> {
        let n_assets = portfolio.assets.len();
        let n_factors = factor_loadings.ncols();

        if factor_loadings.nrows() != n_assets {
            anyhow::bail!("Factor loadings dimension mismatch");
        }

        if factor_covariance.nrows() != n_factors || factor_covariance.ncols() != n_factors {
            anyhow::bail!("Factor covariance dimension mismatch");
        }

        let weights = &portfolio.weights;

        // Factor risk: w^T × B × F × B^T × w
        // where B is factor loadings, F is factor covariance
        let weighted_loadings = weights.dot(factor_loadings);
        let factor_variance = weighted_loadings.dot(&factor_covariance.dot(&weighted_loadings));
        let systematic_risk = factor_variance.sqrt();

        // Idiosyncratic risk (residual)
        let total_variance = portfolio.risk * portfolio.risk;
        let idiosyncratic_variance = (total_variance - factor_variance).max(0.0);
        let idiosyncratic_risk = idiosyncratic_variance.sqrt();

        // Individual factor contributions
        let factor_risks: Vec<f64> = (0..n_factors)
            .map(|f| {
                let factor_exposure = weighted_loadings[f];
                let factor_vol = factor_covariance[[f, f]].sqrt();
                (factor_exposure * factor_vol).abs()
            })
            .collect();

        let total_factor_risk: f64 = factor_risks.iter().sum();
        let factor_percentages: Vec<f64> = if total_factor_risk > 0.0 {
            factor_risks
                .iter()
                .map(|&r| (r / total_factor_risk) * 100.0)
                .collect()
        } else {
            vec![0.0; n_factors]
        };

        let factor_names: Vec<String> = (0..n_factors)
            .map(|i| format!("Factor_{}", i + 1))
            .collect();

        Ok(FactorRiskDecomposition {
            factor_names,
            factor_risks,
            factor_percentages,
            idiosyncratic_risk,
            systematic_risk,
        })
    }
}

impl Default for RiskAnalyzer {
    fn default() -> Self {
        Self::new()
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
        ];

        Portfolio {
            assets,
            weights: arr1(&[0.4, 0.3, 0.3]),
            expected_return: 0.015,
            risk: 0.012,
            sharpe_ratio: 1.25,
        }
    }

    fn create_test_covariance() -> Array2<f64> {
        Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0001, 0.00005, 0.00004,
                0.00005, 0.00012, 0.00006,
                0.00004, 0.00006, 0.00010,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_risk_decomposition() {
        let analyzer = RiskAnalyzer::new();
        let portfolio = create_test_portfolio();
        let covariance = create_test_covariance();

        let result = analyzer.decompose_risk(&portfolio, &covariance);
        assert!(result.is_ok());

        let decomp = result.unwrap();
        assert_eq!(decomp.asset_symbols.len(), 3);
        assert_eq!(decomp.marginal_contributions.len(), 3);
        assert_eq!(decomp.component_contributions.len(), 3);

        // Percentage contributions should sum to ~100%
        let total_pct: f64 = decomp.percentage_contributions.iter().sum();
        assert!((total_pct - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_historical_var() {
        let analyzer = RiskAnalyzer::new();
        let portfolio = create_test_portfolio();

        let result = analyzer.calculate_var(&portfolio, 0.95, VarMethod::Historical, None);
        assert!(result.is_ok());

        let var_result = result.unwrap();
        assert_eq!(var_result.confidence_level, 0.95);
        assert!(var_result.var < 0.0); // VaR should be negative (loss)
        assert!(var_result.cvar <= var_result.var); // CVaR should be worse than VaR
    }

    #[test]
    fn test_parametric_var() {
        let analyzer = RiskAnalyzer::new();
        let portfolio = create_test_portfolio();
        let covariance = create_test_covariance();

        let result = analyzer.calculate_var(&portfolio, 0.95, VarMethod::Parametric, Some(&covariance));
        assert!(result.is_ok());

        let var_result = result.unwrap();
        assert_eq!(var_result.method, VarMethod::Parametric);
        assert!(var_result.cvar <= var_result.var);
    }

    #[test]
    fn test_monte_carlo_var() {
        let analyzer = RiskAnalyzer::new().with_num_simulations(1000);
        let portfolio = create_test_portfolio();
        let covariance = create_test_covariance();

        let result = analyzer.calculate_var(&portfolio, 0.95, VarMethod::MonteCarlo, Some(&covariance));
        assert!(result.is_ok());

        let var_result = result.unwrap();
        assert_eq!(var_result.method, VarMethod::MonteCarlo);
        assert_eq!(var_result.num_scenarios, 1000);
    }

    #[test]
    fn test_top_contributors() {
        let analyzer = RiskAnalyzer::new();
        let portfolio = create_test_portfolio();
        let covariance = create_test_covariance();

        let decomp = analyzer.decompose_risk(&portfolio, &covariance).unwrap();
        let top_2 = decomp.top_contributors(2);

        assert_eq!(top_2.len(), 2);
        assert!(top_2[0].1 >= top_2[1].1); // First should have higher contribution
    }
}
