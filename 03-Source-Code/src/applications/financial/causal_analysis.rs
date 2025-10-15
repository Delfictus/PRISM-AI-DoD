//! Causal Analysis for Financial Markets - Worker 4 Phase 2
//!
//! Integrates Transfer Entropy (Worker 1) with financial portfolio optimization.
//! Identifies causal relationships (Granger causality) between assets using information flow.
//!
//! # Key Concepts
//!
//! **Transfer Entropy (TE)**:
//! - Measures information flow from asset X to asset Y
//! - TE(X→Y) > 0: X contains information about Y's future
//! - TE(X→Y) ≠ TE(Y→X): Directional (asymmetric)
//! - Correlation ≠ Causation: TE detects causal relationships
//!
//! # Use Cases
//!
//! 1. **Portfolio Diversification**: Avoid assets with strong causal links
//! 2. **Lead-Lag Trading**: Trade on leading indicators
//! 3. **Risk Decomposition**: Identify causal vs coincidental risk
//! 4. **Market Regime Detection**: Detect regime changes from TE shifts

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::information_theory::TransferEntropy;
use super::{Asset, MarketRegime, MarketRegimeDetector};

/// Causal relationship between two assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationship {
    /// Source asset (predictor)
    pub source: String,

    /// Target asset (predicted)
    pub target: String,

    /// Transfer entropy from source to target
    pub transfer_entropy: f64,

    /// Statistical significance (p-value)
    pub p_value: f64,

    /// Effective TE (after bias correction)
    pub effective_te: f64,

    /// Time lag (periods)
    pub time_lag: usize,

    /// Relationship strength (weak/moderate/strong)
    pub strength: RelationshipStrength,
}

/// Strength of causal relationship
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RelationshipStrength {
    Weak,
    Moderate,
    Strong,
}

/// Transfer Entropy Matrix for all assets
#[derive(Debug, Clone)]
pub struct TransferEntropyMatrix {
    /// Asset symbols
    pub assets: Vec<String>,

    /// TE matrix[i][j] = TE from asset i to asset j
    pub te_matrix: Vec<Vec<f64>>,

    /// Significance matrix (p-values)
    pub p_value_matrix: Vec<Vec<f64>>,

    /// Network statistics
    pub stats: NetworkStatistics,
}

/// Network-level causal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatistics {
    /// Total information flow in the network
    pub total_information_flow: f64,

    /// Average TE (excluding diagonal)
    pub avg_transfer_entropy: f64,

    /// Network density (% of significant causal links)
    pub network_density: f64,

    /// Leading assets (high out-degree)
    pub leading_assets: Vec<(String, f64)>,

    /// Lagging assets (high in-degree)
    pub lagging_assets: Vec<(String, f64)>,

    /// Hub assets (high total degree)
    pub hub_assets: Vec<(String, f64)>,
}

/// Causal portfolio optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPortfolioResult {
    /// Optimal weights with causal adjustment
    pub optimal_weights: Vec<f64>,

    /// Expected return
    pub expected_return: f64,

    /// Portfolio risk (volatility)
    pub portfolio_risk: f64,

    /// Sharpe ratio
    pub sharpe_ratio: f64,

    /// Causal diversification score (0-1, higher is better)
    pub causal_diversification: f64,

    /// Explanation
    pub explanation: String,
}

/// Configuration for causal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAnalysisConfig {
    /// Minimum data points required for TE calculation
    pub min_data_points: usize,

    /// Significance threshold (p-value)
    pub significance_threshold: f64,

    /// Maximum time lag to consider
    pub max_lag: usize,

    /// TE threshold for "weak" relationship
    pub weak_threshold: f64,

    /// TE threshold for "strong" relationship
    pub strong_threshold: f64,

    /// Weight for causality in portfolio optimization (0.0-1.0)
    pub causality_weight: f64,
}

impl Default for CausalAnalysisConfig {
    fn default() -> Self {
        Self {
            min_data_points: 30,
            significance_threshold: 0.05,
            max_lag: 5,
            weak_threshold: 0.05,
            strong_threshold: 0.15,
            causality_weight: 0.3,
        }
    }
}

/// Causal Analyzer for Financial Markets
pub struct CausalityAnalyzer {
    /// Transfer Entropy calculator
    te_calculator: TransferEntropy,

    /// Configuration
    config: CausalAnalysisConfig,

    /// Cached TE matrix (for efficiency)
    cached_te_matrix: Option<TransferEntropyMatrix>,
}

impl CausalityAnalyzer {
    /// Create new causal analyzer
    pub fn new(config: CausalAnalysisConfig) -> Self {
        Self {
            te_calculator: TransferEntropy::default(),
            config,
            cached_te_matrix: None,
        }
    }

    /// Compute Transfer Entropy matrix for all asset pairs
    pub fn compute_te_matrix(&mut self, assets: &[Asset]) -> Result<TransferEntropyMatrix> {
        let n = assets.len();

        if n < 2 {
            anyhow::bail!("Need at least 2 assets for causal analysis");
        }

        // Initialize matrices
        let mut te_matrix = vec![vec![0.0; n]; n];
        let mut p_value_matrix = vec![vec![1.0; n]; n];

        // Compute pairwise TE
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Check minimum data points
                    let min_len = assets[i].historical_returns.len().min(assets[j].historical_returns.len());

                    if min_len >= self.config.min_data_points {
                        let source = Array1::from_vec(assets[i].historical_returns.clone());
                        let target = Array1::from_vec(assets[j].historical_returns.clone());

                        // Calculate TE from asset i to asset j
                        let result = self.te_calculator.calculate(&source, &target);

                        te_matrix[i][j] = result.te_value;
                        p_value_matrix[i][j] = result.p_value;
                    }
                }
            }
        }

        // Compute network statistics
        let stats = self.compute_network_statistics(assets, &te_matrix, &p_value_matrix)?;

        let te_mat = TransferEntropyMatrix {
            assets: assets.iter().map(|a| a.symbol.clone()).collect(),
            te_matrix,
            p_value_matrix,
            stats,
        };

        // Cache for future use
        self.cached_te_matrix = Some(te_mat.clone());

        Ok(te_mat)
    }

    /// Compute network-level statistics
    fn compute_network_statistics(
        &self,
        assets: &[Asset],
        te_matrix: &[Vec<f64>],
        p_value_matrix: &[Vec<f64>],
    ) -> Result<NetworkStatistics> {
        let n = assets.len();

        // Total information flow
        let total_flow: f64 = te_matrix.iter()
            .flat_map(|row| row.iter())
            .sum();

        // Average TE (excluding diagonal)
        let avg_te: f64 = total_flow / (n * (n - 1)) as f64;

        // Network density (% of significant links)
        let mut num_significant = 0;
        for i in 0..n {
            for j in 0..n {
                if i != j && p_value_matrix[i][j] < self.config.significance_threshold {
                    num_significant += 1;
                }
            }
        }
        let network_density = num_significant as f64 / (n * (n - 1)) as f64;

        // Out-degree (information output) for each asset
        let mut out_degrees: Vec<(String, f64)> = Vec::new();
        for i in 0..n {
            let out_degree: f64 = te_matrix[i].iter()
                .enumerate()
                .filter(|(j, _)| *j != i && p_value_matrix[i][*j] < self.config.significance_threshold)
                .map(|(_, &te)| te)
                .sum();
            out_degrees.push((assets[i].symbol.clone(), out_degree));
        }

        // In-degree (information input) for each asset
        let mut in_degrees: Vec<(String, f64)> = Vec::new();
        for j in 0..n {
            let in_degree: f64 = (0..n)
                .filter(|&i| i != j && p_value_matrix[i][j] < self.config.significance_threshold)
                .map(|i| te_matrix[i][j])
                .sum();
            in_degrees.push((assets[j].symbol.clone(), in_degree));
        }

        // Total degree (hub assets)
        let mut total_degrees: Vec<(String, f64)> = Vec::new();
        for i in 0..n {
            let total_degree = out_degrees[i].1 + in_degrees[i].1;
            total_degrees.push((assets[i].symbol.clone(), total_degree));
        }

        // Sort and get top 5
        out_degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        in_degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        total_degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let leading_assets = out_degrees.into_iter().take(5).collect();
        let lagging_assets = in_degrees.into_iter().take(5).collect();
        let hub_assets = total_degrees.into_iter().take(5).collect();

        Ok(NetworkStatistics {
            total_information_flow: total_flow,
            avg_transfer_entropy: avg_te,
            network_density,
            leading_assets,
            lagging_assets,
            hub_assets,
        })
    }

    /// Identify all significant causal relationships
    pub fn identify_causal_relationships(&self, te_matrix: &TransferEntropyMatrix) -> Vec<CausalRelationship> {
        let mut relationships = Vec::new();

        let n = te_matrix.assets.len();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let te = te_matrix.te_matrix[i][j];
                    let p_value = te_matrix.p_value_matrix[i][j];

                    if p_value < self.config.significance_threshold {
                        let strength = if te < self.config.weak_threshold {
                            RelationshipStrength::Weak
                        } else if te < self.config.strong_threshold {
                            RelationshipStrength::Moderate
                        } else {
                            RelationshipStrength::Strong
                        };

                        relationships.push(CausalRelationship {
                            source: te_matrix.assets[i].clone(),
                            target: te_matrix.assets[j].clone(),
                            transfer_entropy: te,
                            p_value,
                            effective_te: te * 0.9, // Simple bias correction
                            time_lag: 1,
                            strength,
                        });
                    }
                }
            }
        }

        // Sort by TE strength (descending)
        relationships.sort_by(|a, b| b.transfer_entropy.partial_cmp(&a.transfer_entropy).unwrap());

        relationships
    }

    /// Optimize portfolio with causal diversification
    pub fn optimize_with_causality(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        te_matrix: &TransferEntropyMatrix,
        risk_free_rate: f64,
    ) -> Result<CausalPortfolioResult> {
        let n = expected_returns.len();

        if n != te_matrix.assets.len() {
            anyhow::bail!("Dimension mismatch: returns and TE matrix");
        }

        // Compute causal penalty for each asset
        let causal_penalties = self.compute_causal_penalties(te_matrix);

        // Adjust expected returns with causal penalties
        let mut adjusted_returns = expected_returns.to_vec();
        for i in 0..n {
            // Reduce expected return for assets with high causal dependencies
            adjusted_returns[i] *= (1.0 - self.config.causality_weight * causal_penalties[i]);
        }

        // Simple mean-variance optimization
        let weights = self.optimize_mean_variance(
            &adjusted_returns,
            covariance_matrix,
            2.0, // Risk aversion
        )?;

        // Compute portfolio metrics
        let expected_return: f64 = weights.iter().zip(expected_returns.iter()).map(|(w, r)| w * r).sum();

        let mut variance = 0.0;
        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * covariance_matrix[i][j];
            }
        }
        let portfolio_risk = variance.sqrt();

        let sharpe_ratio = (expected_return - risk_free_rate) / portfolio_risk;

        // Compute causal diversification score
        let causal_div = self.compute_causal_diversification(&weights, te_matrix);

        let explanation = format!(
            "Portfolio optimized with causal analysis.\n\
             Causality weight: {:.0}%\n\
             Causal diversification: {:.2} (0-1 scale, higher is better)\n\
             Network density: {:.2}% of asset pairs have significant causal links\n\
             Leading assets avoided: {}\n\
             Method: Transfer Entropy + Mean-Variance Optimization",
            self.config.causality_weight * 100.0,
            causal_div,
            te_matrix.stats.network_density * 100.0,
            te_matrix.stats.leading_assets.iter()
                .take(3)
                .map(|(s, _)| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(CausalPortfolioResult {
            optimal_weights: weights,
            expected_return,
            portfolio_risk,
            sharpe_ratio,
            causal_diversification: causal_div,
            explanation,
        })
    }

    /// Compute causal penalty for each asset (0-1, higher means more causally dependent)
    fn compute_causal_penalties(&self, te_matrix: &TransferEntropyMatrix) -> Vec<f64> {
        let n = te_matrix.assets.len();
        let mut penalties = vec![0.0; n];

        // Penalty = (in-degree + 0.5 * out-degree) / max_degree
        // Rationale: Assets that receive information (high in-degree) are more dependent
        // Assets that send information (high out-degree) are somewhat dependent but less so

        let mut max_degree = 0.0;

        for i in 0..n {
            // In-degree: Sum of significant TE into asset i
            let in_degree: f64 = (0..n)
                .filter(|&j| j != i && te_matrix.p_value_matrix[j][i] < self.config.significance_threshold)
                .map(|j| te_matrix.te_matrix[j][i])
                .sum();

            // Out-degree: Sum of significant TE out of asset i
            let out_degree: f64 = (0..n)
                .filter(|&j| j != i && te_matrix.p_value_matrix[i][j] < self.config.significance_threshold)
                .map(|j| te_matrix.te_matrix[i][j])
                .sum();

            let degree = in_degree + 0.5 * out_degree;
            penalties[i] = degree;

            if degree > max_degree {
                max_degree = degree;
            }
        }

        // Normalize
        if max_degree > 0.0 {
            for p in penalties.iter_mut() {
                *p /= max_degree;
            }
        }

        penalties
    }

    /// Simple mean-variance optimization
    fn optimize_mean_variance(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        risk_aversion: f64,
    ) -> Result<Vec<f64>> {
        let n = expected_returns.len();

        // Initialize equal weights
        let mut weights = vec![1.0 / n as f64; n];

        // Gradient descent
        let learning_rate = 0.01;
        let max_iterations = 500;

        for _ in 0..max_iterations {
            // Gradient = returns - risk_aversion * covariance * weights
            let mut gradient = expected_returns.to_vec();

            for i in 0..n {
                let mut cov_term = 0.0;
                for j in 0..n {
                    cov_term += covariance_matrix[i][j] * weights[j];
                }
                gradient[i] -= risk_aversion * cov_term;
            }

            // Update weights
            for i in 0..n {
                weights[i] += learning_rate * gradient[i];
            }

            // Project onto simplex (sum = 1, all >= 0)
            for w in weights.iter_mut() {
                *w = w.max(0.0);
            }

            let sum: f64 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }

        Ok(weights)
    }

    /// Compute causal diversification score (0-1, higher is better)
    fn compute_causal_diversification(&self, weights: &[f64], te_matrix: &TransferEntropyMatrix) -> f64 {
        let n = weights.len();

        // Weighted average of pairwise TE
        let mut weighted_te = 0.0;
        let mut total_weight = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let weight_prod = weights[i] * weights[j];
                    weighted_te += weight_prod * te_matrix.te_matrix[i][j];
                    total_weight += weight_prod;
                }
            }
        }

        if total_weight > 0.0 {
            weighted_te /= total_weight;
        }

        // Diversification = 1 - normalized_weighted_te
        // Higher TE → Lower diversification
        let max_te = te_matrix.stats.avg_transfer_entropy * 2.0; // Rough estimate

        if max_te > 0.0 {
            (1.0 - (weighted_te / max_te)).max(0.0).min(1.0)
        } else {
            1.0
        }
    }

    /// Detect market regime changes using TE dynamics
    pub fn detect_regime_with_causality(
        &mut self,
        assets: &[Asset],
        lookback_window: usize,
    ) -> Result<(MarketRegime, f64)> {
        if assets.is_empty() || assets[0].historical_returns.len() < lookback_window {
            anyhow::bail!("Insufficient data for regime detection with causality");
        }

        // Compute TE matrix for recent data
        let mut recent_assets: Vec<Asset> = assets.iter().map(|a| {
            let start_idx = a.historical_returns.len().saturating_sub(lookback_window);
            Asset {
                symbol: a.symbol.clone(),
                name: a.name.clone(),
                current_price: a.current_price,
                historical_returns: a.historical_returns[start_idx..].to_vec(),
            }
        }).collect();

        let te_matrix = self.compute_te_matrix(&recent_assets)?;

        // Use network density and information flow to infer regime
        let density = te_matrix.stats.network_density;
        let avg_te = te_matrix.stats.avg_transfer_entropy;

        // Heuristic regime classification based on TE dynamics
        let (regime, confidence) = if density > 0.5 && avg_te > 0.15 {
            // High connectivity + high information flow → Crisis
            (MarketRegime::Crisis, 0.8)
        } else if density > 0.4 && avg_te > 0.10 {
            // Moderate-high connectivity → High Volatility
            (MarketRegime::HighVolatility, 0.7)
        } else if density < 0.2 && avg_te < 0.05 {
            // Low connectivity → Low Volatility / Stable
            (MarketRegime::LowVolatility, 0.6)
        } else if avg_te > 0.12 {
            // Moderate density, high TE → Bull or Bear (need returns to distinguish)
            let avg_return: f64 = recent_assets[0].historical_returns.iter().sum::<f64>()
                / recent_assets[0].historical_returns.len() as f64;

            if avg_return > 0.0 {
                (MarketRegime::Bull, 0.6)
            } else {
                (MarketRegime::Bear, 0.6)
            }
        } else {
            // Default
            (MarketRegime::LowVolatility, 0.5)
        };

        Ok((regime, confidence))
    }

    /// Get cached TE matrix (if available)
    pub fn get_cached_te_matrix(&self) -> Option<&TransferEntropyMatrix> {
        self.cached_te_matrix.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assets(n: usize, periods: usize) -> Vec<Asset> {
        (0..n)
            .map(|i| {
                let returns: Vec<f64> = (0..periods)
                    .map(|t| {
                        0.001 * ((t as f64 + i as f64).sin() + (i as f64) * 0.01)
                    })
                    .collect();

                Asset {
                    symbol: format!("ASSET{}", i),
                    name: format!("Asset {}", i),
                    current_price: 100.0,
                    historical_returns: returns,
                }
            })
            .collect()
    }

    #[test]
    fn test_causality_analyzer_creation() {
        let config = CausalAnalysisConfig::default();
        let analyzer = CausalityAnalyzer::new(config);

        assert_eq!(analyzer.config.min_data_points, 30);
        assert_eq!(analyzer.config.significance_threshold, 0.05);
    }

    #[test]
    fn test_te_matrix_computation() {
        let config = CausalAnalysisConfig::default();
        let mut analyzer = CausalityAnalyzer::new(config);

        let assets = create_test_assets(3, 50);
        let result = analyzer.compute_te_matrix(&assets);

        assert!(result.is_ok());

        let te_matrix = result.unwrap();
        assert_eq!(te_matrix.assets.len(), 3);
        assert_eq!(te_matrix.te_matrix.len(), 3);
        assert_eq!(te_matrix.p_value_matrix.len(), 3);
    }

    #[test]
    fn test_causal_relationships_identification() {
        let config = CausalAnalysisConfig::default();
        let mut analyzer = CausalityAnalyzer::new(config);

        let assets = create_test_assets(3, 50);
        let te_matrix = analyzer.compute_te_matrix(&assets).unwrap();

        let relationships = analyzer.identify_causal_relationships(&te_matrix);

        // Should have some relationships (may be weak)
        assert!(relationships.len() > 0);

        // All relationships should have valid TE
        for rel in &relationships {
            assert!(rel.transfer_entropy >= 0.0);
            assert!(rel.p_value >= 0.0 && rel.p_value <= 1.0);
        }
    }

    #[test]
    fn test_causal_portfolio_optimization() {
        let config = CausalAnalysisConfig::default();
        let mut analyzer = CausalityAnalyzer::new(config);

        let assets = create_test_assets(3, 50);
        let te_matrix = analyzer.compute_te_matrix(&assets).unwrap();

        let expected_returns = vec![0.08, 0.10, 0.09];
        let covariance_matrix = vec![
            vec![0.04, 0.01, 0.01],
            vec![0.01, 0.04, 0.01],
            vec![0.01, 0.01, 0.04],
        ];

        let result = analyzer.optimize_with_causality(
            &expected_returns,
            &covariance_matrix,
            &te_matrix,
            0.02
        );

        assert!(result.is_ok());

        let portfolio = result.unwrap();
        assert_eq!(portfolio.optimal_weights.len(), 3);

        // Weights should sum to 1.0
        let sum: f64 = portfolio.optimal_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // All weights should be non-negative
        for &w in &portfolio.optimal_weights {
            assert!(w >= 0.0);
        }

        // Causal diversification should be in [0, 1]
        assert!(portfolio.causal_diversification >= 0.0 && portfolio.causal_diversification <= 1.0);
    }

    #[test]
    fn test_network_statistics() {
        let config = CausalAnalysisConfig::default();
        let mut analyzer = CausalityAnalyzer::new(config);

        let assets = create_test_assets(5, 50);
        let te_matrix = analyzer.compute_te_matrix(&assets).unwrap();

        let stats = &te_matrix.stats;

        assert!(stats.total_information_flow >= 0.0);
        assert!(stats.avg_transfer_entropy >= 0.0);
        assert!(stats.network_density >= 0.0 && stats.network_density <= 1.0);
        assert!(stats.leading_assets.len() <= 5);
        assert!(stats.lagging_assets.len() <= 5);
        assert!(stats.hub_assets.len() <= 5);
    }

    #[test]
    fn test_regime_detection_with_causality() {
        let config = CausalAnalysisConfig::default();
        let mut analyzer = CausalityAnalyzer::new(config);

        let assets = create_test_assets(4, 60);
        let result = analyzer.detect_regime_with_causality(&assets, 50);

        assert!(result.is_ok());

        let (regime, confidence) = result.unwrap();

        // Should return valid regime
        assert!(matches!(
            regime,
            MarketRegime::Bull
                | MarketRegime::Bear
                | MarketRegime::HighVolatility
                | MarketRegime::LowVolatility
                | MarketRegime::Crisis
                | MarketRegime::Recovery
        ));

        // Confidence should be in [0, 1]
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}
