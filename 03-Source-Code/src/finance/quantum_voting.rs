//! Quantum-Inspired Voting for Portfolio Consensus
//!
//! Implements quantum-inspired voting mechanisms for achieving consensus
//! across multiple portfolio optimization strategies using superposition
//! and entanglement-inspired correlation analysis.
//!
//! Key Features:
//! - Quantum superposition of portfolio states
//! - Entanglement-based correlation voting
//! - Decoherence-weighted consensus
//! - Multi-strategy aggregation

use ndarray::{Array1, Array2};
use anyhow::{Result, Context, bail};
use std::collections::HashMap;

/// Quantum-inspired portfolio state representing superposition
#[derive(Debug, Clone)]
pub struct QuantumPortfolioState {
    /// Superposition of portfolio weights (each column is a strategy)
    pub weight_amplitudes: Array2<f64>,
    /// Probability amplitudes for each strategy
    pub strategy_amplitudes: Array1<f64>,
    /// Coherence factors (measure of state purity)
    pub coherence: Array1<f64>,
    /// Asset labels
    pub assets: Vec<String>,
    /// Strategy labels
    pub strategies: Vec<String>,
}

/// Quantum voting configuration
#[derive(Debug, Clone)]
pub struct QuantumVotingConfig {
    /// Decoherence rate (how fast quantum effects decay)
    pub decoherence_rate: f64,
    /// Entanglement threshold for correlated voting
    pub entanglement_threshold: f64,
    /// Measurement collapse iterations
    pub measurement_iterations: usize,
    /// Temperature for thermal noise (quantum fluctuations)
    pub temperature: f64,
}

impl Default for QuantumVotingConfig {
    fn default() -> Self {
        Self {
            decoherence_rate: 0.1,
            entanglement_threshold: 0.7,
            measurement_iterations: 100,
            temperature: 0.01,
        }
    }
}

/// Quantum voting result
#[derive(Debug, Clone)]
pub struct QuantumVotingResult {
    /// Consensus portfolio weights
    pub consensus_weights: Array1<f64>,
    /// Coherence of consensus (higher = more agreement)
    pub consensus_coherence: f64,
    /// Contribution of each strategy to consensus
    pub strategy_contributions: Array1<f64>,
    /// Entanglement matrix (inter-strategy correlations)
    pub entanglement_matrix: Array2<f64>,
}

/// Quantum-inspired portfolio voting engine
pub struct QuantumVotingEngine {
    config: QuantumVotingConfig,
}

impl QuantumVotingEngine {
    /// Create new quantum voting engine
    pub fn new(config: QuantumVotingConfig) -> Self {
        Self { config }
    }

    /// Create quantum portfolio state from multiple strategies
    pub fn create_quantum_state(
        &self,
        portfolio_weights: &[Array1<f64>],
        strategy_names: &[String],
        asset_names: &[String],
    ) -> Result<QuantumPortfolioState> {
        if portfolio_weights.is_empty() {
            bail!("Need at least one portfolio strategy");
        }

        let n_strategies = portfolio_weights.len();
        let n_assets = portfolio_weights[0].len();

        // Validate all portfolios have same dimension
        for (i, weights) in portfolio_weights.iter().enumerate() {
            if weights.len() != n_assets {
                bail!("Strategy {} has {} assets, expected {}", i, weights.len(), n_assets);
            }
        }

        // Create weight amplitude matrix (assets × strategies)
        let mut weight_amplitudes = Array2::zeros((n_assets, n_strategies));
        for (j, weights) in portfolio_weights.iter().enumerate() {
            for i in 0..n_assets {
                weight_amplitudes[[i, j]] = weights[i];
            }
        }

        // Initialize equal superposition of strategies
        let strategy_amplitudes = Array1::from_elem(n_strategies, 1.0 / (n_strategies as f64).sqrt());

        // Initialize full coherence
        let coherence = Array1::ones(n_strategies);

        Ok(QuantumPortfolioState {
            weight_amplitudes,
            strategy_amplitudes,
            coherence,
            assets: asset_names.to_vec(),
            strategies: strategy_names.to_vec(),
        })
    }

    /// Compute entanglement matrix (inter-strategy correlations)
    fn compute_entanglement(&self, state: &QuantumPortfolioState) -> Result<Array2<f64>> {
        let n_strategies = state.strategies.len();
        let mut entanglement = Array2::zeros((n_strategies, n_strategies));

        // Compute pairwise correlations between strategies
        for i in 0..n_strategies {
            for j in 0..n_strategies {
                if i == j {
                    entanglement[[i, j]] = 1.0;
                } else {
                    // Correlation = dot product of weight vectors
                    let weights_i = state.weight_amplitudes.column(i);
                    let weights_j = state.weight_amplitudes.column(j);

                    let correlation = weights_i.iter()
                        .zip(weights_j.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();

                    entanglement[[i, j]] = correlation.abs();
                }
            }
        }

        Ok(entanglement)
    }

    /// Apply decoherence (quantum to classical transition)
    fn apply_decoherence(&self, state: &mut QuantumPortfolioState, entanglement: &Array2<f64>) {
        let n_strategies = state.strategies.len();

        for i in 0..n_strategies {
            // Decoherence reduces coherence based on:
            // 1. Time evolution (decoherence_rate)
            // 2. Entanglement with other strategies (maintains coherence)

            let mut entanglement_sum = 0.0;
            for j in 0..n_strategies {
                if i != j {
                    entanglement_sum += entanglement[[i, j]];
                }
            }
            let avg_entanglement = entanglement_sum / (n_strategies - 1) as f64;

            // Strategies with high entanglement maintain coherence longer
            let decoherence = self.config.decoherence_rate * (1.0 - avg_entanglement);
            state.coherence[i] *= (1.0 - decoherence).max(0.0);
        }
    }

    /// Perform quantum measurement (collapse to consensus)
    fn measure_consensus(&self, state: &QuantumPortfolioState) -> Result<Array1<f64>> {
        let n_assets = state.assets.len();
        let n_strategies = state.strategies.len();

        // Weighted average based on strategy amplitudes and coherence
        let mut consensus = Array1::zeros(n_assets);

        // Compute effective weights: |amplitude|² × coherence
        let mut effective_weights = Array1::zeros(n_strategies);
        let mut total_weight = 0.0;

        for i in 0..n_strategies {
            let prob = state.strategy_amplitudes[i].powi(2);
            let weight = prob * state.coherence[i];
            effective_weights[i] = weight;
            total_weight += weight;
        }

        // Normalize
        if total_weight > 1e-10 {
            effective_weights /= total_weight;
        } else {
            // Uniform if all weights are zero
            effective_weights = Array1::from_elem(n_strategies, 1.0 / n_strategies as f64);
        }

        // Compute consensus as weighted average
        for i in 0..n_assets {
            for j in 0..n_strategies {
                consensus[i] += state.weight_amplitudes[[i, j]] * effective_weights[j];
            }
        }

        // Normalize to ensure sum = 1
        let sum = consensus.sum();
        if sum > 1e-10 {
            consensus /= sum;
        } else {
            consensus = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
        }

        Ok(consensus)
    }

    /// Perform quantum voting to achieve consensus
    pub fn vote(
        &self,
        portfolio_weights: &[Array1<f64>],
        strategy_names: &[String],
        asset_names: &[String],
    ) -> Result<QuantumVotingResult> {
        // Create initial quantum state
        let mut state = self.create_quantum_state(
            portfolio_weights,
            strategy_names,
            asset_names,
        )?;

        // Compute entanglement matrix
        let entanglement = self.compute_entanglement(&state)?;

        // Apply decoherence to simulate quantum-to-classical transition
        self.apply_decoherence(&mut state, &entanglement);

        // Measure consensus (collapse wavefunction)
        let consensus_weights = self.measure_consensus(&state)?;

        // Compute consensus coherence (average coherence weighted by amplitudes)
        let consensus_coherence = state.coherence.iter()
            .zip(state.strategy_amplitudes.iter())
            .map(|(c, a)| c * a.powi(2))
            .sum::<f64>();

        // Strategy contributions = |amplitude|² × coherence
        let mut strategy_contributions = Array1::zeros(state.strategies.len());
        for i in 0..state.strategies.len() {
            strategy_contributions[i] = state.strategy_amplitudes[i].powi(2) * state.coherence[i];
        }

        // Normalize contributions
        let total_contribution = strategy_contributions.sum();
        if total_contribution > 1e-10 {
            strategy_contributions /= total_contribution;
        }

        Ok(QuantumVotingResult {
            consensus_weights,
            consensus_coherence,
            strategy_contributions,
            entanglement_matrix: entanglement,
        })
    }

    /// Iterative voting with measurement feedback
    pub fn iterative_vote(
        &self,
        portfolio_weights: &[Array1<f64>],
        strategy_names: &[String],
        asset_names: &[String],
    ) -> Result<QuantumVotingResult> {
        let mut best_result: Option<QuantumVotingResult> = None;
        let mut best_coherence = 0.0;

        for _ in 0..self.config.measurement_iterations {
            let result = self.vote(portfolio_weights, strategy_names, asset_names)?;

            if result.consensus_coherence > best_coherence {
                best_coherence = result.consensus_coherence;
                best_result = Some(result);
            }
        }

        best_result.context("Failed to find valid consensus")
    }

    /// Compute quantum discord (non-classical correlations)
    pub fn compute_discord(&self, state: &QuantumPortfolioState) -> Result<f64> {
        let entanglement = self.compute_entanglement(state)?;
        let n_strategies = state.strategies.len();

        // Discord = entropy of entanglement - classical correlations
        // Simplified: measure of non-classical correlations
        let mut discord = 0.0;

        for i in 0..n_strategies {
            for j in (i+1)..n_strategies {
                let e_ij = entanglement[[i, j]];
                if e_ij > self.config.entanglement_threshold {
                    // High entanglement contributes to discord
                    discord += e_ij * (1.0 - e_ij).abs();
                }
            }
        }

        // Normalize by number of pairs
        let n_pairs = (n_strategies * (n_strategies - 1)) / 2;
        if n_pairs > 0 {
            discord /= n_pairs as f64;
        }

        Ok(discord)
    }
}

impl QuantumVotingResult {
    /// Print voting result summary
    pub fn print_summary(&self, asset_names: &[String], strategy_names: &[String]) {
        println!("\n=== Quantum Voting Consensus ===");
        println!("Consensus Coherence: {:.4} (higher = more agreement)", self.consensus_coherence);

        println!("\nConsensus Portfolio Weights:");
        for (i, &weight) in self.consensus_weights.iter().enumerate() {
            if i < asset_names.len() {
                println!("  {}: {:.2}%", asset_names[i], weight * 100.0);
            }
        }

        println!("\nStrategy Contributions:");
        for (i, &contribution) in self.strategy_contributions.iter().enumerate() {
            if i < strategy_names.len() {
                println!("  {}: {:.1}%", strategy_names[i], contribution * 100.0);
            }
        }

        println!("\nEntanglement Matrix (inter-strategy correlations):");
        for i in 0..strategy_names.len() {
            print!("  ");
            for j in 0..strategy_names.len() {
                if i < self.entanglement_matrix.nrows() && j < self.entanglement_matrix.ncols() {
                    print!("{:.2} ", self.entanglement_matrix[[i, j]]);
                }
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_portfolios() -> (Vec<Array1<f64>>, Vec<String>, Vec<String>) {
        // 3 strategies for 4 assets
        let portfolios = vec![
            Array1::from_vec(vec![0.4, 0.3, 0.2, 0.1]), // Aggressive
            Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]), // Balanced
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]), // Conservative
        ];

        let strategies = vec![
            "Aggressive".to_string(),
            "Balanced".to_string(),
            "Conservative".to_string(),
        ];

        let assets = vec![
            "AAPL".to_string(),
            "GOOGL".to_string(),
            "MSFT".to_string(),
            "TSLA".to_string(),
        ];

        (portfolios, strategies, assets)
    }

    #[test]
    fn test_quantum_voting_creation() {
        let engine = QuantumVotingEngine::new(QuantumVotingConfig::default());
        let (portfolios, strategies, assets) = create_test_portfolios();

        let result = engine.vote(&portfolios, &strategies, &assets);
        assert!(result.is_ok());

        let voting_result = result.unwrap();
        assert_eq!(voting_result.consensus_weights.len(), 4);

        // Weights should sum to 1
        let sum: f64 = voting_result.consensus_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantum_state_creation() {
        let engine = QuantumVotingEngine::new(QuantumVotingConfig::default());
        let (portfolios, strategies, assets) = create_test_portfolios();

        let state = engine.create_quantum_state(&portfolios, &strategies, &assets);
        assert!(state.is_ok());

        let quantum_state = state.unwrap();
        assert_eq!(quantum_state.weight_amplitudes.nrows(), 4); // 4 assets
        assert_eq!(quantum_state.weight_amplitudes.ncols(), 3); // 3 strategies
        assert_eq!(quantum_state.strategy_amplitudes.len(), 3);
        assert_eq!(quantum_state.coherence.len(), 3);
    }

    #[test]
    fn test_entanglement_computation() {
        let engine = QuantumVotingEngine::new(QuantumVotingConfig::default());
        let (portfolios, strategies, assets) = create_test_portfolios();

        let state = engine.create_quantum_state(&portfolios, &strategies, &assets).unwrap();
        let entanglement = engine.compute_entanglement(&state);

        assert!(entanglement.is_ok());
        let e_matrix = entanglement.unwrap();

        // Diagonal should be 1
        for i in 0..3 {
            assert!((e_matrix[[i, i]] - 1.0).abs() < 1e-6);
        }

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((e_matrix[[i, j]] - e_matrix[[j, i]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_consensus_coherence() {
        let engine = QuantumVotingEngine::new(QuantumVotingConfig::default());
        let (portfolios, strategies, assets) = create_test_portfolios();

        let result = engine.vote(&portfolios, &strategies, &assets).unwrap();

        // Coherence should be between 0 and 1
        assert!(result.consensus_coherence >= 0.0);
        assert!(result.consensus_coherence <= 1.0);

        // Strategy contributions should sum to 1
        let contrib_sum: f64 = result.strategy_contributions.iter().sum();
        assert!((contrib_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_discord_computation() {
        let engine = QuantumVotingEngine::new(QuantumVotingConfig::default());
        let (portfolios, strategies, assets) = create_test_portfolios();

        let state = engine.create_quantum_state(&portfolios, &strategies, &assets).unwrap();
        let discord = engine.compute_discord(&state);

        assert!(discord.is_ok());
        let discord_value = discord.unwrap();
        assert!(discord_value >= 0.0);
    }
}
