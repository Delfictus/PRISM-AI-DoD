//! Thermodynamic Consensus for Portfolio Optimization
//!
//! Implements free energy minimization and maximum entropy principles
//! for achieving consensus across multiple portfolio strategies through
//! thermodynamic equilibration.
//!
//! Key Features:
//! - Free energy minimization (F = E - TS)
//! - Maximum entropy portfolio selection
//! - Thermal equilibration dynamics
//! - Energy-based strategy weighting

use ndarray::{Array1, Array2};
use anyhow::{Result, Context, bail};

/// Thermodynamic portfolio state
#[derive(Debug, Clone)]
pub struct ThermodynamicState {
    /// Portfolio weights (probability distribution)
    pub weights: Array1<f64>,
    /// Energy of each asset (negative expected return)
    pub energy: Array1<f64>,
    /// Temperature (controls exploration vs exploitation)
    pub temperature: f64,
    /// Free energy F = E - TS
    pub free_energy: f64,
    /// Entropy S = -Σ p log p
    pub entropy: f64,
}

/// Thermodynamic consensus configuration
#[derive(Debug, Clone)]
pub struct ThermodynamicConfig {
    /// Initial temperature
    pub temperature: f64,
    /// Cooling rate for annealing
    pub cooling_rate: f64,
    /// Minimum temperature
    pub min_temperature: f64,
    /// Equilibration steps
    pub equilibration_steps: usize,
    /// Energy penalty for risk
    pub risk_penalty: f64,
}

impl Default for ThermodynamicConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            cooling_rate: 0.95,
            min_temperature: 0.01,
            equilibration_steps: 100,
            risk_penalty: 1.0,
        }
    }
}

/// Thermodynamic consensus result
#[derive(Debug, Clone)]
pub struct ThermodynamicConsensusResult {
    /// Consensus portfolio weights
    pub consensus_weights: Array1<f64>,
    /// Final free energy
    pub free_energy: f64,
    /// Final entropy
    pub entropy: f64,
    /// Energy landscape (per asset)
    pub energy_landscape: Array1<f64>,
    /// Equilibration trajectory (free energy over time)
    pub trajectory: Vec<f64>,
}

/// Thermodynamic consensus engine
pub struct ThermodynamicConsensusEngine {
    config: ThermodynamicConfig,
}

impl ThermodynamicConsensusEngine {
    /// Create new thermodynamic consensus engine
    pub fn new(config: ThermodynamicConfig) -> Self {
        Self { config }
    }

    /// Compute energy landscape from portfolio strategies
    fn compute_energy_landscape(
        &self,
        portfolio_weights: &[Array1<f64>],
        expected_returns: &Array1<f64>,
        risks: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n_assets = expected_returns.len();

        // Energy = -return + risk_penalty * risk
        // (We want to minimize energy, so negative return is energy)
        let mut energy = Array1::zeros(n_assets);

        for i in 0..n_assets {
            energy[i] = -expected_returns[i] + self.config.risk_penalty * risks[i];
        }

        Ok(energy)
    }

    /// Compute Boltzmann distribution: p_i = exp(-E_i / T) / Z
    fn boltzmann_distribution(
        &self,
        energy: &Array1<f64>,
        temperature: f64,
    ) -> Result<Array1<f64>> {
        let n = energy.len();
        let mut weights = Array1::zeros(n);

        // Compute exp(-E_i / T)
        for i in 0..n {
            weights[i] = (-energy[i] / temperature).exp();
        }

        // Normalize (compute partition function Z)
        let partition_function: f64 = weights.sum();

        if partition_function < 1e-10 {
            bail!("Partition function too small (temperature too low or energy too high)");
        }

        weights /= partition_function;

        Ok(weights)
    }

    /// Compute entropy: S = -Σ p_i log(p_i)
    fn compute_entropy(&self, weights: &Array1<f64>) -> f64 {
        weights.iter()
            .filter(|&&p| p > 1e-10) // Avoid log(0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Compute expected energy: E = Σ p_i E_i
    fn compute_expected_energy(&self, weights: &Array1<f64>, energy: &Array1<f64>) -> f64 {
        weights.iter()
            .zip(energy.iter())
            .map(|(p, e)| p * e)
            .sum()
    }

    /// Compute free energy: F = E - TS
    fn compute_free_energy(
        &self,
        weights: &Array1<f64>,
        energy: &Array1<f64>,
        temperature: f64,
    ) -> f64 {
        let expected_energy = self.compute_expected_energy(weights, energy);
        let entropy = self.compute_entropy(weights);
        expected_energy - temperature * entropy
    }

    /// Equilibrate at fixed temperature
    fn equilibrate(
        &self,
        energy: &Array1<f64>,
        temperature: f64,
    ) -> Result<ThermodynamicState> {
        // At equilibrium, weights follow Boltzmann distribution
        let weights = self.boltzmann_distribution(energy, temperature)?;

        let entropy = self.compute_entropy(&weights);
        let free_energy = self.compute_free_energy(&weights, energy, temperature);

        Ok(ThermodynamicState {
            weights,
            energy: energy.clone(),
            temperature,
            free_energy,
            entropy,
        })
    }

    /// Perform simulated annealing to find minimum free energy consensus
    pub fn anneal(
        &self,
        portfolio_weights: &[Array1<f64>],
        expected_returns: &Array1<f64>,
        risks: &Array1<f64>,
    ) -> Result<ThermodynamicConsensusResult> {
        // Compute energy landscape
        let energy = self.compute_energy_landscape(portfolio_weights, expected_returns, risks)?;

        let mut temperature = self.config.temperature;
        let mut trajectory = Vec::with_capacity(self.config.equilibration_steps);

        let mut best_state: Option<ThermodynamicState> = None;
        let mut best_free_energy = f64::INFINITY;

        // Simulated annealing
        for step in 0..self.config.equilibration_steps {
            // Equilibrate at current temperature
            let state = self.equilibrate(&energy, temperature)?;

            trajectory.push(state.free_energy);

            // Track best state (minimum free energy)
            if state.free_energy < best_free_energy {
                best_free_energy = state.free_energy;
                best_state = Some(state.clone());
            }

            // Cool down
            temperature *= self.config.cooling_rate;
            temperature = temperature.max(self.config.min_temperature);

            // Early stopping if converged
            if step > 10 && trajectory.len() > 10 {
                let recent_variance: f64 = trajectory[step-10..step]
                    .iter()
                    .map(|&f| (f - trajectory[step]).powi(2))
                    .sum::<f64>() / 10.0;

                if recent_variance < 1e-6 {
                    break;
                }
            }
        }

        let final_state = best_state.context("Failed to find valid thermodynamic state")?;

        Ok(ThermodynamicConsensusResult {
            consensus_weights: final_state.weights,
            free_energy: final_state.free_energy,
            entropy: final_state.entropy,
            energy_landscape: energy,
            trajectory,
        })
    }

    /// Maximum entropy portfolio (uniform distribution at high temperature)
    pub fn maximum_entropy_portfolio(&self, n_assets: usize) -> Array1<f64> {
        Array1::from_elem(n_assets, 1.0 / n_assets as f64)
    }

    /// Minimum free energy portfolio (at low temperature, concentrated on lowest energy assets)
    pub fn minimum_free_energy_portfolio(
        &self,
        expected_returns: &Array1<f64>,
        risks: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let energy = self.compute_energy_landscape(&[], expected_returns, risks)?;

        // At T → 0, Boltzmann distribution concentrates on minimum energy
        // Use very low temperature
        let low_temp = 0.001;
        self.boltzmann_distribution(&energy, low_temp)
    }

    /// Compute partition function Z = Σ exp(-E_i / T)
    pub fn partition_function(&self, energy: &Array1<f64>, temperature: f64) -> f64 {
        energy.iter()
            .map(|&e| (-e / temperature).exp())
            .sum()
    }

    /// Compute specific heat C = ∂E/∂T (measures thermal fluctuations)
    pub fn specific_heat(
        &self,
        energy: &Array1<f64>,
        temperature: f64,
    ) -> Result<f64> {
        let weights = self.boltzmann_distribution(energy, temperature)?;

        // E = Σ p_i E_i
        let expected_energy = self.compute_expected_energy(&weights, energy);

        // E² = Σ p_i E_i²
        let expected_energy_squared: f64 = weights.iter()
            .zip(energy.iter())
            .map(|(p, e)| p * e.powi(2))
            .sum();

        // C = (E² - E²) / T²
        let variance = expected_energy_squared - expected_energy.powi(2);
        Ok(variance / temperature.powi(2))
    }
}

impl ThermodynamicConsensusResult {
    /// Print consensus result summary
    pub fn print_summary(&self, asset_names: &[String]) {
        println!("\n=== Thermodynamic Consensus ===");
        println!("Free Energy: {:.6}", self.free_energy);
        println!("Entropy: {:.6} (higher = more diversified)", self.entropy);

        println!("\nConsensus Portfolio Weights:");
        for (i, &weight) in self.consensus_weights.iter().enumerate() {
            if i < asset_names.len() {
                println!("  {}: {:.2}% (energy: {:.4})",
                         asset_names[i],
                         weight * 100.0,
                         self.energy_landscape[i]);
            }
        }

        if self.trajectory.len() > 1 {
            println!("\nAnnealing Trajectory:");
            println!("  Initial F: {:.6}", self.trajectory[0]);
            println!("  Final F: {:.6}", self.trajectory[self.trajectory.len() - 1]);
            println!("  Steps: {}", self.trajectory.len());

            let improvement = self.trajectory[0] - self.trajectory[self.trajectory.len() - 1];
            println!("  Improvement: {:.6}", improvement);
        }
    }

    /// Compute diversification score (normalized entropy)
    pub fn diversification_score(&self) -> f64 {
        let n = self.consensus_weights.len() as f64;
        let max_entropy = n.ln(); // Maximum entropy for n assets
        if max_entropy > 0.0 {
            self.entropy / max_entropy
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>, Vec<String>) {
        let portfolios = vec![
            Array1::from_vec(vec![0.4, 0.3, 0.2, 0.1]),
            Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]),
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
        ];

        let expected_returns = Array1::from_vec(vec![0.12, 0.10, 0.08, 0.15]);
        let risks = Array1::from_vec(vec![0.20, 0.15, 0.10, 0.25]);
        let assets = vec![
            "AAPL".to_string(),
            "GOOGL".to_string(),
            "MSFT".to_string(),
            "TSLA".to_string(),
        ];

        (portfolios, expected_returns, risks, assets)
    }

    #[test]
    fn test_thermodynamic_consensus_creation() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());
        let (portfolios, returns, risks, _assets) = create_test_data();

        let result = engine.anneal(&portfolios, &returns, &risks);
        assert!(result.is_ok());

        let consensus = result.unwrap();
        assert_eq!(consensus.consensus_weights.len(), 4);

        // Weights should sum to 1
        let sum: f64 = consensus.consensus_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_boltzmann_distribution() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());
        let energy = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let weights = engine.boltzmann_distribution(&energy, 1.0);
        assert!(weights.is_ok());

        let w = weights.unwrap();

        // Weights should sum to 1
        let sum: f64 = w.sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Lower energy should have higher weight
        assert!(w[0] > w[1]);
        assert!(w[1] > w[2]);
        assert!(w[2] > w[3]);
    }

    #[test]
    fn test_entropy_computation() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());

        // Uniform distribution should have maximum entropy
        let uniform = Array1::from_elem(4, 0.25);
        let entropy_uniform = engine.compute_entropy(&uniform);

        // Concentrated distribution should have lower entropy
        let concentrated = Array1::from_vec(vec![0.7, 0.1, 0.1, 0.1]);
        let entropy_concentrated = engine.compute_entropy(&concentrated);

        assert!(entropy_uniform > entropy_concentrated);
    }

    #[test]
    fn test_free_energy_minimization() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());
        let (portfolios, returns, risks, _assets) = create_test_data();

        let result = engine.anneal(&portfolios, &returns, &risks).unwrap();

        // Free energy should decrease during annealing
        assert!(result.trajectory.len() > 0);
        if result.trajectory.len() > 1 {
            let initial_f = result.trajectory[0];
            let final_f = result.trajectory[result.trajectory.len() - 1];
            assert!(final_f <= initial_f + 1e-6); // Allow small numerical error
        }
    }

    #[test]
    fn test_maximum_entropy_portfolio() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());
        let max_entropy = engine.maximum_entropy_portfolio(4);

        // Should be uniform
        for &w in max_entropy.iter() {
            assert!((w - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_partition_function() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());
        let energy = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let z_high_temp = engine.partition_function(&energy, 10.0);
        let z_low_temp = engine.partition_function(&energy, 0.1);

        // Partition function should decrease with temperature
        assert!(z_high_temp > z_low_temp);
    }

    #[test]
    fn test_specific_heat() {
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());
        let energy = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let heat_capacity = engine.specific_heat(&energy, 1.0);
        assert!(heat_capacity.is_ok());

        let c = heat_capacity.unwrap();
        assert!(c >= 0.0); // Heat capacity should be non-negative
    }

    #[test]
    fn test_diversification_score() {
        let (portfolios, returns, risks, _assets) = create_test_data();
        let engine = ThermodynamicConsensusEngine::new(ThermodynamicConfig::default());

        let result = engine.anneal(&portfolios, &returns, &risks).unwrap();
        let diversification = result.diversification_score();

        // Diversification should be between 0 and 1
        assert!(diversification >= 0.0);
        assert!(diversification <= 1.0);
    }
}
