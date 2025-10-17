//! Replica Exchange Framework for Thermodynamic Optimization
//!
//! Week 3: Task 2.3.1-2.3.3 - Full Replica Exchange Implementation
//!
//! Integrates replica exchange with advanced energy model for parallel GPU optimization.
//! Includes Gelman-Rubin convergence diagnostics and adaptive temperature spacing.
//!
//! Mathematical Framework:
//! - Replica Exchange: Multiple parallel chains at different temperatures
//! - Swap criterion: P_swap = min(1, exp((β_i - β_j)(E_j - E_i)))
//! - Gelman-Rubin statistic: R̂ = √(W + B/n) / W
//!   where W = within-chain variance, B = between-chain variance
//! - Convergence: R̂ < 1.1 indicates chains have mixed

use anyhow::{Result, Context};
use std::collections::HashMap;

use super::temperature_schedules::ReplicaExchangeManager;
use super::advanced_energy::{AdvancedEnergyModel, AdvancedLLMModel, TaskType};

/// State of a single replica
#[derive(Debug, Clone)]
pub struct ReplicaState {
    /// Replica ID
    pub id: usize,
    /// Temperature (inverse beta = 1/T)
    pub temperature: f64,
    /// Current energy
    pub energy: f64,
    /// Current model selection
    pub selected_model: usize,
    /// Energy history for diagnostics
    pub energy_history: Vec<f64>,
    /// Model selection history
    pub selection_history: Vec<usize>,
}

impl ReplicaState {
    pub fn new(id: usize, temperature: f64) -> Self {
        Self {
            id,
            temperature,
            energy: 0.0,
            selected_model: 0,
            energy_history: Vec::new(),
            selection_history: Vec::new(),
        }
    }

    /// Record current state in history
    pub fn record(&mut self) {
        self.energy_history.push(self.energy);
        self.selection_history.push(self.selected_model);
    }

    /// Get recent energy samples for diagnostics
    pub fn recent_energies(&self, n: usize) -> &[f64] {
        let start = self.energy_history.len().saturating_sub(n);
        &self.energy_history[start..]
    }
}

/// Full Replica Exchange System
///
/// Combines ReplicaExchangeManager with AdvancedEnergyModel
/// for parallel thermodynamic optimization
pub struct ReplicaExchangeSystem {
    /// Energy model for computing model energies
    energy_model: AdvancedEnergyModel,
    /// Replica exchange manager (temperature ladder)
    exchange_manager: ReplicaExchangeManager,
    /// State of each replica
    replicas: Vec<ReplicaState>,
    /// Task type for quality estimation
    current_task: TaskType,
    /// Convergence threshold for Gelman-Rubin
    convergence_threshold: f64,
    /// Iteration counter
    iteration: usize,
}

impl ReplicaExchangeSystem {
    /// Create new replica exchange system
    pub fn new(
        models: Vec<AdvancedLLMModel>,
        n_replicas: usize,
        t_max: f64,
        t_min: f64,
        task_type: TaskType,
    ) -> Result<Self> {
        let energy_model = AdvancedEnergyModel::new(models)?;
        let exchange_manager = ReplicaExchangeManager::new(n_replicas, t_max, t_min)?;

        let mut replicas = Vec::new();
        for i in 0..n_replicas {
            let temp = exchange_manager.temperature(i);
            replicas.push(ReplicaState::new(i, temp));
        }

        Ok(Self {
            energy_model,
            exchange_manager,
            replicas,
            current_task: task_type,
            convergence_threshold: 1.1,
            iteration: 0,
        })
    }

    /// Run one iteration of replica exchange optimization
    ///
    /// Steps:
    /// 1. For each replica: select model using Boltzmann distribution at its temperature
    /// 2. Compute energies for all replicas
    /// 3. Attempt replica exchanges
    /// 4. Record states
    pub fn step(
        &mut self,
        budget_constraint: f64,
        latency_budget_ms: f64,
    ) -> Result<()> {
        // Step 1 & 2: Select models and compute energies for all replicas
        let energies = self.energy_model.compute_energies(
            self.current_task,
            budget_constraint,
            latency_budget_ms,
        )?;

        for replica in &mut self.replicas {
            // Select model using Boltzmann distribution at this replica's temperature
            let temp_scaled_energies = Self::scale_energies_by_temperature(
                &energies,
                replica.temperature,
            );

            let selected_model = Self::select_model_boltzmann(&temp_scaled_energies)?;
            replica.selected_model = selected_model;
            replica.energy = energies[selected_model];

            // Update exchange manager with current energy
            self.exchange_manager.update_energy(replica.id, replica.energy);
        }

        // Step 3: Attempt replica exchanges
        let n_accepted = self.exchange_manager.exchange_round();

        // Step 4: Record states for diagnostics
        for replica in &mut self.replicas {
            replica.record();
        }

        self.iteration += 1;

        // Periodically adapt temperature spacing
        if self.iteration % 100 == 0 {
            self.exchange_manager.adapt_temperatures();
        }

        Ok(())
    }

    /// Scale energies by temperature for Boltzmann sampling
    fn scale_energies_by_temperature(energies: &[f64], temperature: f64) -> Vec<f64> {
        energies.iter()
            .map(|&e| e / temperature)
            .collect()
    }

    /// Select model using Boltzmann distribution
    ///
    /// P(model) ∝ exp(-E/T)
    fn select_model_boltzmann(energies: &[f64]) -> Result<usize> {
        // Compute Boltzmann probabilities
        let mut exp_sum = 0.0;
        let mut exp_values = Vec::new();

        for &e in energies {
            let exp_e = (-e).exp();
            exp_values.push(exp_e);
            exp_sum += exp_e;
        }

        // Normalize
        let probabilities: Vec<f64> = exp_values.iter()
            .map(|&exp_e| exp_e / exp_sum)
            .collect();

        // Sample using cumulative distribution
        // Simple deterministic sampling for reproducibility
        let mut seed = (energies.len() * 1103515245 + 12345) as u64;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let uniform = (seed % 1000) as f64 / 1000.0;

        let mut cumulative = 0.0;
        for (i, &p) in probabilities.iter().enumerate() {
            cumulative += p;
            if uniform < cumulative {
                return Ok(i);
            }
        }

        Ok(probabilities.len() - 1)
    }

    /// Compute Gelman-Rubin convergence diagnostic
    ///
    /// R̂ = √[(W + B/n) / W]
    /// where:
    /// - W = within-chain variance (average variance across chains)
    /// - B = between-chain variance (variance of chain means)
    /// - n = number of samples per chain
    ///
    /// Interpretation:
    /// - R̂ ≈ 1: chains have converged (well-mixed)
    /// - R̂ > 1.1: chains have not converged (need more iterations)
    pub fn gelman_rubin_statistic(&self) -> Result<f64> {
        let n_samples = 100; // Use last 100 samples

        // Need at least 2 replicas and sufficient samples
        if self.replicas.len() < 2 {
            anyhow::bail!("Need at least 2 replicas for Gelman-Rubin");
        }

        if self.replicas[0].energy_history.len() < n_samples {
            anyhow::bail!("Need at least {} samples", n_samples);
        }

        // Step 1: Compute chain means
        let mut chain_means = Vec::new();
        for replica in &self.replicas {
            let samples = replica.recent_energies(n_samples);
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            chain_means.push(mean);
        }

        // Step 2: Compute overall mean
        let overall_mean = chain_means.iter().sum::<f64>() / chain_means.len() as f64;

        // Step 3: Compute between-chain variance (B)
        let m = self.replicas.len() as f64;
        let n = n_samples as f64;
        let b = n * chain_means.iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>() / (m - 1.0);

        // Step 4: Compute within-chain variance (W)
        let mut chain_variances = Vec::new();
        for (i, replica) in self.replicas.iter().enumerate() {
            let samples = replica.recent_energies(n_samples);
            let mean = chain_means[i];
            let variance = samples.iter()
                .map(|&e| (e - mean).powi(2))
                .sum::<f64>() / (n - 1.0);
            chain_variances.push(variance);
        }

        let w = chain_variances.iter().sum::<f64>() / m;

        // Step 5: Compute R̂
        // Handle edge case where w ≈ 0 (perfect convergence or constant chains)
        if w < 1e-10 {
            // If within-chain variance is effectively zero, chains have converged
            return Ok(1.0);
        }

        let var_plus = w * (n - 1.0) / n + b / n;
        let r_hat = (var_plus / w).sqrt();

        Ok(r_hat)
    }

    /// Check if replicas have converged
    pub fn has_converged(&self) -> bool {
        if let Ok(r_hat) = self.gelman_rubin_statistic() {
            r_hat < self.convergence_threshold
        } else {
            false
        }
    }

    /// Get best model from ground state replica (coldest temperature)
    pub fn get_best_model(&self) -> usize {
        let ground_replica_idx = self.exchange_manager.get_ground_state_replica();
        self.replicas[ground_replica_idx].selected_model
    }

    /// Get best energy from ground state replica
    pub fn get_best_energy(&self) -> f64 {
        let ground_replica_idx = self.exchange_manager.get_ground_state_replica();
        self.replicas[ground_replica_idx].energy
    }

    /// Get detailed statistics
    pub fn statistics(&self) -> ReplicaExchangeSystemStats {
        let exchange_stats = self.exchange_manager.statistics();

        let r_hat = self.gelman_rubin_statistic().ok();
        let converged = self.has_converged();

        let replica_stats: Vec<_> = self.replicas.iter()
            .map(|r| ReplicaStats {
                id: r.id,
                temperature: r.temperature,
                current_energy: r.energy,
                current_model: r.selected_model,
                n_samples: r.energy_history.len(),
            })
            .collect();

        ReplicaExchangeSystemStats {
            iteration: self.iteration,
            n_replicas: self.replicas.len(),
            gelman_rubin: r_hat,
            converged,
            exchange_stats,
            replica_stats,
        }
    }

    /// Update feedback for selected model
    pub fn update_feedback(&mut self, model_idx: usize, actual_quality: f64) -> Result<()> {
        self.energy_model.update_quality_bayesian(
            model_idx,
            self.current_task,
            actual_quality,
        )?;

        self.energy_model.update_energy_feedback(model_idx, actual_quality)?;

        Ok(())
    }

    /// Learn energy weights from accumulated feedback
    pub fn learn_weights(&mut self) -> Result<()> {
        self.energy_model.learn_weights_from_feedback()
    }

    /// Get number of iterations run
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get number of replicas
    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }
}

/// Statistics for individual replica
#[derive(Debug, Clone)]
pub struct ReplicaStats {
    pub id: usize,
    pub temperature: f64,
    pub current_energy: f64,
    pub current_model: usize,
    pub n_samples: usize,
}

/// Comprehensive statistics for replica exchange system
#[derive(Debug, Clone)]
pub struct ReplicaExchangeSystemStats {
    pub iteration: usize,
    pub n_replicas: usize,
    pub gelman_rubin: Option<f64>,
    pub converged: bool,
    pub exchange_stats: super::temperature_schedules::ReplicaExchangeStats,
    pub replica_stats: Vec<ReplicaStats>,
}

impl ReplicaExchangeSystemStats {
    /// Print formatted statistics
    pub fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║          Replica Exchange System Statistics                 ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!("Iteration: {}", self.iteration);
        println!("Replicas: {}", self.n_replicas);

        if let Some(r_hat) = self.gelman_rubin {
            let status = if self.converged { "✓ CONVERGED" } else { "⏳ MIXING" };
            println!("Gelman-Rubin: {:.4} [{}]", r_hat, status);
        } else {
            println!("Gelman-Rubin: Insufficient samples");
        }

        println!("\nSwap Statistics:");
        println!("  Total swaps: {}", self.exchange_stats.total_swaps);
        println!("  Accepted: {} ({:.1}%)",
                 self.exchange_stats.accepted_swaps,
                 self.exchange_stats.overall_acceptance * 100.0);

        println!("\nPer-Pair Acceptance Rates:");
        for (i, &rate) in self.exchange_stats.per_pair_acceptance.iter().enumerate() {
            println!("  Replicas {}-{}: {:.1}%", i, i + 1, rate * 100.0);
        }

        println!("\nReplica States:");
        for stat in &self.replica_stats {
            println!("  Replica {}: T={:.3}, E={:.3}, Model={}, Samples={}",
                     stat.id,
                     stat.temperature,
                     stat.current_energy,
                     stat.current_model,
                     stat.n_samples);
        }

        println!("\n═══════════════════════════════════════════════════════════════\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::thermodynamic::advanced_energy::create_advanced_models;

    #[test]
    fn test_replica_exchange_system_creation() {
        let models = create_advanced_models();
        let system = ReplicaExchangeSystem::new(
            models,
            4,
            2.0,
            0.1,
            TaskType::Reasoning,
        ).unwrap();

        assert_eq!(system.num_replicas(), 4);
        assert_eq!(system.iteration(), 0);
    }

    #[test]
    fn test_replica_step() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            3,
            1.0,
            0.1,
            TaskType::Coding,
        ).unwrap();

        // Run one step
        system.step(0.01, 2000.0).unwrap();

        assert_eq!(system.iteration(), 1);

        // Each replica should have recorded one sample
        for replica in &system.replicas {
            assert_eq!(replica.energy_history.len(), 1);
            assert_eq!(replica.selection_history.len(), 1);
        }
    }

    #[test]
    fn test_multiple_steps() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            4,
            2.0,
            0.2,
            TaskType::General,
        ).unwrap();

        // Run 50 steps
        for _ in 0..50 {
            system.step(0.015, 1500.0).unwrap();
        }

        assert_eq!(system.iteration(), 50);

        // All replicas should have 50 samples
        for replica in &system.replicas {
            assert_eq!(replica.energy_history.len(), 50);
        }
    }

    #[test]
    fn test_gelman_rubin_computation() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            3,
            1.5,
            0.1,
            TaskType::Reasoning,
        ).unwrap();

        // Need at least 100 samples
        for _ in 0..100 {
            system.step(0.01, 2000.0).unwrap();
        }

        let r_hat = system.gelman_rubin_statistic().unwrap();

        // R̂ should be positive
        assert!(r_hat > 0.0);

        println!("Gelman-Rubin after 100 steps: {:.4}", r_hat);
    }

    #[test]
    fn test_convergence_check() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            2,
            1.0,
            0.1,
            TaskType::Creative,
        ).unwrap();

        // Initially should not have converged (not enough samples)
        assert!(!system.has_converged());

        // Run many steps
        for _ in 0..200 {
            system.step(0.01, 2000.0).unwrap();
        }

        // Check convergence (may or may not have converged depending on dynamics)
        let _converged = system.has_converged();
        // Don't assert convergence as it depends on problem dynamics
    }

    #[test]
    fn test_best_model_selection() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            4,
            2.0,
            0.1,
            TaskType::Coding,
        ).unwrap();

        // Run some steps
        for _ in 0..20 {
            system.step(0.01, 2000.0).unwrap();
        }

        // Should be able to get best model
        let best_model = system.get_best_model();
        assert!(best_model < system.energy_model.num_models());

        let best_energy = system.get_best_energy();
        assert!(best_energy.is_finite());
    }

    #[test]
    fn test_feedback_update() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            3,
            1.0,
            0.1,
            TaskType::QA,
        ).unwrap();

        // Run a step
        system.step(0.01, 2000.0).unwrap();

        let best_model = system.get_best_model();

        // Update feedback
        system.update_feedback(best_model, 0.85).unwrap();

        // Should not error
    }

    #[test]
    fn test_statistics() {
        let models = create_advanced_models();
        let mut system = ReplicaExchangeSystem::new(
            models,
            3,
            1.5,
            0.2,
            TaskType::Summarization,
        ).unwrap();

        // Run some steps
        for _ in 0..50 {
            system.step(0.015, 1800.0).unwrap();
        }

        let stats = system.statistics();

        assert_eq!(stats.iteration, 50);
        assert_eq!(stats.n_replicas, 3);
        assert_eq!(stats.replica_stats.len(), 3);

        // Print for visual inspection
        stats.print();
    }

    #[test]
    fn test_temperature_ordering() {
        let models = create_advanced_models();
        let system = ReplicaExchangeSystem::new(
            models,
            5,
            2.0,
            0.1,
            TaskType::General,
        ).unwrap();

        // Temperatures should be in descending order
        for i in 0..4 {
            assert!(system.replicas[i].temperature > system.replicas[i + 1].temperature);
        }
    }
}
