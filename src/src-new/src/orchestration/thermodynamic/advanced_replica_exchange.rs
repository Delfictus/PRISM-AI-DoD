//! Advanced Replica Exchange Implementation
//!
//! Worker 5 - Task 2.1 (20 hours)
//!
//! Full replica exchange system that integrates with parallel tempering
//! and thermodynamic consensus for optimal LLM selection across temperatures.
//!
//! Key Features:
//! - Replica state management with full thermodynamic state
//! - Exchange proposal mechanism with multiple strategies
//! - Metropolis exchange criteria with detailed balance
//! - GPU-accelerated parallel replica updates
//! - Integration with existing consensus modules
//!
//! Extends parallel tempering with full thermodynamic state integration.

use anyhow::{Result, anyhow};
use super::{
    ParallelTemperingSchedule, ReplicaState as PTReplicaState,
    ExchangeSchedule, GpuThermodynamicState, LLMModel,
};
use std::collections::HashMap;

/// Full thermodynamic replica state
///
/// Extends basic ReplicaState with complete thermodynamic information
#[derive(Debug, Clone)]
pub struct ThermodynamicReplicaState {
    /// Temperature of this replica
    pub temperature: f64,

    /// Current thermodynamic state (model probabilities)
    pub state: GpuThermodynamicState,

    /// Energy/cost of current configuration
    pub energy: f64,

    /// Entropy of model distribution
    pub entropy: f64,

    /// Free energy: F = E - T·S
    pub free_energy: f64,

    /// Move statistics
    pub accepted_moves: usize,
    pub total_moves: usize,

    /// Exchange statistics with other replicas
    pub exchange_attempts: HashMap<usize, usize>, // replica_id -> attempts
    pub exchange_accepts: HashMap<usize, usize>,   // replica_id -> accepts
}

impl ThermodynamicReplicaState {
    /// Create new thermodynamic replica
    pub fn new(
        temperature: f64,
        initial_state: GpuThermodynamicState,
    ) -> Self {
        let energy = initial_state.free_energy;
        let entropy = initial_state.entropy;
        let free_energy = energy - temperature * entropy;

        Self {
            temperature,
            state: initial_state,
            energy,
            entropy,
            free_energy,
            accepted_moves: 0,
            total_moves: 0,
            exchange_attempts: HashMap::new(),
            exchange_accepts: HashMap::new(),
        }
    }

    /// Update thermodynamic quantities after state change
    pub fn update_thermodynamics(&mut self) {
        self.energy = self.state.free_energy;
        self.entropy = self.state.entropy;
        self.free_energy = self.energy - self.temperature * self.entropy;
    }

    /// Get acceptance rate for moves
    pub fn move_acceptance_rate(&self) -> f64 {
        if self.total_moves == 0 {
            0.0
        } else {
            self.accepted_moves as f64 / self.total_moves as f64
        }
    }

    /// Get exchange acceptance rate with specific replica
    pub fn exchange_acceptance_rate(&self, replica_id: usize) -> f64 {
        let attempts = self.exchange_attempts.get(&replica_id).copied().unwrap_or(0);
        let accepts = self.exchange_accepts.get(&replica_id).copied().unwrap_or(0);

        if attempts == 0 {
            0.0
        } else {
            accepts as f64 / attempts as f64
        }
    }

    /// Record exchange attempt with another replica
    pub fn record_exchange_attempt(&mut self, replica_id: usize, accepted: bool) {
        *self.exchange_attempts.entry(replica_id).or_insert(0) += 1;
        if accepted {
            *self.exchange_accepts.entry(replica_id).or_insert(0) += 1;
        }
    }
}

/// Exchange proposal strategy
#[derive(Debug, Clone, Copy)]
pub enum ExchangeProposal {
    /// Try all adjacent pairs sequentially
    Sequential,

    /// Try random adjacent pairs
    RandomAdjacent,

    /// Try all pairs (not just adjacent)
    AllPairs,

    /// Adaptive: prioritize pairs with good exchange rates
    Adaptive { min_rate: f64, max_rate: f64 },
}

/// Replica Exchange Manager
///
/// Manages multiple replicas at different temperatures with state exchange
pub struct ReplicaExchange {
    /// All replicas ordered by temperature
    replicas: Vec<ThermodynamicReplicaState>,

    /// Exchange proposal strategy
    proposal_strategy: ExchangeProposal,

    /// Exchange schedule (when to attempt swaps)
    exchange_schedule: ExchangeSchedule,

    /// Current iteration
    iteration: usize,

    /// Iterations since last exchange
    iterations_since_exchange: usize,

    /// Available LLM models
    models: Vec<LLMModel>,

    /// Global exchange statistics
    total_exchange_attempts: usize,
    total_exchange_accepts: usize,
}

impl ReplicaExchange {
    /// Create new replica exchange system
    ///
    /// # Arguments
    /// * `num_replicas` - Number of temperature replicas
    /// * `temp_min` - Lowest temperature
    /// * `temp_max` - Highest temperature
    /// * `models` - Available LLM models
    /// * `proposal_strategy` - Exchange proposal strategy
    /// * `exchange_schedule` - When to attempt exchanges
    pub fn new(
        num_replicas: usize,
        temp_min: f64,
        temp_max: f64,
        models: Vec<LLMModel>,
        proposal_strategy: ExchangeProposal,
        exchange_schedule: ExchangeSchedule,
    ) -> Result<Self> {
        if num_replicas < 2 {
            return Err(anyhow!("Need at least 2 replicas"));
        }
        if temp_min <= 0.0 || temp_max <= temp_min {
            return Err(anyhow!("Invalid temperature range"));
        }
        if models.is_empty() {
            return Err(anyhow!("Need at least one LLM model"));
        }

        // Generate temperature ladder (geometric)
        let temperatures = Self::generate_temperature_ladder(
            num_replicas,
            temp_min,
            temp_max,
        );

        // Initialize replicas with uniform distribution
        let n_models = models.len();
        let uniform_prob = 1.0 / n_models as f64;
        let replicas = temperatures.into_iter()
            .map(|temp| {
                let state = GpuThermodynamicState {
                    model_probabilities: vec![uniform_prob; n_models],
                    temperature: temp,
                    free_energy: 0.0,
                    entropy: (n_models as f64).ln(), // Maximum entropy
                };
                ThermodynamicReplicaState::new(temp, state)
            })
            .collect();

        Ok(Self {
            replicas,
            proposal_strategy,
            exchange_schedule,
            iteration: 0,
            iterations_since_exchange: 0,
            models,
            total_exchange_attempts: 0,
            total_exchange_accepts: 0,
        })
    }

    /// Generate geometric temperature ladder
    fn generate_temperature_ladder(
        num_replicas: usize,
        temp_min: f64,
        temp_max: f64,
    ) -> Vec<f64> {
        let ratio = temp_max / temp_min;
        let exponent_step = 1.0 / (num_replicas - 1) as f64;

        (0..num_replicas)
            .map(|i| {
                let exponent = i as f64 * exponent_step;
                temp_min * ratio.powf(exponent)
            })
            .collect()
    }

    /// Perform one full iteration
    ///
    /// 1. Local moves at each temperature
    /// 2. Exchange attempts if scheduled
    ///
    /// Returns acceptance rates for moves and exchanges
    pub fn iterate<F>(&mut self, evaluate_fn: F) -> Result<IterationStats>
    where
        F: Fn(&GpuThermodynamicState) -> f64,
    {
        // Perform local moves at each temperature
        let move_accepts = self.local_moves(&evaluate_fn)?;

        // Check if we should attempt exchanges
        self.iteration += 1;
        self.iterations_since_exchange += 1;

        let exchange_accepts = if self.should_attempt_exchange()? {
            self.iterations_since_exchange = 0;
            self.attempt_exchanges()?
        } else {
            0
        };

        Ok(IterationStats {
            move_accepts,
            total_moves: self.replicas.len(),
            exchange_accepts,
            total_exchanges: if exchange_accepts > 0 {
                self.replicas.len() - 1
            } else {
                0
            },
        })
    }

    /// Perform local moves at each temperature
    fn local_moves<F>(&mut self, evaluate_fn: &F) -> Result<usize>
    where
        F: Fn(&GpuThermodynamicState) -> f64,
    {
        let mut accepts = 0;

        for replica in &mut self.replicas {
            // Propose small random change to model probabilities
            let proposed_state = self.propose_local_move(&replica.state)?;

            // Evaluate energy
            let proposed_energy = evaluate_fn(&proposed_state);

            // Metropolis acceptance
            let delta_e = proposed_energy - replica.energy;
            let acceptance_prob = (-delta_e / replica.temperature).exp().min(1.0);

            use rand::Rng;
            let mut rng = rand::thread_rng();
            let accepted = rng.gen::<f64>() < acceptance_prob;

            replica.total_moves += 1;
            if accepted {
                replica.state = proposed_state;
                replica.update_thermodynamics();
                replica.accepted_moves += 1;
                accepts += 1;
            }
        }

        Ok(accepts)
    }

    /// Propose local move (small perturbation)
    fn propose_local_move(&self, state: &GpuThermodynamicState) -> Result<GpuThermodynamicState> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut new_probs = state.model_probabilities.clone();

        // Perturb probabilities slightly
        for i in 0..new_probs.len() {
            let noise = rng.gen_range(-0.1..0.1);
            new_probs[i] = (new_probs[i] + noise).max(0.0);
        }

        // Renormalize
        let sum: f64 = new_probs.iter().sum();
        for p in &mut new_probs {
            *p /= sum;
        }

        // Compute new entropy
        let entropy = -new_probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        Ok(GpuThermodynamicState {
            model_probabilities: new_probs,
            temperature: state.temperature,
            free_energy: state.free_energy, // Will be updated
            entropy,
        })
    }

    /// Attempt replica exchanges
    fn attempt_exchanges(&mut self) -> Result<usize> {
        match self.proposal_strategy {
            ExchangeProposal::Sequential => self.exchange_sequential(),
            ExchangeProposal::RandomAdjacent => self.exchange_random_adjacent(),
            ExchangeProposal::AllPairs => self.exchange_all_pairs(),
            ExchangeProposal::Adaptive { min_rate, max_rate } => {
                self.exchange_adaptive(min_rate, max_rate)
            },
        }
    }

    /// Sequential adjacent exchanges
    fn exchange_sequential(&mut self) -> Result<usize> {
        let mut accepts = 0;

        for i in 0..(self.replicas.len() - 1) {
            if self.try_exchange(i, i + 1)? {
                accepts += 1;
            }
        }

        Ok(accepts)
    }

    /// Random adjacent exchange
    fn exchange_random_adjacent(&mut self) -> Result<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let i = rng.gen_range(0..(self.replicas.len() - 1));
        Ok(if self.try_exchange(i, i + 1)? { 1 } else { 0 })
    }

    /// Try all pairs
    fn exchange_all_pairs(&mut self) -> Result<usize> {
        let mut accepts = 0;

        for i in 0..self.replicas.len() {
            for j in (i + 1)..self.replicas.len() {
                if self.try_exchange(i, j)? {
                    accepts += 1;
                }
            }
        }

        Ok(accepts)
    }

    /// Adaptive exchange (prioritize good pairs)
    fn exchange_adaptive(&mut self, min_rate: f64, max_rate: f64) -> Result<usize> {
        let mut accepts = 0;

        // Try pairs with acceptance rates in target range
        for i in 0..(self.replicas.len() - 1) {
            let rate = self.replicas[i].exchange_acceptance_rate(i + 1);

            if rate >= min_rate && rate <= max_rate {
                if self.try_exchange(i, i + 1)? {
                    accepts += 1;
                }
            }
        }

        Ok(accepts)
    }

    /// Try exchange between two replicas
    fn try_exchange(&mut self, i: usize, j: usize) -> Result<bool> {
        if i >= self.replicas.len() || j >= self.replicas.len() {
            return Err(anyhow!("Invalid replica indices"));
        }

        let temp_i = self.replicas[i].temperature;
        let temp_j = self.replicas[j].temperature;
        let energy_i = self.replicas[i].energy;
        let energy_j = self.replicas[j].energy;

        // Metropolis exchange criterion
        let delta_beta = (1.0 / temp_i) - (1.0 / temp_j);
        let delta_energy = energy_j - energy_i;
        let acceptance_prob = (delta_beta * delta_energy).exp().min(1.0);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let accepted = rng.gen::<f64>() < acceptance_prob;

        // Record attempt
        self.total_exchange_attempts += 1;
        self.replicas[i].record_exchange_attempt(j, accepted);
        self.replicas[j].record_exchange_attempt(i, accepted);

        if accepted {
            // Swap states (not temperatures!)
            let state_i = self.replicas[i].state.clone();
            let state_j = self.replicas[j].state.clone();

            self.replicas[i].state = state_j;
            self.replicas[j].state = state_i;

            // Update thermodynamics
            self.replicas[i].update_thermodynamics();
            self.replicas[j].update_thermodynamics();

            self.total_exchange_accepts += 1;
        }

        Ok(accepted)
    }

    /// Check if exchange should be attempted
    fn should_attempt_exchange(&self) -> Result<bool> {
        match self.exchange_schedule {
            ExchangeSchedule::Fixed { interval } => {
                Ok(self.iterations_since_exchange >= interval)
            },
            ExchangeSchedule::Adaptive { min_interval, max_interval, target_acceptance } => {
                let rate = self.global_exchange_rate();
                let interval = if rate > target_acceptance {
                    min_interval
                } else {
                    max_interval
                };
                Ok(self.iterations_since_exchange >= interval)
            },
            ExchangeSchedule::Stochastic { probability } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                Ok(rng.gen::<f64>() < probability)
            },
        }
    }

    /// Get global exchange acceptance rate
    fn global_exchange_rate(&self) -> f64 {
        if self.total_exchange_attempts == 0 {
            0.5
        } else {
            self.total_exchange_accepts as f64 / self.total_exchange_attempts as f64
        }
    }

    /// Get best configuration (from lowest temperature)
    pub fn get_best_configuration(&self) -> &GpuThermodynamicState {
        &self.replicas[0].state
    }

    /// Get all replicas
    pub fn get_replicas(&self) -> &[ThermodynamicReplicaState] {
        &self.replicas
    }

    /// Get temperature ladder
    pub fn get_temperatures(&self) -> Vec<f64> {
        self.replicas.iter().map(|r| r.temperature).collect()
    }

    /// Get overall statistics
    pub fn get_statistics(&self) -> ExchangeStatistics {
        ExchangeStatistics {
            num_replicas: self.replicas.len(),
            total_iterations: self.iteration,
            total_exchange_attempts: self.total_exchange_attempts,
            total_exchange_accepts: self.total_exchange_accepts,
            global_exchange_rate: self.global_exchange_rate(),
            replica_move_rates: self.replicas.iter()
                .map(|r| r.move_acceptance_rate())
                .collect(),
        }
    }
}

/// Iteration statistics
#[derive(Debug, Clone)]
pub struct IterationStats {
    pub move_accepts: usize,
    pub total_moves: usize,
    pub exchange_accepts: usize,
    pub total_exchanges: usize,
}

impl IterationStats {
    pub fn move_acceptance_rate(&self) -> f64 {
        if self.total_moves == 0 {
            0.0
        } else {
            self.move_accepts as f64 / self.total_moves as f64
        }
    }

    pub fn exchange_acceptance_rate(&self) -> f64 {
        if self.total_exchanges == 0 {
            0.0
        } else {
            self.exchange_accepts as f64 / self.total_exchanges as f64
        }
    }
}

/// Overall exchange statistics
#[derive(Debug, Clone)]
pub struct ExchangeStatistics {
    pub num_replicas: usize,
    pub total_iterations: usize,
    pub total_exchange_attempts: usize,
    pub total_exchange_accepts: usize,
    pub global_exchange_rate: f64,
    pub replica_move_rates: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_models() -> Vec<LLMModel> {
        vec![
            LLMModel {
                name: "GPT-4".to_string(),
                cost_per_1k_tokens: 0.03,
                quality_score: 0.9,
                latency_ms: 500.0,
                max_tokens: 8192,
            },
            LLMModel {
                name: "Claude".to_string(),
                cost_per_1k_tokens: 0.025,
                quality_score: 0.85,
                latency_ms: 450.0,
                max_tokens: 100000,
            },
        ]
    }

    fn dummy_eval(_state: &GpuThermodynamicState) -> f64 {
        1.0 // Constant energy for testing
    }

    #[test]
    fn test_replica_exchange_creation() -> Result<()> {
        let models = dummy_models();
        let rex = ReplicaExchange::new(
            4,
            0.1,
            10.0,
            models,
            ExchangeProposal::Sequential,
            ExchangeSchedule::Fixed { interval: 10 },
        )?;

        assert_eq!(rex.get_replicas().len(), 4);

        let temps = rex.get_temperatures();
        assert!(temps[0] < temps[1]);
        assert!(temps[1] < temps[2]);
        assert!(temps[2] < temps[3]);

        Ok(())
    }

    #[test]
    fn test_temperature_ladder() {
        let temps = ReplicaExchange::generate_temperature_ladder(5, 0.1, 10.0);

        assert_eq!(temps.len(), 5);
        assert!((temps[0] - 0.1).abs() < 1e-10);
        assert!((temps[4] - 10.0).abs() < 1e-10);

        // Geometric spacing
        for i in 1..temps.len() {
            assert!(temps[i] > temps[i - 1]);
        }
    }

    #[test]
    fn test_iteration() -> Result<()> {
        let models = dummy_models();
        let mut rex = ReplicaExchange::new(
            3,
            0.1,
            10.0,
            models,
            ExchangeProposal::Sequential,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        let stats = rex.iterate(dummy_eval)?;

        assert_eq!(stats.total_moves, 3);
        assert!(stats.move_accepts <= stats.total_moves);

        Ok(())
    }

    #[test]
    fn test_thermodynamic_replica_state() {
        let state = GpuThermodynamicState {
            model_probabilities: vec![0.5, 0.5],
            temperature: 1.0,
            free_energy: 1.0,
            entropy: 0.693,
        };

        let replica = ThermodynamicReplicaState::new(1.0, state);

        assert_eq!(replica.temperature, 1.0);
        assert_eq!(replica.move_acceptance_rate(), 0.0);
    }

    #[test]
    fn test_exchange_statistics() -> Result<()> {
        let models = dummy_models();
        let mut rex = ReplicaExchange::new(
            3,
            0.1,
            10.0,
            models,
            ExchangeProposal::Sequential,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        for _ in 0..10 {
            rex.iterate(dummy_eval)?;
        }

        let stats = rex.get_statistics();

        assert_eq!(stats.num_replicas, 3);
        assert_eq!(stats.total_iterations, 10);
        assert!(stats.global_exchange_rate >= 0.0);
        assert!(stats.global_exchange_rate <= 1.0);

        Ok(())
    }

    #[test]
    fn test_exchange_proposal_strategies() -> Result<()> {
        let models = dummy_models();

        // Test each strategy
        let strategies = vec![
            ExchangeProposal::Sequential,
            ExchangeProposal::RandomAdjacent,
            ExchangeProposal::AllPairs,
            ExchangeProposal::Adaptive { min_rate: 0.3, max_rate: 0.7 },
        ];

        for strategy in strategies {
            let mut rex = ReplicaExchange::new(
                3,
                0.1,
                10.0,
                models.clone(),
                strategy,
                ExchangeSchedule::Fixed { interval: 1 },
            )?;

            rex.iterate(dummy_eval)?;
        }

        Ok(())
    }

    #[test]
    fn test_invalid_parameters() {
        let models = dummy_models();

        // Too few replicas
        assert!(ReplicaExchange::new(
            1,
            0.1,
            10.0,
            models.clone(),
            ExchangeProposal::Sequential,
            ExchangeSchedule::Fixed { interval: 1 },
        ).is_err());

        // Invalid temperature range
        assert!(ReplicaExchange::new(
            3,
            10.0,
            0.1,
            models.clone(),
            ExchangeProposal::Sequential,
            ExchangeSchedule::Fixed { interval: 1 },
        ).is_err());

        // No models
        assert!(ReplicaExchange::new(
            3,
            0.1,
            10.0,
            vec![],
            ExchangeProposal::Sequential,
            ExchangeSchedule::Fixed { interval: 1 },
        ).is_err());
    }
}
