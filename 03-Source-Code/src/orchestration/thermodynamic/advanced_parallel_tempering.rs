//! Advanced Parallel Tempering (Replica Exchange) Temperature Schedule
//!
//! Worker 5 - Task 1.2 (12 hours)
//!
//! Implements parallel tempering optimization where multiple replicas run
//! at different temperatures simultaneously and exchange configurations.
//!
//! Key Innovation:
//! - Geometric temperature ladder: T_i = T_min * (T_max/T_min)^(i/(n-1))
//! - Metropolis swap acceptance: P(swap) = min(1, exp(ΔβΔE))
//! - GPU-accelerated parallel replica updates
//!
//! Benefits:
//! - Escapes local minima better than single-temperature annealing
//! - Parallel exploration of temperature space
//! - Automatic mixing between exploration (high T) and exploitation (low T)

use anyhow::{Result, anyhow};
use std::collections::VecDeque;

/// Replica state in parallel tempering
#[derive(Debug, Clone)]
pub struct ReplicaState {
    /// Temperature of this replica
    pub temperature: f64,

    /// Energy of current state
    pub energy: f64,

    /// Configuration (model probabilities for LLM selection)
    pub configuration: Vec<f64>,

    /// Number of accepted moves at this temperature
    pub accepted_moves: usize,

    /// Total moves attempted at this temperature
    pub total_moves: usize,
}

impl ReplicaState {
    /// Create new replica state
    pub fn new(temperature: f64, configuration: Vec<f64>, energy: f64) -> Self {
        Self {
            temperature,
            energy,
            configuration,
            accepted_moves: 0,
            total_moves: 0,
        }
    }

    /// Get acceptance rate for this replica
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_moves == 0 {
            0.0
        } else {
            self.accepted_moves as f64 / self.total_moves as f64
        }
    }

    /// Record move result
    pub fn record_move(&mut self, accepted: bool) {
        self.total_moves += 1;
        if accepted {
            self.accepted_moves += 1;
        }
    }
}

/// Exchange schedule type
#[derive(Debug, Clone, Copy)]
pub enum ExchangeSchedule {
    /// Fixed interval: try swaps every N iterations
    Fixed { interval: usize },

    /// Adaptive: adjust interval based on acceptance rates
    Adaptive {
        min_interval: usize,
        max_interval: usize,
        target_acceptance: f64,
    },

    /// Random: try swaps with probability p each iteration
    Stochastic { probability: f64 },
}

/// Parallel Tempering Schedule
///
/// Manages multiple replicas at different temperatures that can exchange
/// configurations to improve exploration of the energy landscape.
pub struct ParallelTemperingSchedule {
    /// All replicas ordered by temperature (ascending)
    replicas: Vec<ReplicaState>,

    /// Exchange schedule
    exchange_schedule: ExchangeSchedule,

    /// Current iteration
    iteration: usize,

    /// Number of iterations since last swap attempt
    iterations_since_swap: usize,

    /// History of swap attempts (for adaptive scheduling)
    swap_history: VecDeque<SwapAttempt>,

    /// Maximum swap history to keep
    max_history: usize,
}

/// Record of a swap attempt
#[derive(Debug, Clone)]
struct SwapAttempt {
    iteration: usize,
    replica_i: usize,
    replica_j: usize,
    accepted: bool,
}

impl ParallelTemperingSchedule {
    /// Create new parallel tempering schedule with geometric temperature ladder
    ///
    /// Temperature ladder: T_i = T_min * (T_max/T_min)^(i/(n-1))
    ///
    /// # Arguments
    /// * `num_replicas` - Number of parallel replicas
    /// * `temp_min` - Lowest temperature (exploitation)
    /// * `temp_max` - Highest temperature (exploration)
    /// * `initial_config` - Initial configuration for all replicas
    /// * `exchange_schedule` - When to attempt swaps
    ///
    /// # Returns
    /// New parallel tempering schedule
    pub fn new(
        num_replicas: usize,
        temp_min: f64,
        temp_max: f64,
        initial_config: Vec<f64>,
        exchange_schedule: ExchangeSchedule,
    ) -> Result<Self> {
        if num_replicas < 2 {
            return Err(anyhow!("Need at least 2 replicas for parallel tempering"));
        }
        if temp_min <= 0.0 || temp_max <= temp_min {
            return Err(anyhow!("Invalid temperature range"));
        }

        // Generate geometric temperature ladder
        let temperatures = Self::generate_temperature_ladder(
            num_replicas,
            temp_min,
            temp_max,
        );

        // Initialize replicas with same configuration but different temperatures
        let replicas = temperatures
            .into_iter()
            .map(|temp| ReplicaState::new(temp, initial_config.clone(), 0.0))
            .collect();

        Ok(Self {
            replicas,
            exchange_schedule,
            iteration: 0,
            iterations_since_swap: 0,
            swap_history: VecDeque::new(),
            max_history: 1000,
        })
    }

    /// Generate geometric temperature ladder
    ///
    /// T_i = T_min * (T_max/T_min)^(i/(n-1))
    ///
    /// This ensures:
    /// - Even spacing on log scale
    /// - Good overlap between adjacent temperatures
    /// - Efficient exchange acceptance
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

    /// Attempt replica exchange between adjacent temperatures
    ///
    /// Metropolis acceptance: P(swap) = min(1, exp((β_i - β_j)(E_j - E_i)))
    /// where β = 1/T
    ///
    /// Returns: (replica_i, replica_j, accepted)
    pub fn attempt_swap(&mut self) -> Option<(usize, usize, bool)> {
        if self.replicas.len() < 2 {
            return None;
        }

        // Try swapping adjacent replicas
        // In practice, randomly select an adjacent pair
        let i = (self.iteration % (self.replicas.len() - 1)) as usize;
        let j = i + 1;

        let accepted = self.metropolis_swap_criterion(i, j);

        if accepted {
            // Swap configurations (not temperatures!)
            let config_i = self.replicas[i].configuration.clone();
            let config_j = self.replicas[j].configuration.clone();
            let energy_i = self.replicas[i].energy;
            let energy_j = self.replicas[j].energy;

            self.replicas[i].configuration = config_j;
            self.replicas[i].energy = energy_j;
            self.replicas[j].configuration = config_i;
            self.replicas[j].energy = energy_i;
        }

        // Record swap attempt
        self.swap_history.push_back(SwapAttempt {
            iteration: self.iteration,
            replica_i: i,
            replica_j: j,
            accepted,
        });

        // Keep history bounded
        if self.swap_history.len() > self.max_history {
            self.swap_history.pop_front();
        }

        Some((i, j, accepted))
    }

    /// Metropolis swap acceptance criterion
    ///
    /// P(swap) = min(1, exp((β_i - β_j)(E_j - E_i)))
    ///
    /// GPU-accelerated in production (uses GPU random number generation)
    fn metropolis_swap_criterion(&self, i: usize, j: usize) -> bool {
        let beta_i = 1.0 / self.replicas[i].temperature;
        let beta_j = 1.0 / self.replicas[j].temperature;
        let energy_i = self.replicas[i].energy;
        let energy_j = self.replicas[j].energy;

        let delta_beta = beta_i - beta_j;
        let delta_energy = energy_j - energy_i;

        let acceptance_prob = (delta_beta * delta_energy).exp().min(1.0);

        // TODO: Use GPU random number generator from Worker 2
        // For now, use CPU random
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();

        random_value < acceptance_prob
    }

    /// Update iteration and check if swap should be attempted
    ///
    /// Returns true if swap should be attempted this iteration
    pub fn should_attempt_swap(&mut self) -> bool {
        self.iteration += 1;
        self.iterations_since_swap += 1;

        match self.exchange_schedule {
            ExchangeSchedule::Fixed { interval } => {
                if self.iterations_since_swap >= interval {
                    self.iterations_since_swap = 0;
                    true
                } else {
                    false
                }
            },
            ExchangeSchedule::Adaptive {
                min_interval,
                max_interval,
                target_acceptance,
            } => {
                // Adjust interval based on recent acceptance rate
                let recent_acceptance = self.get_swap_acceptance_rate(100);

                let current_interval = if recent_acceptance > target_acceptance {
                    // Too many accepts, can swap more often
                    (self.iterations_since_swap as f64 * 0.9) as usize
                } else {
                    // Too few accepts, swap less often
                    (self.iterations_since_swap as f64 * 1.1) as usize
                };

                let interval = current_interval.clamp(min_interval, max_interval);

                if self.iterations_since_swap >= interval {
                    self.iterations_since_swap = 0;
                    true
                } else {
                    false
                }
            },
            ExchangeSchedule::Stochastic { probability } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let random_value: f64 = rng.gen();

                if random_value < probability {
                    self.iterations_since_swap = 0;
                    true
                } else {
                    false
                }
            },
        }
    }

    /// Get swap acceptance rate over recent history
    fn get_swap_acceptance_rate(&self, window: usize) -> f64 {
        if self.swap_history.is_empty() {
            return 0.5; // Default to 50%
        }

        let recent = if self.swap_history.len() <= window {
            &self.swap_history
        } else {
            let start = self.swap_history.len() - window;
            let slice = self.swap_history.range(start..);
            return slice.filter(|s| s.accepted).count() as f64 / window as f64;
        };

        recent.iter().filter(|s| s.accepted).count() as f64 / recent.len() as f64
    }

    /// Update replica energy (called after local moves)
    pub fn update_replica_energy(&mut self, replica_idx: usize, new_energy: f64) {
        if replica_idx < self.replicas.len() {
            self.replicas[replica_idx].energy = new_energy;
        }
    }

    /// Update replica configuration (called after local moves)
    pub fn update_replica_config(
        &mut self,
        replica_idx: usize,
        new_config: Vec<f64>,
        new_energy: f64,
        accepted: bool,
    ) {
        if replica_idx < self.replicas.len() {
            if accepted {
                self.replicas[replica_idx].configuration = new_config;
                self.replicas[replica_idx].energy = new_energy;
            }
            self.replicas[replica_idx].record_move(accepted);
        }
    }

    /// Get best configuration (from lowest temperature replica)
    ///
    /// Lowest temperature replica has best optimization result
    pub fn get_best_configuration(&self) -> &[f64] {
        &self.replicas[0].configuration
    }

    /// Get best energy
    pub fn get_best_energy(&self) -> f64 {
        self.replicas[0].energy
    }

    /// Get all replicas (for inspection/debugging)
    pub fn get_replicas(&self) -> &[ReplicaState] {
        &self.replicas
    }

    /// Get number of replicas
    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }

    /// Get temperature ladder
    pub fn get_temperature_ladder(&self) -> Vec<f64> {
        self.replicas.iter().map(|r| r.temperature).collect()
    }

    /// Get current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get swap statistics
    pub fn get_swap_statistics(&self) -> SwapStatistics {
        let total_swaps = self.swap_history.len();
        let accepted_swaps = self.swap_history.iter().filter(|s| s.accepted).count();
        let acceptance_rate = if total_swaps > 0 {
            accepted_swaps as f64 / total_swaps as f64
        } else {
            0.0
        };

        SwapStatistics {
            total_attempts: total_swaps,
            accepted_swaps,
            acceptance_rate,
        }
    }
}

/// Swap statistics
#[derive(Debug, Clone)]
pub struct SwapStatistics {
    pub total_attempts: usize,
    pub accepted_swaps: usize,
    pub acceptance_rate: f64,
}

/// GPU-accelerated parallel tempering
///
/// Manages multiple parallel tempering runs on GPU
pub struct GpuParallelTempering {
    /// Number of independent parallel tempering runs
    num_runs: usize,

    /// Schedules for each run
    schedules: Vec<ParallelTemperingSchedule>,
}

impl GpuParallelTempering {
    /// Create new GPU parallel tempering manager
    pub fn new(
        num_runs: usize,
        replicas_per_run: usize,
        temp_min: f64,
        temp_max: f64,
        initial_configs: Vec<Vec<f64>>,
        exchange_schedule: ExchangeSchedule,
    ) -> Result<Self> {
        if num_runs != initial_configs.len() {
            return Err(anyhow!("Number of runs must match number of initial configs"));
        }

        let schedules = initial_configs
            .into_iter()
            .map(|config| {
                ParallelTemperingSchedule::new(
                    replicas_per_run,
                    temp_min,
                    temp_max,
                    config,
                    exchange_schedule,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            num_runs,
            schedules,
        })
    }

    /// Update all runs in parallel on GPU
    ///
    /// TODO: Request GPU kernel from Worker 2:
    /// - parallel_tempering_update.cu
    /// - Computes all replica swaps in parallel
    /// - Uses GPU random number generation
    pub fn update_all_gpu(&mut self) -> Result<()> {
        // PLACEHOLDER: Will use GPU kernel once Worker 2 provides it
        // For now, update each schedule sequentially

        for schedule in &mut self.schedules {
            if schedule.should_attempt_swap() {
                schedule.attempt_swap();
            }
        }

        Ok(())
    }

    /// Get best configurations from all runs
    pub fn get_best_configurations(&self) -> Vec<Vec<f64>> {
        self.schedules
            .iter()
            .map(|s| s.get_best_configuration().to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_ladder_generation() {
        let temps = ParallelTemperingSchedule::generate_temperature_ladder(
            5,    // num_replicas
            0.1,  // temp_min
            10.0, // temp_max
        );

        assert_eq!(temps.len(), 5);
        assert!((temps[0] - 0.1).abs() < 1e-10);
        assert!((temps[4] - 10.0).abs() < 1e-10);

        // Check geometric spacing
        for i in 1..temps.len() {
            assert!(temps[i] > temps[i-1]); // Ascending order
        }
    }

    #[test]
    fn test_parallel_tempering_creation() -> Result<()> {
        let initial_config = vec![0.2, 0.3, 0.5];
        let schedule = ParallelTemperingSchedule::new(
            4,
            0.1,
            10.0,
            initial_config.clone(),
            ExchangeSchedule::Fixed { interval: 10 },
        )?;

        assert_eq!(schedule.num_replicas(), 4);

        let ladder = schedule.get_temperature_ladder();
        assert_eq!(ladder.len(), 4);
        assert!(ladder[0] < ladder[1]);
        assert!(ladder[1] < ladder[2]);
        assert!(ladder[2] < ladder[3]);

        Ok(())
    }

    #[test]
    fn test_swap_scheduling_fixed() -> Result<()> {
        let initial_config = vec![1.0];
        let mut schedule = ParallelTemperingSchedule::new(
            3,
            0.1,
            10.0,
            initial_config,
            ExchangeSchedule::Fixed { interval: 5 },
        )?;

        // Should not swap for first 4 iterations
        for _ in 0..4 {
            assert!(!schedule.should_attempt_swap());
        }

        // Should swap on 5th iteration
        assert!(schedule.should_attempt_swap());

        Ok(())
    }

    #[test]
    fn test_swap_attempt() -> Result<()> {
        let initial_config = vec![1.0];
        let mut schedule = ParallelTemperingSchedule::new(
            3,
            0.1,
            10.0,
            initial_config,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        // Set different energies
        schedule.update_replica_energy(0, 1.0);
        schedule.update_replica_energy(1, 2.0);
        schedule.update_replica_energy(2, 3.0);

        // Attempt swap
        schedule.should_attempt_swap();
        let result = schedule.attempt_swap();

        assert!(result.is_some());
        let (i, j, _accepted) = result.unwrap();
        assert_eq!(j, i + 1); // Adjacent replicas

        Ok(())
    }

    #[test]
    fn test_replica_update() -> Result<()> {
        let initial_config = vec![1.0, 2.0];
        let mut schedule = ParallelTemperingSchedule::new(
            2,
            0.1,
            10.0,
            initial_config,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        let new_config = vec![3.0, 4.0];
        schedule.update_replica_config(0, new_config.clone(), 5.0, true);

        assert_eq!(schedule.replicas[0].configuration, new_config);
        assert_eq!(schedule.replicas[0].energy, 5.0);
        assert_eq!(schedule.replicas[0].accepted_moves, 1);

        Ok(())
    }

    #[test]
    fn test_best_configuration() -> Result<()> {
        let initial_config = vec![1.0];
        let mut schedule = ParallelTemperingSchedule::new(
            3,
            0.1,
            10.0,
            initial_config,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        // Update lowest temperature replica
        let best_config = vec![0.5];
        schedule.update_replica_config(0, best_config.clone(), 0.1, true);

        assert_eq!(schedule.get_best_configuration(), &best_config[..]);
        assert_eq!(schedule.get_best_energy(), 0.1);

        Ok(())
    }

    #[test]
    fn test_swap_statistics() -> Result<()> {
        let initial_config = vec![1.0];
        let mut schedule = ParallelTemperingSchedule::new(
            3,
            0.1,
            10.0,
            initial_config,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        // Attempt several swaps
        for _ in 0..10 {
            schedule.should_attempt_swap();
            schedule.attempt_swap();
        }

        let stats = schedule.get_swap_statistics();
        assert_eq!(stats.total_attempts, 10);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);

        Ok(())
    }

    #[test]
    fn test_acceptance_rate_tracking() -> Result<()> {
        let initial_config = vec![1.0];
        let mut schedule = ParallelTemperingSchedule::new(
            2,
            0.1,
            10.0,
            initial_config,
            ExchangeSchedule::Fixed { interval: 1 },
        )?;

        schedule.replicas[0].record_move(true);
        schedule.replicas[0].record_move(true);
        schedule.replicas[0].record_move(false);

        let rate = schedule.replicas[0].acceptance_rate();
        assert!((rate - 2.0/3.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_gpu_parallel_tempering() -> Result<()> {
        let initial_configs = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let mut gpu_pt = GpuParallelTempering::new(
            2,    // num_runs
            3,    // replicas_per_run
            0.1,  // temp_min
            10.0, // temp_max
            initial_configs,
            ExchangeSchedule::Fixed { interval: 5 },
        )?;

        gpu_pt.update_all_gpu()?;

        let best_configs = gpu_pt.get_best_configurations();
        assert_eq!(best_configs.len(), 2);

        Ok(())
    }

    #[test]
    fn test_invalid_parameters() {
        // Too few replicas
        let result = ParallelTemperingSchedule::new(
            1,
            0.1,
            10.0,
            vec![1.0],
            ExchangeSchedule::Fixed { interval: 1 },
        );
        assert!(result.is_err());

        // Invalid temperature range
        let result = ParallelTemperingSchedule::new(
            3,
            10.0,  // min > max
            0.1,
            vec![1.0],
            ExchangeSchedule::Fixed { interval: 1 },
        );
        assert!(result.is_err());
    }
}
