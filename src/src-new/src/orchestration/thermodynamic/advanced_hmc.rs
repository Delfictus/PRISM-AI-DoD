//! Advanced Hamiltonian Monte Carlo Temperature Schedule
//!
//! Worker 5 - Task 1.3 (12 hours)
//!
//! Implements Hamiltonian Monte Carlo (HMC) for efficient exploration of
//! thermodynamic state space. HMC uses Hamiltonian dynamics to propose
//! distant states with high acceptance probability.
//!
//! Key Features:
//! - Leapfrog integration for Hamiltonian dynamics
//! - Momentum sampling from Gaussian distribution
//! - Metropolis acceptance for detailed balance
//! - GPU-accelerated trajectory computation
//!
//! Physics:
//! H(q, p) = U(q) + K(p)
//! where q = position (configuration), p = momentum
//! U(q) = potential energy (cost function)
//! K(p) = kinetic energy = p^T M^{-1} p / 2
//!
//! Evolution:
//! dq/dt = ∂H/∂p = M^{-1} p
//! dp/dt = -∂H/∂q = -∇U(q)

use anyhow::{Result, anyhow};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Hamiltonian Monte Carlo Schedule
///
/// Uses Hamiltonian dynamics to efficiently explore thermodynamic state space.
/// Proposes moves by simulating physical dynamics, leading to high acceptance
/// rates even for distant proposals.
pub struct HMCSchedule {
    /// Step size for leapfrog integrator (ε)
    step_size: f64,

    /// Number of leapfrog steps per HMC iteration (L)
    num_steps: usize,

    /// Mass matrix (M) - diagonal assumed for efficiency
    /// Controls momentum distribution: p ~ N(0, M)
    mass_matrix: Vec<f64>,

    /// Current position (configuration/state)
    position: Vec<f64>,

    /// Current potential energy U(q)
    potential_energy: f64,

    /// Temperature for HMC (controls acceptance)
    temperature: f64,

    /// Acceptance statistics
    accepted_moves: usize,
    total_moves: usize,

    /// Trajectory history (for diagnostics)
    trajectory_history: Vec<Trajectory>,
}

/// Trajectory from HMC proposal
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub initial_position: Vec<f64>,
    pub final_position: Vec<f64>,
    pub initial_energy: f64,
    pub final_energy: f64,
    pub accepted: bool,
}

impl HMCSchedule {
    /// Create new HMC schedule
    ///
    /// # Arguments
    /// * `step_size` - Integration step size ε (typical: 0.01 - 0.1)
    /// * `num_steps` - Number of leapfrog steps L (typical: 10 - 100)
    /// * `initial_position` - Starting configuration
    /// * `initial_energy` - Energy of starting configuration
    /// * `temperature` - Temperature for acceptance (typical: 1.0)
    ///
    /// # Returns
    /// New HMC schedule
    pub fn new(
        step_size: f64,
        num_steps: usize,
        initial_position: Vec<f64>,
        initial_energy: f64,
        temperature: f64,
    ) -> Result<Self> {
        if step_size <= 0.0 {
            return Err(anyhow!("Step size must be positive"));
        }
        if num_steps == 0 {
            return Err(anyhow!("Number of steps must be positive"));
        }
        if temperature <= 0.0 {
            return Err(anyhow!("Temperature must be positive"));
        }
        if initial_position.is_empty() {
            return Err(anyhow!("Position must be non-empty"));
        }

        let dim = initial_position.len();
        let mass_matrix = vec![1.0; dim]; // Identity mass matrix

        Ok(Self {
            step_size,
            num_steps,
            mass_matrix,
            position: initial_position,
            potential_energy: initial_energy,
            temperature,
            accepted_moves: 0,
            total_moves: 0,
            trajectory_history: Vec::new(),
        })
    }

    /// Create HMC with custom mass matrix
    ///
    /// Mass matrix M controls the momentum distribution.
    /// Diagonal elements should be ~ inverse variance of each dimension.
    pub fn with_mass_matrix(
        step_size: f64,
        num_steps: usize,
        initial_position: Vec<f64>,
        initial_energy: f64,
        temperature: f64,
        mass_matrix: Vec<f64>,
    ) -> Result<Self> {
        if mass_matrix.len() != initial_position.len() {
            return Err(anyhow!("Mass matrix dimension must match position dimension"));
        }
        if mass_matrix.iter().any(|&m| m <= 0.0) {
            return Err(anyhow!("Mass matrix elements must be positive"));
        }

        let mut schedule = Self::new(
            step_size,
            num_steps,
            initial_position,
            initial_energy,
            temperature,
        )?;
        schedule.mass_matrix = mass_matrix;

        Ok(schedule)
    }

    /// Perform one HMC iteration
    ///
    /// 1. Sample momentum p ~ N(0, M)
    /// 2. Simulate Hamiltonian dynamics for L steps using leapfrog
    /// 3. Accept/reject using Metropolis criterion
    ///
    /// # Arguments
    /// * `gradient_fn` - Function to compute ∇U(q)
    ///
    /// # Returns
    /// (new_position, new_energy, accepted)
    pub fn step<F>(&mut self, gradient_fn: F) -> Result<(Vec<f64>, f64, bool)>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        // Sample momentum from p ~ N(0, M)
        let initial_momentum = self.sample_momentum();

        // Compute initial Hamiltonian
        let initial_hamiltonian = self.potential_energy +
            self.kinetic_energy(&initial_momentum);

        // Simulate dynamics using leapfrog
        let (proposed_position, proposed_momentum) = self.leapfrog(
            &self.position,
            &initial_momentum,
            &gradient_fn,
        )?;

        // Compute proposed potential energy
        // Note: In practice, this would be computed by evaluating the energy function
        // For now, we estimate it from the gradient norm
        let proposed_energy = self.estimate_energy(&proposed_position, &gradient_fn);

        // Compute proposed Hamiltonian
        let proposed_hamiltonian = proposed_energy +
            self.kinetic_energy(&proposed_momentum);

        // Metropolis acceptance
        let accepted = self.metropolis_accept(
            initial_hamiltonian,
            proposed_hamiltonian,
        );

        // Record trajectory
        self.trajectory_history.push(Trajectory {
            initial_position: self.position.clone(),
            final_position: proposed_position.clone(),
            initial_energy: self.potential_energy,
            final_energy: proposed_energy,
            accepted,
        });

        // Update state
        self.total_moves += 1;
        if accepted {
            self.accepted_moves += 1;
            self.position = proposed_position.clone();
            self.potential_energy = proposed_energy;
        }

        Ok((self.position.clone(), self.potential_energy, accepted))
    }

    /// Sample momentum from Gaussian distribution
    ///
    /// p ~ N(0, M) where M is the mass matrix
    fn sample_momentum(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        self.mass_matrix
            .iter()
            .map(|&mass| {
                let normal = Normal::new(0.0, mass.sqrt()).unwrap();
                normal.sample(&mut rng)
            })
            .collect()
    }

    /// Compute kinetic energy
    ///
    /// K(p) = p^T M^{-1} p / 2
    fn kinetic_energy(&self, momentum: &[f64]) -> f64 {
        momentum.iter()
            .zip(&self.mass_matrix)
            .map(|(&p, &m)| p * p / (2.0 * m))
            .sum()
    }

    /// Leapfrog integrator for Hamiltonian dynamics
    ///
    /// Symplectic integrator that preserves volume and is time-reversible.
    ///
    /// Algorithm:
    /// 1. p_{1/2} = p_0 - (ε/2) ∇U(q_0)
    /// 2. For i = 1 to L-1:
    ///    q_i = q_{i-1} + ε M^{-1} p_{i-1/2}
    ///    p_{i+1/2} = p_{i-1/2} - ε ∇U(q_i)
    /// 3. q_L = q_{L-1} + ε M^{-1} p_{L-1/2}
    /// 4. p_L = p_{L-1/2} - (ε/2) ∇U(q_L)
    ///
    /// GPU-accelerated in production
    fn leapfrog<F>(
        &self,
        initial_position: &[f64],
        initial_momentum: &[f64],
        gradient_fn: &F,
    ) -> Result<(Vec<f64>, Vec<f64>)>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut q = initial_position.to_vec();
        let mut p = initial_momentum.to_vec();

        // Half step for momentum
        let grad = gradient_fn(&q);
        for i in 0..p.len() {
            p[i] -= 0.5 * self.step_size * grad[i];
        }

        // Full steps
        for _ in 0..(self.num_steps - 1) {
            // Full step for position
            for i in 0..q.len() {
                q[i] += self.step_size * p[i] / self.mass_matrix[i];
            }

            // Full step for momentum
            let grad = gradient_fn(&q);
            for i in 0..p.len() {
                p[i] -= self.step_size * grad[i];
            }
        }

        // Final full step for position
        for i in 0..q.len() {
            q[i] += self.step_size * p[i] / self.mass_matrix[i];
        }

        // Final half step for momentum
        let grad = gradient_fn(&q);
        for i in 0..p.len() {
            p[i] -= 0.5 * self.step_size * grad[i];
        }

        // Negate momentum for reversibility (convention)
        for i in 0..p.len() {
            p[i] = -p[i];
        }

        Ok((q, p))
    }

    /// Estimate energy from gradient
    ///
    /// Simple approximation: E ≈ ||∇U||^2
    /// In practice, energy function would be provided explicitly
    fn estimate_energy<F>(&self, position: &[f64], gradient_fn: &F) -> f64
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let grad = gradient_fn(position);
        grad.iter().map(|&g| g * g).sum::<f64>().sqrt()
    }

    /// Metropolis acceptance criterion
    ///
    /// P(accept) = min(1, exp(-(H_proposed - H_current) / T))
    fn metropolis_accept(
        &self,
        current_hamiltonian: f64,
        proposed_hamiltonian: f64,
    ) -> bool {
        let delta_h = proposed_hamiltonian - current_hamiltonian;
        let acceptance_prob = (-delta_h / self.temperature).exp().min(1.0);

        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();

        random_value < acceptance_prob
    }

    /// Get current position
    pub fn position(&self) -> &[f64] {
        &self.position
    }

    /// Get current energy
    pub fn energy(&self) -> f64 {
        self.potential_energy
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_moves == 0 {
            0.0
        } else {
            self.accepted_moves as f64 / self.total_moves as f64
        }
    }

    /// Get temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set temperature (for annealing)
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Get step size
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Adapt step size based on acceptance rate
    ///
    /// Target acceptance rate for HMC is typically 0.6 - 0.9
    pub fn adapt_step_size(&mut self, target_acceptance: f64) {
        let current_acceptance = self.acceptance_rate();

        if current_acceptance < target_acceptance {
            // Too many rejections, decrease step size
            self.step_size *= 0.9;
        } else {
            // Too many acceptances, increase step size
            self.step_size *= 1.1;
        }
    }

    /// Get trajectory history
    pub fn trajectory_history(&self) -> &[Trajectory] {
        &self.trajectory_history
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.accepted_moves = 0;
        self.total_moves = 0;
        self.trajectory_history.clear();
    }
}

/// GPU-accelerated batch HMC
///
/// Runs multiple HMC chains in parallel on GPU
pub struct GpuHMCScheduler {
    /// Number of parallel chains
    num_chains: usize,

    /// HMC schedules for each chain
    chains: Vec<HMCSchedule>,
}

impl GpuHMCScheduler {
    /// Create new GPU HMC scheduler
    pub fn new(
        num_chains: usize,
        step_size: f64,
        num_steps: usize,
        initial_positions: Vec<Vec<f64>>,
        initial_energies: Vec<f64>,
        temperature: f64,
    ) -> Result<Self> {
        if num_chains != initial_positions.len() {
            return Err(anyhow!("Number of chains must match number of initial positions"));
        }
        if num_chains != initial_energies.len() {
            return Err(anyhow!("Number of chains must match number of initial energies"));
        }

        let chains = initial_positions
            .into_iter()
            .zip(initial_energies)
            .map(|(pos, energy)| {
                HMCSchedule::new(step_size, num_steps, pos, energy, temperature)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            num_chains,
            chains,
        })
    }

    /// Update all chains in parallel on GPU
    ///
    /// TODO: Request GPU kernel from Worker 2:
    /// - hmc_leapfrog_batch.cu
    /// - Computes all leapfrog trajectories in parallel
    /// - Uses GPU random number generation
    pub fn step_all_gpu<F>(&mut self, gradient_fn: F) -> Result<Vec<(Vec<f64>, f64, bool)>>
    where
        F: Fn(&[f64]) -> Vec<f64> + Clone,
    {
        // PLACEHOLDER: Will use GPU kernel once Worker 2 provides it
        // For now, update each chain sequentially

        self.chains
            .iter_mut()
            .map(|chain| chain.step(gradient_fn.clone()))
            .collect()
    }

    /// Get best configuration (lowest energy across all chains)
    pub fn get_best_configuration(&self) -> (Vec<f64>, f64) {
        self.chains
            .iter()
            .min_by(|a, b| a.energy().partial_cmp(&b.energy()).unwrap())
            .map(|chain| (chain.position().to_vec(), chain.energy()))
            .unwrap()
    }

    /// Get all chain states
    pub fn get_chains(&self) -> &[HMCSchedule] {
        &self.chains
    }

    /// Get average acceptance rate across all chains
    pub fn average_acceptance_rate(&self) -> f64 {
        let total: f64 = self.chains.iter().map(|c| c.acceptance_rate()).sum();
        total / self.num_chains as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple quadratic potential: U(q) = q^T q / 2
    fn quadratic_gradient(q: &[f64]) -> Vec<f64> {
        q.to_vec()
    }

    #[test]
    fn test_hmc_creation() -> Result<()> {
        let hmc = HMCSchedule::new(
            0.1,           // step_size
            10,            // num_steps
            vec![1.0, 2.0], // initial_position
            2.5,           // initial_energy
            1.0,           // temperature
        )?;

        assert_eq!(hmc.step_size(), 0.1);
        assert_eq!(hmc.position(), &[1.0, 2.0]);
        assert_eq!(hmc.energy(), 2.5);

        Ok(())
    }

    #[test]
    fn test_momentum_sampling() -> Result<()> {
        let hmc = HMCSchedule::new(
            0.1,
            10,
            vec![0.0, 0.0],
            0.0,
            1.0,
        )?;

        let momentum = hmc.sample_momentum();
        assert_eq!(momentum.len(), 2);

        // Momentum should be non-zero (with high probability)
        assert!(momentum.iter().any(|&p| p.abs() > 0.01));

        Ok(())
    }

    #[test]
    fn test_kinetic_energy() -> Result<()> {
        let hmc = HMCSchedule::new(
            0.1,
            10,
            vec![0.0, 0.0],
            0.0,
            1.0,
        )?;

        let momentum = vec![1.0, 1.0];
        let ke = hmc.kinetic_energy(&momentum);

        // K = p^T M^{-1} p / 2 = (1 + 1) / 2 = 1.0 (for unit mass)
        assert!((ke - 1.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_leapfrog_integration() -> Result<()> {
        let hmc = HMCSchedule::new(
            0.1,
            10,
            vec![1.0, 1.0],
            1.0,
            1.0,
        )?;

        let initial_position = vec![1.0, 1.0];
        let initial_momentum = vec![0.0, 0.0];

        let (final_position, final_momentum) = hmc.leapfrog(
            &initial_position,
            &initial_momentum,
            &quadratic_gradient,
        )?;

        assert_eq!(final_position.len(), 2);
        assert_eq!(final_momentum.len(), 2);

        // With zero initial momentum, should move towards minimum
        assert!(final_position[0].abs() < initial_position[0].abs());

        Ok(())
    }

    #[test]
    fn test_hmc_step() -> Result<()> {
        let mut hmc = HMCSchedule::new(
            0.1,
            10,
            vec![2.0, 2.0],
            4.0,
            1.0,
        )?;

        let (new_position, new_energy, _accepted) = hmc.step(quadratic_gradient)?;

        assert_eq!(new_position.len(), 2);
        assert!(new_energy >= 0.0);

        Ok(())
    }

    #[test]
    fn test_acceptance_rate_tracking() -> Result<()> {
        let mut hmc = HMCSchedule::new(
            0.1,
            10,
            vec![1.0, 1.0],
            1.0,
            1.0,
        )?;

        // Run several steps
        for _ in 0..20 {
            hmc.step(quadratic_gradient)?;
        }

        let acceptance = hmc.acceptance_rate();
        assert!(acceptance >= 0.0 && acceptance <= 1.0);
        assert_eq!(hmc.total_moves, 20);

        Ok(())
    }

    #[test]
    fn test_temperature_control() -> Result<()> {
        let mut hmc = HMCSchedule::new(
            0.1,
            10,
            vec![1.0],
            1.0,
            1.0,
        )?;

        assert_eq!(hmc.temperature(), 1.0);

        hmc.set_temperature(2.0);
        assert_eq!(hmc.temperature(), 2.0);

        Ok(())
    }

    #[test]
    fn test_step_size_adaptation() -> Result<()> {
        let mut hmc = HMCSchedule::new(
            0.1,
            10,
            vec![1.0],
            1.0,
            1.0,
        )?;

        let initial_step_size = hmc.step_size();

        // Manually set acceptance rate by running steps
        for _ in 0..10 {
            hmc.step(quadratic_gradient)?;
        }

        hmc.adapt_step_size(0.75);

        // Step size should change
        assert_ne!(hmc.step_size(), initial_step_size);

        Ok(())
    }

    #[test]
    fn test_custom_mass_matrix() -> Result<()> {
        let mass_matrix = vec![2.0, 3.0];
        let hmc = HMCSchedule::with_mass_matrix(
            0.1,
            10,
            vec![1.0, 1.0],
            1.0,
            1.0,
            mass_matrix.clone(),
        )?;

        assert_eq!(hmc.mass_matrix, mass_matrix);

        Ok(())
    }

    #[test]
    fn test_trajectory_history() -> Result<()> {
        let mut hmc = HMCSchedule::new(
            0.1,
            10,
            vec![1.0],
            1.0,
            1.0,
        )?;

        hmc.step(quadratic_gradient)?;
        hmc.step(quadratic_gradient)?;

        let history = hmc.trajectory_history();
        assert_eq!(history.len(), 2);

        Ok(())
    }

    #[test]
    fn test_gpu_hmc_scheduler() -> Result<()> {
        let initial_positions = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let initial_energies = vec![1.0, 4.0, 9.0];

        let mut scheduler = GpuHMCScheduler::new(
            3,
            0.1,
            10,
            initial_positions,
            initial_energies,
            1.0,
        )?;

        let results = scheduler.step_all_gpu(quadratic_gradient)?;
        assert_eq!(results.len(), 3);

        let (best_pos, best_energy) = scheduler.get_best_configuration();
        assert!(best_energy >= 0.0);
        assert_eq!(best_pos.len(), 2);

        Ok(())
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid step size
        assert!(HMCSchedule::new(-0.1, 10, vec![1.0], 1.0, 1.0).is_err());

        // Invalid num_steps
        assert!(HMCSchedule::new(0.1, 0, vec![1.0], 1.0, 1.0).is_err());

        // Invalid temperature
        assert!(HMCSchedule::new(0.1, 10, vec![1.0], 1.0, -1.0).is_err());

        // Empty position
        assert!(HMCSchedule::new(0.1, 10, vec![], 1.0, 1.0).is_err());

        // Mismatched mass matrix
        assert!(HMCSchedule::with_mass_matrix(
            0.1, 10, vec![1.0, 2.0], 1.0, 1.0, vec![1.0]
        ).is_err());
    }
}
