//! GPU Kernel Wrappers for Advanced Thermodynamic Schedules
//!
//! This module provides high-level Rust wrappers around CUDA kernels
//! implemented by Worker 2 for thermodynamic optimization.
//!
//! Worker 5 Enhancement: Week 2, Task 2.3
//!
//! Design Philosophy:
//! - Worker 2 owns the CUDA kernel implementations (.cu files)
//! - Worker 5 owns the high-level Rust API wrappers (this file)
//! - Minimize CPU↔GPU transfers through batching
//! - Persistent GPU data where possible
//! - Graceful error handling with detailed context

use anyhow::{Context, Result};
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig};
use crate::gpu::GpuKernelExecutor;

/// GPU-accelerated Boltzmann probability computation
///
/// Computes softmax probabilities from energy values using Boltzmann distribution:
/// P_i = exp(-E_i / kT) / sum_j(exp(-E_j / kT))
///
/// Used by: Simulated Annealing, HMC, Bayesian Optimization
pub struct BoltzmannKernel {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

impl BoltzmannKernel {
    pub fn new(context: Arc<CudaContext>, executor: Arc<std::sync::Mutex<GpuKernelExecutor>>) -> Self {
        Self { context, executor }
    }

    /// Compute Boltzmann probabilities on GPU
    ///
    /// # Arguments
    /// * `energies` - Energy values (on CPU, will be uploaded)
    /// * `temperature` - Temperature parameter (kT)
    ///
    /// # Returns
    /// Normalized probability distribution
    ///
    /// # Performance
    /// Target: <0.1ms for 1000 energies
    pub fn compute_probabilities(&self, energies: &[f32], temperature: f32) -> Result<Vec<f32>> {
        let stream = self.context.default_stream();
        let exec = self.executor.lock().unwrap();

        // Upload energies to GPU
        let energies_gpu = stream.memcpy_stod(energies)
            .context("Failed to upload energies to GPU")?;

        // Compute -E/kT on GPU
        let neg_e_kt: Vec<f32> = energies.iter().map(|&e| -e / temperature).collect();
        let neg_e_kt_gpu = stream.memcpy_stod(&neg_e_kt)
            .context("Failed to upload neg_e_kt to GPU")?;

        // Allocate output
        let mut probs_gpu = stream.alloc_zeros::<f32>(energies.len())
            .context("Failed to allocate GPU memory for probabilities")?;

        // Call fused exp+normalize kernel (Worker 2 implementation)
        let kernel = exec.get_kernel("fused_exp_normalize")
            .context("Boltzmann kernel not registered - Worker 2 needs to implement")?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&neg_e_kt_gpu)
                .arg(&mut probs_gpu)
                .arg(&(energies.len() as i32))
                .launch(cfg)
                .context("Failed to launch Boltzmann kernel")?;
        }

        // Download results
        let probs = stream.memcpy_dtov(&probs_gpu)
            .context("Failed to download probabilities from GPU")?;

        drop(exec);
        Ok(probs)
    }

    /// Batch compute Boltzmann probabilities for multiple energy arrays
    ///
    /// # Performance
    /// Target: <1ms for 100 arrays of 100 energies each
    pub fn compute_probabilities_batch(
        &self,
        energy_batches: &[Vec<f32>],
        temperatures: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        // TODO: Implement once Worker 2 provides batch kernel
        // For now, process sequentially
        energy_batches.iter()
            .zip(temperatures.iter())
            .map(|(energies, &temp)| self.compute_probabilities(energies, temp))
            .collect()
    }
}

/// GPU-accelerated replica exchange swap acceptance
///
/// Computes Metropolis acceptance probabilities for replica pair swaps:
/// P_accept = min(1, exp((beta_i - beta_j) * (E_j - E_i)))
///
/// Used by: Parallel Tempering, Replica Exchange
pub struct ReplicaSwapKernel {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

impl ReplicaSwapKernel {
    pub fn new(context: Arc<CudaContext>, executor: Arc<std::sync::Mutex<GpuKernelExecutor>>) -> Self {
        Self { context, executor }
    }

    /// Compute swap acceptance probabilities for replica pairs
    ///
    /// # Arguments
    /// * `energies` - Energy of each replica
    /// * `temperatures` - Temperature of each replica
    /// * `swap_pairs` - Pairs to attempt swapping (i, j indices)
    ///
    /// # Returns
    /// Acceptance probability for each swap pair
    ///
    /// # Performance
    /// Target: <0.5ms for 100 replicas
    pub fn compute_swap_acceptance(
        &self,
        energies: &[f32],
        temperatures: &[f32],
        swap_pairs: &[(usize, usize)],
    ) -> Result<Vec<f32>> {
        // TODO: Implement once Worker 2 provides replica_swap_acceptance kernel
        // For now, compute on CPU as fallback
        let acceptance_probs: Vec<f32> = swap_pairs.iter().map(|&(i, j)| {
            let beta_i = 1.0 / temperatures[i];
            let beta_j = 1.0 / temperatures[j];
            let delta_beta = beta_i - beta_j;
            let delta_energy = energies[j] - energies[i];
            let log_prob = delta_beta * delta_energy;

            if log_prob >= 0.0 {
                1.0
            } else {
                log_prob.exp()
            }
        }).collect();

        Ok(acceptance_probs)
    }
}

/// GPU-accelerated Hamiltonian leapfrog integration
///
/// Performs symplectic integration for HMC trajectory generation
///
/// Used by: Hamiltonian Monte Carlo
pub struct LeapfrogKernel {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

impl LeapfrogKernel {
    pub fn new(context: Arc<CudaContext>, executor: Arc<std::sync::Mutex<GpuKernelExecutor>>) -> Self {
        Self { context, executor }
    }

    /// Integrate Hamiltonian dynamics using leapfrog method
    ///
    /// # Arguments
    /// * `position` - Initial position
    /// * `momentum` - Initial momentum
    /// * `gradient_fn` - Function to compute gradient of potential energy
    /// * `mass_matrix` - Mass matrix (diagonal)
    /// * `step_size` - Integration step size
    /// * `num_steps` - Number of leapfrog steps
    ///
    /// # Returns
    /// (final_position, final_momentum)
    ///
    /// # Performance
    /// Target: <1ms for 50 steps in 100-D space
    pub fn integrate<F>(
        &self,
        position: &[f32],
        momentum: &[f32],
        gradient_fn: F,
        mass_matrix: &[f32],
        step_size: f32,
        num_steps: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        // TODO: Implement once Worker 2 provides leapfrog_integrate kernel
        // For now, use CPU implementation
        let mut q = position.to_vec();
        let mut p = momentum.to_vec();

        for _ in 0..num_steps {
            // Half step for momentum
            let grad = gradient_fn(&q);
            for i in 0..p.len() {
                p[i] += 0.5 * step_size * grad[i];
            }

            // Full step for position
            for i in 0..q.len() {
                q[i] += step_size * p[i] / mass_matrix[i];
            }

            // Half step for momentum
            let grad = gradient_fn(&q);
            for i in 0..p.len() {
                p[i] += 0.5 * step_size * grad[i];
            }
        }

        Ok((q, p))
    }
}

/// GPU-accelerated Gaussian Process covariance computation
///
/// Computes covariance matrices for GP surrogate models
///
/// Used by: Bayesian Optimization
pub struct GaussianProcessKernel {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

#[derive(Debug, Clone, Copy)]
pub enum GPKernelType {
    SquaredExponential,
    Matern52,
    RationalQuadratic,
}

impl GaussianProcessKernel {
    pub fn new(context: Arc<CudaContext>, executor: Arc<std::sync::Mutex<GpuKernelExecutor>>) -> Self {
        Self { context, executor }
    }

    /// Compute covariance matrix between two sets of points
    ///
    /// # Arguments
    /// * `X1` - First set of points (n1 x d)
    /// * `X2` - Second set of points (n2 x d)
    /// * `kernel_type` - Type of kernel function
    /// * `kernel_params` - Kernel hyperparameters [length_scale, signal_variance, ...]
    ///
    /// # Returns
    /// Covariance matrix (n1 x n2)
    ///
    /// # Performance
    /// Target: <2ms for 100x100 matrix
    pub fn compute_covariance(
        &self,
        X1: &[Vec<f32>],
        X2: &[Vec<f32>],
        kernel_type: GPKernelType,
        kernel_params: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        // TODO: Implement once Worker 2 provides gp_covariance kernel
        // For now, use CPU implementation
        let n1 = X1.len();
        let n2 = X2.len();
        let mut cov = vec![vec![0.0; n2]; n1];

        let length_scale = kernel_params[0];
        let signal_variance = kernel_params[1];

        for i in 0..n1 {
            for j in 0..n2 {
                let dist_sq: f32 = X1[i].iter()
                    .zip(X2[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                cov[i][j] = match kernel_type {
                    GPKernelType::SquaredExponential => {
                        signal_variance * (-dist_sq / (2.0 * length_scale * length_scale)).exp()
                    },
                    GPKernelType::Matern52 => {
                        let r = dist_sq.sqrt();
                        let sqrt5 = 5.0_f32.sqrt();
                        signal_variance * (1.0 + sqrt5 * r / length_scale + 5.0 * r * r / (3.0 * length_scale * length_scale))
                            * (-sqrt5 * r / length_scale).exp()
                    },
                    GPKernelType::RationalQuadratic => {
                        let alpha = if kernel_params.len() > 2 { kernel_params[2] } else { 1.0 };
                        signal_variance * (1.0 + dist_sq / (2.0 * alpha * length_scale * length_scale)).powf(-alpha)
                    },
                };
            }
        }

        Ok(cov)
    }
}

/// GPU-accelerated Pareto dominance checking
///
/// Checks dominance relationships for multi-objective optimization
///
/// Used by: Multi-Objective Schedule
pub struct ParetoDominanceKernel {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

impl ParetoDominanceKernel {
    pub fn new(context: Arc<CudaContext>, executor: Arc<std::sync::Mutex<GpuKernelExecutor>>) -> Self {
        Self { context, executor }
    }

    /// Check dominance relationships between solutions
    ///
    /// # Arguments
    /// * `objectives` - Matrix of objective values (n_solutions x n_objectives)
    ///
    /// # Returns
    /// Dominance matrix where [i][j] = 1 if i dominates j, -1 if j dominates i, 0 if non-dominated
    ///
    /// # Performance
    /// Target: <1ms for 100 solutions with 5 objectives
    pub fn compute_dominance(&self, objectives: &[Vec<f32>]) -> Result<Vec<Vec<i32>>> {
        // TODO: Implement once Worker 2 provides pareto_dominance kernel
        // For now, use CPU implementation
        let n = objectives.len();
        let mut dominance = vec![vec![0; n]; n];

        for i in 0..n {
            for j in (i+1)..n {
                let mut i_dominates = false;
                let mut j_dominates = false;

                for k in 0..objectives[i].len() {
                    if objectives[i][k] < objectives[j][k] {
                        i_dominates = true;
                    } else if objectives[i][k] > objectives[j][k] {
                        j_dominates = true;
                    }
                }

                if i_dominates && !j_dominates {
                    dominance[i][j] = 1;
                    dominance[j][i] = -1;
                } else if j_dominates && !i_dominates {
                    dominance[i][j] = -1;
                    dominance[j][i] = 1;
                }
            }
        }

        Ok(dominance)
    }
}

/// GPU-accelerated batch temperature updates
///
/// Updates temperatures for multiple schedules in parallel
///
/// Used by: All schedule types for batch processing
pub struct BatchTemperatureKernel {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

#[derive(Debug, Clone, Copy)]
pub enum CoolingStrategy {
    Exponential,
    Logarithmic,
    Adaptive,
}

impl BatchTemperatureKernel {
    pub fn new(context: Arc<CudaContext>, executor: Arc<std::sync::Mutex<GpuKernelExecutor>>) -> Self {
        Self { context, executor }
    }

    /// Update temperatures for multiple schedules in parallel
    ///
    /// # Arguments
    /// * `temperatures` - Current temperatures (modified in place)
    /// * `cooling_rates` - Cooling rate for each schedule
    /// * `cooling_strategies` - Cooling strategy for each schedule
    /// * `acceptance_rates` - Current acceptance rates (for adaptive cooling)
    /// * `iteration` - Current iteration number
    ///
    /// # Performance
    /// Target: <0.1ms for 1000 schedules
    pub fn update_batch(
        &self,
        temperatures: &mut [f32],
        cooling_rates: &[f32],
        cooling_strategies: &[CoolingStrategy],
        acceptance_rates: &[f32],
        iteration: usize,
    ) -> Result<()> {
        // TODO: Implement once Worker 2 provides batch_temperature_update kernel
        // For now, use CPU implementation
        for i in 0..temperatures.len() {
            temperatures[i] = match cooling_strategies[i] {
                CoolingStrategy::Exponential => {
                    temperatures[i] * cooling_rates[i]
                },
                CoolingStrategy::Logarithmic => {
                    temperatures[i] / (1.0 + iteration as f32).ln()
                },
                CoolingStrategy::Adaptive => {
                    let target_acceptance = 0.44; // Optimal for many problems
                    let adjustment = 0.1 * (target_acceptance - acceptance_rates[i]);
                    temperatures[i] * (1.0 + adjustment)
                },
            };
        }

        Ok(())
    }
}

/// Factory for creating GPU kernel wrappers
///
/// Centralizes GPU context and executor management
pub struct GpuScheduleKernels {
    context: Arc<CudaContext>,
    executor: Arc<std::sync::Mutex<GpuKernelExecutor>>,
}

impl GpuScheduleKernels {
    /// Create kernel factory with shared GPU context
    pub fn new() -> Result<Self> {
        let context = Arc::new(CudaContext::new(0).context("Failed to create CUDA context")?);
        let mut executor = GpuKernelExecutor::new(0).context("Failed to create GPU executor")?;

        // Register standard kernels
        executor.register_standard_kernels().context("Failed to register standard kernels")?;

        // TODO: Register thermodynamic kernels once Worker 2 implements them
        // executor.register_thermodynamic_kernels()?;

        let executor = Arc::new(std::sync::Mutex::new(executor));

        Ok(Self { context, executor })
    }

    /// Create Boltzmann probability kernel
    pub fn boltzmann(&self) -> BoltzmannKernel {
        BoltzmannKernel::new(self.context.clone(), self.executor.clone())
    }

    /// Create replica swap kernel
    pub fn replica_swap(&self) -> ReplicaSwapKernel {
        ReplicaSwapKernel::new(self.context.clone(), self.executor.clone())
    }

    /// Create leapfrog integration kernel
    pub fn leapfrog(&self) -> LeapfrogKernel {
        LeapfrogKernel::new(self.context.clone(), self.executor.clone())
    }

    /// Create Gaussian process kernel
    pub fn gaussian_process(&self) -> GaussianProcessKernel {
        GaussianProcessKernel::new(self.context.clone(), self.executor.clone())
    }

    /// Create Pareto dominance kernel
    pub fn pareto_dominance(&self) -> ParetoDominanceKernel {
        ParetoDominanceKernel::new(self.context.clone(), self.executor.clone())
    }

    /// Create batch temperature update kernel
    pub fn batch_temperature(&self) -> BatchTemperatureKernel {
        BatchTemperatureKernel::new(self.context.clone(), self.executor.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzmann_kernel_cpu_fallback() -> Result<()> {
        let kernels = GpuScheduleKernels::new()?;
        let boltzmann = kernels.boltzmann();

        let energies = vec![1.0, 2.0, 3.0];
        let temperature = 1.0;

        let probs = boltzmann.compute_probabilities(&energies, temperature)?;

        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check probabilities are in descending order (lower energy = higher prob)
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);

        Ok(())
    }

    #[test]
    fn test_replica_swap_kernel() -> Result<()> {
        let kernels = GpuScheduleKernels::new()?;
        let swap = kernels.replica_swap();

        let energies = vec![1.0, 2.0, 3.0, 4.0];
        let temperatures = vec![1.0, 1.5, 2.0, 2.5];
        let swap_pairs = vec![(0, 1), (1, 2), (2, 3)];

        let acceptance = swap.compute_swap_acceptance(&energies, &temperatures, &swap_pairs)?;

        // Check all probabilities are in [0, 1]
        for &prob in &acceptance {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_gaussian_process_kernel() -> Result<()> {
        let kernels = GpuScheduleKernels::new()?;
        let gp = kernels.gaussian_process();

        let X1 = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let X2 = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let params = vec![1.0, 1.0]; // length_scale, signal_variance

        let cov = gp.compute_covariance(&X1, &X2, GPKernelType::SquaredExponential, &params)?;

        // Check diagonal elements are 1.0 (same point)
        assert!((cov[0][0] - 1.0).abs() < 1e-5);
        assert!((cov[1][1] - 1.0).abs() < 1e-5);

        // Check symmetry
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_pareto_dominance_kernel() -> Result<()> {
        let kernels = GpuScheduleKernels::new()?;
        let pareto = kernels.pareto_dominance();

        // Solution 0 dominates solution 1 (better in both objectives)
        let objectives = vec![
            vec![1.0, 1.0],  // Solution 0: low cost, low latency (BEST)
            vec![2.0, 2.0],  // Solution 1: high cost, high latency
            vec![1.5, 1.5],  // Solution 2: middle
        ];

        let dominance = pareto.compute_dominance(&objectives)?;

        // Solution 0 should dominate both others
        assert_eq!(dominance[0][1], 1);
        assert_eq!(dominance[0][2], 1);

        // Solution 1 should be dominated by both others
        assert_eq!(dominance[1][0], -1);
        assert_eq!(dominance[1][2], -1);

        Ok(())
    }

    #[test]
    fn test_batch_temperature_kernel() -> Result<()> {
        let kernels = GpuScheduleKernels::new()?;
        let batch_temp = kernels.batch_temperature();

        let mut temperatures = vec![10.0, 10.0, 10.0];
        let cooling_rates = vec![0.95, 0.9, 0.85];
        let strategies = vec![
            CoolingStrategy::Exponential,
            CoolingStrategy::Exponential,
            CoolingStrategy::Exponential,
        ];
        let acceptance_rates = vec![0.5, 0.5, 0.5];

        batch_temp.update_batch(&mut temperatures, &cooling_rates, &strategies, &acceptance_rates, 1)?;

        // Check temperatures decreased
        assert!(temperatures[0] < 10.0);
        assert!(temperatures[1] < 10.0);
        assert!(temperatures[2] < 10.0);

        // Check different cooling rates applied correctly
        assert!(temperatures[0] > temperatures[1]); // 0.95 > 0.9
        assert!(temperatures[1] > temperatures[2]); // 0.9 > 0.85

        Ok(())
    }
}
