//! Physics Coupling Adapter - GPU Accelerated
//!
//! Wraps PhysicsCoupling from platform-foundation (commit 35963c6) to implement PhysicsCouplingPort.
//! GPU-accelerated Kuramoto synchronization and transfer entropy.

use prct_core::ports::PhysicsCouplingPort;
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use platform_foundation::PhysicsCoupling;
use num_complex::Complex64;
use nalgebra::DMatrix;
use cudarc::driver::{CudaContext, LaunchConfig, CudaModule, PushKernelArg};
use std::sync::Arc;

/// Adapter connecting PRCT domain to GPU-accelerated physics coupling service
pub struct CouplingAdapter {
    gpu_device: Option<Arc<CudaContext>>,
    gpu_module: Option<Arc<CudaModule>>,
    use_gpu: bool,
}

impl CouplingAdapter {
    /// Create new GPU-accelerated coupling adapter
    pub fn new() -> Self {
        // Try to initialize GPU
        let (gpu_device, gpu_module, use_gpu) = match CudaContext::new(0) {
            Ok(device_arc) => {
                // cudarc 0.17 returns Arc<CudaContext> directly
                // Try to load GPU kernels
                match Self::load_gpu_module(&device_arc) {
                    Ok(module) => {
                        println!("✓ Coupling GPU initialized (CUDA device 0)");
                        (Some(device_arc), Some(module), true)
                    }
                    Err(e) => {
                        eprintln!("⚠ GPU kernel load failed: {}. Using CPU fallback.", e);
                        (None, None, false)
                    }
                }
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU fallback.", e);
                (None, None, false)
            }
        };

        Self {
            gpu_device,
            gpu_module,
            use_gpu,
        }
    }

    /// Load GPU module for coupling operations
    fn load_gpu_module(device: &Arc<CudaContext>) -> std::result::Result<Arc<CudaModule>, String> {
        // Load PTX from runtime location
        let ptx_path = "target/ptx/coupling_kernels.ptx";
        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| format!("Failed to load PTX: {}", e))?;

        // Load module using cudarc 0.17 API (returns Arc<CudaModule>)
        let module = device.load_module(ptx.into())
            .map_err(|e| format!("PTX load failed: {}", e))?;

        Ok(module)
    }

    /// GPU-accelerated Kuramoto synchronization step
    fn kuramoto_step_gpu(
        &self,
        phases: &[f64],
        natural_frequencies: &[f64],
        coupling_strength: f64,
        dt: f64,
    ) -> Result<Vec<f64>> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU not initialized".into()))?;

        let n = phases.len();

        // Build coupling matrix (all-to-all with uniform strength)
        let coupling_matrix = vec![coupling_strength; n * n];

        // Convert to f32 for GPU
        let phases_f32: Vec<f32> = phases.iter().map(|&x| x as f32).collect();
        let frequencies_f32: Vec<f32> = natural_frequencies.iter().map(|&x| x as f32).collect();
        let coupling_f32: Vec<f32> = coupling_matrix.iter().map(|&x| x as f32).collect();

        // Allocate GPU memory using stream-based API
        let stream = device.default_stream();
        let gpu_phases = stream.memcpy_stod(&phases_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_frequencies = stream.memcpy_stod(&frequencies_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_coupling = stream.memcpy_stod(&coupling_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_new_phases = stream.alloc_zeros::<f32>(n)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU alloc failed: {}", e)))?;

        // Launch Kuramoto step kernel
        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get module and function
        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU module not loaded".into()))?;
        let func = module.load_function("kuramoto_step")
            .map_err(|e| PRCTError::CouplingFailed(format!("Kernel not found: {}", e)))?;

        let n_u32 = n as u32;
        let coupling_strength_f32 = coupling_strength as f32;
        let dt_f32 = dt as f32;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&gpu_phases);
        launch_args.arg(&gpu_frequencies);
        launch_args.arg(&gpu_coupling);
        launch_args.arg(&mut gpu_new_phases);
        launch_args.arg(&n_u32);
        launch_args.arg(&coupling_strength_f32);
        launch_args.arg(&dt_f32);

        unsafe {
            launch_args.launch(cfg).map_err(|e| PRCTError::CouplingFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy results back using stream
        let new_phases_f32: Vec<f32> = stream.memcpy_dtov(&gpu_new_phases)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy back failed: {}", e)))?;

        Ok(new_phases_f32.iter().map(|&x| x as f64).collect())
    }

    /// GPU-accelerated order parameter computation
    fn compute_order_parameter_gpu(&self, phases: &[f64]) -> Result<f64> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU not initialized".into()))?;

        let n = phases.len();
        let phases_f32: Vec<f32> = phases.iter().map(|&x| x as f32).collect();

        let stream = device.default_stream();
        let gpu_phases = stream.memcpy_stod(&phases_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_order_param = stream.alloc_zeros::<f32>(1)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_mean_phase = stream.alloc_zeros::<f32>(1)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU alloc failed: {}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU module not loaded".into()))?;
        let func = module.load_function("kuramoto_order_parameter")
            .map_err(|e| PRCTError::CouplingFailed(format!("Kernel not found: {}", e)))?;

        let n_u32 = n as u32;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&gpu_phases);
        launch_args.arg(&mut gpu_order_param);
        launch_args.arg(&mut gpu_mean_phase);
        launch_args.arg(&n_u32);

        unsafe {
            launch_args.launch(cfg).map_err(|e| PRCTError::CouplingFailed(format!("Kernel launch failed: {}", e)))?;
        }

        let order_vec: Vec<f32> = stream.memcpy_dtov(&gpu_order_param)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy back failed: {}", e)))?;

        Ok(order_vec[0] as f64)
    }

    /// GPU-accelerated transfer entropy calculation
    fn calculate_transfer_entropy_gpu(
        &self,
        source: &[f64],
        target: &[f64],
        lag: usize,
    ) -> Result<f64> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU not initialized".into()))?;

        let n = source.len().min(target.len());
        let source_f32: Vec<f32> = source[..n].iter().map(|&x| x as f32).collect();
        let target_f32: Vec<f32> = target[..n].iter().map(|&x| x as f32).collect();

        let stream = device.default_stream();
        let gpu_source = stream.memcpy_stod(&source_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_target = stream.memcpy_stod(&target_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_te = stream.alloc_zeros::<f32>(1)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU alloc failed: {}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU module not loaded".into()))?;
        let func = module.load_function("transfer_entropy")
            .map_err(|e| PRCTError::CouplingFailed(format!("Kernel not found: {}", e)))?;

        let n_u32 = n as u32;
        let lag_u32 = lag as u32;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&gpu_source);
        launch_args.arg(&gpu_target);
        launch_args.arg(&mut gpu_te);
        launch_args.arg(&n_u32);
        launch_args.arg(&lag_u32);

        unsafe {
            launch_args.launch(cfg).map_err(|e| PRCTError::CouplingFailed(format!("Kernel launch failed: {}", e)))?;
        }

        let te_vec: Vec<f32> = stream.memcpy_dtov(&gpu_te)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy back failed: {}", e)))?;

        Ok(te_vec[0] as f64)
    }

    /// GPU-accelerated coupling strength computation
    fn compute_coupling_strength_gpu(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
    ) -> Result<f64> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU not initialized".into()))?;

        let neuro_f32: Vec<f32> = neuro_phases.iter().map(|&x| x as f32).collect();
        let quantum_f32: Vec<f32> = quantum_phases.iter().map(|&x| x as f32).collect();

        let stream = device.default_stream();
        let gpu_neuro = stream.memcpy_stod(&neuro_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_quantum = stream.memcpy_stod(&quantum_f32)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_strength = stream.alloc_zeros::<f32>(1)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU alloc failed: {}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (((neuro_phases.len() + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::CouplingFailed("GPU module not loaded".into()))?;
        let func = module.load_function("compute_coupling_strength")
            .map_err(|e| PRCTError::CouplingFailed(format!("Kernel not found: {}", e)))?;

        let neuro_len_u32 = neuro_phases.len() as u32;
        let quantum_len_u32 = quantum_phases.len() as u32;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&gpu_neuro);
        launch_args.arg(&gpu_quantum);
        launch_args.arg(&mut gpu_strength);
        launch_args.arg(&neuro_len_u32);
        launch_args.arg(&quantum_len_u32);

        unsafe {
            launch_args.launch(cfg).map_err(|e| PRCTError::CouplingFailed(format!("Kernel launch failed: {}", e)))?;
        }

        let strength_vec: Vec<f32> = stream.memcpy_dtov(&gpu_strength)
            .map_err(|e| PRCTError::CouplingFailed(format!("GPU copy back failed: {}", e)))?;

        Ok(strength_vec[0] as f64)
    }
}

impl PhysicsCouplingPort for CouplingAdapter {
    fn compute_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<CouplingStrength> {
        // Build coupling matrix from quantum state
        let n = quantum_state.amplitudes.len();
        let mut coupling_matrix = DMatrix::from_element(n, n, Complex64::new(0.0, 0.0));

        for i in 0..n {
            for j in 0..n {
                let phase_i = Complex64::new(quantum_state.amplitudes[i].0, quantum_state.amplitudes[i].1).arg();
                let phase_j = Complex64::new(quantum_state.amplitudes[j].0, quantum_state.amplitudes[j].1).arg();
                let phase_diff = (phase_i - phase_j).cos();
                coupling_matrix[(i, j)] = Complex64::new(phase_diff, 0.0);
            }
        }

        // Use PhysicsCoupling from commit 35963c6
        let quantum_vec: Vec<Complex64> = quantum_state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();

        let physics = PhysicsCoupling::from_system_state(
            &neuro_state.neuron_states,
            &neuro_state.spike_pattern.iter().map(|&s| s as f64).collect::<Vec<_>>(),
            &quantum_vec,
            &coupling_matrix,
        ).map_err(|e| PRCTError::CouplingFailed(e.to_string()))?;

        Ok(CouplingStrength {
            neuro_to_quantum: physics.neuro_to_quantum.pattern_to_hamiltonian,
            quantum_to_neuro: physics.quantum_to_neuro.energy_to_learning_rate,
            bidirectional_coherence: physics.phase_sync.order_parameter,
            timestamp_ns: 0,
        })
    }

    fn update_kuramoto_sync(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
        dt: f64,
    ) -> Result<KuramotoState> {
        // Combine phases from both subsystems
        let mut phases = neuro_phases.to_vec();
        phases.extend_from_slice(quantum_phases);

        let n = phases.len();
        let natural_frequencies = vec![1.0; n];
        let coupling_strength = 0.5;

        // GPU path: Kuramoto on GPU
        if self.use_gpu {
            let new_phases = self.kuramoto_step_gpu(&phases, &natural_frequencies, coupling_strength, dt)?;
            let order_parameter = self.compute_order_parameter_gpu(&new_phases)?;
            let mean_phase = new_phases.iter().sum::<f64>() / n as f64;

            return Ok(KuramotoState {
                phases: new_phases,
                natural_frequencies,
                coupling_matrix: vec![coupling_strength; n * n],
                order_parameter,
                mean_phase,
            });
        }

        // CPU fallback
        let new_phases: Vec<f64> = (0..n).map(|i| {
            let mut coupling_sum = 0.0;
            for j in 0..n {
                if i != j {
                    coupling_sum += (phases[j] - phases[i]).sin();
                }
            }
            let dphase = natural_frequencies[i] + (coupling_strength / n as f64) * coupling_sum;
            (phases[i] + dphase * dt) % (2.0 * core::f64::consts::PI)
        }).collect();

        let sum_real: f64 = new_phases.iter().map(|p| p.cos()).sum();
        let sum_imag: f64 = new_phases.iter().map(|p| p.sin()).sum();
        let order_parameter = ((sum_real / n as f64).powi(2) + (sum_imag / n as f64).powi(2)).sqrt();

        Ok(KuramotoState {
            phases: new_phases.clone(),
            natural_frequencies,
            coupling_matrix: vec![coupling_strength; n * n],
            order_parameter,
            mean_phase: new_phases.iter().sum::<f64>() / n as f64,
        })
    }

    fn calculate_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        lag: f64,
    ) -> Result<TransferEntropy> {
        let n = source.len().min(target.len());

        if n < 2 {
            return Ok(TransferEntropy {
                entropy_bits: 0.0,
                confidence: 0.0,
                lag_ms: lag,
            });
        }

        // GPU path: Transfer entropy on GPU
        if self.use_gpu {
            let lag_samples = (lag / 10.0).max(1.0) as usize;
            let te = self.calculate_transfer_entropy_gpu(source, target, lag_samples)?;

            return Ok(TransferEntropy {
                entropy_bits: te,
                confidence: 0.9,
                lag_ms: lag,
            });
        }

        // CPU fallback
        let mut te = 0.0;
        for i in 1..n {
            let dy = target[i] - target[i - 1];
            let x_prev = source[i - 1];
            te += (dy * x_prev).abs();
        }
        te /= (n - 1) as f64;

        Ok(TransferEntropy {
            entropy_bits: te,
            confidence: 0.9,
            lag_ms: lag,
        })
    }

    fn get_bidirectional_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<BidirectionalCoupling> {
        // Compute coupling strength
        let coupling_strength = self.compute_coupling(neuro_state, quantum_state)?;

        // Compute transfer entropy in both directions
        let neuro_to_quantum_te = self.calculate_transfer_entropy(
            &neuro_state.neuron_states,
            &quantum_state.amplitudes.iter().map(|&(re, _)| re).collect::<Vec<_>>(),
            10.0,
        )?;

        let quantum_to_neuro_te = self.calculate_transfer_entropy(
            &quantum_state.amplitudes.iter().map(|&(re, _)| re).collect::<Vec<_>>(),
            &neuro_state.neuron_states,
            10.0,
        )?;

        // Extract phases for Kuramoto
        let neuro_phases: Vec<f64> = neuro_state.neuron_states.iter()
            .map(|&x| (x * core::f64::consts::TAU) % core::f64::consts::TAU)
            .collect();

        let quantum_phases: Vec<f64> = quantum_state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im).arg())
            .collect();

        let kuramoto_state = self.update_kuramoto_sync(&neuro_phases, &quantum_phases, 0.01)?;

        Ok(BidirectionalCoupling {
            neuro_to_quantum_entropy: neuro_to_quantum_te,
            quantum_to_neuro_entropy: quantum_to_neuro_te,
            kuramoto_state,
            coupling_quality: coupling_strength.bidirectional_coherence,
        })
    }
}

impl Default for CouplingAdapter {
    fn default() -> Self {
        Self::new()
    }
}
