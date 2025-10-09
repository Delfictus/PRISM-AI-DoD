//! Adapter Implementations (Hexagonal Architecture)
//!
//! Infrastructure implementations of domain ports.
//! GPU-accelerated adapters using shared CUDA context.

use std::sync::Arc;
use ndarray::{Array1, Array2};
use anyhow::Result;
use cudarc::driver::CudaContext;

use neuromorphic_engine::GpuReservoirComputer;
use crate::statistical_mechanics::{ThermodynamicNetwork, ThermodynamicState};
use super::ports::{NeuromorphicPort, InformationFlowPort, ThermodynamicPort, QuantumPort, ActiveInferencePort};
use super::quantum_mlir_integration::{QuantumMlirIntegration, QuantumGate};

#[cfg(feature = "cuda")]
use crate::information_theory::TransferEntropyGpu;

#[cfg(not(feature = "cuda"))]
use crate::information_theory::TransferEntropy;

/// GPU-accelerated neuromorphic adapter
pub struct NeuromorphicAdapter {
    reservoir: GpuReservoirComputer,
    spike_history: Vec<Array1<bool>>,
    threshold: f64,
}

impl NeuromorphicAdapter {
    pub fn new_gpu(context: Arc<CudaContext>, input_size: usize, reservoir_size: usize) -> Result<Self> {
        println!("[NEURO-ADAPTER] Creating GPU reservoir with SHARED context (Article V)");

        Ok(Self {
            reservoir: GpuReservoirComputer::new_shared(
                neuromorphic_engine::reservoir::ReservoirConfig {
                    size: reservoir_size.max(1000),  // Use parameter or default to 1000
                    input_size,
                    spectral_radius: 0.9,
                    connection_prob: 0.1,
                    leak_rate: 0.3,
                    input_scaling: 1.0,
                    noise_level: 0.01,
                    enable_plasticity: false,
                    stdp_profile: neuromorphic_engine::stdp_profiles::STDPProfile::Balanced,
                },
                context,  // Pass shared context!
            )?,
            spike_history: Vec::new(),
            threshold: 0.01,  // Lowered from 0.5 to handle small values (financial data, etc.)
        })
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_spikes(&mut self, input: &Array1<f64>) -> Result<Array1<bool>> {
        use neuromorphic_engine::{SpikePattern, Spike};

        let start_total = std::time::Instant::now();
        println!("[NEURO-ADAPTER] encode_spikes() ENTRY");

        // Convert Array1<f64> to SpikePattern
        let conv_start = std::time::Instant::now();
        let mut spikes_vec = Vec::new();
        for (i, &val) in input.iter().enumerate() {
            if val > self.threshold {
                spikes_vec.push(Spike {
                    neuron_id: i,
                    time_ms: 0.0,
                    amplitude: Some(val as f32),
                });
            }
        }

        let spike_pattern = SpikePattern {
            spikes: spikes_vec,
            duration_ms: 1.0,
            metadata: neuromorphic_engine::types::PatternMetadata {
                strength: 1.0,
                pattern_type: None,
                source: None,
                custom: std::collections::HashMap::new(),
            },
        };
        println!("[NEURO-ADAPTER] Spike pattern conversion took {:?}", conv_start.elapsed());

        // Process on GPU reservoir
        let gpu_start = std::time::Instant::now();
        let reservoir_state = self.reservoir.process_gpu(&spike_pattern)?;
        println!("[NEURO-ADAPTER] GPU reservoir.process_gpu() took {:?}", gpu_start.elapsed());

        // Extract spike encoding from reservoir state (Vec<f64> -> Array1<bool>)
        let extract_start = std::time::Instant::now();
        let spikes = Array1::from_vec(
            reservoir_state.activations.iter().map(|&x| x > self.threshold).collect()
        );
        println!("[NEURO-ADAPTER] Spike extraction took {:?}", extract_start.elapsed());

        // Store in history
        let hist_start = std::time::Instant::now();
        self.spike_history.push(spikes.clone());
        if self.spike_history.len() > 100 {
            self.spike_history.remove(0);
        }
        println!("[NEURO-ADAPTER] History update took {:?}", hist_start.elapsed());

        let total_elapsed = start_total.elapsed();
        println!("[NEURO-ADAPTER] TOTAL encode_spikes() time: {:?}", total_elapsed);

        Ok(spikes)
    }

    fn get_spike_history(&self) -> &[Array1<bool>] {
        &self.spike_history
    }
}

/// Information flow adapter (GPU-accelerated when CUDA enabled)
pub struct InformationFlowAdapter {
    #[cfg(feature = "cuda")]
    te_calculator: TransferEntropyGpu,

    #[cfg(not(feature = "cuda"))]
    te_calculator: TransferEntropy,
}

impl InformationFlowAdapter {
    pub fn new_gpu(context: Arc<CudaContext>, embedding_dim: usize, tau: usize, k: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // GPU implementation (Article VI: data stays on GPU)
            let te_calculator = TransferEntropyGpu::new(context, embedding_dim, tau, Some(16))?;
            Ok(Self { te_calculator })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to CPU if CUDA not enabled
            let _ = context;  // Suppress unused warning
            let te_calculator = TransferEntropy::new(embedding_dim, tau, k);
            Ok(Self { te_calculator })
        }
    }
}

impl InformationFlowPort for InformationFlowAdapter {
    fn compute_transfer_entropy(&mut self, source: &Array1<bool>, target: &Array1<bool>) -> Result<f64> {
        // Convert bool to f64 for processing
        let source_f64 = source.mapv(|x| if x { 1.0 } else { 0.0 });
        let target_f64 = target.mapv(|x| if x { 1.0 } else { 0.0 });

        #[cfg(feature = "cuda")]
        {
            // GPU path
            self.te_calculator.compute_transfer_entropy(&source_f64, &target_f64)
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU path
            let result = self.te_calculator.calculate(&source_f64, &target_f64);
            Ok(result.te_value)
        }
    }

    fn compute_coupling_matrix(&mut self, spike_history: &[Array1<bool>]) -> Result<Array2<f64>> {
        if spike_history.len() < 2 {
            return Ok(Array2::zeros((0, 0)));
        }

        let n = spike_history[0].len();
        let mut coupling = Array2::zeros((n, n));

        // Compute pairwise transfer entropy
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                // Extract time series for nodes i and j
                let source: Array1<bool> = Array1::from_vec(
                    spike_history.iter().map(|s| s[i]).collect()
                );
                let target: Array1<bool> = Array1::from_vec(
                    spike_history.iter().map(|s| s[j]).collect()
                );

                coupling[[i, j]] = self.compute_transfer_entropy(&source, &target)?;
            }
        }

        Ok(coupling)
    }
}

/// Thermodynamic adapter (GPU-accelerated when CUDA enabled)
pub struct ThermodynamicAdapter {
    #[cfg(feature = "cuda")]
    network: crate::statistical_mechanics::ThermodynamicGpu,

    #[cfg(not(feature = "cuda"))]
    network: ThermodynamicNetwork,

    config: crate::statistical_mechanics::NetworkConfig,
}

impl ThermodynamicAdapter {
    pub fn new_gpu(context: Arc<CudaContext>, n_oscillators: usize) -> Result<Self> {
        use crate::statistical_mechanics::NetworkConfig;

        let config = NetworkConfig {
            n_oscillators,
            temperature: 1.0,
            damping: 0.1,
            dt: 0.001,
            coupling_strength: 0.5,
            enable_information_gating: true,
            seed: 42,
        };

        #[cfg(feature = "cuda")]
        {
            // GPU implementation (Article VI: evolution on GPU)
            let config_copy = config.clone();
            let network = crate::statistical_mechanics::ThermodynamicGpu::new(context, config)?;
            Ok(Self { network, config: config_copy })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to CPU if CUDA not enabled
            let _ = context;  // Suppress unused warning
            let network = ThermodynamicNetwork::new(config);
            Ok(Self { network, config })
        }
    }
}

impl ThermodynamicPort for ThermodynamicAdapter {
    fn evolve(&mut self, coupling: &Array2<f64>, dt: f64) -> Result<ThermodynamicState> {
        #[cfg(feature = "cuda")]
        {
            // GPU path: update coupling and evolve
            self.network.update_coupling(coupling)?;

            // Compute steps from dt
            let n_steps = ((dt / self.config.dt).max(1.0)) as usize;

            // Evolve on GPU
            let mut state = self.network.get_state()?;
            for _ in 0..n_steps {
                state = self.network.evolve_step()?;
            }

            Ok(state)
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU path: use existing evolution
            let _ = coupling;  // TODO: Use coupling matrix
            let n_steps = ((dt / self.config.dt).max(1.0)) as usize;
            let result = self.network.evolve(n_steps);
            Ok(result.state.clone())
        }
    }

    fn entropy_production(&self) -> f64 {
        #[cfg(feature = "cuda")]
        {
            self.network.entropy_production()
        }

        #[cfg(not(feature = "cuda"))]
        {
            let history = self.network.entropy_history();
            if history.len() > 1 {
                let n = history.len();
                let delta_s = history[n-1] - history[n-2];
                delta_s / self.config.dt
            } else {
                0.0
            }
        }
    }

    fn get_kuramoto_state(&self) -> Option<shared_types::KuramotoState> {
        #[cfg(feature = "cuda")]
        {
            self.network.get_kuramoto_state().ok()
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU path: extract from network state
            let state = self.network.state();
            let n = state.phases.len();
            Some(shared_types::KuramotoState {
                phases: state.phases.clone(),
                natural_frequencies: vec![1.0; n], // Default frequencies
                coupling_matrix: vec![self.config.coupling_strength; n * n],
                order_parameter: 0.0, // TODO: Compute from phases
                mean_phase: state.phases.iter().sum::<f64>() / n as f64,
            })
        }
    }
}

/// GPU-accelerated quantum adapter
pub struct QuantumAdapter {
    quantum_mlir: QuantumMlirIntegration,
    phases: Array1<f64>,
    amplitudes: Array1<f64>,
}

impl QuantumAdapter {
    pub fn new_gpu(_context: Arc<CudaContext>, num_qubits: usize) -> Result<Self> {
        // QuantumMlirIntegration creates its own context internally
        // TODO: Refactor to accept shared context
        let quantum_mlir = QuantumMlirIntegration::new(num_qubits)?;
        let state_dim = 1 << num_qubits;

        Ok(Self {
            quantum_mlir,
            phases: Array1::zeros(state_dim),
            amplitudes: Array1::ones(state_dim),
        })
    }
}

impl QuantumPort for QuantumAdapter {
    fn quantum_process(&mut self, thermo_state: &ThermodynamicState) -> Result<Array1<f64>> {
        // Apply quantum gates based on thermodynamic state
        let gates = vec![
            QuantumGate::Hadamard(0),  // Create superposition
            QuantumGate::RZ(0, thermo_state.phases[0]),  // Phase rotation
        ];

        self.quantum_mlir.apply_gates(gates)?;

        // Get quantum state and extract observables
        let state = self.quantum_mlir.get_state()?;

        // Update phases and amplitudes from quantum state
        for (i, amp) in state.amplitudes.iter().enumerate().take(self.phases.len()) {
            // Complex64 methods: norm() and arg()
            self.amplitudes[i] = (amp.real * amp.real + amp.imag * amp.imag).sqrt();
            self.phases[i] = amp.imag.atan2(amp.real);
        }

        Ok(self.amplitudes.clone())
    }

    fn get_observables(&self) -> Array1<f64> {
        self.amplitudes.clone()
    }

    fn get_phase_field(&self) -> Option<shared_types::PhaseField> {
        let n = self.phases.len();

        // Compute coherence matrix (simple version: phase difference)
        let mut coherence = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let phase_diff = (self.phases[i] - self.phases[j]).abs();
                coherence[i * n + j] = (1.0 - phase_diff / std::f64::consts::PI).max(0.0);
            }
        }

        // Compute order parameter
        let (sum_cos, sum_sin): (f64, f64) = self.phases.iter()
            .map(|&p| (p.cos(), p.sin()))
            .fold((0.0, 0.0), |(c, s), (pc, ps)| (c + pc, s + ps));
        let order_parameter = ((sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n as f64).min(1.0);

        Some(shared_types::PhaseField {
            phases: self.phases.to_vec(),
            coherence_matrix: coherence,
            order_parameter,
            resonance_frequency: 1.0, // TODO: Extract from quantum state
        })
    }
}

/// Active inference adapter (GPU-accelerated when CUDA enabled)
pub struct ActiveInferenceAdapter {
    hierarchical_model: crate::active_inference::HierarchicalModel,

    #[cfg(feature = "cuda")]
    inference_engine: crate::active_inference::ActiveInferenceGpu,

    #[cfg(not(feature = "cuda"))]
    inference_engine: crate::active_inference::VariationalInference,

    controller: crate::active_inference::ActiveInferenceController,
}

impl ActiveInferenceAdapter {
    pub fn new_gpu(context: Arc<CudaContext>, _n_dimensions: usize) -> Result<Self> {
        use crate::active_inference::{
            HierarchicalModel, VariationalInference,
            PolicySelector, ActiveInferenceController, SensingStrategy,
            ObservationModel, TransitionModel,
        };

        let hierarchical_model = HierarchicalModel::new();
        let n_windows = 900;
        let obs_model = ObservationModel::new(100, n_windows, 8.0, 0.01);
        let trans_model = TransitionModel::default_timescales();
        let cpu_inference = VariationalInference::new(
            obs_model.clone(),
            trans_model.clone(),
            &hierarchical_model
        );

        let preferred_obs = Array1::zeros(100);
        let mut selector = PolicySelector::new(3, 5, preferred_obs, cpu_inference.clone(), trans_model);

        #[cfg(feature = "cuda")]
        {
            // Create GPU policy evaluator
            println!("[ADAPTER] Creating GPU policy evaluator...");
            match crate::active_inference::GpuPolicyEvaluator::new(
                context.clone(),
                5,   // n_policies
                3,   // horizon
                10,  // substeps for window evolution
            ) {
                Ok(gpu_policy_eval) => {
                    let gpu_eval_arc = std::sync::Arc::new(std::sync::Mutex::new(gpu_policy_eval));
                    selector.set_gpu_evaluator(gpu_eval_arc);
                    println!("[ADAPTER] GPU policy evaluator created and wired successfully");
                }
                Err(e) => {
                    eprintln!("[ADAPTER] Failed to create GPU policy evaluator: {}", e);
                    eprintln!("[ADAPTER] Will use CPU policy evaluation");
                }
            }
        }

        let controller = ActiveInferenceController::new(selector, SensingStrategy::Adaptive);

        #[cfg(feature = "cuda")]
        {
            // GPU implementation (Article VI: variational inference on GPU)
            let inference_engine = crate::active_inference::ActiveInferenceGpu::new(context, cpu_inference)?;
            Ok(Self {
                hierarchical_model,
                inference_engine,
                controller,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to CPU if CUDA not enabled
            let _ = context;
            Ok(Self {
                hierarchical_model,
                inference_engine: cpu_inference,
                controller,
            })
        }
    }
}

impl ActiveInferencePort for ActiveInferenceAdapter {
    fn infer(&mut self, observations: &Array1<f64>, _quantum_obs: &Array1<f64>) -> Result<f64> {
        let start_total = std::time::Instant::now();
        println!("[ADAPTER] ========================================");
        println!("[ADAPTER] infer() ENTRY - observations.len()={}", observations.len());

        // Resize observations to 100 dimensions (ObservationModel requirement)
        let resize_start = std::time::Instant::now();
        let obs_resized = if observations.len() != 100 {
            let mut resized = Array1::zeros(100);
            let n_copy = observations.len().min(100);
            for i in 0..n_copy {
                resized[i] = observations[i];
            }
            resized
        } else {
            observations.clone()
        };
        let resize_elapsed = resize_start.elapsed();
        println!("[ADAPTER] Resize/clone took {:?}", resize_elapsed);

        #[cfg(feature = "cuda")]
        {
            println!("[ADAPTER] CUDA feature ENABLED - Using GPU path");
            let gpu_start = std::time::Instant::now();
            // GPU path: accelerated variational inference
            let result = self.inference_engine.infer_gpu(&mut self.hierarchical_model, &obs_resized);
            let gpu_elapsed = gpu_start.elapsed();
            println!("[ADAPTER] inference_engine.infer_gpu() returned in {:?}", gpu_elapsed);

            let total_elapsed = start_total.elapsed();
            println!("[ADAPTER] TOTAL infer() time: {:?}", total_elapsed);
            println!("[ADAPTER] ========================================");
            result
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!("[ADAPTER] CUDA feature DISABLED - Using CPU path");
            // CPU path: standard variational inference
            let _components = self.inference_engine.infer(&mut self.hierarchical_model, &obs_resized);
            let free_energy = self.hierarchical_model.compute_free_energy(&obs_resized);

            let total_elapsed = start_total.elapsed();
            println!("[ADAPTER] TOTAL infer() time: {:?}", total_elapsed);
            println!("[ADAPTER] ========================================");
            Ok(free_energy)
        }
    }

    fn select_action(&mut self, _targets: &Array1<f64>) -> Result<Array1<f64>> {
        // Select action via active inference controller
        let action = self.controller.control(&self.hierarchical_model);

        Ok(action.phase_correction)
    }
}
