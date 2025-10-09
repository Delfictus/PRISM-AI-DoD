//! Neuromorphic Engine Adapter - GPU Accelerated
//!
//! Wraps GPU-accelerated neuromorphic processing using CUDA kernels.
//! CPU fallback only if GPU unavailable.

use prct_core::ports::{NeuromorphicPort, NeuromorphicEncodingParams};
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use neuromorphic_engine::{SpikeEncoder, ReservoirComputer, InputData};
use cudarc::driver::{CudaContext, LaunchConfig, CudaModule, PushKernelArg};
use std::sync::Arc;

/// Adapter connecting PRCT domain to GPU-accelerated neuromorphic engine
pub struct NeuromorphicAdapter {
    window_ms: f64,
    gpu_device: Option<Arc<CudaContext>>,
    gpu_module: Option<Arc<CudaModule>>,
    use_gpu: bool,
}

impl NeuromorphicAdapter {
    /// Create new GPU-accelerated neuromorphic adapter
    pub fn new() -> Result<Self> {
        // Try to initialize GPU
        let (gpu_device, gpu_module, use_gpu) = match CudaContext::new(0) {
            Ok(device_arc) => {
                // cudarc 0.17 returns Arc<CudaContext> directly
                // Try to load GPU kernels
                match Self::load_gpu_module(&device_arc) {
                    Ok(module) => {
                        println!("✓ Neuromorphic GPU initialized (CUDA device 0)");
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

        Ok(Self {
            window_ms: 100.0,
            gpu_device,
            gpu_module,
            use_gpu,
        })
    }

    /// Load GPU module for neuromorphic processing
    fn load_gpu_module(device: &Arc<CudaContext>) -> Result<Arc<CudaModule>> {
        // Load PTX from runtime location
        let ptx_path = "target/ptx/neuromorphic_kernels.ptx";
        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Failed to load PTX: {}", e)))?;

        // Load module using cudarc 0.17 API (returns Arc<CudaModule>)
        let module = device.load_module(ptx.into())
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("PTX load failed: {}", e)))?;

        Ok(module)
    }

    /// Calculate optimal neuron count for graph size
    fn neuron_count_for_graph(&self, graph: &Graph) -> usize {
        // Scale with graph size: min 10, max 1000
        // Use 10x vertices as a reasonable scaling factor
        (graph.num_vertices * 10).clamp(10, 1000)
    }

    /// GPU-accelerated spike encoding
    fn encode_spikes_gpu(
        &self,
        features: &[f64],
        neuron_count: usize,
    ) -> Result<SpikePattern> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::NeuromorphicFailed("GPU not initialized".into()))?;

        // Allocate GPU memory using stream-based API
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let stream = device.default_stream();
        let gpu_features = stream.memcpy_stod(&features_f32)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy failed: {}", e)))?;

        let max_spikes_per_neuron = 1000;
        let mut gpu_spike_times = stream.alloc_zeros::<f32>(neuron_count * max_spikes_per_neuron)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_spike_counts = stream.alloc_zeros::<u32>(neuron_count)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU alloc failed: {}", e)))?;

        // Launch encoding kernel
        let cfg = LaunchConfig {
            grid_dim: (((neuron_count + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get module and function
        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::NeuromorphicFailed("GPU module not loaded".into()))?;
        let func = module.load_function("encode_spikes_rate")
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel not found: {}", e)))?;

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let neuron_count_u32 = neuron_count as u32;
        let features_len_u32 = features.len() as u32;
        let window_ms_f32 = self.window_ms as f32;
        let max_rate = 100.0f32;
        let min_rate = 1.0f32;

        // Use builder pattern for kernel launch
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&gpu_features);
        launch_args.arg(&mut gpu_spike_times);
        launch_args.arg(&mut gpu_spike_counts);
        launch_args.arg(&neuron_count_u32);
        launch_args.arg(&features_len_u32);
        launch_args.arg(&window_ms_f32);
        launch_args.arg(&max_rate);
        launch_args.arg(&min_rate);
        launch_args.arg(&seed);

        unsafe {
            launch_args.launch(cfg).map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy results back using stream
        let spike_times: Vec<f32> = stream.memcpy_dtov(&gpu_spike_times)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy back failed: {}", e)))?;
        let spike_counts: Vec<u32> = stream.memcpy_dtov(&gpu_spike_counts)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy back failed: {}", e)))?;

        // Convert to spike pattern
        let mut spikes = Vec::new();
        for neuron_id in 0..neuron_count {
            let count = spike_counts[neuron_id] as usize;
            let offset = neuron_id * max_spikes_per_neuron;
            for i in 0..count {
                spikes.push(Spike {
                    neuron_id,
                    time_ms: spike_times[offset + i] as f64,
                    amplitude: 1.0,
                });
            }
        }

        Ok(SpikePattern {
            spikes,
            duration_ms: self.window_ms,
            num_neurons: neuron_count,
        })
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        _params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern> {
        let neuron_count = self.neuron_count_for_graph(graph);

        // Convert graph to input data (use vertex degrees as features)
        let features: Vec<f64> = (0..graph.num_vertices)
            .map(|v| {
                let degree = graph.edges.iter()
                    .filter(|(u, w, _)| *u == v || *w == v)
                    .count();
                degree as f64 / graph.num_vertices as f64
            })
            .collect();

        // GPU-accelerated spike encoding if available
        if self.use_gpu {
            return self.encode_spikes_gpu(&features, neuron_count);
        }

        // CPU fallback
        let input_data = InputData::new("graph_encoding".to_string(), features);
        let mut encoder = SpikeEncoder::new(neuron_count, self.window_ms)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;
        let engine_spikes = encoder.encode(&input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        let spikes: Vec<Spike> = engine_spikes.spikes.iter().map(|s| {
            Spike {
                neuron_id: s.neuron_id,
                time_ms: s.time_ms,
                amplitude: 1.0,
            }
        }).collect();

        Ok(SpikePattern {
            spikes,
            duration_ms: self.window_ms,
            num_neurons: neuron_count,
        })
    }

    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState> {
        let neuron_count = spikes.num_neurons;

        // GPU-accelerated reservoir processing if available
        if self.use_gpu {
            return self.process_reservoir_gpu(spikes, neuron_count);
        }

        // CPU fallback
        let engine_spikes = neuromorphic_engine::SpikePattern::new(
            spikes.spikes.iter().map(|s| neuromorphic_engine::Spike::new(
                s.neuron_id,
                s.time_ms,
            )).collect(),
            self.window_ms
        );

        let mut reservoir = ReservoirComputer::new(
            neuron_count,
            spikes.spikes.len().max(10),
            0.9, 0.1, 0.3,
        ).map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        let reservoir_state = reservoir.process(&engine_spikes)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        let pattern_strength = reservoir_state.average_activation as f64;
        let coherence = reservoir_state.dynamics.memory_capacity;

        Ok(NeuroState {
            neuron_states: reservoir_state.activations.clone(),
            spike_pattern: vec![0; neuron_count],
            coherence,
            pattern_strength,
            timestamp_ns: 0,
        })
    }

    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
        // Simplified for now
        Ok(vec![])
    }
}

impl NeuromorphicAdapter {
    /// GPU-accelerated reservoir processing (private helper)
    fn process_reservoir_gpu(
        &self,
        spikes: &SpikePattern,
        neuron_count: usize,
    ) -> Result<NeuroState> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::NeuromorphicFailed("GPU not initialized".into()))?;

        // Convert spikes to input vector (spike counts per neuron)
        let mut input_spikes = vec![0.0f32; neuron_count];
        for spike in &spikes.spikes {
            if spike.neuron_id < neuron_count {
                input_spikes[spike.neuron_id] += 1.0;
            }
        }

        // Generate random reservoir weights (simplified - should be cached)
        let input_size = neuron_count;
        let reservoir_size = neuron_count;
        let w_in: Vec<f32> = (0..reservoir_size * input_size)
            .map(|i| ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let w_reservoir: Vec<f32> = (0..reservoir_size * reservoir_size)
            .map(|i| ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        // Allocate GPU memory using stream-based API
        let stream = device.default_stream();
        let gpu_input = stream.memcpy_stod(&input_spikes)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_w_in = stream.memcpy_stod(&w_in)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_w_reservoir = stream.memcpy_stod(&w_reservoir)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy failed: {}", e)))?;

        let initial_state = vec![0.0f32; reservoir_size];
        let gpu_state = stream.memcpy_stod(&initial_state)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy failed: {}", e)))?;
        let mut gpu_new_state = stream.alloc_zeros::<f32>(reservoir_size)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU alloc failed: {}", e)))?;

        // Launch reservoir update kernel
        let cfg = LaunchConfig {
            grid_dim: (((reservoir_size + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::NeuromorphicFailed("GPU module not loaded".into()))?;
        let func = module.load_function("reservoir_update")
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel not found: {}", e)))?;

        let reservoir_size_u32 = reservoir_size as u32;
        let input_size_u32 = input_size as u32;
        let leak_rate = 0.3f32;
        let spectral_radius = 0.9f32;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&gpu_input);
        launch_args.arg(&gpu_state);
        launch_args.arg(&gpu_w_in);
        launch_args.arg(&gpu_w_reservoir);
        launch_args.arg(&mut gpu_new_state);
        launch_args.arg(&reservoir_size_u32);
        launch_args.arg(&input_size_u32);
        launch_args.arg(&leak_rate);
        launch_args.arg(&spectral_radius);

        unsafe {
            launch_args.launch(cfg).map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy results back using stream
        let reservoir_states: Vec<f32> = stream.memcpy_dtov(&gpu_new_state)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy back failed: {}", e)))?;

        // Compute coherence on GPU
        let mut gpu_coherence = stream.alloc_zeros::<f32>(1)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU alloc failed: {}", e)))?;

        let func_coherence = module.load_function("compute_coherence")
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel not found: {}", e)))?;

        let reservoir_size_u32_2 = reservoir_size as u32;

        let mut launch_args2 = stream.launch_builder(&func_coherence);
        launch_args2.arg(&gpu_new_state);
        launch_args2.arg(&mut gpu_coherence);
        launch_args2.arg(&reservoir_size_u32_2);

        unsafe {
            launch_args2.launch(cfg).map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel launch failed: {}", e)))?;
        }

        let coherence_vec: Vec<f32> = stream.memcpy_dtov(&gpu_coherence)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy back failed: {}", e)))?;
        let coherence = coherence_vec[0] as f64;

        // Pattern strength from mean activation
        let pattern_strength = reservoir_states.iter().sum::<f32>() as f64 / reservoir_size as f64;

        Ok(NeuroState {
            neuron_states: reservoir_states.iter().map(|&x| x as f64).collect(),
            spike_pattern: vec![0; neuron_count],
            coherence,
            pattern_strength,
            timestamp_ns: 0,
        })
    }
}

impl Default for NeuromorphicAdapter {
    fn default() -> Self {
        Self::new().expect("Failed to create NeuromorphicAdapter")
    }
}
