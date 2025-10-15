//! GPU Neuromorphic Processing Demo
//!
//! Demonstrates GPU-accelerated spiking neural network simulation
//! with Izhikevich neurons and STDP learning using custom CUDA kernels.

use prism_ai::orchestration::neuromorphic::{GpuNeuromorphicProcessor, UnifiedNeuromorphicProcessor};
use cudarc::driver::CudaDevice;
use nalgebra::DVector;
use std::sync::Arc;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧠 GPU Neuromorphic Processing Demo with Custom CUDA Kernels");
    println!("==============================================================\n");

    // Initialize CUDA device
    println!("1. Initializing CUDA device...");
    let device = match CudaDevice::new(0) {
        Ok(dev) => {
            println!("   ✅ CUDA device 0 available");
            Some(Arc::new(dev))
        }
        Err(e) => {
            println!("   ⚠️  CUDA not available: {}", e);
            println!("   Using CPU fallback\n");
            None
        }
    };

    if let Some(cuda_device) = device {
        // Demo 1: GPU-Accelerated Spiking Network
        demo_gpu_spiking_network(cuda_device.clone())?;

        // Demo 2: STDP Learning
        demo_stdp_learning(cuda_device.clone())?;

        // Demo 3: Large-Scale Simulation
        demo_large_scale_simulation(cuda_device)?;
    }

    // Demo 4: CPU Unified Neuromorphic Processor
    demo_unified_processor()?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

fn demo_gpu_spiking_network(device: Arc<CudaDevice>) -> Result<()> {
    println!("\n2. Demo: GPU-Accelerated Spiking Network with CUDA Kernels");
    println!("   Creating 100-neuron network...");

    let n_neurons = 100;
    let n_synapses = 500;

    let mut processor = GpuNeuromorphicProcessor::new(device, n_neurons, n_synapses)?;

    // Create random connectivity
    let mut connectivity = Vec::new();
    for i in 0..50 {
        for j in 50..100 {
            if rand::random::<f32>() < 0.2 {
                connectivity.push((i, j));
            }
        }
    }

    processor.initialize_network(&connectivity, None)?;

    // Set different neuron types
    let excitatory: Vec<usize> = (0..80).collect();
    let inhibitory: Vec<usize> = (80..100).collect();

    let excitatory_params = vec![(0.02, 0.2, -65.0, 8.0); 80]; // Regular spiking
    let inhibitory_params = vec![(0.1, 0.2, -65.0, 2.0); 20];   // Fast spiking

    processor.set_neuron_params(&excitatory, &excitatory_params)?;
    processor.set_neuron_params(&inhibitory, &inhibitory_params)?;

    // Apply input to first 10 neurons
    let input_neurons: Vec<usize> = (0..10).collect();
    let input_currents = vec![20.0; 10];
    processor.apply_input(&input_neurons, &input_currents)?;

    // Simulate for 50ms
    println!("   Simulating 50ms...");
    let result = processor.simulate(50.0)?;

    println!("   ✅ Simulation complete");
    println!("   Total spikes: {}", result.total_spikes);
    println!("   Mean firing rate: {:.2} Hz", result.mean_firing_rate);
    println!("   Active neurons: {}/{}",
        result.spike_counts.iter().filter(|&&c| c > 0).count(),
        n_neurons
    );

    Ok(())
}

fn demo_stdp_learning(device: Arc<CudaDevice>) -> Result<()> {
    println!("\n3. Demo: STDP Learning with GPU Kernels");
    println!("   Creating 50-neuron network with STDP...");

    let n_neurons = 50;
    let n_synapses = 200;

    let mut processor = GpuNeuromorphicProcessor::new(device, n_neurons, n_synapses)?;

    // Create feedforward connectivity
    let mut connectivity = Vec::new();
    for i in 0..25 {
        for j in 25..50 {
            if rand::random::<f32>() < 0.3 {
                connectivity.push((i, j));
            }
        }
    }

    // Initialize with weak weights
    let initial_weights = vec![0.1; connectivity.len()];
    processor.initialize_network(&connectivity, Some(&initial_weights))?;

    // Record initial weights
    let initial_state = processor.get_state();
    let initial_mean_weight: f32 = initial_state.synapse_weights.iter().sum::<f32>()
        / initial_state.synapse_weights.len() as f32;

    // Apply repeated input pattern
    println!("   Training with repeated pattern (200ms)...");
    for _ in 0..20 {
        let input_neurons: Vec<usize> = (0..10).collect();
        let input_currents = vec![25.0; 10];
        processor.apply_input(&input_neurons, &input_currents)?;

        // Simulate 10ms
        processor.simulate(10.0)?;
    }

    // Record final weights
    let final_state = processor.get_state();
    let final_mean_weight: f32 = final_state.synapse_weights.iter().sum::<f32>()
        / final_state.synapse_weights.len() as f32;

    println!("   ✅ Learning complete");
    println!("   Initial mean weight: {:.4}", initial_mean_weight);
    println!("   Final mean weight: {:.4}", final_mean_weight);
    println!("   Weight change: {:.4}", final_mean_weight - initial_mean_weight);

    // Check for potentiation
    if final_mean_weight > initial_mean_weight {
        println!("   🎓 STDP potentiation detected!");
    }

    Ok(())
}

fn demo_large_scale_simulation(device: Arc<CudaDevice>) -> Result<()> {
    println!("\n4. Demo: Large-Scale GPU Simulation with Custom Kernels");
    println!("   Creating 1000-neuron network...");

    let n_neurons = 1000;
    let n_synapses = 10000;

    let mut processor = GpuNeuromorphicProcessor::new(device, n_neurons, n_synapses)?;

    // Create sparse random connectivity
    let mut connectivity = Vec::new();
    for _ in 0..n_synapses {
        let pre = rand::random::<usize>() % n_neurons;
        let post = rand::random::<usize>() % n_neurons;
        if pre != post {
            connectivity.push((pre, post));
        }
    }

    processor.initialize_network(&connectivity, None)?;

    // Apply random input
    let n_input = 100;
    let input_neurons: Vec<usize> = (0..n_input).collect();
    let input_currents: Vec<f32> = (0..n_input)
        .map(|_| rand::random::<f32>() * 30.0)
        .collect();
    processor.apply_input(&input_neurons, &input_currents)?;

    // Benchmark simulation speed
    println!("   Simulating 100ms on GPU...");
    let start = std::time::Instant::now();
    let result = processor.simulate(100.0)?;
    let duration = start.elapsed();

    println!("   ✅ Large-scale simulation complete");
    println!("   Total spikes: {}", result.total_spikes);
    println!("   Mean firing rate: {:.2} Hz", result.mean_firing_rate);
    println!("   Simulation time: {:?}", duration);
    println!("   Speed: {:.2}× realtime",
        100.0 / duration.as_secs_f32()
    );

    Ok(())
}

fn demo_unified_processor() -> Result<()> {
    println!("\n5. Demo: Unified Neuromorphic Processor (CPU)");
    println!("   Creating unified processor with 10 input, 20 hidden, 5 output...");

    let mut processor = UnifiedNeuromorphicProcessor::new(10, 20, 5)?;

    // Create test input
    let input = DVector::from_element(10, 0.5);

    println!("   Processing 100ms...");
    let result = processor.process(&input, 100.0)?;

    println!("   ✅ Processing complete");
    println!("   Spike count: {}", result.spike_count);
    println!("   Energy consumed: {:.2} pJ", result.energy_consumed);
    println!("   Mean firing rate: {:.2} Hz", result.metrics.mean_firing_rate);
    println!("   Synchrony index: {:.4}", result.metrics.synchrony_index);
    println!("   Energy efficiency: {:.2} spikes/pJ", result.metrics.energy_efficiency);
    println!("   Learning progress: {:.4}", result.metrics.learning_progress);

    // Test LLM response processing
    println!("\n   Testing LLM response processing...");
    let responses = vec![
        "Response A: This is a test".to_string(),
        "Response B: Another test".to_string(),
        "Response C: Final test".to_string(),
    ];

    let consensus = processor.process_llm_responses(&responses)?;

    println!("   ✅ Consensus computed");
    println!("   Spike coherence: {:.4}", consensus.spike_coherence);
    println!("   Energy efficiency: {:.2} spikes/pJ", consensus.energy_efficiency);
    println!("   Confidence: {:.4}", consensus.confidence);

    Ok(())
}
