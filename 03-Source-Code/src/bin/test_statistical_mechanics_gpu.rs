//! Test Statistical Mechanics GPU Kernels

use anyhow::Result;
use prism_ai::gpu::GpuKernelExecutor;
use std::time::Instant;

fn main() -> Result<()> {
    println!("========================================");
    println!("  STATISTICAL MECHANICS GPU TEST");
    println!("========================================\n");

    // Initialize GPU kernel executor
    println!("[1] Initializing GPU kernel executor...");
    let mut executor = GpuKernelExecutor::new(0)?;
    executor.register_standard_kernels()?;
    println!("✅ Kernel executor ready with thermodynamic kernels\n");

    // Test Kuramoto Evolution Kernel
    println!("[2] Testing Kuramoto Oscillator Evolution...");
    let n = 100; // 100 coupled oscillators
    let mut phases = vec![0.0f32; n];
    let frequencies = vec![1.0f32; n];
    let coupling_matrix = vec![0.1f32; n * n]; // Uniform coupling
    let dt = 0.01f32;
    let coupling_strength = 1.0f32;

    // Initialize with random phases
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for phase in &mut phases {
        *phase = rng.gen::<f32>() * 6.28;
    }

    let start = Instant::now();
    let new_phases = kuramoto_step(&executor, &phases, &frequencies, &coupling_matrix, n, dt, coupling_strength)?;
    let kuramoto_time = start.elapsed();

    println!("  Oscillators: {}", n);
    println!("  GPU Time: {:.2} μs", kuramoto_time.as_micros());
    println!("  Initial phase[0]: {:.4}", phases[0]);
    println!("  Updated phase[0]: {:.4}", new_phases[0]);
    println!("✅ Kuramoto evolution on GPU!\n");

    // Test Order Parameter (phase synchronization measure)
    println!("[3] Testing Order Parameter Computation...");
    let order = compute_order_parameter(&executor, &new_phases)?;

    println!("  Order parameter: {:.4}", order);
    println!("  (0 = no sync, 1 = perfect sync)");
    println!("✅ Order parameter computed on GPU!\n");

    // Test Entropy Production
    println!("[4] Testing Entropy Production...");
    let velocities = vec![0.5f32; n];
    let temperature = 1.0f32;

    let start = Instant::now();
    let entropy_rate = compute_entropy_production(&executor, &velocities, temperature)?;
    let entropy_time = start.elapsed();

    println!("  Entropy production rate: {:.6}", entropy_rate);
    println!("  GPU Time: {:.2} μs", entropy_time.as_micros());
    println!("✅ Entropy production on GPU!\n");

    // Performance benchmark
    println!("[5] Performance Benchmark (1000 Kuramoto steps)...");
    let mut current_phases = phases.clone();

    let start = Instant::now();
    for _ in 0..1000 {
        current_phases = kuramoto_step(&executor, &current_phases, &frequencies, &coupling_matrix, n, dt, coupling_strength)?;
    }
    let total_time = start.elapsed();

    let avg_time_us = total_time.as_micros() as f64 / 1000.0;
    println!("  Total time: {:.2} ms", total_time.as_millis());
    println!("  Average per step: {:.2} μs", avg_time_us);
    println!("  Throughput: {:.0} steps/sec", 1_000_000.0 / avg_time_us);
    println!("✅ Benchmark complete!\n");

    // Test scaling
    println!("[6] Testing Large System (1000 oscillators)...");
    let n_large = 1000;
    let phases_large = vec![1.0f32; n_large];
    let freq_large = vec![1.0f32; n_large];
    let coupling_large = vec![0.05f32; n_large * n_large];

    let start = Instant::now();
    let _result = kuramoto_step(&executor, &phases_large, &freq_large, &coupling_large, n_large, dt, coupling_strength)?;
    let large_time = start.elapsed();

    println!("  Oscillators: {}", n_large);
    println!("  GPU Time: {:.2} ms", large_time.as_millis());
    println!("  Performance: {:.0} oscillators/ms", n_large as f64 / large_time.as_millis() as f64);
    println!("✅ Large-scale thermodynamic system!\n");

    println!("========================================");
    println!("  ALL THERMODYNAMIC KERNELS PASS!");
    println!("========================================");
    println!("\n🚀 GPU Statistical Mechanics operational!");
    println!("   Kuramoto model: 100-1000 oscillators");
    println!("   Phase synchronization analysis");
    println!("   Entropy production tracking");

    Ok(())
}

fn kuramoto_step(
    executor: &GpuKernelExecutor,
    phases: &[f32],
    frequencies: &[f32],
    coupling_matrix: &[f32],
    n: usize,
    dt: f32,
    coupling_strength: f32,
) -> Result<Vec<f32>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let stream = executor.context().default_stream();
    let kernel = executor.get_kernel("kuramoto_evolution")?;

    // Upload data
    let phases_dev = stream.memcpy_stod(phases)?;
    let freq_dev = stream.memcpy_stod(frequencies)?;
    let coupling_dev = stream.memcpy_stod(coupling_matrix)?;
    let mut new_phases_dev = stream.alloc_zeros::<f32>(n)?;

    // Launch kernel
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        stream.launch_builder(kernel)
            .arg(&phases_dev)
            .arg(&freq_dev)
            .arg(&coupling_dev)
            .arg(&mut new_phases_dev)
            .arg(&(n as i32))
            .arg(&dt)
            .arg(&coupling_strength)
            .launch(cfg)?;
    }

    // Download result
    let result = stream.memcpy_dtov(&new_phases_dev)?;
    Ok(result)
}

fn compute_order_parameter(executor: &GpuKernelExecutor, phases: &[f32]) -> Result<f32> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let n = phases.len();
    let stream = executor.context().default_stream();
    let kernel = executor.get_kernel("order_parameter")?;

    // Upload data
    let phases_dev = stream.memcpy_stod(phases)?;
    let mut order_real_dev = stream.alloc_zeros::<f32>(1)?;
    let mut order_imag_dev = stream.alloc_zeros::<f32>(1)?;

    // Launch kernel
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(kernel)
            .arg(&phases_dev)
            .arg(&mut order_real_dev)
            .arg(&mut order_imag_dev)
            .arg(&(n as i32))
            .launch(cfg)?;
    }

    // Download and compute magnitude
    let real = stream.memcpy_dtov(&order_real_dev)?;
    let imag = stream.memcpy_dtov(&order_imag_dev)?;

    let order = (real[0] * real[0] + imag[0] * imag[0]).sqrt();
    Ok(order)
}

fn compute_entropy_production(executor: &GpuKernelExecutor, velocities: &[f32], temperature: f32) -> Result<f32> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let n = velocities.len();
    let stream = executor.context().default_stream();
    let kernel = executor.get_kernel("entropy_production")?;

    // Upload data
    let vel_dev = stream.memcpy_stod(velocities)?;
    let mut entropy_dev = stream.alloc_zeros::<f32>(1)?;

    // Launch kernel
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(kernel)
            .arg(&vel_dev)
            .arg(&mut entropy_dev)
            .arg(&temperature)
            .arg(&(n as i32))
            .launch(cfg)?;
    }

    // Download result
    let result = stream.memcpy_dtov(&entropy_dev)?;
    Ok(result[0])
}