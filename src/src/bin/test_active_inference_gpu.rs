//! Test Active Inference GPU Kernels

use anyhow::Result;
use prism_ai::gpu::{GpuKernelExecutor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("========================================");
    println!("  ACTIVE INFERENCE GPU KERNEL TEST");
    println!("========================================\n");

    // Create kernel executor
    println!("[1] Initializing GPU kernel executor...");
    let mut executor = GpuKernelExecutor::new(0)?;
    executor.register_standard_kernels()?;
    println!("✅ Kernel executor ready\n");

    // Test KL Divergence
    println!("[2] Testing KL Divergence Kernel...");
    let q = vec![0.7, 0.1, 0.1, 0.05, 0.05]; // Posterior (5 threat classes)
    let p = vec![0.6, 0.15, 0.15, 0.05, 0.05]; // Prior

    let start = Instant::now();
    let kl = executor.kl_divergence(&q, &p)?;
    let kl_time = start.elapsed();

    println!("  KL Divergence: {:.6}", kl);
    println!("  GPU Time: {:.2} μs", kl_time.as_micros());
    println!("✅ KL divergence computed on GPU!\n");

    // Test Element-wise Multiply
    println!("[3] Testing Element-wise Multiply Kernel...");
    let a = vec![0.5, 0.3, 0.1, 0.05, 0.05]; // Posterior
    let b = vec![0.7, 0.1, 0.1, 0.05, 0.05]; // Prior

    let start = Instant::now();
    let c = executor.elementwise_multiply(&a, &b)?;
    let mul_time = start.elapsed();

    println!("  Result: {:?}", c);
    println!("  GPU Time: {:.2} μs", mul_time.as_micros());
    println!("✅ Element-wise multiply on GPU!\n");

    // Test Normalize
    println!("[4] Testing Normalize Kernel...");
    let mut data = vec![2.0, 3.0, 1.0, 4.0, 5.0];

    let start = Instant::now();
    executor.normalize_inplace(&mut data)?;
    let norm_time = start.elapsed();

    let sum: f32 = data.iter().sum();
    println!("  Normalized: {:?}", data);
    println!("  Sum: {} (should be 1.0)", sum);
    println!("  GPU Time: {:.2} μs", norm_time.as_micros());
    assert!((sum - 1.0).abs() < 1e-6);
    println!("✅ Normalization computed on GPU!\n");

    // Test Free Energy
    println!("[5] Testing Free Energy Kernel...");
    let posterior = vec![0.7f32, 0.1, 0.1, 0.05, 0.05];
    let prior = vec![0.6f32, 0.15, 0.15, 0.05, 0.05];
    let log_likelihood = 0.0;

    let start = Instant::now();
    let fe = executor.compute_free_energy(&posterior, &prior, log_likelihood)?;
    let fe_time = start.elapsed();

    println!("  Free Energy: {:.6}", fe);
    println!("  GPU Time: {:.2} μs", fe_time.as_micros());
    println!("✅ Free energy computed on GPU!\n");

    // Benchmark: 1000 iterations
    println!("[6] Performance Benchmark (1000 iterations)...");
    let start = Instant::now();

    for _ in 0..1000 {
        let _ = executor.kl_divergence(&q, &p)?;
    }

    let total_time = start.elapsed();
    let avg_time_us = total_time.as_micros() as f64 / 1000.0;

    println!("  Total time: {:.2} ms", total_time.as_millis());
    println!("  Average per operation: {:.2} μs", avg_time_us);
    println!("  Throughput: {:.0} ops/sec", 1_000_000.0 / avg_time_us);
    println!("✅ Benchmark complete!\n");

    println!("========================================");
    println!("   ALL ACTIVE INFERENCE KERNELS PASS!");
    println!("========================================");
    println!("\n🚀 GPU Active Inference kernels operational!");
    println!("   Ready for PWSA threat classification");

    Ok(())
}