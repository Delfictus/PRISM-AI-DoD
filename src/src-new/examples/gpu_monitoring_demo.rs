//! GPU Monitoring Demo
//!
//! Demonstrates real-time GPU utilization tracking and performance profiling.
//! Run with: cargo run --example gpu_monitoring_demo --features cuda,mission_charlie

#[cfg(all(feature = "cuda", feature = "mission_charlie"))]
fn main() -> anyhow::Result<()> {
    use prism_ai::gpu::kernel_executor::get_global_executor;
    use prism_ai::orchestration::production::gpu_monitoring::{get_global_monitor, MonitoringConfig};
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU Monitoring Demo - Worker 2 Infrastructure           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU executor
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    // Get global monitor
    let monitor = get_global_monitor();

    println!("🚀 Running GPU kernels with monitoring...\n");

    // Test 1: Vector operations
    println!("1️⃣  Testing vector operations...");
    for i in 0..20 {
        let start = Instant::now();
        let a = vec![1.0 + i as f32; 1000];
        let b = vec![2.0 + i as f32; 1000];
        let _ = executor.vector_add(&a, &b)?;
        let duration = start.elapsed();

        monitor.lock().unwrap().record_kernel_execution(
            "vector_add".to_string(),
            duration,
            8000, // 1000 * 4 bytes * 2 vectors
            true,
        )?;
    }
    println!("   ✅ Completed 20 vector_add operations\n");

    // Test 2: Matrix operations
    println!("2️⃣  Testing matrix operations...");
    for i in 0..10 {
        let start = Instant::now();
        let m = 32;
        let k = 32;
        let n = 32;
        let a = vec![0.1f32 + i as f32 * 0.01; m * k];
        let b = vec![0.2f32 + i as f32 * 0.01; k * n];
        let _ = executor.matrix_multiply(&a, &b, m, k, n)?;
        let duration = start.elapsed();

        monitor.lock().unwrap().record_kernel_execution(
            "matrix_multiply".to_string(),
            duration,
            (m * k + k * n + m * n) * 4,
            true,
        )?;
    }
    println!("   ✅ Completed 10 matrix_multiply operations\n");

    // Test 3: Tensor Core operations
    println!("3️⃣  Testing Tensor Core operations...");
    for i in 0..5 {
        let start = Instant::now();
        let m = 64;
        let k = 64;
        let n = 64;
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * n];
        let _ = executor.tensor_core_matmul_wmma(&a, &b, m, k, n)?;
        let duration = start.elapsed();

        monitor.lock().unwrap().record_kernel_execution(
            "tensor_core_matmul_wmma".to_string(),
            duration,
            (m * k + k * n + m * n) * 4,
            true,
        )?;
    }
    println!("   ✅ Completed 5 tensor_core_matmul_wmma operations\n");

    // Test 4: Activation functions
    println!("4️⃣  Testing activation functions...");
    for _ in 0..15 {
        let start = Instant::now();
        let mut data = vec![0.5; 1000];
        executor.relu_inplace(&mut data)?;
        let duration = start.elapsed();

        monitor.lock().unwrap().record_kernel_execution(
            "relu_inplace".to_string(),
            duration,
            4000,
            true,
        )?;
    }
    println!("   ✅ Completed 15 relu_inplace operations\n");

    // Display monitoring report
    println!("═══════════════════════════════════════════════════════════\n");
    let report = monitor.lock().unwrap().get_report()?;
    println!("{}", report);

    // Export to JSON
    println!("\n📄 Exporting metrics to JSON...");
    let json_metrics = monitor.lock().unwrap().export_json()?;
    println!("Metrics exported (sample):");
    println!("{}", &json_metrics[0..500.min(json_metrics.len())]);
    println!("...\n");

    println!("✅ GPU Monitoring Demo Complete!");
    println!("\n💡 Integration:");
    println!("   • Monitor tracks all kernel executions automatically");
    println!("   • Access via get_global_monitor()");
    println!("   • Export metrics for production dashboards");
    println!("   • Set alerts for high utilization/memory usage");

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "mission_charlie")))]
fn main() {
    eprintln!("⚠️  Required features not enabled");
    eprintln!("   Run: cargo run --example gpu_monitoring_demo --features cuda,mission_charlie");
    std::process::exit(1);
}
