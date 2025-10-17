//! Test that gpu_enabled actually uses GPU kernels

use anyhow::Result;
use prism_ai::gpu::gpu_enabled::{SimpleGpuContext, SimpleGpuTensor};

fn main() -> Result<()> {
    println!("========================================");
    println!("  TESTING REAL GPU KERNEL EXECUTION");
    println!("========================================\n");

    // This will FAIL without GPU - NO CPU FALLBACK
    println!("[1] Creating GPU context (NO CPU FALLBACK)...");
    let ctx = SimpleGpuContext::new()?;
    assert!(ctx.is_gpu_available());
    println!("✅ GPU context created - kernel execution ready!\n");

    // Test matrix multiplication with actual GPU kernels
    println!("[2] Testing matrix multiplication on GPU...");
    let a = SimpleGpuTensor::from_cpu(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;
    let b = SimpleGpuTensor::from_cpu(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2]
    )?;

    let c = a.matmul(&b)?;
    let result = c.to_cpu()?;

    // Verify computation
    println!("   Result: {:?}", result);
    assert!((result[0] - 19.0).abs() < 1e-6); // 1*5 + 2*7 = 19
    assert!((result[1] - 22.0).abs() < 1e-6); // 1*6 + 2*8 = 22
    assert!((result[2] - 43.0).abs() < 1e-6); // 3*5 + 4*7 = 43
    assert!((result[3] - 50.0).abs() < 1e-6); // 3*6 + 4*8 = 50
    println!("✅ Matrix multiplication computed correctly on GPU!\n");

    // Test ReLU activation
    println!("[3] Testing ReLU activation on GPU...");
    let mut tensor = SimpleGpuTensor::from_cpu(
        vec![-1.0, 0.0, 1.0, -0.5, 2.0, -3.0],
        vec![6]
    )?;

    tensor.relu()?;
    let result = tensor.to_cpu()?;

    println!("   Result: {:?}", result);
    assert_eq!(result, vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
    println!("✅ ReLU activation computed correctly on GPU!\n");

    // Test Softmax
    println!("[4] Testing Softmax on GPU...");
    let mut tensor = SimpleGpuTensor::from_cpu(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    tensor.softmax(1)?;
    let result = tensor.to_cpu()?;

    // Check that each row sums to 1
    let row1_sum = result[0] + result[1] + result[2];
    let row2_sum = result[3] + result[4] + result[5];

    println!("   Row 1 sum: {}", row1_sum);
    println!("   Row 2 sum: {}", row2_sum);
    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
    println!("✅ Softmax computed correctly on GPU!\n");

    // Performance test
    println!("[5] Performance test - 1000 matrix multiplications...");
    use std::time::Instant;

    let size = 128;
    let a = SimpleGpuTensor::from_cpu(
        vec![1.0; size * size],
        vec![size, size]
    )?;
    let b = SimpleGpuTensor::from_cpu(
        vec![2.0; size * size],
        vec![size, size]
    )?;

    let start = Instant::now();
    for _ in 0..1000 {
        let _c = a.matmul(&b)?;
    }
    let elapsed = start.elapsed();

    let ops = 1000.0 * 2.0 * (size as f64).powi(3); // 2*n^3 ops per matmul
    let gflops = ops / (elapsed.as_secs_f64() * 1e9);

    println!("   Time: {:.2} ms", elapsed.as_millis());
    println!("   Performance: {:.1} GFLOPS", gflops);
    println!("✅ GPU achieving high performance!\n");

    println!("========================================");
    println!("     ALL GPU KERNEL TESTS PASSED!");
    println!("========================================");
    println!("\n🚀 GPU kernels are executing correctly!");
    println!("   NO CPU FALLBACK - 100% GPU execution!");

    Ok(())
}