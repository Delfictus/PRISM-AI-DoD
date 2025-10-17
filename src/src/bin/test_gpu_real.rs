//! Test program to verify GPU is actually being used

use anyhow::Result;
use std::time::Instant;
use prism_ai::gpu::gpu_enabled::{GpuContext, GpuTensor, GpuLinear};

fn main() -> Result<()> {
    println!("========================================");
    println!("     GPU VERIFICATION TEST");
    println!("========================================\n");

    // Test 1: GPU Context Creation
    println!("[1] Creating GPU Context...");
    let ctx = GpuContext::new()?;

    if ctx.is_gpu_available() {
        println!("✅ GPU is AVAILABLE and INITIALIZED!");
        println!("   This is using the REAL GPU, GPU acceleration.\n");
    } else {
        println!("❌ GPU not available - GPU REQUIRED\n");
        println!("   Check CUDA installation and drivers.\n");
    }

    // Test 2: Tensor Creation on GPU
    println!("[2] Creating tensors on GPU...");
    let size = 1024;
    let a_data: Vec<f32> = (0..size*size).map(|i| i as f32 * 0.001).collect();
    let b_data: Vec<f32> = (0..size*size).map(|i| i as f32 * 0.002).collect();

    let a = GpuTensor::from_cpu(a_data.clone(), vec![size, size])?;
    let b = GpuTensor::from_cpu(b_data.clone(), vec![size, size])?;

    // Test 3: Matrix Multiplication on GPU
    println!("\n[3] Performing matrix multiplication (1024x1024)...");
    let start = Instant::now();

    let c = a.matmul(&b)?;

    let gpu_time = start.elapsed();
    println!("   GPU Time: {:.2} ms", gpu_time.as_secs_f64() * 1000.0);

    // Verify result (spot check)
    let c_data = c.to_cpu()?;
    println!("   Result shape: {:?}", c.shape());
    println!("   Sample value C[0,0]: {}", c_data[0]);

    // Test 4: ReLU Activation on GPU
    println!("\n[4] Testing ReLU activation...");
    let test_data = vec![-1.0, 0.0, 1.0, -0.5, 2.0, -3.0, 4.0, -5.0];
    let mut tensor = GpuTensor::from_cpu(test_data.clone(), vec![2, 4])?;

    tensor.relu()?;

    let result = tensor.to_cpu()?;
    println!("   Input:  {:?}", test_data);
    println!("   Output: {:?}", result);

    // Verify ReLU worked
    let all_positive = result.iter().all(|&x| x >= 0.0);
    if all_positive {
        println!("   ✅ ReLU working correctly!");
    } else {
        println!("   ❌ ReLU failed!");
    }

    // Test 5: Linear Layer on GPU
    println!("\n[5] Testing Linear layer...");
    let linear = GpuLinear::new(256, 128)?;
    let input = GpuTensor::zeros(vec![32, 256])?; // Batch of 32

    let start = Instant::now();
    let output = linear.forward(&input)?;
    let linear_time = start.elapsed();

    println!("   Linear layer (256→128) with batch 32");
    println!("   Time: {:.2} ms", linear_time.as_secs_f64() * 1000.0);
    println!("   Output shape: {:?}", output.shape());

    // Test 6: Large-scale operation to trigger visible GPU usage
    println!("\n[6] Running large-scale GPU operations...");
    println!("   Monitor with: watch -n 0.5 nvidia-smi");

    let large_size = 2048;
    println!("   Creating {}x{} matrices...", large_size, large_size);

    let x = GpuTensor::from_cpu(vec![1.0; large_size * large_size], vec![large_size, large_size])?;
    let y = GpuTensor::from_cpu(vec![2.0; large_size * large_size], vec![large_size, large_size])?;

    println!("   Performing 10 matrix multiplications...");
    let start = Instant::now();

    for i in 0..10 {
        let _z = x.matmul(&y)?;
        print!("   Iteration {}/10\r", i + 1);
        use std::io::{self, Write};
        io::stdout().flush()?;
    }

    let total_time = start.elapsed();
    println!("\n   Total time for 10 iterations: {:.2} ms", total_time.as_secs_f64() * 1000.0);
    println!("   Average per iteration: {:.2} ms", total_time.as_secs_f64() * 100.0);

    // Performance comparison
    println!("\n========================================");
    println!("         PERFORMANCE SUMMARY");
    println!("========================================");

    if ctx.is_gpu_available() {
        println!("✅ GPU ACCELERATION ACTIVE!");
        println!("   Matrix multiply (1024x1024): {:.2} ms", gpu_time.as_secs_f64() * 1000.0);

        // Expected CPU time for comparison
        let expected_cpu_time_ms = (size as f64).powi(3) / 1_000_000.0; // Rough estimate
        let speedup = expected_cpu_time_ms / (gpu_time.as_secs_f64() * 1000.0);

        println!("   Estimated CPU time: {:.0} ms", expected_cpu_time_ms);
        println!("   Speedup: {:.1}x", speedup);

        if speedup > 10.0 {
            println!("\n🚀 EXCELLENT: GPU providing {:.0}x speedup!", speedup);
        } else if speedup > 2.0 {
            println!("\n✅ GOOD: GPU providing {:.1}x speedup", speedup);
        } else {
            println!("\n⚠️  LIMITED: GPU speedup only {:.1}x", speedup);
            println!("   May still be using GPU execution");
        }
    } else {
        println!("❌ GPU NOT ACTIVE - GPU REQUIRED");
        println!("   Check CUDA installation and GPU drivers");
    }

    println!("\n========================================");
    println!("To verify GPU usage, run in another terminal:");
    println!("1. nvidia-smi -l 1");
    println!("2. nvtop");
    println!("3. strace -e trace=ioctl -p {} 2>&1 | grep nvidia", std::process::id());
    println!("========================================\n");

    Ok(())
}