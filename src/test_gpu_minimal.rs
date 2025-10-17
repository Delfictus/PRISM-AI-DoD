// Minimal GPU test using cudarc directly with CUDA 13
// This bypasses candle to test pure GPU functionality

use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

const VECTOR_SIZE: usize = 10_000_000;

// CUDA kernel source for vector addition
#[cfg(feature = "cuda")]
const VECTOR_ADD_KERNEL: &str = r#"
extern "C" __global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

// CPU version for comparison
fn vector_add_cpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[cfg(feature = "cuda")]
fn test_gpu() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Testing GPU with CUDA 13...\n");

    // Initialize CUDA device
    println!("Initializing CUDA device...");
    let device = CudaDevice::new(0)?;
    println!("✅ CUDA device initialized");

    // Get device properties
    let props = device.props()?;
    println!("\n📊 GPU Properties:");
    println!("  Name: {}", props.name()?);
    println!("  Compute Capability: {}.{}", props.major, props.minor);
    println!("  Total Memory: {:.2} GB", props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  Multiprocessors: {}", props.multi_processor_count);

    // Compile kernel
    println!("\nCompiling CUDA kernel...");
    let ptx = Ptx::compile_ptx(VECTOR_ADD_KERNEL)?;
    device.load_ptx(ptx, "vector_module", &["vector_add"])?;
    let kernel = device.get_func("vector_module", "vector_add")?;
    println!("✅ Kernel compiled and loaded");

    // Prepare test data
    println!("\nPreparing test data ({} elements)...", VECTOR_SIZE);
    let a_host: Vec<f32> = (0..VECTOR_SIZE).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..VECTOR_SIZE).map(|i| (i * 2) as f32).collect();

    // Allocate GPU memory
    println!("Allocating GPU memory...");
    let a_gpu = device.htod_copy(a_host.clone())?;
    let b_gpu = device.htod_copy(b_host.clone())?;
    let mut c_gpu = device.alloc::<f32>(VECTOR_SIZE)?;
    println!("✅ GPU memory allocated");

    // Configure kernel launch
    let block_size = 256;
    let grid_size = (VECTOR_SIZE as u32 + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm up GPU
    println!("\nWarming up GPU...");
    unsafe {
        kernel.launch(config, (&a_gpu, &b_gpu, &mut c_gpu, VECTOR_SIZE as i32))?;
    }
    device.synchronize()?;

    // Benchmark GPU
    println!("\n⚡ Benchmarking GPU...");
    let gpu_start = Instant::now();
    for _ in 0..100 {
        unsafe {
            kernel.launch(config, (&a_gpu, &b_gpu, &mut c_gpu, VECTOR_SIZE as i32))?;
        }
    }
    device.synchronize()?;
    let gpu_duration = gpu_start.elapsed();

    // Get results
    let c_host = device.dtoh_copy(&c_gpu)?;

    // Verify results
    let first_results: Vec<f32> = c_host.iter().take(5).copied().collect();
    println!("First 5 results: {:?}", first_results);

    // Benchmark CPU for comparison
    println!("\n🐌 Benchmarking CPU...");
    let cpu_start = Instant::now();
    for _ in 0..100 {
        let _ = vector_add_cpu(&a_host, &b_host);
    }
    let cpu_duration = cpu_start.elapsed();

    // Results
    println!("\n📈 Performance Results:");
    println!("  GPU Time (100 iterations): {:.2} ms", gpu_duration.as_secs_f64() * 1000.0);
    println!("  CPU Time (100 iterations): {:.2} ms", cpu_duration.as_secs_f64() * 1000.0);
    println!("  Speedup: {:.2}x", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());

    // Memory info
    let (free, total) = device.memory_info()?;
    println!("\n💾 GPU Memory:");
    println!("  Used: {:.2} MB", (total - free) as f64 / (1024.0 * 1024.0));
    println!("  Free: {:.2} MB", free as f64 / (1024.0 * 1024.0));
    println!("  Total: {:.2} MB", total as f64 / (1024.0 * 1024.0));

    println!("\n✅ GPU test completed successfully!");
    println!("🎯 GPU acceleration is WORKING with CUDA 13!");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn test_gpu() -> Result<(), Box<dyn std::error::Error>> {
    println!("❌ CUDA feature not enabled. GPU test cannot run.");
    println!("   Build with: cargo build --features cuda");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║         PRISM-AI GPU Test - CUDA 13 Direct          ║");
    println!("╚══════════════════════════════════════════════════════╝");

    test_gpu()?;

    Ok(())
}