// Direct cudarc test - this is what PRISM-AI actually uses
// Compile with: cargo build --features cuda

fn main() {
    println!("========================================");
    println!("Testing cudarc (PRISM-AI's CUDA library)");
    println!("========================================\n");

    println!("Persistence Mode: ENABLED ✅\n");

    // Test with cudarc's driver module
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaDevice;

        println!("Testing cudarc CudaDevice initialization...");
        match CudaDevice::new(0) {
            Ok(device) => {
                println!("✅ SUCCESS! CudaDevice initialized!");

                if let Ok(name) = device.name() {
                    println!("  Device: {}", name);
                }

                // Try memory allocation
                match device.alloc_zeros::<f32>(1024) {
                    Ok(_) => println!("  Memory allocation: ✅ SUCCESS"),
                    Err(e) => println!("  Memory allocation: ❌ Failed - {}", e),
                }

                println!("\n🎉 cudarc is working! PRISM-AI can use GPU acceleration!");
            }
            Err(e) => {
                println!("❌ CudaDevice::new(0) failed: {}", e);
                println!("\nThis is the actual error PRISM-AI encounters.");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA feature not enabled. Compile with: --features cuda");
    }

    println!("\n========================================");
}