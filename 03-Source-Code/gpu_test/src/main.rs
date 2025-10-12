// Minimal GPU test - just test CUDA context creation

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║         PRISM-AI GPU Test - CUDA 13 Direct          ║");
    println!("╚══════════════════════════════════════════════════════╝");

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;

        println!("\n🚀 Testing CUDA 13 GPU support...\n");

        println!("Attempting to create CUDA context...");
        match CudaContext::new(0) {
            Ok(_device) => {
                println!("\n{}", "=".repeat(60));
                println!("✅ SUCCESS: CUDA 13 CONTEXT CREATED!");
                println!("   Your RTX 5070 is detected and accessible!");
                println!("   GPU acceleration is possible with proper integration");
                println!("{}", "=".repeat(60));
            }
            Err(e) => {
                println!("\n{}", "=".repeat(60));
                println!("❌ FAILED TO CREATE CUDA CONTEXT");
                println!("   Error: {:?}", e);
                println!("\n   Possible causes:");
                println!("   - NVIDIA driver not loaded");
                println!("   - CUDA libraries not found");
                println!("   - GPU not available");
                println!("{}", "=".repeat(60));
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\n❌ CUDA feature not enabled");
        println!("   Build with: cargo build --features cuda");
    }
}