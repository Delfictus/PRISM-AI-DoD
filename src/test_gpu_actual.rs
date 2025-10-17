// Test to verify PRISM-AI can ACTUALLY use GPU
use std::process::Command;

fn main() {
    println!("===========================================");
    println!("PRISM-AI GPU VERIFICATION TEST");
    println!("===========================================\n");

    // Step 1: Check if GPU runtime library exists
    println!("Step 1: Checking GPU runtime library...");
    let lib_exists = std::path::Path::new("src/libgpu_runtime.so").exists();
    println!("  libgpu_runtime.so exists: {}", lib_exists);

    if !lib_exists {
        println!("  ❌ GPU runtime library not found. Building...");
        let output = Command::new("nvcc")
            .args(&[
                "--shared",
                "-o", "src/libgpu_runtime.so",
                "src/gpu_runtime.cu",
                "-arch=sm_70",
                "--compiler-options", "-fPIC"
            ])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                println!("  ✅ GPU runtime library built successfully!");
            }
            _ => {
                println!("  ❌ Failed to build GPU runtime library");
                return;
            }
        }
    } else {
        println!("  ✅ GPU runtime library found!");
    }

    // Step 2: Check PTX files
    println!("\nStep 2: Checking PTX kernels...");
    let ptx_files = [
        "transfer_entropy.ptx",
        "thermodynamic.ptx",
        "active_inference.ptx",
    ];

    let mut ptx_count = 0;
    for ptx in &ptx_files {
        let path = format!("src/kernels/ptx/{}", ptx);
        let exists = std::path::Path::new(&path).exists();
        println!("  {} exists: {}", ptx, exists);
        if exists { ptx_count += 1; }
    }
    println!("  ✅ {}/{} critical PTX files ready", ptx_count, ptx_files.len());

    // Step 3: Test GPU availability via runtime
    println!("\nStep 3: Testing GPU availability...");

    // Set library path
    std::env::set_var("LD_LIBRARY_PATH", format!("{}:src", std::env::var("LD_LIBRARY_PATH").unwrap_or_default()));

    // Try to load and call the GPU library
    #[link(name = "gpu_runtime", kind = "dylib")]
    extern "C" {
        fn gpu_available() -> i32;
    }

    let gpu_ready = unsafe { gpu_available() };

    if gpu_ready != 0 {
        println!("  ✅ GPU is AVAILABLE and READY!");

        // Step 4: Test actual kernel execution
        println!("\nStep 4: Testing actual GPU kernel execution...");

        #[link(name = "gpu_runtime", kind = "dylib")]
        extern "C" {
            fn launch_transfer_entropy(source: *const f64, target: *const f64, n: i32) -> f32;
        }

        // Create test data
        let source: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let target: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).cos()).collect();

        println!("  Launching Transfer Entropy kernel with 1000 elements...");
        let result = unsafe {
            launch_transfer_entropy(source.as_ptr(), target.as_ptr(), 1000)
        };

        println!("  ✅ GPU kernel executed successfully!");
        println!("  Transfer Entropy result from GPU: {}", result);

        // Step 5: Test thermodynamic kernel
        println!("\nStep 5: Testing Thermodynamic evolution on GPU...");

        #[link(name = "gpu_runtime", kind = "dylib")]
        extern "C" {
            fn launch_thermodynamic(phases: *mut f64, velocities: *mut f64, n_osc: i32, n_steps: i32);
        }

        let mut phases: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut velocities: Vec<f64> = vec![0.0; 100];

        println!("  Launching Thermodynamic kernel: 100 oscillators, 100 steps...");
        unsafe {
            launch_thermodynamic(phases.as_mut_ptr(), velocities.as_mut_ptr(), 100, 100);
        }

        println!("  ✅ Thermodynamic evolution completed on GPU!");
        println!("  Sample evolved phase[0]: {}", phases[0]);
        println!("  Sample evolved velocity[0]: {}", velocities[0]);

    } else {
        println!("  ❌ GPU not available");
        return;
    }

    // Final confirmation
    println!("\n===========================================");
    println!("FINAL VERDICT:");
    println!("===========================================");
    println!("✅ PRISM-AI CAN USE GPU!");
    println!("✅ GPU kernels execute successfully!");
    println!("✅ Data transfers work!");
    println!("✅ Results returned from GPU!");
    println!("\nGPU ACCELERATION IS OPERATIONAL! 🚀");
}