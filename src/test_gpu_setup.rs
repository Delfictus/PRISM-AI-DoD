#!/usr/bin/env rust-script
//! Test GPU setup for PRISM-AI CUDA functionality
//!
//! ```cargo
//! [dependencies]
//! cudarc = "0.17"
//! ```

fn main() {
    println!("=" .repeat(70));
    println!("PRISM-AI GPU/CUDA Setup Validation");
    println!("=" .repeat(70));
    println!();

    // Try using cudarc to detect GPUs
    match cudarc::driver::CudaDevice::new(0) {
        Ok(device) => {
            println!("✅ SUCCESS: CUDA device detected via cudarc!");
            println!();

            // Get device properties
            if let Ok(name) = device.name() {
                println!("  Device Name: {}", name);
            }

            if let Ok(major) = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) {
                if let Ok(minor) = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) {
                    println!("  Compute Capability: {}.{}", major, minor);
                }
            }

            if let Ok(mp_count) = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) {
                println!("  Multiprocessors: {}", mp_count);
            }

            // Test memory allocation
            println!();
            println!("Testing GPU memory allocation...");
            match device.alloc_zeros::<f32>(1024) {
                Ok(_mem) => {
                    println!("✅ GPU memory allocation successful!");
                },
                Err(e) => {
                    println!("❌ GPU memory allocation failed: {}", e);
                }
            }

            println!();
            println!("🎉 Your GPU is ready for PRISM-AI CUDA operations!");
        },
        Err(e) => {
            println!("❌ FAILED: Could not initialize CUDA device");
            println!("  Error: {}", e);
            println!();
            println!("Troubleshooting:");
            println!("  1. nvidia-smi shows your GPU is present");
            println!("  2. CUDA 12.8 is installed");
            println!("  3. But CUDA runtime cannot access the device");
            println!();
            println!("Possible solutions:");
            println!("  - Restart your system to reload drivers");
            println!("  - Check if another process is using the GPU exclusively");
            println!("  - Try: nvidia-smi -pm 1 (persistence mode)");
            println!("  - Check dmesg for driver errors: dmesg | grep nvidia");
        }
    }

    println!();
    println!("=" .repeat(70));
}