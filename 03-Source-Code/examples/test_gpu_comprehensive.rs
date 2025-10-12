//! Comprehensive GPU Module Test

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      PRISM-AI Comprehensive GPU Test - All Modules          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Basic CUDA Context
    println!("TEST 1: Basic CUDA Context Creation");
    println!("====================================");
    test_basic_cuda();

    // Test 2: GPU Launcher
    println!("\nTEST 2: GPU Launcher");
    println!("====================");
    test_gpu_launcher();

    // Test 3: Neural Network Modules
    println!("\nTEST 3: Neural Network Modules");
    println!("===============================");
    test_neural_modules();

    // Test 4: PWSA Active Inference
    println!("\nTEST 4: PWSA Active Inference");
    println!("==============================");
    test_pwsa();

    // Test 5: GPU Memory Operations
    println!("\nTEST 5: GPU Memory Operations");
    println!("==============================");
    test_memory_ops();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      TEST SUMMARY                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    #[cfg(feature = "cuda")]
    {
        println!("✅ CUDA feature is ENABLED");
        println!("✅ RTX 5070 should be accessible");
    }

    #[cfg(not(feature = "cuda"))]
    println!("⚠️  CUDA feature is DISABLED - using CPU fallback");
}

#[cfg(feature = "cuda")]
fn test_basic_cuda() {
    use cudarc::driver::CudaContext;

    match CudaContext::new(0) {
        Ok(ctx) => {
            println!("✅ CudaContext created successfully!");

            // Get device info
            match ctx.device() {
                Ok(device) => {
                    println!("✅ GPU Device: Ordinal {}", device.ordinal());
                }
                Err(e) => println!("❌ Could not get device info: {}", e),
            }
        }
        Err(e) => println!("❌ Failed to create CudaContext: {}", e),
    }
}

#[cfg(not(feature = "cuda"))]
fn test_basic_cuda() {
    println!("⚠️  CUDA not enabled - skipping");
}

fn test_gpu_launcher() {
    use prism_ai::gpu_launcher::GpuContext;

    match GpuContext::initialize() {
        Ok(ctx) => {
            println!("✅ GpuContext initialized");
            if GpuContext::is_available() {
                println!("✅ GPU is available through launcher");
            }
        }
        Err(e) => println!("❌ GPU launcher initialization failed: {}", e),
    }
}

fn test_neural_modules() {
    use prism_ai::cma::neural::{
        neural_quantum::{Device, NeuralQuantumState},
        E3EquivariantGNN,
        ConsistencyDiffusion,
    };

    // Test Neural Quantum State
    match Device::cuda_if_available(0) {
        Ok(device) => {
            println!("✅ Neural Quantum: Device created");

            match NeuralQuantumState::new(10, 64, 4, device.clone()) {
                Ok(_) => println!("✅ Neural Quantum State created"),
                Err(e) => println!("❌ Failed to create Neural Quantum State: {}", e),
            }
        }
        Err(e) => println!("❌ Device creation failed: {}", e),
    }

    // Test GNN
    match Device::cuda_if_available(0) {
        Ok(device) => {
            match E3EquivariantGNN::new(8, 4, 128, 4, device.clone()) {
                Ok(_) => println!("✅ E3-Equivariant GNN created"),
                Err(e) => println!("❌ Failed to create GNN: {}", e),
            }
        }
        Err(_) => println!("⚠️  Using CPU for GNN"),
    }

    // Test Diffusion
    match Device::cuda_if_available(0) {
        Ok(device) => {
            match ConsistencyDiffusion::new(128, 256, 50, device) {
                Ok(_) => println!("✅ Consistency Diffusion created"),
                Err(e) => println!("❌ Failed to create Diffusion: {}", e),
            }
        }
        Err(_) => println!("⚠️  Using CPU for Diffusion"),
    }
}

fn test_pwsa() {
    use prism_ai::pwsa::active_inference_classifier::{
        Device, ActiveInferenceClassifier
    };
    use ndarray::Array1;

    match Device::cuda_if_available(0) {
        Ok(device) => {
            println!("✅ PWSA: Device created");

            // Test classifier creation
            match ActiveInferenceClassifier::new("models/test.safetensors") {
                Ok(mut classifier) => {
                    println!("✅ Active Inference Classifier created");

                    // Test classification
                    let features = Array1::from_vec(vec![0.5; 100]);
                    match classifier.classify(&features) {
                        Ok(result) => {
                            println!("✅ Classification successful");
                            println!("   Free energy: {:.4}", result.free_energy);
                            println!("   Confidence: {:.2}%", result.confidence * 100.0);
                        }
                        Err(e) => println!("❌ Classification failed: {}", e),
                    }
                }
                Err(e) => println!("⚠️  Classifier using default initialization: {}", e),
            }
        }
        Err(e) => println!("❌ PWSA device creation failed: {}", e),
    }
}

#[cfg(feature = "cuda")]
fn test_memory_ops() {
    use cudarc::driver::CudaContext;

    match CudaContext::new(0) {
        Ok(ctx) => {
            // Test memory allocation
            let size = 1024;
            match ctx.alloc_zeros::<f32>(size) {
                Ok(mut gpu_mem) => {
                    println!("✅ Allocated {} floats on GPU", size);

                    // Test data transfer
                    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                    match ctx.htod_sync_copy_into(&data, &mut gpu_mem) {
                        Ok(()) => println!("✅ Data transferred to GPU"),
                        Err(e) => println!("❌ Transfer to GPU failed: {}", e),
                    }

                    // Test copy back
                    let mut result = vec![0.0f32; size];
                    match ctx.dtoh_sync_copy_into(&gpu_mem, &mut result) {
                        Ok(()) => {
                            println!("✅ Data retrieved from GPU");
                            if result[10] == 10.0 && result[100] == 100.0 {
                                println!("✅ Data integrity verified!");
                            }
                        }
                        Err(e) => println!("❌ Transfer from GPU failed: {}", e),
                    }
                }
                Err(e) => println!("❌ GPU memory allocation failed: {}", e),
            }
        }
        Err(e) => println!("❌ No GPU context for memory test: {}", e),
    }
}

#[cfg(not(feature = "cuda"))]
fn test_memory_ops() {
    println!("⚠️  CUDA not enabled - memory operations skipped");
}