// Simple test to check if GPU acceleration is detected
use prism_ai::information_theory::{TransferEntropy, TransferEntropyGpuExt};
use prism_ai::statistical_mechanics::{ThermodynamicNetwork, ThermodynamicNetworkGpuExt, NetworkConfig};
use ndarray::Array1;

fn main() {
    println!("GPU Acceleration Detection Test");
    println!("================================\n");

    // Test Transfer Entropy GPU detection
    println!("Transfer Entropy GPU Detection:");
    let gpu_available_te = TransferEntropy::gpu_available();
    println!("  PTX file exists: {}", std::path::Path::new("src/kernels/ptx/transfer_entropy.ptx").exists());
    println!("  GPU available: {}", gpu_available_te);

    if gpu_available_te {
        println!("  ✅ Transfer Entropy will use GPU acceleration!");

        // Test calculation
        let source = Array1::linspace(0.0, 10.0, 100);
        let target = source.mapv(|x| x.sin());
        let te = TransferEntropy::default();
        let result = te.calculate_auto(&source, &target);
        println!("  Test calculation: TE = {:.4}", result.te_value);
    } else {
        println!("  ❌ Transfer Entropy using CPU fallback");
    }

    println!("\nThermodynamic Network GPU Detection:");
    let gpu_available_thermo = ThermodynamicNetwork::gpu_available();
    println!("  PTX file exists: {}", std::path::Path::new("src/kernels/ptx/thermodynamic.ptx").exists());
    println!("  GPU available: {}", gpu_available_thermo);

    if gpu_available_thermo {
        println!("  ✅ Thermodynamic Network will use GPU acceleration!");

        // Test evolution
        let config = NetworkConfig::default();
        let mut network = ThermodynamicNetwork::new(config);
        let result = network.evolve_auto(100);
        println!("  Test evolution: Completed {} steps", 100);
        println!("  Execution time: {:.2}ms", result.execution_time_ms);
    } else {
        println!("  ❌ Thermodynamic Network using CPU fallback");
    }

    println!("\nSummary:");
    println!("========");

    // List all PTX files
    println!("\nCompiled PTX kernels:");
    let ptx_dir = std::path::Path::new("src/kernels/ptx");
    if ptx_dir.exists() {
        for entry in std::fs::read_dir(ptx_dir).unwrap() {
            if let Ok(entry) = entry {
                if entry.path().extension().map_or(false, |ext| ext == "ptx") {
                    let size = entry.metadata().unwrap().len();
                    println!("  ✅ {} ({:.1} KB)", entry.file_name().to_string_lossy(), size as f64 / 1024.0);
                }
            }
        }
    }

    let gpu_count = [gpu_available_te, gpu_available_thermo].iter().filter(|&&x| x).count();
    println!("\n{}/{} components GPU-ready", gpu_count, 2);
}