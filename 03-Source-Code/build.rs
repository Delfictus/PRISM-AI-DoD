use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/cutlass_protein_kernels.cu");

    // Only compile CUDA kernels if cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        compile_cuda_kernels();
    }
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::process::Command;

    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc_path = format!("{}/bin/nvcc", cuda_path);
    let cutlass_path = env::var("CUTLASS_PATH")
        .unwrap_or_else(|_| format!("{}/.cutlass", env::var("HOME").unwrap_or_default()));

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    // cuDNN is optional - only needed for training
    // println!("cargo:rustc-link-lib=cudnn");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile CUTLASS protein kernels
    let kernel_file = "kernels/cutlass_protein_kernels.cu";
    let output_file = out_dir.join("cutlass_protein_kernels.ptx");

    println!("Compiling CUDA kernels: {} -> {:?}", kernel_file, output_file);

    // Detect GPU architecture
    let arch = detect_gpu_arch();
    println!("Detected GPU architecture: {}", arch);

    // nvcc compilation command
    let status = Command::new(&nvcc_path)
        .args(&[
            kernel_file,
            "-o", output_file.to_str().unwrap(),
            &format!("-arch={}", arch),
            "-ptx",                          // Generate PTX for runtime compilation
            "--use_fast_math",               // Enable fast math
            "-O3",                           // Maximum optimization
            "-std=c++17",                    // C++17 standard
            &format!("-I{}/include", cutlass_path),  // CUTLASS headers
            &format!("-I{}/include", cuda_path),     // CUDA headers
            "-DCUDA_ENABLED",
            "--expt-relaxed-constexpr",      // Relaxed constexpr for CUTLASS
            "--extended-lambda",             // Extended lambda support
        ])
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("CUDA kernel compilation successful!");

            // Copy PTX file to a predictable location
            let target_ptx = "target/cutlass_protein_kernels.ptx";
            std::fs::copy(&output_file, target_ptx)
                .expect("Failed to copy PTX file to target");

            println!("PTX file available at: {}", target_ptx);
        }
        Ok(status) => {
            eprintln!("CUDA kernel compilation failed with status: {}", status);
            eprintln!("Note: CUTLASS 3.8 headers required at: {}", cutlass_path);
            eprintln!("Download from: https://github.com/NVIDIA/cutlass/releases/tag/v3.8.0");
        }
        Err(e) => {
            eprintln!("Failed to execute nvcc: {}", e);
            eprintln!("Make sure CUDA Toolkit 12.0+ is installed");
            eprintln!("Set CUDA_PATH environment variable if needed");
        }
    }
}

#[cfg(feature = "cuda")]
fn detect_gpu_arch() -> String {
    use std::process::Command;

    // Try to detect GPU compute capability
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();

    if let Ok(output) = output {
        if let Ok(cap_str) = String::from_utf8(output.stdout) {
            let cap = cap_str.trim().replace(".", "");
            if !cap.is_empty() {
                // Map compute capability to architecture
                return match cap.as_str() {
                    "86" => "sm_86".to_string(),  // RTX 3090, A6000
                    "89" => "sm_89".to_string(),  // RTX 4090, L40
                    "90" => "sm_90".to_string(),  // H100 (Hopper)
                    "100" => "sm_100".to_string(), // B100 (Blackwell)
                    _ => format!("sm_{}", cap),
                };
            }
        }
    }

    // Fallback: Try reading from cuda-config
    let cuda_config = env::var("CUDA_ARCH").ok();
    if let Some(arch) = cuda_config {
        return arch;
    }

    // Default to Ampere (sm_86) for RTX 30 series
    println!("cargo:warning=Could not detect GPU architecture, defaulting to sm_86 (Ampere)");
    "sm_86".to_string()
}