// Build script to compile CUDA kernels with nvcc
// This runs during `cargo build` and produces PTX files for Tensor Core operations

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA if the cuda feature is enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rerun-if-changed=cuda_kernels/tensor_core_matmul.cu");
        println!("cargo:rerun-if-changed=cuda/neuromorphic_kernels.cu");
        println!("cargo:rerun-if-changed=src/gpu/cublas_interposer.c");

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        // CRITICAL: Compile and link CUBLAS shim library FIRST
        // This provides the missing symbols for CUDA 12.8 compatibility
        compile_cublas_shim(&out_dir);

        // Detect CUDA installation
        let nvcc = find_nvcc().expect("nvcc not found. Please install CUDA Toolkit.");

        println!("cargo:warning=Compiling CUDA kernels with nvcc: {}", nvcc);

        // Detect GPU architecture
        let arch = detect_gpu_arch().unwrap_or_else(|| {
            println!("cargo:warning=Could not detect GPU, using sm_89 (Ada Lovelace)");
            "sm_89".to_string()
        });

        println!("cargo:warning=Compiling for GPU architecture: {}", arch);

        // Compile Tensor Core kernels to PTX
        compile_cuda_kernel(
            &nvcc,
            "cuda_kernels/tensor_core_matmul.cu",
            &out_dir.join("tensor_core_matmul.ptx"),
            &arch,
        );

        println!("cargo:rustc-env=TENSOR_CORE_PTX_PATH={}", out_dir.join("tensor_core_matmul.ptx").display());

        // Compile Neuromorphic kernels to shared library
        compile_neuromorphic_kernels(&nvcc, &out_dir, &arch);
    }
}

fn compile_cublas_shim(out_dir: &PathBuf) {
    println!("cargo:warning=Compiling CUBLAS interposer library for CUDA 12.8 compatibility");

    let source = "src/gpu/cublas_interposer.c";
    let output = out_dir.join("libcublas_interposer.so");

    // Compile the interposer library with gcc, linking against dl for dlopen/dlsym
    let status = Command::new("gcc")
        .args(&[
            "-shared",
            "-fPIC",
            "-o", output.to_str().unwrap(),
            source,
            "-ldl",  // Link against libdl for dlopen/dlsym
            "-D_GNU_SOURCE",
        ])
        .status()
        .expect("Failed to compile CUBLAS interposer library");

    if !status.success() {
        panic!("Failed to compile CUBLAS interposer library");
    }

    println!("cargo:warning=Successfully compiled CUBLAS interposer library");
    println!("cargo:warning=Interposer library at: {}", output.display());

    // Use LD_PRELOAD to inject our interposer
    println!("cargo:rustc-env=LD_PRELOAD={}", output.display());

    // Also export path for manual testing
    println!("cargo:rustc-env=CUBLAS_INTERPOSER_PATH={}", output.display());
}

fn compile_cuda_kernel(nvcc: &str, source: &str, output: &PathBuf, arch: &str) {
    println!("cargo:warning=Compiling {}", source);

    let status = Command::new(nvcc)
        .args(&[
            "--ptx",                          // Generate PTX instead of binary
            source,
            "-o", output.to_str().unwrap(),
            &format!("-arch={}", arch),       // Target architecture
            "-O3",                            // Optimization level
            "--use_fast_math",                // Fast math for better performance
            "-std=c++17",                     // C++17 for modern CUDA
            "-I/usr/local/cuda/include",      // CUDA include path
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed to compile {}", source);
    }

    println!("cargo:warning=Successfully compiled {} to PTX", source);
}

fn compile_neuromorphic_kernels(nvcc: &str, out_dir: &PathBuf, arch: &str) {
    println!("cargo:warning=Compiling neuromorphic kernels to shared library");

    let source = "cuda/neuromorphic_kernels.cu";
    let output = out_dir.join("libneuromorphic_kernels.so");

    let status = Command::new(nvcc)
        .args(&[
            "--shared",                       // Build shared library
            "-Xcompiler", "-fPIC",            // Position independent code
            source,
            "-o", output.to_str().unwrap(),
            &format!("-arch={}", arch),       // Target architecture
            "-O3",                            // Optimization level
            "--use_fast_math",                // Fast math for better performance
            "-std=c++17",                     // C++17 for modern CUDA
            "-I/usr/local/cuda/include",      // CUDA include path
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed to compile neuromorphic kernels");
    }

    println!("cargo:warning=Successfully compiled neuromorphic kernels to shared library");
    println!("cargo:warning=Library: {}", output.display());

    // Tell cargo where to find the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=neuromorphic_kernels");
    println!("cargo:rustc-env=NEUROMORPHIC_KERNELS_PATH={}", output.display());
}

fn find_nvcc() -> Option<String> {
    // Try common locations for nvcc
    let common_paths = vec![
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "nvcc",  // In PATH
    ];

    for path in common_paths {
        if Command::new(path).arg("--version").status().is_ok() {
            return Some(path.to_string());
        }
    }

    // Try to find via which
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            if let Ok(path) = String::from_utf8(output.stdout) {
                return Some(path.trim().to_string());
            }
        }
    }

    None
}

fn detect_gpu_arch() -> Option<String> {
    // Try to detect GPU compute capability using nvidia-smi
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if output.status.success() {
        let cap = String::from_utf8(output.stdout).ok()?;
        let cap = cap.trim().replace(".", "");

        // Map compute capability to SM architecture
        // 7.0 -> sm_70 (Volta)
        // 7.5 -> sm_75 (Turing)
        // 8.0 -> sm_80 (Ampere)
        // 8.6 -> sm_86 (Ampere)
        // 8.9 -> sm_89 (Ada Lovelace - but use sm_90 as sm_89 may not exist)
        // 9.0 -> sm_90 (Hopper)

        // Note: For compute 12.0, we should use sm_90 (closest available)
        if cap == "120" {
            println!("cargo:warning=Detected Compute 12.0, using sm_90");
            return Some("sm_90".to_string());
        }

        return Some(format!("sm_{}", cap));
    }

    None
}
