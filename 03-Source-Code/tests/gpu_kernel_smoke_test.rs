//! GPU Kernel Smoke Test - Minimal Test for Worker 2 Deliverables
//!
//! This is a simplified test that validates core GPU functionality
//! without pulling in cross-worker dependencies that cause build failures.
//!
//! Run with: cargo test --lib --features cuda gpu_kernel

#[cfg(feature = "cuda")]
#[cfg(test)]
mod gpu_smoke_tests {
    use prism_ai::gpu::kernel_executor::get_global_executor;
    use anyhow::Result;

    #[test]
    fn test_gpu_executor_initialization() -> Result<()> {
        let executor = get_global_executor()?;
        let _guard = executor.lock();
        println!("✅ GPU executor initialized successfully");
        Ok(())
    }

    #[test]
    fn test_basic_vector_ops() -> Result<()> {
        let executor = get_global_executor()?;
        let executor = executor.lock();

        // Vector addition
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = executor.vector_add(&a, &b)?;

        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 6.0);
        assert_eq!(c[3], 12.0);

        println!("✅ Vector addition works");
        Ok(())
    }

    #[test]
    fn test_activation_functions() -> Result<()> {
        let executor = get_global_executor()?;
        let executor = executor.lock();

        // ReLU
        let mut data = vec![-1.0, 2.0, -3.0, 4.0];
        executor.relu_inplace(&mut data)?;
        assert!(data.iter().all(|&x| x >= 0.0));
        assert_eq!(data[1], 2.0);
        assert_eq!(data[3], 4.0);

        // Sigmoid
        let data = vec![0.0, 1.0, -1.0];
        let result = executor.sigmoid(&data)?;
        assert!(result.iter().all(|&x| x >= 0.0 && x <= 1.0));

        println!("✅ Activation functions work");
        Ok(())
    }

    #[test]
    fn test_matrix_multiply() -> Result<()> {
        let executor = get_global_executor()?;
        let executor = executor.lock();

        // 2x2 identity-like matrix multiply
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = executor.matmul(&a, &b, 2, 2, 2)?;

        assert_eq!(c.len(), 4);
        // Identity * B = B
        assert!((c[0] - 2.0).abs() < 1e-5);
        assert!((c[1] - 3.0).abs() < 1e-5);

        println!("✅ Matrix multiplication works");
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[cfg(test)]
mod no_cuda_warning {
    #[test]
    fn warn_no_cuda() {
        println!("⚠️  CUDA feature not enabled. GPU tests skipped.");
        println!("   Run: cargo test --features cuda");
    }
}
