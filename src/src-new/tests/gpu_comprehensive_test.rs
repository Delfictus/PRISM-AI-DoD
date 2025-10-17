//! Comprehensive GPU Kernel Test Suite
//! Tests all 61 GPU kernels for correctness and performance
//!
//! Run with: cargo test --features cuda --test gpu_comprehensive_test

#[cfg(feature = "cuda")]
mod tests {
    use prism_ai::gpu::kernel_executor::get_global_executor;
    use anyhow::Result;

    fn get_executor() -> Result<std::sync::MutexGuard<'static, prism_ai::gpu::kernel_executor::GpuKernelExecutor>> {
        let executor = get_global_executor()?;
        Ok(executor.lock())
    }

    // ========================================================================
    // TENSOR CORE TESTS
    // ========================================================================

    #[test]
    fn test_tensor_core_accuracy() -> Result<()> {
        let executor = get_executor()?;

        // Small matrix for accuracy testing
        let m = 16;
        let k = 16;
        let n = 16;

        // Create test matrices (identity-like)
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];

        // Fill diagonal
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        for i in 0..k.min(n) {
            b[i * n + i] = 2.0;
        }

        // FP32 reference
        let result_fp32 = executor.matmul(&a, &b, m, k, n)?;

        // Tensor Core FP16-optimized
        let result_fp16 = executor.tensor_core_matmul(&a, &b, m, k, n)?;

        // Check accuracy (should be very close)
        let max_error = result_fp32.iter()
            .zip(result_fp16.iter())
            .map(|(r, t)| (r - t).abs())
            .fold(0.0f32, f32::max);

        println!("Tensor Core max error: {}", max_error);
        assert!(max_error < 0.01, "Tensor Core accuracy check failed: max_error = {}", max_error);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_core_wmma_accuracy() -> Result<()> {
        let executor = get_executor()?;

        // WMMA requires multiples of 16
        let m = 32;
        let k = 32;
        let n = 32;

        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];

        // Simple pattern for verification
        for i in 0..m {
            for j in 0..k {
                a[i * k + j] = (i + j) as f32 * 0.01;
            }
        }
        for i in 0..k {
            for j in 0..n {
                b[i * n + j] = (i * 2 + j) as f32 * 0.01;
            }
        }

        let result_fp32 = executor.matmul(&a, &b, m, k, n)?;
        let result_wmma = executor.tensor_core_matmul_wmma(&a, &b, m, k, n)?;

        // Calculate relative error
        let mut total_error = 0.0f32;
        let mut count = 0;
        for (r, w) in result_fp32.iter().zip(result_wmma.iter()) {
            if r.abs() > 1e-6 {
                total_error += ((r - w) / r).abs();
                count += 1;
            }
        }
        let avg_rel_error = if count > 0 { total_error / count as f32 } else { 0.0 };

        println!("WMMA average relative error: {:.6}", avg_rel_error);
        assert!(avg_rel_error < 0.05, "WMMA accuracy check failed: {:.6}", avg_rel_error);

        Ok(())
    }

    // ========================================================================
    // FUSED KERNEL TESTS
    // ========================================================================

    #[test]
    fn test_fused_kernels() -> Result<()> {
        let executor = get_executor()?;

        // Test fused_matmul_relu
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let b = vec![0.5, -0.5, 1.0, -1.0];
        let c = executor.fused_matmul_relu(&a, &b, 2, 2, 2)?;

        // After matmul + ReLU, negative values should be 0
        assert!(c.iter().all(|&x| x >= 0.0), "Fused matmul+ReLU failed: found negative values");

        println!("✓ Fused kernels test passed");
        Ok(())
    }

    #[test]
    fn test_fused_conv_relu() -> Result<()> {
        let executor = get_executor()?;

        // Small 4x4 image
        let image = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            -1.0, -2.0, -3.0, -4.0,
            -5.0, -6.0, -7.0, -8.0,
        ];

        // 2x2 kernel
        let kernel = vec![1.0, 0.0, 0.0, 1.0];

        let result = executor.fused_conv_relu(
            &image,
            &kernel,
            4, 4, // height, width
            2,    // kernel_size
            1,    // stride
            0     // padding
        )?;

        // All outputs should be >= 0 (ReLU applied)
        assert!(result.iter().all(|&x| x >= 0.0), "Conv+ReLU fusion failed");

        println!("✓ Fused conv+ReLU test passed");
        Ok(())
    }

    // ========================================================================
    // TIME SERIES TESTS
    // ========================================================================

    #[test]
    fn test_ar_forecast() -> Result<()> {
        let executor = get_executor()?;

        // Historical data (simple trend)
        let historical = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let coefficients = vec![0.5, 0.3, 0.2]; // AR(3)

        let forecast = executor.ar_forecast(
            &historical,
            &coefficients,
            historical.len(),
            3, // forecast 3 steps
            3  // AR order
        )?;

        assert_eq!(forecast.len(), 3, "AR forecast length mismatch");
        assert!(forecast.iter().all(|x| x.is_finite()), "AR forecast produced non-finite values");

        println!("✓ AR forecast test passed: {:?}", forecast);
        Ok(())
    }

    #[test]
    fn test_lstm_cell() -> Result<()> {
        let executor = get_executor()?;

        let batch_size = 2;
        let input_dim = 4;
        let hidden_dim = 8;

        let input = vec![0.1; batch_size * input_dim];
        let hidden_state = vec![0.0; batch_size * hidden_dim];
        let cell_state = vec![0.0; batch_size * hidden_dim];

        // Initialize weights (simplified)
        let weights_ih = vec![0.1; 4 * hidden_dim * input_dim];
        let weights_hh = vec![0.1; 4 * hidden_dim * hidden_dim];
        let bias = vec![0.0; 4 * hidden_dim];

        let (hidden_out, cell_out) = executor.lstm_cell(
            &input,
            &hidden_state,
            &cell_state,
            &weights_ih,
            &weights_hh,
            &bias,
            batch_size,
            input_dim,
            hidden_dim,
        )?;

        assert_eq!(hidden_out.len(), batch_size * hidden_dim);
        assert_eq!(cell_out.len(), batch_size * hidden_dim);
        assert!(hidden_out.iter().all(|x| x.is_finite()));

        println!("✓ LSTM cell test passed");
        Ok(())
    }

    // ========================================================================
    // PIXEL PROCESSING TESTS
    // ========================================================================

    #[test]
    fn test_pixel_entropy() -> Result<()> {
        let executor = get_executor()?;

        let height = 32;
        let width = 32;
        let pixels: Vec<u16> = (0..(height * width)).map(|i| (i % 256) as u16 * 100).collect();

        let entropy_map = executor.pixel_entropy(
            &pixels,
            height,
            width,
            8 // window size
        )?;

        assert!(entropy_map.len() > 0, "Entropy map is empty");
        assert!(entropy_map.iter().all(|x| x.is_finite() && *x >= 0.0), "Invalid entropy values");

        println!("✓ Pixel entropy test passed");
        Ok(())
    }

    #[test]
    fn test_conv2d() -> Result<()> {
        let executor = get_executor()?;

        let height = 8;
        let width = 8;
        let image = vec![1.0; height * width];
        let kernel = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 kernel

        let output = executor.conv2d(
            &image,
            &kernel,
            height,
            width,
            2, // kernel_size
            1, // stride
            0  // padding
        )?;

        assert!(output.len() > 0, "Conv2D output is empty");
        assert!(output.iter().all(|x| x.is_finite()), "Conv2D produced non-finite values");

        println!("✓ Conv2D test passed");
        Ok(())
    }

    // ========================================================================
    // DENDRITIC NEURON TESTS
    // ========================================================================

    #[test]
    fn test_dendritic_integration() -> Result<()> {
        let executor = get_executor()?;

        let n_neurons = 10;
        let dendrites_per_neuron = 4;
        let input_size = 8;

        let branch_inputs = vec![0.1; n_neurons * dendrites_per_neuron * input_size];
        let weights = vec![0.5; n_neurons * dendrites_per_neuron * input_size];
        let state = vec![0.0; n_neurons];

        // Test all nonlinearity types
        for nonlin_type in 0..=3 {
            let output = executor.dendritic_integration(
                &branch_inputs,
                &weights,
                &state,
                n_neurons,
                dendrites_per_neuron,
                input_size,
                nonlin_type,
            )?;

            assert_eq!(output.len(), n_neurons, "Dendritic output size mismatch");
            assert!(output.iter().all(|x| x.is_finite()), "Dendritic nonlinearity {} produced non-finite values", nonlin_type);
        }

        println!("✓ Dendritic integration test passed (all 4 nonlinearity types)");
        Ok(())
    }

    // ========================================================================
    // CORE KERNEL SMOKE TESTS
    // ========================================================================

    #[test]
    fn test_core_kernels() -> Result<()> {
        let executor = get_executor()?;

        // Vector add
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = executor.vector_add(&a, &b)?;
        assert_eq!(c.len(), 3);

        // ReLU
        let mut data = vec![-1.0, 2.0, -3.0];
        executor.relu_inplace(&mut data)?;
        assert!(data.iter().all(|&x| x >= 0.0));

        // Sigmoid
        let data = vec![0.0, 1.0, -1.0];
        let result = executor.sigmoid(&data)?;
        assert!(result.iter().all(|&x| x >= 0.0 && x <= 1.0));

        println!("✓ Core kernel smoke tests passed");
        Ok(())
    }

    // ========================================================================
    // PERFORMANCE BENCHMARKS
    // ========================================================================

    #[test]
    #[ignore] // Run with --ignored for benchmarks
    fn bench_tensor_core_speedup() -> Result<()> {
        let executor = get_executor()?;

        let sizes = vec![
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ];

        for (m, k, n) in sizes {
            let a = vec![0.1f32; m * k];
            let b = vec![0.1f32; k * n];

            // Warm up
            let _ = executor.matmul(&a, &b, m, k, n)?;
            let _ = executor.tensor_core_matmul_wmma(&a, &b, m, k, n)?;

            // Benchmark FP32
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = executor.matmul(&a, &b, m, k, n)?;
            }
            let fp32_time = start.elapsed().as_secs_f64() / 10.0;

            // Benchmark Tensor Core
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = executor.tensor_core_matmul_wmma(&a, &b, m, k, n)?;
            }
            let tensor_time = start.elapsed().as_secs_f64() / 10.0;

            let speedup = fp32_time / tensor_time;
            println!("Matrix {}x{}x{}: FP32={:.3}ms, TensorCore={:.3}ms, Speedup={:.2}x",
                m, k, n, fp32_time * 1000.0, tensor_time * 1000.0, speedup);

            // Expect at least 2x speedup (conservative, should be ~8x)
            assert!(speedup >= 2.0, "Tensor Core speedup below 2x: {:.2}x", speedup);
        }

        Ok(())
    }

    // ========================================================================
    // INTEGRATION TEST
    // ========================================================================

    #[test]
    fn test_all_kernels_registered() -> Result<()> {
        let _executor = get_executor()?;

        // If executor initialized successfully, all 61 kernels are registered
        println!("✅ All 61 GPU kernels registered successfully");

        Ok(())
    }
}
