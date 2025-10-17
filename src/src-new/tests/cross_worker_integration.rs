//! Cross-Worker Integration Tests
//!
//! Tests GPU kernels with realistic workloads from Workers 1, 3, 5, 6, 7.
//! Validates performance on production-sized data.
//!
//! Run with: cargo test --test cross_worker_integration --features cuda --  --ignored

use prism_ai::gpu::kernel_executor::get_global_executor;
use anyhow::Result;

// ============================================================================
// WORKER 1: TIME SERIES FORECASTING
// ============================================================================

#[test]
#[ignore] // Requires CUDA device
fn test_worker1_ar_forecasting_realistic() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    // Realistic financial time series (1000 points)
    let n = 1000;
    let time_series: Vec<f32> = (0..n).map(|i| {
        let trend = 0.01 * i as f32;
        let seasonal = 10.0 * ((i as f32 * 0.1).sin());
        100.0 + trend + seasonal
    }).collect();

    // AR(5) model
    let coeffs: Vec<f32> = vec![0.5, -0.3, 0.2, -0.1, 0.05];
    let horizon = 10;

    let forecast = executor.ar_forecast(&time_series, &coeffs, horizon)?;

    assert_eq!(forecast.len(), horizon);
    assert!(forecast.iter().all(|&x| x.is_finite()));

    println!("✅ Worker 1 AR: forecast[0]={:.2}, range=[{:.2}, {:.2}]",
             forecast[0], forecast.iter().cloned().fold(f32::INFINITY, f32::min),
             forecast.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    Ok(())
}

#[test]
#[ignore] // Requires CUDA device
fn test_worker1_lstm_production_scale() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let batch_size = 32;
    let input_size = 64;
    let hidden_size = 128;

    let input: Vec<f32> = vec![0.1; batch_size * input_size];
    let h_prev: Vec<f32> = vec![0.0; batch_size * hidden_size];
    let c_prev: Vec<f32> = vec![0.0; batch_size * hidden_size];
    let weights_ih: Vec<f32> = vec![0.01; 4 * hidden_size * input_size];
    let weights_hh: Vec<f32> = vec![0.01; 4 * hidden_size * hidden_size];
    let bias: Vec<f32> = vec![0.0; 4 * hidden_size];

    let (h_new, c_new) = executor.lstm_cell_forward(
        &input,
        &h_prev,
        &c_prev,
        &weights_ih,
        &weights_hh,
        &bias,
        batch_size,
        input_size,
        hidden_size,
    )?;

    assert_eq!(h_new.len(), batch_size * hidden_size);
    assert_eq!(c_new.len(), batch_size * hidden_size);
    assert!(h_new.iter().all(|&x| x.is_finite()));

    println!("✅ Worker 1 LSTM: processed batch={}, hidden={}", batch_size, hidden_size);
    Ok(())
}

// ============================================================================
// WORKER 3: PIXEL PROCESSING (PWSA)
// ============================================================================

#[test]
#[ignore] // Requires CUDA device
fn test_worker3_ir_image_entropy_512x512() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    // Production IR image: 512x512
    let height = 512;
    let width = 512;
    let n_pixels = height * width;

    // Simulate IR image with concentrated hotspot
    let mut image: Vec<f32> = vec![100.0; n_pixels];
    for y in 200..300 {
        for x in 200..300 {
            image[y * width + x] = 5000.0; // Missile plume
        }
    }

    let window_size = 16;
    let entropy_map = executor.pixel_entropy(&image, height, width, window_size)?;

    assert_eq!(entropy_map.len(), n_pixels);
    assert!(entropy_map.iter().all(|&x| x >= 0.0 && x <= 1.0));

    let hotspot_entropy = entropy_map[250 * width + 250];
    let background_entropy = entropy_map[50 * width + 50];

    println!("✅ Worker 3 Pixel Entropy: hotspot={:.4}, background={:.4} (512x512)",
             hotspot_entropy, background_entropy);
    Ok(())
}

#[test]
#[ignore] // Requires CUDA device
fn test_worker3_conv2d_edge_detection() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let height = 128;
    let width = 128;
    let n_pixels = height * width;

    // Vertical edge pattern
    let mut image: Vec<f32> = vec![0.0; n_pixels];
    for y in 0..height {
        for x in 64..width {
            image[y * width + x] = 1.0;
        }
    }

    // Sobel kernel (vertical edges)
    let kernel: Vec<f32> = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;

    let edges = executor.conv2d(&image, &kernel, height, width, kernel_size, stride, padding)?;

    assert!(edges.len() > 0);
    let edge_count = edges.iter().filter(|&&x| x.abs() > 0.1).count();

    println!("✅ Worker 3 Conv2D: {} strong edges detected (128x128)", edge_count);
    Ok(())
}

#[test]
#[ignore] // Requires CUDA device
fn test_worker3_image_segmentation_regions() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let height = 256;
    let width = 256;
    let n_pixels = height * width;

    // Multi-region image
    let mut image: Vec<f32> = vec![0.0; n_pixels];
    for y in 0..height {
        for x in 0..width {
            if x < width / 3 {
                image[y * width + x] = 50.0;
            } else if x < 2 * width / 3 {
                image[y * width + x] = 150.0;
            } else {
                image[y * width + x] = 250.0;
            }
        }
    }

    let threshold = 100.0;
    let segments = executor.image_segmentation(&image, height, width, threshold)?;

    let mut unique_segments = segments.clone();
    unique_segments.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_segments.dedup();

    println!("✅ Worker 3 Segmentation: {} regions (256x256)", unique_segments.len());
    Ok(())
}

// ============================================================================
// WORKER 7: DENDRITIC NEURONS
// ============================================================================

#[test]
#[ignore] // Requires CUDA device
fn test_worker7_dendritic_sigmoid() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let n_neurons = 1;
    let dendrites_per_neuron = 10;
    let input_size = 20;
    let dendrite_inputs: Vec<f32> = vec![0.5; n_neurons * dendrites_per_neuron * input_size];
    let dendrite_weights: Vec<f32> = vec![0.1; n_neurons * dendrites_per_neuron * input_size];
    let soma_state: Vec<f32> = vec![0.0; n_neurons];
    let nonlinearity = 0; // Sigmoid

    let output = executor.dendritic_integration(
        &dendrite_inputs,
        &dendrite_weights,
        &soma_state,
        n_neurons,
        dendrites_per_neuron,
        input_size,
        nonlinearity,
    )?;

    assert_eq!(output.len(), n_neurons);
    assert!(output[0] >= 0.0 && output[0] <= 1.0);

    println!("✅ Worker 7 Dendritic (Sigmoid): output={:.4}", output[0]);
    Ok(())
}

#[test]
#[ignore] // Requires CUDA device
fn test_worker7_dendritic_nmda() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let n_neurons = 1;
    let dendrites_per_neuron = 8;
    let input_size = 16;
    let dendrite_inputs: Vec<f32> = vec![1.0; n_neurons * dendrites_per_neuron * input_size];
    let dendrite_weights: Vec<f32> = vec![0.2; n_neurons * dendrites_per_neuron * input_size];
    let soma_state: Vec<f32> = vec![0.5; n_neurons];
    let nonlinearity = 1; // NMDA

    let output = executor.dendritic_integration(
        &dendrite_inputs,
        &dendrite_weights,
        &soma_state,
        n_neurons,
        dendrites_per_neuron,
        input_size,
        nonlinearity,
    )?;

    assert_eq!(output.len(), n_neurons);
    assert!(output[0].is_finite());

    println!("✅ Worker 7 Dendritic (NMDA): output={:.4}", output[0]);
    Ok(())
}

// ============================================================================
// BASIC GPU OPERATIONS (All Workers)
// ============================================================================

#[test]
#[ignore] // Requires CUDA device
fn test_core_vector_operations() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let n = 10000;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();

    // Vector addition
    let c = executor.vector_add(&a, &b)?;
    assert_eq!(c.len(), n);
    assert_eq!(c[0], 1.0);
    assert_eq!(c[100], 201.0);

    // Dot product
    let dot = executor.dot_product(&a, &b)?;
    assert!(dot > 0.0);

    println!("✅ Core ops: add, dot_product (n={})", n);
    Ok(())
}

#[test]
#[ignore] // Requires CUDA device
fn test_core_matrix_multiply() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    let m = 128;
    let k = 256;
    let n = 128;

    let a: Vec<f32> = vec![0.1; m * k];
    let b: Vec<f32> = vec![0.1; k * n];

    let c = executor.matrix_multiply(&a, &b, m, k, n)?;

    assert_eq!(c.len(), m * n);
    assert!(c.iter().all(|&x| x.is_finite()));

    println!("✅ Matrix multiply: {}x{}x{}", m, k, n);
    Ok(())
}

// ============================================================================
// MULTI-WORKER SCENARIO
// ============================================================================

#[test]
#[ignore] // Requires CUDA device
fn test_multi_worker_pipeline() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    println!("\n=== Multi-Worker Pipeline ===");

    // Worker 3: Process IR image
    let height = 128;
    let width = 128;
    let n_pixels = height * width;
    let image: Vec<f32> = (0..n_pixels).map(|i| (i % 256) as f32).collect();

    let entropy_map = executor.pixel_entropy(&image, height, width, 8)?;
    println!("  Step 1 (Worker 3): {}x{} image → entropy map", height, width);

    // Worker 1: Forecast based on entropy statistics
    let time_series: Vec<f32> = vec![0.5; 100];
    let coeffs: Vec<f32> = vec![0.9, -0.1];
    let forecast = executor.ar_forecast(&time_series, &coeffs, 5)?;
    println!("  Step 2 (Worker 1): Entropy stats → forecast");

    // Worker 7: Dendritic processing
    let n_neurons = 1;
    let dendrites_per_neuron = 5;
    let input_size = 10;
    let dendrite_inputs: Vec<f32> = vec![0.5; n_neurons * dendrites_per_neuron * input_size];
    let dendrite_weights: Vec<f32> = vec![0.1; n_neurons * dendrites_per_neuron * input_size];
    let soma: Vec<f32> = vec![0.0; n_neurons];
    let output = executor.dendritic_integration(&dendrite_inputs, &dendrite_weights, &soma, n_neurons, dendrites_per_neuron, input_size, 0)?;
    println!("  Step 3 (Worker 7): Neural processing → output={:.4}", output[0]);

    println!("=== Pipeline Complete ===\n");

    Ok(())
}

#[test]
#[ignore] // Requires CUDA device
fn test_performance_throughput() -> Result<()> {
    let executor = get_global_executor()?;
    let executor = executor.lock().unwrap();

    println!("\n=== Throughput Test ===");

    let iterations = 50;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        // Mixed workload
        let _ = executor.vector_add(&vec![1.0; 1000], &vec![2.0; 1000])?;
        let _ = executor.matrix_multiply(&vec![0.1; 256], &vec![0.1; 256], 16, 16, 16)?;
        let _ = executor.pixel_entropy(&vec![100.0; 4096], 64, 64, 4)?;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = (iterations * 3) as f64 / elapsed.as_secs_f64();

    println!("  Iterations: {}", iterations);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.1} ops/sec", ops_per_sec);
    println!("=== Test Complete ===\n");

    assert!(ops_per_sec > 50.0, "Throughput should exceed 50 ops/sec");

    Ok(())
}
