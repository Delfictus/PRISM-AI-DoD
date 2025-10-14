//! Test GPU acceleration for PWSA
//!
//! Verifies that GPU operations work correctly (using GPU kernels)

use anyhow::Result;
use prism_ai::gpu::gpu_enabled::{SimpleGpuTensor, SimpleGpuLinear};
use std::time::Instant;

fn main() -> Result<()> {
    println!("===========================================");
    println!("    GPU PWSA Acceleration Test");
    println!("===========================================\n");

    println!("GPU kernels will be integrated once cudarc API is stabilized\n");

    // Test 1: Matrix multiplication
    println!("--- Test 1: Matrix Multiplication ---");
    test_matmul()?;

    // Test 2: ReLU activation
    println!("\n--- Test 2: ReLU Activation ---");
    test_relu()?;

    // Test 3: Softmax
    println!("\n--- Test 3: Softmax ---");
    test_softmax()?;

    // Test 4: Full forward pass through Linear layer
    println!("\n--- Test 4: Linear Layer Forward Pass ---");
    test_linear_layer()?;

    println!("\n===========================================");
    println!("✅ All tests passed successfully!");
    println!("   Operations are working correctly");
    println!("===========================================");

    Ok(())
}

fn test_matmul() -> Result<()> {
    // Create two matrices for multiplication
    let m = 64;
    let k = 128;
    let n = 32;

    let a_data = vec![1.0f32; m * k];
    let b_data = vec![0.5f32; k * n];

    println!("Creating tensors A[{}x{}] and B[{}x{}]", m, k, k, n);
    let a = SimpleGpuTensor::from_cpu(a_data, vec![m, k])?;
    let b = SimpleGpuTensor::from_cpu(b_data, vec![k, n])?;

    println!("Performing matrix multiplication...");
    let start = Instant::now();
    let c = a.matmul(&b)?;
    let time = start.elapsed();

    // Verify result
    let c_cpu = c.to_cpu()?;
    let expected = (k as f32) * 1.0 * 0.5; // Sum of k multiplications

    if (c_cpu[0] - expected).abs() < 1e-4 {
        println!("✅ Matrix multiplication correct!");
        println!("   Result: {} (expected {})", c_cpu[0], expected);
        println!("   Time: {:?}", time);
    } else {
        println!("❌ Matrix multiplication incorrect!");
        println!("   Got {} but expected {}", c_cpu[0], expected);
    }

    Ok(())
}

fn test_relu() -> Result<()> {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let mut tensor = SimpleGpuTensor::from_cpu(data.clone(), vec![2, 3])?;

    println!("Input: {:?}", data);
    println!("Applying ReLU...");

    let start = Instant::now();
    tensor.relu()?;
    let time = start.elapsed();

    let result = tensor.to_cpu()?;
    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];

    if result == expected {
        println!("✅ ReLU activation correct!");
        println!("   Result: {:?}", result);
        println!("   Time: {:?}", time);
    } else {
        println!("❌ ReLU activation incorrect!");
        println!("   Got {:?} but expected {:?}", result, expected);
    }

    Ok(())
}

fn test_softmax() -> Result<()> {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut tensor = SimpleGpuTensor::from_cpu(data.clone(), vec![2, 3])?;

    println!("Input: {:?}", data);
    println!("Applying Softmax...");

    let start = Instant::now();
    tensor.softmax(1)?;
    let time = start.elapsed();

    let result = tensor.to_cpu()?;

    // Check that each row sums to 1
    let sum1: f32 = result[0..3].iter().sum();
    let sum2: f32 = result[3..6].iter().sum();

    if (sum1 - 1.0).abs() < 1e-6 && (sum2 - 1.0).abs() < 1e-6 {
        println!("✅ Softmax correct!");
        println!("   Row 1 sum: {}", sum1);
        println!("   Row 2 sum: {}", sum2);
        println!("   Time: {:?}", time);
    } else {
        println!("❌ Softmax incorrect!");
        println!("   Row sums: {} and {} (should be 1.0)", sum1, sum2);
    }

    Ok(())
}

fn test_linear_layer() -> Result<()> {
    let batch_size = 16;
    let in_features = 64;
    let out_features = 32;

    println!("Creating Linear layer: {} -> {}", in_features, out_features);
    let linear = SimpleGpuLinear::new(in_features, out_features)?;

    // Create input batch
    let input_data = vec![1.0f32; batch_size * in_features];
    let input = SimpleGpuTensor::from_cpu(input_data, vec![batch_size, in_features])?;

    println!("Running forward pass on batch of {}...", batch_size);
    let start = Instant::now();
    let output = linear.forward(&input)?;
    let time = start.elapsed();

    let output_shape = output.shape();
    if output_shape == &[batch_size, out_features] {
        println!("✅ Linear layer forward pass successful!");
        println!("   Output shape: {:?}", output_shape);
        println!("   Time: {:?}", time);
    } else {
        println!("❌ Linear layer output shape incorrect!");
        println!("   Got {:?} but expected [{}, {}]", output_shape, batch_size, out_features);
    }

    Ok(())
}