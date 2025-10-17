//! GPU Pixel Processing Kernel Tests
//! Tests for the 4 new pixel processing kernels

#[cfg(feature = "cuda")]
use prism_ai::gpu::kernel_executor::GpuKernelExecutor;

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    // Simple 4x4 image
    #[rustfmt::skip]
    let image = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    // 3x3 edge detection kernel
    #[rustfmt::skip]
    let kernel = vec![
        -1.0, -1.0, -1.0,
        -1.0,  8.0, -1.0,
        -1.0, -1.0, -1.0,
    ];

    let height = 4;
    let width = 4;
    let kernel_size = 3;
    let stride = 1;
    let padding = 0;

    let output = executor.conv2d(&image, &kernel, height, width, kernel_size, stride, padding)
        .expect("Conv2D failed");

    // Output should be 2x2 (no padding, stride 1)
    let expected_output_size = ((height + 2 * padding - kernel_size) / stride + 1)
        * ((width + 2 * padding - kernel_size) / stride + 1);
    assert_eq!(output.len(), expected_output_size);

    println!("✅ Conv2D test passed: output size = {}", output.len());
}

#[test]
#[cfg(feature = "cuda")]
fn test_pixel_entropy_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let height = 8;
    let width = 8;

    // Create checkerboard pattern (high local entropy)
    let mut pixels = vec![0.0; height * width];
    for row in 0..height {
        for col in 0..width {
            pixels[row * width + col] = if (row + col) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }

    let window_size = 3;

    let entropy_map = executor.pixel_entropy(&pixels, height, width, window_size)
        .expect("Pixel entropy failed");

    assert_eq!(entropy_map.len(), height * width);

    // Entropy should be positive for checkerboard pattern
    let avg_entropy: f32 = entropy_map.iter().sum::<f32>() / entropy_map.len() as f32;
    assert!(avg_entropy > 0.0, "Entropy should be positive for mixed pattern");

    println!("✅ Pixel entropy test passed: avg entropy = {:.3}", avg_entropy);
}

#[test]
#[cfg(feature = "cuda")]
fn test_pixel_tda_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let height = 8;
    let width = 8;

    // Create gradient image
    let mut pixels = vec![0.0; height * width];
    for row in 0..height {
        for col in 0..width {
            pixels[row * width + col] = (row as f32) / (height as f32);
        }
    }

    let threshold = 0.5;

    let features = executor.pixel_tda(&pixels, height, width, threshold)
        .expect("Pixel TDA failed");

    assert_eq!(features.len(), height * width);

    // Features should vary across the image
    let min_feature = features.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_feature = features.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_feature > min_feature, "TDA features should vary across image");

    println!("✅ Pixel TDA test passed: feature range = [{:.3}, {:.3}]", min_feature, max_feature);
}

#[test]
#[cfg(feature = "cuda")]
fn test_image_segmentation_kernel() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    let height = 8;
    let width = 8;

    // Create image with distinct regions
    let mut pixels = vec![0.0; height * width];
    for row in 0..height {
        for col in 0..width {
            if row < height / 2 {
                pixels[row * width + col] = 0.2;  // Dark region
            } else {
                pixels[row * width + col] = 0.8;  // Bright region
            }
        }
    }

    let threshold = 0.5;

    let labels = executor.image_segmentation(&pixels, height, width, threshold)
        .expect("Image segmentation failed");

    assert_eq!(labels.len(), height * width);

    // Should have at least 2 different labels (top and bottom regions)
    let mut unique_labels: Vec<i32> = labels.clone();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    assert!(unique_labels.len() >= 2, "Should have at least 2 segments");

    // Top half should be mostly one label, bottom half another
    let top_labels: Vec<i32> = labels.iter().take(height * width / 2).copied().collect();
    let bottom_labels: Vec<i32> = labels.iter().skip(height * width / 2).copied().collect();

    let top_most_common = most_common(&top_labels);
    let bottom_most_common = most_common(&bottom_labels);

    assert_ne!(top_most_common, bottom_most_common, "Top and bottom regions should have different labels");

    println!("✅ Image segmentation test passed: {} unique segments, top={}, bottom={}",
        unique_labels.len(), top_most_common, bottom_most_common);
}

#[test]
#[cfg(feature = "cuda")]
fn test_all_pixel_kernels_registered() {
    let executor = GpuKernelExecutor::new(0).expect("Failed to create GPU executor");

    // Verify all 4 kernels are registered
    assert!(executor.get_kernel("conv2d").is_ok(), "Conv2D kernel not registered");
    assert!(executor.get_kernel("pixel_entropy").is_ok(), "Pixel entropy kernel not registered");
    assert!(executor.get_kernel("pixel_tda").is_ok(), "Pixel TDA kernel not registered");
    assert!(executor.get_kernel("image_segmentation").is_ok(), "Image segmentation kernel not registered");

    println!("✅ All 4 pixel processing kernels registered successfully");
}

// Helper function
fn most_common(values: &[i32]) -> i32 {
    let mut counts = std::collections::HashMap::new();
    for &val in values {
        *counts.entry(val).or_insert(0) += 1;
    }
    *counts.iter().max_by_key(|(_, &count)| count).unwrap().0
}
