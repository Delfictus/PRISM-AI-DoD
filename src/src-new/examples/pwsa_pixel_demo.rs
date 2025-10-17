//! PWSA Pixel Processing Demo
//!
//! Demonstrates full pixel-level IR analysis:
//! 1. Load IR sensor frame
//! 2. Compute Shannon entropy map
//! 3. Extract convolutional features
//! 4. Perform TDA analysis
//! 5. Segment image
//!
//! # Usage
//! ```bash
//! cargo run --example pwsa_pixel_demo --features cuda
//! ```

use prism_ai::pwsa::pixel_processor::{PixelProcessor, ConvFeatures, PixelTdaFeatures};
use ndarray::Array2;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM-AI PWSA Pixel Processing Demo ===\n");

    // 1. Initialize pixel processor
    println!("1. Initializing pixel processor...");
    let mut processor = PixelProcessor::new()?;
    println!("   ✅ Processor initialized (GPU: {})\n", if cfg!(feature = "cuda") { "enabled" } else { "disabled" });

    // 2. Generate synthetic IR frame
    println!("2. Loading IR sensor frame...");
    let ir_frame = generate_synthetic_ir_frame();
    let (height, width) = ir_frame.dim();
    println!("   Frame dimensions: {}x{} pixels", height, width);
    println!("   Bit depth: 16-bit (SWIR sensor)\n");

    // 3. Compute Shannon entropy map
    println!("3. Computing pixel-level Shannon entropy map...");
    let entropy_map = processor.compute_entropy_map(&ir_frame, 16)?;
    let avg_entropy: f32 = entropy_map.iter().sum::<f32>() / entropy_map.len() as f32;
    println!("   ✅ Entropy map computed");
    println!("   Average entropy: {:.4}", avg_entropy);
    println!("   Min entropy: {:.4}", entropy_map.iter().cloned().fold(f32::INFINITY, f32::min));
    println!("   Max entropy: {:.4}", entropy_map.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!();

    // 4. Extract convolutional features
    println!("4. Extracting convolutional features (edge detection, blobs)...");
    let conv_features = processor.extract_conv_features(&ir_frame)?;

    let edge_strength: f32 = conv_features.edge_magnitude.iter().sum();
    let blob_count = conv_features.blob_response.iter().filter(|&&x| x > 100.0).count();

    println!("   ✅ Convolution complete");
    println!("   Total edge strength: {:.2}", edge_strength);
    println!("   Blob detections: {}", blob_count);
    println!("   Feature map dimensions: {}x{}", conv_features.edge_magnitude.nrows(), conv_features.edge_magnitude.ncols());
    println!();

    // 5. Perform TDA analysis
    println!("5. Computing pixel-level TDA features...");
    let tda_threshold = 500u16;
    let tda_features = processor.compute_pixel_tda(&ir_frame, tda_threshold)?;

    println!("   ✅ TDA analysis complete");
    println!("   Betti-0 (connected components): {}", tda_features.betti_0);
    println!("   Betti-1 (holes): {}", tda_features.betti_1);
    println!("   Persistence range: ({:.0}, {:.0})",
        tda_features.persistence_range.0,
        tda_features.persistence_range.1
    );
    println!();

    // 6. Segment image
    println!("6. Performing image segmentation...");
    let n_segments = 4;
    let segmentation = processor.segment_image(&ir_frame, n_segments)?;

    // Count pixels in each segment
    let mut segment_counts = vec![0; n_segments];
    for &seg_id in segmentation.iter() {
        segment_counts[seg_id as usize] += 1;
    }

    println!("   ✅ Segmentation complete");
    println!("   Number of segments: {}", n_segments);
    for (i, count) in segment_counts.iter().enumerate() {
        let percentage = (*count as f64 / (height * width) as f64) * 100.0;
        println!("   Segment {}: {} pixels ({:.1}%)", i, count, percentage);
    }
    println!();

    // 7. Threat detection summary
    println!("7. Threat Analysis Summary:");
    println!("   ┌─────────────────────────────────────────────┐");
    println!("   │ Metric                     │ Value          │");
    println!("   ├─────────────────────────────────────────────┤");
    println!("   │ Spatial Entropy            │ {:<14.4} │", avg_entropy);
    println!("   │ Edge Strength              │ {:<14.0} │", edge_strength);
    println!("   │ Hotspots Detected          │ {:<14} │", tda_features.betti_0);
    println!("   │ Topological Complexity     │ {:<14} │",
        if tda_features.betti_0 > 5 { "High" } else { "Low" });
    println!("   │ Threat Classification      │ {:<14} │",
        classify_threat(avg_entropy, tda_features.betti_0));
    println!("   └─────────────────────────────────────────────┘");
    println!();

    println!("✅ Pixel processing demo complete!");
    println!("\n💡 Note: This demo uses synthetic data.");
    println!("   In production, integrate with SDA IR sensors for real-time analysis.");

    Ok(())
}

/// Generate synthetic IR frame with thermal features
fn generate_synthetic_ir_frame() -> Array2<u16> {
    let height = 128;
    let width = 128;

    Array2::from_shape_fn((height, width), |(y, x)| {
        // Background: ~500 intensity
        let mut intensity = 500;

        // Add multiple hotspots
        let hotspots = vec![
            (32, 32, 2000),   // Strong hotspot
            (96, 96, 1500),   // Medium hotspot
            (64, 32, 1200),   // Weak hotspot
        ];

        for (hy, hx, peak) in hotspots {
            let dx = (x as i32 - hx as i32).abs();
            let dy = (y as i32 - hy as i32).abs();
            let dist = ((dx * dx + dy * dy) as f64).sqrt();

            if dist < 15.0 {
                let falloff = (1.0 - dist / 15.0);
                intensity += ((peak as f64 * falloff) as u16);
            }
        }

        // Add noise
        let noise = ((y * 7 + x * 11) % 20) as i32 - 10;
        ((intensity as i32 + noise).max(0).min(4095)) as u16
    })
}

/// Simple threat classification based on features
fn classify_threat(entropy: f32, hotspot_count: usize) -> &'static str {
    if entropy > 0.6 && hotspot_count > 5 {
        "High"
    } else if entropy > 0.4 || hotspot_count > 2 {
        "Medium"
    } else {
        "Low"
    }
}
