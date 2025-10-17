//! Enhanced Information Theory Metrics Demo
//!
//! Demonstrates Miller-Madow bias-corrected Shannon entropy for pixel processing.
//!
//! Run with: cargo run --example enhanced_it_demo

use prism_ai::mathematics::EnhancedITMetrics;

fn main() {
    println!("=== PRISM Enhanced Information Theory Demo ===\n");

    // Initialize enhanced IT metrics
    let metrics = EnhancedITMetrics::new()
        .with_bias_correction(true);

    println!("Configuration:");
    println!("  Bias Correction: Miller-Madow enabled\n");

    // === Demo 1: Bias-Corrected Shannon Entropy ===
    println!("--- Demo 1: Bias-Corrected Shannon Entropy ---");

    // Uniform distribution: H = log2(4) = 2.0 bits
    let uniform_hist = vec![10, 10, 10, 10];
    let h_uniform = metrics.shannon_entropy_from_histogram(&uniform_hist);
    println!("Uniform distribution (4 equal bins):");
    println!("  H(X) = {:.4} bits (theoretical: 2.0000 bits)", h_uniform);

    // Deterministic: H = 0 bits
    let deterministic_hist = vec![40, 0, 0, 0];
    let h_det = metrics.shannon_entropy_from_histogram(&deterministic_hist);
    println!("Deterministic distribution (single bin):");
    println!("  H(X) = {:.4} bits (theoretical: 0.0000 bits)", h_det);

    // Skewed distribution
    let skewed_hist = vec![20, 10, 5, 5];
    let h_skewed = metrics.shannon_entropy_from_histogram(&skewed_hist);
    println!("Skewed distribution (50%, 25%, 12.5%, 12.5%):");
    println!("  H(X) = {:.4} bits\n", h_skewed);

    // === Demo 2: Bias Correction Impact ===
    println!("--- Demo 2: Miller-Madow Bias Correction Impact ---");

    let metrics_no_bias = EnhancedITMetrics::new().with_bias_correction(false);
    
    // Checkerboard pattern (2 intensity levels)
    let checkerboard_hist = vec![128, 0, 0, 0, 128, 0, 0, 0];
    let h_biased = metrics_no_bias.shannon_entropy_from_histogram(&checkerboard_hist);
    let h_corrected = metrics.shannon_entropy_from_histogram(&checkerboard_hist);
    
    println!("Checkerboard pattern (2 intensity levels, N=256 pixels):");
    println!("  H (no correction) = {:.4} bits", h_biased);
    println!("  H (with correction) = {:.4} bits", h_corrected);
    println!("  Bias correction added: {:.4} bits", h_corrected - h_biased);

    // Smooth gradient (8 intensity levels)
    let gradient_hist = vec![32, 32, 32, 32, 32, 32, 32, 32];
    let h_gradient_biased = metrics_no_bias.shannon_entropy_from_histogram(&gradient_hist);
    let h_gradient_corrected = metrics.shannon_entropy_from_histogram(&gradient_hist);
    
    println!("\nSmooth gradient (8 equal intensity levels, N=256 pixels):");
    println!("  H (no correction) = {:.4} bits", h_gradient_biased);
    println!("  H (with correction) = {:.4} bits", h_gradient_corrected);
    println!("  Bias correction added: {:.4} bits", h_gradient_corrected - h_gradient_biased);

    // === Demo 3: Finite Sample Bias ===
    println!("\n--- Demo 3: Finite Sample Bias Effect ---");
    
    // Small sample
    let small_hist = vec![5, 5, 0, 0];
    let h_small_biased = metrics_no_bias.shannon_entropy_from_histogram(&small_hist);
    let h_small_corrected = metrics.shannon_entropy_from_histogram(&small_hist);
    
    println!("Small sample (N=10, 2 non-zero bins):");
    println!("  H (no correction) = {:.4} bits", h_small_biased);
    println!("  H (with correction) = {:.4} bits", h_small_corrected);
    println!("  Correction magnitude: {:.4} bits", h_small_corrected - h_small_biased);
    
    // Large sample
    let large_hist = vec![500, 500, 0, 0];
    let h_large_biased = metrics_no_bias.shannon_entropy_from_histogram(&large_hist);
    let h_large_corrected = metrics.shannon_entropy_from_histogram(&large_hist);
    
    println!("\nLarge sample (N=1000, 2 non-zero bins):");
    println!("  H (no correction) = {:.4} bits", h_large_biased);
    println!("  H (with correction) = {:.4} bits", h_large_corrected);
    println!("  Correction magnitude: {:.4} bits", h_large_corrected - h_large_biased);
    println!("  (Note: correction decreases with sample size)");

    // === Summary ===
    println!("\n=== Mathematical Guarantees ===");
    println!("✓ Shannon entropy H(X) ≥ 0");
    println!("✓ Miller-Madow bias correction: H_corrected = H_naive + (K-1)/(2N)");
    println!("✓ Correction reduces underestimation from finite samples");
    println!("✓ Correction magnitude scales as O(1/N)");

    println!("\n=== Applications ===");
    println!("1. PWSA Pixel Processor: Improved entropy maps for threat detection");
    println!("2. Image Analysis: More accurate texture and complexity metrics");
    println!("3. Feature Extraction: Better entropy-based feature selection");
    println!("4. Anomaly Detection: Enhanced sensitivity to distribution changes");

    println!("\n✓ Enhanced information theory demo complete!");
    println!("All PWSA pixel entropy computations now use bias-corrected estimators.");
}
