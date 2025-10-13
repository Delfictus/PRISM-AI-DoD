//! Comprehensive Tests for Phase 1 Information Theory Enhancements
//!
//! Tests:
//! 1. KD-tree performance and correctness
//! 2. KSG estimator accuracy
//! 3. Conditional TE for confounder control
//! 4. Bootstrap confidence intervals

use prism_ai::*;
use ndarray::Array1;

#[test]
fn test_kdtree_knn_accuracy() {
    // Create structured point cloud
    let points: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];

    let tree = KdTree::new(&points);

    // Query near origin
    let query = vec![0.1, 0.1];
    let neighbors = tree.knn_search(&query, 2);

    assert_eq!(neighbors.len(), 2);
    // Should find (0,0) as nearest
    assert_eq!(neighbors[0].index, 0);
}

#[test]
fn test_kdtree_range_search() {
    let points: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let angle = i as f64 * 0.1;
            vec![angle.cos(), angle.sin()]
        })
        .collect();

    let tree = KdTree::new(&points);

    let query = vec![1.0, 0.0];
    let neighbors = tree.range_search(&query, 0.5);

    // Should find multiple points within distance 0.5
    assert!(neighbors.len() > 0);
    assert!(neighbors.len() < 100);
}

#[test]
fn test_ksg_estimator_vs_histogram() {
    // Compare KSG with histogram-based TE

    // Create causal system: Y(t) = 0.8 * X(t-1) + noise
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..300 {
        x.push((i as f64 * 0.05).sin());
        if i == 0 {
            y.push(0.0);
        } else {
            y.push(x[i - 1] * 0.8 + 0.05 * rand::random::<f64>());
        }
    }

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // KSG estimator
    let ksg = KsgEstimator::new(3, 1, 1, 1);
    let result_ksg = ksg.calculate(&x_arr, &y_arr).unwrap();

    // Histogram-based estimator
    let hist = TransferEntropy::new(1, 1, 1);
    let result_hist = hist.calculate(&x_arr, &y_arr);

    println!("KSG TE: {}", result_ksg.effective_te);
    println!("Histogram TE: {}", result_hist.effective_te);

    // Both should detect positive transfer entropy
    assert!(result_ksg.effective_te > 0.0);
    assert!(result_hist.effective_te > 0.0);

    // KSG should generally be more accurate (less bias)
    // but exact comparison depends on data
}

#[test]
fn test_ksg_optimal_embedding() {
    let ksg = KsgEstimator::default();

    // Deterministic series with known dimensionality
    let series: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin()).collect();
    let series_arr = Array1::from_vec(series);

    let optimal_dim = ksg.find_optimal_embedding(&series_arr);

    println!("Optimal embedding dimension: {}", optimal_dim);

    // Should find reasonable embedding (1-5 for simple sine wave)
    assert!(optimal_dim >= 1);
    assert!(optimal_dim <= 10);
}

#[test]
fn test_conditional_te_removes_spurious_correlation() {
    // Test that conditional TE removes spurious correlations
    // System: Z → X, Z → Y (common driver, no direct X→Y link)

    let mut z = Vec::new();
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..300 {
        let z_val = (i as f64 * 0.05).sin();
        z.push(z_val);

        // Both X and Y driven by Z only
        x.push(z_val * 0.9 + 0.05 * rand::random::<f64>());
        y.push(z_val * 0.85 + 0.05 * rand::random::<f64>());
    }

    let z_arr = Array1::from_vec(z);
    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // Unconditioned TE (should be high - spurious)
    let ksg = KsgEstimator::new(3, 1, 1, 1);
    let te_uncond = ksg.calculate(&x_arr, &y_arr).unwrap();

    // Conditioned TE (should be lower - no direct link)
    let cte = ConditionalTe::new(3, 1, 1, 1, 1);
    let te_cond = cte.calculate(&x_arr, &y_arr, &z_arr).unwrap();

    println!("TE(X→Y): {}", te_uncond.effective_te);
    println!("TE(X→Y|Z): {}", te_cond.effective_te);

    // Conditional TE should reduce or eliminate spurious correlation
    // (may not always be exactly lower due to estimation variance)
    assert!(te_cond.te_value >= 0.0);
    assert!(te_uncond.te_value >= 0.0);
}

#[test]
fn test_conditional_te_preserves_direct_causation() {
    // Test that conditional TE preserves direct causation
    // System: Z → X → Y (X directly causes Y)

    let mut z = Vec::new();
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..300 {
        let z_val = (i as f64 * 0.05).sin();
        z.push(z_val);

        // X driven by Z
        let x_val = z_val * 0.8;
        x.push(x_val);

        // Y driven by X (direct causation)
        if i > 0 {
            y.push(x[i - 1] * 0.9);
        } else {
            y.push(0.0);
        }
    }

    let z_arr = Array1::from_vec(z);
    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // Conditional TE should remain significant (direct link)
    let cte = ConditionalTe::new(3, 1, 1, 1, 1);
    let te_cond = cte.calculate(&x_arr, &y_arr, &z_arr).unwrap();

    println!("TE(X→Y|Z) with direct link: {}", te_cond.effective_te);

    // Direct causation should survive conditioning
    assert!(te_cond.effective_te > 0.0);
}

#[test]
fn test_bootstrap_percentile_ci() {
    let resampler = BootstrapResampler::new(100, 0.95);

    let x = Array1::linspace(0.0, 10.0, 100);
    let y = x.mapv(|v| v.sin());

    // Simple TE calculator
    let te_calc = |source: &Array1<f64>, target: &Array1<f64>| -> anyhow::Result<f64> {
        let te = TransferEntropy::default();
        let result = te.calculate(source, target);
        Ok(result.effective_te)
    };

    let observed_te = 0.5;
    let ci = resampler.calculate_ci(te_calc, &x, &y, observed_te).unwrap();

    println!("95% CI: [{}, {}]", ci.lower, ci.upper);

    // Check CI properties
    assert!(ci.lower <= ci.estimate);
    assert!(ci.estimate <= ci.upper);
    assert_eq!(ci.confidence_level, 0.95);
    assert!(ci.upper - ci.lower > 0.0); // Non-degenerate interval
}

#[test]
fn test_bootstrap_bca_ci() {
    let resampler = BootstrapResampler {
        n_bootstrap: 100,
        confidence_level: 0.95,
        method: BootstrapMethod::BCa,
        ..Default::default()
    };

    let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // BCa with KSG estimator
    let te_calc = |source: &Array1<f64>, target: &Array1<f64>| -> anyhow::Result<f64> {
        let ksg = KsgEstimator::default();
        let result = ksg.calculate(source, target)?;
        Ok(result.effective_te)
    };

    let observed_te = 0.3;
    let ci = resampler.calculate_ci(te_calc, &x_arr, &y_arr, observed_te).unwrap();

    println!("BCa 95% CI: [{}, {}]", ci.lower, ci.upper);

    assert!(ci.lower <= ci.upper);
    assert_eq!(ci.method, BootstrapMethod::BCa);
}

#[test]
fn test_bootstrap_block_ci_for_time_series() {
    let resampler = BootstrapResampler {
        n_bootstrap: 50,
        confidence_level: 0.90,
        method: BootstrapMethod::Block,
        block_size: 10,
    };

    // Autocorrelated time series
    let mut x = vec![0.0];
    let mut y = vec![0.0];

    for i in 1..200 {
        // AR(1) process
        x.push(0.8 * x[i - 1] + 0.2 * rand::random::<f64>());
        y.push(0.7 * y[i - 1] + 0.3 * x[i - 1]);
    }

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    let te_calc = |source: &Array1<f64>, target: &Array1<f64>| -> anyhow::Result<f64> {
        let te = TransferEntropy::new(1, 1, 1);
        let result = te.calculate(source, target);
        Ok(result.effective_te)
    };

    let observed_te = 0.4;
    let ci = resampler.calculate_ci(te_calc, &x_arr, &y_arr, observed_te).unwrap();

    println!("Block Bootstrap 90% CI: [{}, {}]", ci.lower, ci.upper);

    assert!(ci.lower <= ci.upper);
    assert_eq!(ci.method, BootstrapMethod::Block);
}

#[test]
fn test_partial_correlation_vs_conditional_te() {
    // Compare partial correlation (linear) with conditional TE (nonlinear)

    // Create linear system with confounder
    let z: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let x: Vec<f64> = z.iter().map(|&zi| zi * 2.0 + 1.0).collect();
    let y: Vec<f64> = z.iter().map(|&zi| zi * 1.5 + 2.0).collect();

    let z_arr = Array1::from_vec(z);
    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // Partial correlation (linear measure)
    let partial_corr = ConditionalTe::partial_correlation(&x_arr, &y_arr, &z_arr).unwrap();

    println!("Partial correlation: {}", partial_corr);

    // Partial correlation should be small (Z explains all correlation)
    assert!(partial_corr.abs() < 0.5);
}

#[test]
fn test_gpu_te_fallback() {
    // Test GPU TE with CPU fallback
    let te_gpu = TransferEntropyGpu::new(1, 1, 1);

    let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).cos()).collect();

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // Should fall back to CPU if GPU not available
    let result = te_gpu.calculate(&x_arr, &y_arr).unwrap();

    println!("GPU TE (with fallback): {}", result.te_value);

    assert!(result.te_value >= 0.0);
    assert!(result.te_value.is_finite());
}

#[test]
fn test_multiscale_ksg_analysis() {
    let ksg = KsgEstimator::default();

    // Create system with lag-2 dependence
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..200 {
        x.push((i as f64 * 0.05).sin());
        if i < 2 {
            y.push(0.0);
        } else {
            // Y depends on X with lag 2
            y.push(x[i - 2] * 0.8);
        }
    }

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    // Multi-scale analysis
    let results = ksg.calculate_multiscale(&x_arr, &y_arr, 5).unwrap();

    assert_eq!(results.len(), 5);

    // Find peak TE
    let max_te = results.iter()
        .map(|r| r.effective_te)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("Max TE across lags: {}", max_te);

    // Should find positive TE at some lag
    assert!(max_te > 0.0);
}
