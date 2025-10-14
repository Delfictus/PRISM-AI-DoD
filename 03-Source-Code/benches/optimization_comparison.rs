//! Performance Comparison: Baseline vs Optimized Information Metrics
//!
//! Benchmarks comparing the original O(n²) implementation with the
//! optimized O(n log n) KD-tree implementation.
//!
//! Worker 7 Quality Enhancement - Task 2 (Performance Optimization)
//!
//! Expected Results:
//! - Small (n=50): 2-3x speedup
//! - Medium (n=200): 5-10x speedup
//! - Large (n=500): 10-20x speedup

#![feature(test)]
extern crate test;

use test::Bencher;
use prism_ai::applications::information_metrics::ExperimentInformationMetrics;
use prism_ai::applications::information_metrics_optimized::OptimizedExperimentInformationMetrics;
use ndarray::Array2;
use std::f64::consts::PI;

// ============================================================================
// Differential Entropy: Baseline vs Optimized
// ============================================================================

#[bench]
fn bench_entropy_baseline_n50(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (50, 2),
        (0..100).map(|i| (i as f64 / 100.0 * 2.0 * PI).sin()).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_optimized_n50(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (50, 2),
        (0..100).map(|i| (i as f64 / 100.0 * 2.0 * PI).sin()).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_baseline_n100(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_optimized_n100(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_baseline_n200(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (200, 3),
        (0..600).map(|i| (i as f64 / 600.0) * 10.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_optimized_n200(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (200, 3),
        (0..600).map(|i| (i as f64 / 600.0) * 10.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_baseline_n400(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (400, 5),
        (0..2000).map(|i| i as f64 / 2000.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_optimized_n400(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (400, 5),
        (0..2000).map(|i| i as f64 / 2000.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

// ============================================================================
// Mutual Information: Baseline vs Optimized
// ============================================================================

#[bench]
fn bench_mi_baseline_n100(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    let x = Array2::from_shape_vec(
        (100, 2),
        (0..200).map(|i| i as f64 / 200.0).collect()
    ).unwrap();

    let y = Array2::from_shape_vec(
        (100, 2),
        (0..200).map(|i| (i as f64 / 200.0) + 0.1 * (i as f64).sin()).collect()
    ).unwrap();

    b.iter(|| metrics.mutual_information(&x, &y).unwrap());
}

#[bench]
fn bench_mi_optimized_n100(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

    let x = Array2::from_shape_vec(
        (100, 2),
        (0..200).map(|i| i as f64 / 200.0).collect()
    ).unwrap();

    let y = Array2::from_shape_vec(
        (100, 2),
        (0..200).map(|i| (i as f64 / 200.0) + 0.1 * (i as f64).sin()).collect()
    ).unwrap();

    b.iter(|| metrics.mutual_information(&x, &y).unwrap());
}

// ============================================================================
// KL Divergence: Baseline vs Optimized
// ============================================================================

#[bench]
fn bench_kl_baseline_n100(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    let p_samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    let q_samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| (i as f64 / 300.0) + 0.2).collect()
    ).unwrap();

    b.iter(|| metrics.kl_divergence(&p_samples, &q_samples).unwrap());
}

#[bench]
fn bench_kl_optimized_n100(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

    let p_samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    let q_samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| (i as f64 / 300.0) + 0.2).collect()
    ).unwrap();

    b.iter(|| metrics.kl_divergence(&p_samples, &q_samples).unwrap());
}

// ============================================================================
// Scalability Analysis
// ============================================================================

#[bench]
fn bench_scalability_baseline_n50(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_scalability_optimized_n50(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_scalability_baseline_n150(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((150, 2), (0..300).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_scalability_optimized_n150(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((150, 2), (0..300).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_scalability_baseline_n300(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((300, 2), (0..600).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_scalability_optimized_n300(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((300, 2), (0..600).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}
