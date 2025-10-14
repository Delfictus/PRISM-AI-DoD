//! Worker 7 Performance Benchmarks
//!
//! Benchmarks comparing baseline O(n²) vs optimized O(n log n) implementations
//! of information-theoretic metrics.
//!
//! Run with: cargo bench --bench worker7_performance
//!
//! Constitution: Worker 7 - Drug Discovery & Robotics

#![feature(test)]
extern crate test;

use test::Bencher;
use ndarray::Array2;
use prism_ai::applications::{
    information_metrics::ExperimentInformationMetrics,
    information_metrics_optimized::OptimizedExperimentInformationMetrics,
};

// Baseline benchmarks

#[bench]
fn bench_baseline_entropy_n100(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_baseline_entropy_n200(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (200, 3),
        (0..600).map(|i| i as f64 / 600.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_baseline_entropy_n400(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (400, 5),
        (0..2000).map(|i| i as f64 / 2000.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

// Optimized benchmarks

#[bench]
fn bench_optimized_entropy_n100(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_optimized_entropy_n200(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (200, 3),
        (0..600).map(|i| i as f64 / 600.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_optimized_entropy_n400(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec(
        (400, 5),
        (0..2000).map(|i| i as f64 / 2000.0).collect()
    ).unwrap();

    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

// Mutual Information benchmarks

#[bench]
fn bench_baseline_mi_n100(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64).collect()).unwrap();
    let y = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64 + 1.0).collect()).unwrap();

    b.iter(|| metrics.mutual_information(&x, &y).unwrap());
}

#[bench]
fn bench_optimized_mi_n100(b: &mut Bencher) {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64).collect()).unwrap();
    let y = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64 + 1.0).collect()).unwrap();

    b.iter(|| metrics.mutual_information(&x, &y).unwrap());
}
