//! Performance Benchmarks for Worker 7 Information Metrics
//!
//! Measures performance of information-theoretic calculations to identify
//! optimization opportunities and validate GPU acceleration benefits.
//!
//! Worker 7 Quality Enhancement - Task 2 (Performance Optimization)

#![feature(test)]
extern crate test;

use test::Bencher;
use prism_ai::applications::information_metrics::{
    ExperimentInformationMetrics,
    MolecularInformationMetrics,
    RoboticsInformationMetrics,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// ============================================================================
// Differential Entropy Benchmarks
// ============================================================================

#[bench]
fn bench_differential_entropy_small(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    // Small dataset: 50 samples, 2D
    let samples = Array2::from_shape_vec(
        (50, 2),
        (0..100).map(|i| (i as f64 / 100.0).sin()).collect()
    ).unwrap();

    b.iter(|| {
        metrics.differential_entropy(&samples).unwrap()
    });
}

#[bench]
fn bench_differential_entropy_medium(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    // Medium dataset: 200 samples, 3D
    let samples = Array2::from_shape_vec(
        (200, 3),
        (0..600).map(|i| (i as f64 / 600.0) * 10.0).collect()
    ).unwrap();

    b.iter(|| {
        metrics.differential_entropy(&samples).unwrap()
    });
}

#[bench]
fn bench_differential_entropy_large(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    // Large dataset: 500 samples, 5D
    let samples = Array2::from_shape_vec(
        (500, 5),
        (0..2500).map(|i| {
            let t = i as f64 / 2500.0 * 2.0 * PI;
            t.sin() + t.cos()
        }).collect()
    ).unwrap();

    b.iter(|| {
        metrics.differential_entropy(&samples).unwrap()
    });
}

// ============================================================================
// Mutual Information Benchmarks
// ============================================================================

#[bench]
fn bench_mutual_information_small(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    let x = Array2::from_shape_vec(
        (50, 2),
        (0..100).map(|i| i as f64 / 100.0).collect()
    ).unwrap();

    let y = Array2::from_shape_vec(
        (50, 2),
        (0..100).map(|i| (i as f64 / 100.0) + 0.1 * (i as f64).sin()).collect()
    ).unwrap();

    b.iter(|| {
        metrics.mutual_information(&x, &y).unwrap()
    });
}

#[bench]
fn bench_mutual_information_medium(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    let x = Array2::from_shape_vec(
        (150, 3),
        (0..450).map(|i| i as f64 / 450.0).collect()
    ).unwrap();

    let y = Array2::from_shape_vec(
        (150, 3),
        (0..450).map(|i| (i as f64 / 450.0) * 2.0).collect()
    ).unwrap();

    b.iter(|| {
        metrics.mutual_information(&x, &y).unwrap()
    });
}

// ============================================================================
// KL Divergence Benchmarks
// ============================================================================

#[bench]
fn bench_kl_divergence_small(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    let p_samples = Array2::from_shape_vec(
        (60, 2),
        (0..120).map(|i| {
            let t = i as f64 / 120.0 * 2.0 * PI;
            t.cos()
        }).collect()
    ).unwrap();

    let q_samples = Array2::from_shape_vec(
        (60, 2),
        (0..120).map(|i| {
            let t = i as f64 / 120.0 * 2.0 * PI;
            t.cos() + 0.1
        }).collect()
    ).unwrap();

    b.iter(|| {
        metrics.kl_divergence(&p_samples, &q_samples).unwrap()
    });
}

#[bench]
fn bench_kl_divergence_medium(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    let p_samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| i as f64 / 300.0).collect()
    ).unwrap();

    let q_samples = Array2::from_shape_vec(
        (100, 3),
        (0..300).map(|i| (i as f64 / 300.0) + 0.2).collect()
    ).unwrap();

    b.iter(|| {
        metrics.kl_divergence(&p_samples, &q_samples).unwrap()
    });
}

// ============================================================================
// Molecular Similarity Benchmarks
// ============================================================================

#[bench]
fn bench_molecular_similarity(b: &mut Bencher) {
    let metrics = MolecularInformationMetrics::new();

    // 200D molecular descriptor (typical for fingerprints)
    let mol1 = Array1::from_vec((0..200).map(|i| i as f64 / 200.0).collect());
    let mol2 = Array1::from_vec((0..200).map(|i| (i as f64 / 200.0) + 0.1).collect());

    b.iter(|| {
        metrics.molecular_similarity(&mol1, &mol2)
    });
}

#[bench]
fn bench_chemical_space_entropy(b: &mut Bencher) {
    let metrics = MolecularInformationMetrics::new();

    // Library of 100 molecules with 50 descriptors each
    let descriptors = Array2::from_shape_vec(
        (100, 50),
        (0..5000).map(|i| (i as f64 / 5000.0) * 100.0).collect()
    ).unwrap();

    b.iter(|| {
        metrics.chemical_space_entropy(&descriptors)
    });
}

// ============================================================================
// Robotics Metrics Benchmarks
// ============================================================================

#[bench]
fn bench_trajectory_entropy(b: &mut Bencher) {
    let metrics = RoboticsInformationMetrics::new();

    // 1000 trajectory points in 3D space
    let mut trajectories = Array2::zeros((1000, 3));
    for i in 0..1000 {
        let t = i as f64 * 0.01;
        trajectories[[i, 0]] = t + 0.1 * (t * 2.0).sin();
        trajectories[[i, 1]] = 0.5 * t + 0.1 * (t * 3.0).cos();
        trajectories[[i, 2]] = 0.2 * t;
    }

    b.iter(|| {
        metrics.trajectory_entropy(&trajectories)
    });
}

#[bench]
fn bench_sensor_information_gain(b: &mut Bencher) {
    let metrics = RoboticsInformationMetrics::new();

    let prior_var = 1.0;
    let posterior_var = 0.1;

    b.iter(|| {
        metrics.sensor_information_gain(prior_var, posterior_var)
    });
}

// ============================================================================
// End-to-End Workflow Benchmarks
// ============================================================================

#[bench]
fn bench_experiment_design_workflow(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();

    // Simulate experiment design: prior → posterior, compute EIG
    let prior = Array2::from_shape_vec(
        (80, 2),
        (0..160).map(|i| {
            let t = i as f64 / 160.0 * 2.0 * PI;
            1.5 * t.cos()
        }).collect()
    ).unwrap();

    let posterior = Array2::from_shape_vec(
        (80, 2),
        (0..160).map(|i| {
            let t = i as f64 / 160.0 * 2.0 * PI;
            0.8 * t.cos()
        }).collect()
    ).unwrap();

    b.iter(|| {
        metrics.expected_information_gain(&prior, &posterior).unwrap()
    });
}

#[bench]
fn bench_drug_discovery_workflow(b: &mut Bencher) {
    let mol_metrics = MolecularInformationMetrics::new();

    // Simulate comparing 50 molecules pairwise
    let molecules: Vec<Array1<f64>> = (0..50)
        .map(|i| Array1::from_vec((0..100).map(|j| (i * 100 + j) as f64 / 5000.0).collect()))
        .collect();

    b.iter(|| {
        let mut similarities = Vec::new();
        for i in 0..5 {
            for j in (i+1)..10 {
                let sim = mol_metrics.molecular_similarity(&molecules[i], &molecules[j]);
                similarities.push(sim);
            }
        }
        similarities
    });
}

// ============================================================================
// Scalability Benchmarks
// ============================================================================

#[bench]
fn bench_entropy_n100_d2(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_n200_d2(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((200, 2), (0..400).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_n400_d2(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((400, 2), (0..800).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_n100_d5(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((100, 5), (0..500).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}

#[bench]
fn bench_entropy_n100_d10(b: &mut Bencher) {
    let metrics = ExperimentInformationMetrics::new().unwrap();
    let samples = Array2::from_shape_vec((100, 10), (0..1000).map(|i| i as f64).collect()).unwrap();
    b.iter(|| metrics.differential_entropy(&samples).unwrap());
}
