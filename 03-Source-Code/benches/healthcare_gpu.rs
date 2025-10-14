//! Healthcare Risk Trajectory GPU Benchmarks
//!
//! Phase 3 Task 2: Validate GPU acceleration performance
//!
//! Benchmarks:
//! 1. Single patient risk trajectory (CPU vs GPU)
//! 2. Batch processing (100 patients)
//! 3. Real-time monitoring throughput
//! 4. Different forecast horizons (6h, 12h, 24h, 48h)
//!
//! Expected Results:
//! - Single patient: 10-25x speedup (60ms → 2.4-6ms)
//! - Batch processing: 10-25x speedup (6s → 240-600ms for 100 patients)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_ai::applications::healthcare::risk_trajectory::{
    RiskTrajectoryForecaster, TrajectoryConfig, RiskTimePoint
};
use prism_ai::applications::healthcare::risk_predictor::RiskCategory;
use prism_ai::time_series::ArimaConfig;

fn create_stable_patient() -> Vec<RiskTimePoint> {
    vec![
        RiskTimePoint {
            time_hours: 0.0,
            mortality_risk: 0.15,
            sepsis_risk: 0.20,
            severity_score: 8.0,
            risk_category: RiskCategory::Low,
        },
        RiskTimePoint {
            time_hours: 4.0,
            mortality_risk: 0.14,
            sepsis_risk: 0.18,
            severity_score: 7.5,
            risk_category: RiskCategory::Low,
        },
        RiskTimePoint {
            time_hours: 8.0,
            mortality_risk: 0.13,
            sepsis_risk: 0.17,
            severity_score: 7.0,
            risk_category: RiskCategory::Low,
        },
        RiskTimePoint {
            time_hours: 12.0,
            mortality_risk: 0.12,
            sepsis_risk: 0.16,
            severity_score: 6.5,
            risk_category: RiskCategory::Low,
        },
    ]
}

fn create_deteriorating_patient() -> Vec<RiskTimePoint> {
    vec![
        RiskTimePoint {
            time_hours: 0.0,
            mortality_risk: 0.20,
            sepsis_risk: 0.30,
            severity_score: 10.0,
            risk_category: RiskCategory::Moderate,
        },
        RiskTimePoint {
            time_hours: 4.0,
            mortality_risk: 0.28,
            sepsis_risk: 0.42,
            severity_score: 14.0,
            risk_category: RiskCategory::Moderate,
        },
        RiskTimePoint {
            time_hours: 8.0,
            mortality_risk: 0.38,
            sepsis_risk: 0.55,
            severity_score: 18.5,
            risk_category: RiskCategory::High,
        },
        RiskTimePoint {
            time_hours: 12.0,
            mortality_risk: 0.48,
            sepsis_risk: 0.68,
            severity_score: 23.0,
            risk_category: RiskCategory::High,
        },
    ]
}

fn create_high_risk_patient() -> Vec<RiskTimePoint> {
    vec![
        RiskTimePoint {
            time_hours: 0.0,
            mortality_risk: 0.55,
            sepsis_risk: 0.70,
            severity_score: 25.0,
            risk_category: RiskCategory::Critical,
        },
        RiskTimePoint {
            time_hours: 4.0,
            mortality_risk: 0.60,
            sepsis_risk: 0.75,
            severity_score: 28.0,
            risk_category: RiskCategory::Critical,
        },
        RiskTimePoint {
            time_hours: 8.0,
            mortality_risk: 0.63,
            sepsis_risk: 0.78,
            severity_score: 30.0,
            risk_category: RiskCategory::Critical,
        },
        RiskTimePoint {
            time_hours: 12.0,
            mortality_risk: 0.65,
            sepsis_risk: 0.80,
            severity_score: 31.5,
            risk_category: RiskCategory::Critical,
        },
    ]
}

fn bench_single_patient_trajectory(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_single_patient");
    group.sample_size(50);

    let config = TrajectoryConfig::default(); // 24-hour horizon

    // Benchmark stable patient
    let stable_history = create_stable_patient();
    group.bench_function("stable_patient_24h", |b| {
        b.iter(|| {
            let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
            forecaster.forecast_trajectory(black_box(&stable_history)).unwrap()
        })
    });

    // Benchmark deteriorating patient
    let deteriorating_history = create_deteriorating_patient();
    group.bench_function("deteriorating_patient_24h", |b| {
        b.iter(|| {
            let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
            forecaster.forecast_trajectory(black_box(&deteriorating_history)).unwrap()
        })
    });

    // Benchmark high-risk patient
    let high_risk_history = create_high_risk_patient();
    group.bench_function("critical_patient_24h", |b| {
        b.iter(|| {
            let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
            forecaster.forecast_trajectory(black_box(&high_risk_history)).unwrap()
        })
    });

    group.finish();
}

fn bench_forecast_horizons(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_forecast_horizons");
    group.sample_size(30);

    let patient_history = create_deteriorating_patient();

    for horizon in [6, 12, 24, 48].iter() {
        let mut config = TrajectoryConfig::default();
        config.horizon_hours = *horizon;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}h_horizon", horizon)),
            horizon,
            |b, _| {
                b.iter(|| {
                    let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
                    forecaster.forecast_trajectory(black_box(&patient_history)).unwrap()
                })
            }
        );
    }

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_batch_processing");
    group.sample_size(10);

    let config = TrajectoryConfig::default();

    // Create batch of patients (mix of stable, deteriorating, and critical)
    let mut patient_batch = Vec::new();
    for i in 0..100 {
        let patient = match i % 3 {
            0 => create_stable_patient(),
            1 => create_deteriorating_patient(),
            _ => create_high_risk_patient(),
        };
        patient_batch.push(patient);
    }

    group.bench_function("100_patients", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(100);
            for patient in &patient_batch {
                let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
                let trajectory = forecaster.forecast_trajectory(black_box(patient)).unwrap();
                results.push(trajectory);
            }
            results
        })
    });

    group.finish();
}

fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_batch_sizes");
    group.sample_size(15);

    let config = TrajectoryConfig::default();

    for batch_size in [10, 25, 50, 100].iter() {
        let mut patient_batch = Vec::new();
        for i in 0..*batch_size {
            let patient = match i % 3 {
                0 => create_stable_patient(),
                1 => create_deteriorating_patient(),
                _ => create_high_risk_patient(),
            };
            patient_batch.push(patient);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patients", batch_size)),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(*batch_size);
                    for patient in &patient_batch {
                        let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
                        let trajectory = forecaster.forecast_trajectory(black_box(patient)).unwrap();
                        results.push(trajectory);
                    }
                    results
                })
            }
        );
    }

    group.finish();
}

fn bench_real_time_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_real_time_monitoring");
    group.sample_size(20);

    let mut config = TrajectoryConfig::default();
    config.horizon_hours = 6; // Shorter horizon for real-time updates

    // Simulate ICU ward with 50 patients
    let mut icu_ward = Vec::new();
    for i in 0..50 {
        let patient = match i % 3 {
            0 => create_stable_patient(),
            1 => create_deteriorating_patient(),
            _ => create_high_risk_patient(),
        };
        icu_ward.push(patient);
    }

    group.bench_function("50_patient_icu_update", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(50);
            for patient in &icu_ward {
                let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
                let trajectory = forecaster.forecast_trajectory(black_box(patient)).unwrap();
                results.push(trajectory);
            }
            results
        })
    });

    group.finish();
}

fn bench_treatment_impact_assessment(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_treatment_impact");
    group.sample_size(30);

    let config = TrajectoryConfig::default();
    let patient_history = create_deteriorating_patient();

    group.bench_function("baseline_forecast_plus_impact", |b| {
        b.iter(|| {
            let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
            let baseline = forecaster.forecast_trajectory(black_box(&patient_history)).unwrap();

            // Assess treatment impact (simulated post-treatment risk)
            let post_treatment_risk = 0.35;
            let impact = forecaster.assess_treatment_impact(
                black_box(&baseline),
                black_box(post_treatment_risk)
            ).unwrap();

            (baseline, impact)
        })
    });

    group.finish();
}

fn bench_arima_configuration_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_arima_config");
    group.sample_size(20);

    let patient_history = create_deteriorating_patient();

    // Test different ARIMA configurations
    let arima_configs = vec![
        ("AR(1)", ArimaConfig { p: 1, d: 0, q: 0, include_constant: true }),
        ("AR(2)", ArimaConfig { p: 2, d: 0, q: 0, include_constant: true }),
        ("ARIMA(2,0,1)", ArimaConfig { p: 2, d: 0, q: 1, include_constant: true }),
        ("ARIMA(2,1,1)", ArimaConfig { p: 2, d: 1, q: 1, include_constant: true }),
    ];

    for (name, arima_config) in arima_configs {
        let mut config = TrajectoryConfig::default();
        config.arima_config = arima_config;

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &name,
            |b, _| {
                b.iter(|| {
                    let mut forecaster = RiskTrajectoryForecaster::new(config.clone());
                    forecaster.forecast_trajectory(black_box(&patient_history)).unwrap()
                })
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_patient_trajectory,
    bench_forecast_horizons,
    bench_batch_processing,
    bench_batch_sizes,
    bench_real_time_monitoring,
    bench_treatment_impact_assessment,
    bench_arima_configuration_impact
);

criterion_main!(benches);
