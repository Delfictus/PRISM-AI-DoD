//! PWSA Performance Benchmarking Suite
//!
//! Measures fusion pipeline performance for Week 2 optimization validation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_ai::pwsa::satellite_adapters::*;
use std::time::SystemTime;

fn create_sample_oct_telemetry() -> OctTelemetry {
    OctTelemetry {
        sv_id: 42,
        link_id: 2,
        timestamp: SystemTime::now(),
        optical_power_dbm: -12.5,
        bit_error_rate: 1e-9,
        pointing_error_urad: 3.2,
        data_rate_gbps: 10.0,
        temperature_c: 22.0,
    }
}

fn create_sample_ir_frame() -> IrSensorFrame {
    IrSensorFrame {
        sv_id: 17,
        timestamp: SystemTime::now(),
        width: 1024,
        height: 1024,
        max_intensity: 2500.0,
        background_level: 150.0,
        hotspot_count: 2,
        centroid_x: 512.0,
        centroid_y: 512.0,
        velocity_estimate_mps: 1800.0,  // Hypersonic
        acceleration_estimate: 45.0,
        swir_band_ratio: 1.2,
        thermal_signature: 0.85,
        geolocation: (38.0, 127.0),  // Korean peninsula
    }
}

fn create_sample_ground_data() -> GroundStationData {
    GroundStationData {
        station_id: 3,
        timestamp: SystemTime::now(),
        uplink_power_dbm: 48.0,
        downlink_snr_db: 22.5,
        command_queue_depth: 8,
    }
}

fn bench_fusion_pipeline_baseline(c: &mut Criterion) {
    let mut platform = PwsaFusionPlatform::new_tranche1()
        .expect("Failed to initialize platform");

    c.bench_function("fusion_pipeline_baseline", |b| {
        b.iter(|| {
            let transport = create_sample_oct_telemetry();
            let tracking = create_sample_ir_frame();
            let ground = create_sample_ground_data();

            platform.fuse_mission_data(
                black_box(&transport),
                black_box(&tracking),
                black_box(&ground),
            ).unwrap()
        });
    });
}

fn bench_fusion_with_history(c: &mut Criterion) {
    let mut platform = PwsaFusionPlatform::new_tranche1()
        .expect("Failed to initialize platform");

    // Build up history for real TE computation
    for _ in 0..30 {
        let _ = platform.fuse_mission_data(
            &create_sample_oct_telemetry(),
            &create_sample_ir_frame(),
            &create_sample_ground_data(),
        );
    }

    c.bench_function("fusion_with_real_te", |b| {
        b.iter(|| {
            let transport = create_sample_oct_telemetry();
            let tracking = create_sample_ir_frame();
            let ground = create_sample_ground_data();

            platform.fuse_mission_data(
                black_box(&transport),
                black_box(&tracking),
                black_box(&ground),
            ).unwrap()
        });
    });
}

fn bench_throughput_sustained(c: &mut Criterion) {
    c.bench_function("fusion_throughput_100samples", |b| {
        b.iter(|| {
            let mut platform = PwsaFusionPlatform::new_tranche1().unwrap();

            for _ in 0..100 {
                let transport = create_sample_oct_telemetry();
                let tracking = create_sample_ir_frame();
                let ground = create_sample_ground_data();

                let _ = platform.fuse_mission_data(
                    black_box(&transport),
                    black_box(&tracking),
                    black_box(&ground),
                );
            }
        });
    });
}

fn bench_transfer_entropy_computation(c: &mut Criterion) {
    use prism_ai::information_theory::transfer_entropy::TransferEntropy;
    use ndarray::Array1;

    let te_calc = TransferEntropy::new(3, 3, 1);

    // Create sample time-series
    let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    c.bench_function("transfer_entropy_single_pair", |b| {
        b.iter(|| {
            te_calc.calculate(black_box(&x_arr), black_box(&y_arr))
        });
    });
}

criterion_group!(
    benches,
    bench_fusion_pipeline_baseline,
    bench_fusion_with_history,
    bench_throughput_sustained,
    bench_transfer_entropy_computation,
);
criterion_main!(benches);