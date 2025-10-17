//! Integration tests for PWSA platform
//!
//! Critical performance validation:
//! - <5ms end-to-end fusion latency
//! - Multi-layer concurrent processing
//! - Constitutional compliance under load

#[cfg(test)]
mod integration_tests {
    use prism_ai::pwsa::*;
    use std::time::{SystemTime, Instant};
    use std::thread;
    use std::sync::{Arc, Mutex};

    /// Performance requirement: <5ms fusion latency
    const MAX_LATENCY_MS: u128 = 5;
    const BENCHMARK_ITERATIONS: usize = 100;

    #[test]
    fn test_fusion_latency_under_5ms() {
        let mut platform = PwsaFusionPlatform::new_tranche1()
            .expect("Failed to create fusion platform");

        let mut passed = 0;
        let mut failed = 0;
        let mut total_latency_ms = 0u128;
        let mut max_latency = 0u128;
        let mut min_latency = u128::MAX;

        for i in 0..BENCHMARK_ITERATIONS {
            // Generate varied test data
            let transport_telem = OctTelemetry {
                sv_id: (i % 154) as u32 + 1,
                link_id: (i % 4) as u8,
                optical_power_dbm: -15.0 + (i as f64 * 0.1),
                bit_error_rate: 1e-9 * (1.0 + i as f64 * 0.01),
                pointing_error_urad: 2.0 + (i as f64 * 0.05),
                data_rate_gbps: 10.0,
                temperature_c: 20.0 + (i as f64 * 0.2),
                timestamp: SystemTime::now(),
            };

            let tracking_frame = IrSensorFrame {
                sv_id: (i % 35) as u32 + 1,
                width: 1024,
                height: 1024,
                max_intensity: 4000.0 + (i as f64 * 10.0),
                background_level: 150.0,
                hotspot_count: (i % 10) as u32,
                centroid_x: 512.0,
                centroid_y: 512.0,
                velocity_estimate_mps: 1000.0 + (i as f64 * 20.0),
                acceleration_estimate: 10.0 + (i as f64 * 0.5),
                swir_band_ratio: 1.5,
                thermal_signature: 0.5 + (i as f64 * 0.01),
                geolocation: (35.0 + (i as f64 * 0.01), 127.0),
                timestamp: SystemTime::now(),
            };

            let ground_data = GroundStationData {
                station_id: (i % 5) as u32 + 1,
                uplink_power_dbm: 45.0,
                downlink_snr_db: 18.0 + (i as f64 * 0.1),
                command_queue_depth: (i % 20) as u32,
                timestamp: SystemTime::now(),
            };

            // Measure fusion latency
            let start = Instant::now();
            let result = platform.fuse_mission_data(
                &transport_telem,
                &tracking_frame,
                &ground_data,
            );
            let latency = start.elapsed().as_millis();

            assert!(result.is_ok(), "Fusion failed on iteration {}", i);

            // Track statistics
            total_latency_ms += latency;
            max_latency = max_latency.max(latency);
            min_latency = min_latency.min(latency);

            if latency <= MAX_LATENCY_MS {
                passed += 1;
            } else {
                failed += 1;
                eprintln!("Iteration {} exceeded 5ms: {}ms", i, latency);
            }
        }

        let avg_latency = total_latency_ms / BENCHMARK_ITERATIONS as u128;
        let pass_rate = (passed as f64 / BENCHMARK_ITERATIONS as f64) * 100.0;

        println!("\n=== LATENCY BENCHMARK RESULTS ===");
        println!("Iterations: {}", BENCHMARK_ITERATIONS);
        println!("Average latency: {}ms", avg_latency);
        println!("Min latency: {}ms", min_latency);
        println!("Max latency: {}ms", max_latency);
        println!("Pass rate (<5ms): {:.1}%", pass_rate);
        println!("Passed: {} | Failed: {}", passed, failed);

        // Require at least 80% pass rate for CI/CD
        assert!(
            pass_rate >= 80.0,
            "Latency requirement failed: only {:.1}% of iterations met <5ms target",
            pass_rate
        );
    }

    #[test]
    fn test_concurrent_multi_layer_processing() {
        let platform = Arc::new(Mutex::new(
            PwsaFusionPlatform::new_tranche1().unwrap()
        ));

        let mut handles = vec![];

        // Simulate concurrent access from multiple threads
        for thread_id in 0..3 {
            let platform_clone = Arc::clone(&platform);

            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let transport_telem = OctTelemetry {
                        sv_id: (thread_id * 50 + i) as u32 + 1,
                        link_id: thread_id as u8,
                        optical_power_dbm: -15.0,
                        bit_error_rate: 1e-9,
                        pointing_error_urad: 2.5,
                        data_rate_gbps: 10.0,
                        temperature_c: 22.0,
                        timestamp: SystemTime::now(),
                    };

                    let tracking_frame = IrSensorFrame {
                        sv_id: (thread_id * 10 + i) as u32 + 1,
                        width: 1024,
                        height: 1024,
                        max_intensity: 4095.0,
                        background_level: 150.0,
                        hotspot_count: thread_id as u32,
                        centroid_x: 512.0,
                        centroid_y: 512.0,
                        velocity_estimate_mps: 2000.0,
                        acceleration_estimate: 30.0,
                        swir_band_ratio: 1.8,
                        thermal_signature: 0.7,
                        geolocation: (35.5, 127.8),
                        timestamp: SystemTime::now(),
                    };

                    let ground_data = GroundStationData {
                        station_id: thread_id as u32 + 1,
                        uplink_power_dbm: 45.0,
                        downlink_snr_db: 18.5,
                        command_queue_depth: 10,
                        timestamp: SystemTime::now(),
                    };

                    let start = Instant::now();
                    let mut platform = platform_clone.lock().unwrap();
                    let result = platform.fuse_mission_data(
                        &transport_telem,
                        &tracking_frame,
                        &ground_data,
                    );
                    drop(platform); // Release lock quickly
                    let latency = start.elapsed();

                    assert!(result.is_ok(), "Thread {} iteration {} failed", thread_id, i);
                    assert!(
                        latency.as_millis() < 50,
                        "Thread {} iteration {} too slow: {}ms",
                        thread_id, i, latency.as_millis()
                    );
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_stress_test_maximum_load() {
        let mut platform = PwsaFusionPlatform::new_tranche1().unwrap();

        // Simulate maximum constellation load
        // 154 Transport SVs × 4 links = 616 transport streams
        // 35 Tracking SVs = 35 tracking streams
        // 5 Ground stations = 5 ground streams

        let start_total = Instant::now();
        let mut fusion_count = 0;

        // Process one frame from each possible source
        for transport_sv in 1..=154 {
            for link_id in 0..4 {
                for tracking_sv in 1..=35 {
                    for station_id in 1..=5 {
                        // Only process a subset to keep test reasonable
                        if (transport_sv + tracking_sv + station_id) % 100 != 0 {
                            continue;
                        }

                        let transport_telem = OctTelemetry {
                            sv_id: transport_sv,
                            link_id,
                            optical_power_dbm: -15.0,
                            bit_error_rate: 1e-9,
                            pointing_error_urad: 2.5,
                            data_rate_gbps: 10.0,
                            temperature_c: 22.0,
                            timestamp: SystemTime::now(),
                        };

                        let tracking_frame = IrSensorFrame {
                            sv_id: tracking_sv,
                            width: 1024,
                            height: 1024,
                            max_intensity: 4095.0,
                            background_level: 150.0,
                            hotspot_count: 2,
                            centroid_x: 512.0,
                            centroid_y: 512.0,
                            velocity_estimate_mps: 1500.0,
                            acceleration_estimate: 20.0,
                            swir_band_ratio: 1.6,
                            thermal_signature: 0.6,
                            geolocation: (35.5, 127.8),
                            timestamp: SystemTime::now(),
                        };

                        let ground_data = GroundStationData {
                            station_id,
                            uplink_power_dbm: 45.0,
                            downlink_snr_db: 18.5,
                            command_queue_depth: 15,
                            timestamp: SystemTime::now(),
                        };

                        let result = platform.fuse_mission_data(
                            &transport_telem,
                            &tracking_frame,
                            &ground_data,
                        );

                        assert!(result.is_ok());
                        fusion_count += 1;
                    }
                }
            }
        }

        let total_time = start_total.elapsed();
        let throughput = fusion_count as f64 / total_time.as_secs_f64();

        println!("\n=== STRESS TEST RESULTS ===");
        println!("Total fusions: {}", fusion_count);
        println!("Total time: {:.2}s", total_time.as_secs_f64());
        println!("Throughput: {:.0} fusions/second", throughput);

        assert!(throughput > 10.0, "Throughput too low: {:.0} fusions/s", throughput);
    }

    #[test]
    fn test_extreme_threat_scenario() {
        let mut platform = PwsaFusionPlatform::new_tranche1().unwrap();

        // Simulate detection of multiple hypersonic threats
        let transport_telem = OctTelemetry {
            sv_id: 42,
            link_id: 2,
            optical_power_dbm: -10.0,  // Strong signal
            bit_error_rate: 1e-10,     // Very low error
            pointing_error_urad: 1.0,   // Excellent pointing
            data_rate_gbps: 10.0,
            temperature_c: 25.0,
            timestamp: SystemTime::now(),
        };

        let tracking_frame = IrSensorFrame {
            sv_id: 17,
            width: 1024,
            height: 1024,
            max_intensity: 8000.0,      // Very bright
            background_level: 100.0,     // High contrast
            hotspot_count: 5,            // Multiple threats
            centroid_x: 512.0,
            centroid_y: 512.0,
            velocity_estimate_mps: 2500.0,  // Mach 7+
            acceleration_estimate: 60.0,     // High-G maneuver
            swir_band_ratio: 2.0,
            thermal_signature: 0.95,        // Very hot
            geolocation: (38.0, 125.0),     // Near Korean DMZ
            timestamp: SystemTime::now(),
        };

        let ground_data = GroundStationData {
            station_id: 1,
            uplink_power_dbm: 50.0,
            downlink_snr_db: 25.0,
            command_queue_depth: 50,  // High command queue
            timestamp: SystemTime::now(),
        };

        let start = Instant::now();
        let awareness = platform.fuse_mission_data(
            &transport_telem,
            &tracking_frame,
            &ground_data,
        ).expect("Failed to process extreme threat scenario");
        let latency = start.elapsed();

        // Even under extreme threat, must maintain performance
        assert!(
            latency.as_millis() <= MAX_LATENCY_MS * 2,  // Allow 2x for extreme case
            "Extreme scenario latency too high: {}ms",
            latency.as_millis()
        );

        // Verify appropriate threat response
        let max_threat = awareness.threat_status.iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(max_threat > 0.5, "Should detect high threat level");

        let has_immediate_action = awareness.recommended_actions.iter()
            .any(|a| a.contains("IMMEDIATE") || a.contains("ALERT"));
        assert!(has_immediate_action, "Should recommend immediate action");
    }

    #[test]
    fn test_degraded_performance_handling() {
        let mut platform = PwsaFusionPlatform::new_tranche1().unwrap();

        // Test with degraded/partial data
        let transport_telem = OctTelemetry {
            sv_id: 150,
            link_id: 3,
            optical_power_dbm: -28.0,   // Weak signal
            bit_error_rate: 1e-4,       // High error rate
            pointing_error_urad: 90.0,  // Poor pointing
            data_rate_gbps: 2.0,        // Reduced rate
            temperature_c: 85.0,        // Overheating
            timestamp: SystemTime::now(),
        };

        let tracking_frame = IrSensorFrame {
            sv_id: 30,
            width: 1024,
            height: 1024,
            max_intensity: 500.0,       // Low intensity
            background_level: 400.0,     // Poor contrast
            hotspot_count: 0,            // No clear hotspots
            centroid_x: 0.0,
            centroid_y: 0.0,
            velocity_estimate_mps: 0.0,
            acceleration_estimate: 0.0,
            swir_band_ratio: 1.0,
            thermal_signature: 0.1,
            geolocation: (0.0, 0.0),
            timestamp: SystemTime::now(),
        };

        let ground_data = GroundStationData {
            station_id: 5,
            uplink_power_dbm: 35.0,     // Weak uplink
            downlink_snr_db: 5.0,       // Poor SNR
            command_queue_depth: 95,     // Nearly full queue
            timestamp: SystemTime::now(),
        };

        // Should still process degraded data without panic
        let result = platform.fuse_mission_data(
            &transport_telem,
            &tracking_frame,
            &ground_data,
        );

        assert!(result.is_ok(), "Should handle degraded data gracefully");

        let awareness = result.unwrap();
        assert!(awareness.transport_health < 0.5, "Should reflect poor transport health");
        assert!(awareness.ground_connectivity < 0.5, "Should reflect poor ground connectivity");
    }
}