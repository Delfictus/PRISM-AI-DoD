//! Spatial Entropy Enhancement Tests
//!
//! **Enhancement 2:** Validates real pixel processing and Shannon entropy

use prism_ai::pwsa::satellite_adapters::*;
use ndarray::Array2;

#[test]
fn test_from_pixels_single_hotspot() {
    // Create pixel array with single concentrated hotspot
    let mut pixels = Array2::from_elem((1024, 1024), 100u16);

    // Add hotspot at center (24×24 region)
    for y in 500..524 {
        for x in 500..524 {
            pixels[[y, x]] = 4000;
        }
    }

    let frame = IrSensorFrame::from_pixels(
        1,  // sv_id
        pixels,
        (38.0, 127.0),  // Korean peninsula
        1800.0,  // Hypersonic velocity
        45.0,    // High acceleration
    ).unwrap();

    // Validate computed metadata
    assert_eq!(frame.width, 1024);
    assert_eq!(frame.height, 1024);
    assert_eq!(frame.max_intensity, 4000.0);
    assert!((frame.background_level - 100.0).abs() < 10.0, "Background should be ~100");
    assert_eq!(frame.hotspot_count, 1, "Should detect 1 hotspot");

    // Validate spatial entropy (should be low for single hotspot)
    assert!(frame.spatial_entropy.is_some(), "Spatial entropy should be computed");
    let entropy = frame.spatial_entropy.unwrap();
    assert!(entropy < 0.3, "Single hotspot should have low entropy, got {}", entropy);
}

#[test]
fn test_from_pixels_multiple_dispersed_hotspots() {
    // Create pixel array with 5 dispersed hotspots
    let mut pixels = Array2::from_elem((1024, 1024), 100u16);

    let hotspot_locations = [(200, 200), (200, 800), (500, 500), (800, 200), (800, 800)];

    for (cx, cy) in hotspot_locations {
        for y in (cy - 10)..(cy + 10) {
            for x in (cx - 10)..(cx + 10) {
                pixels[[y, x]] = 3000;
            }
        }
    }

    let frame = IrSensorFrame::from_pixels(1, pixels, (38.0, 127.0), 500.0, 10.0).unwrap();

    // Should detect ~5 hotspots
    assert!(frame.hotspot_count >= 4 && frame.hotspot_count <= 6,
        "Should detect ~5 hotspots, got {}", frame.hotspot_count);

    // Entropy should be higher (dispersed pattern)
    let entropy = frame.spatial_entropy.unwrap();
    assert!(entropy > 0.4, "Dispersed hotspots should have higher entropy, got {}", entropy);
}

#[test]
fn test_shannon_entropy_uniform_distribution() {
    // Uniform distribution should have maximum entropy
    let pixels = Array2::from_shape_fn((100, 100), |(y, x)| {
        ((y + x) % 256) as u16  // Uniform distribution
    });

    let frame = IrSensorFrame::from_pixels(1, pixels, (0.0, 0.0), 0.0, 0.0).unwrap();

    let entropy = frame.spatial_entropy.unwrap();

    // Should be close to 1.0 (maximum)
    assert!(entropy > 0.8, "Uniform distribution should have high entropy, got {}", entropy);
}

#[test]
fn test_shannon_entropy_single_value() {
    // All pixels same value should have zero entropy
    let pixels = Array2::from_elem((100, 100), 500u16);

    let frame = IrSensorFrame::from_pixels(1, pixels, (0.0, 0.0), 0.0, 0.0).unwrap();

    let entropy = frame.spatial_entropy.unwrap();

    // Should be 0.0 (no disorder)
    assert!(entropy < 0.01, "Single intensity should have zero entropy, got {}", entropy);
}

#[test]
fn test_backward_compatibility_metadata_mode() {
    // Ensure metadata-only mode still works (existing demos)
    let frame = IrSensorFrame {
        sv_id: 1,
        timestamp: std::time::SystemTime::now(),
        width: 1024,
        height: 1024,
        pixels: None,  // No pixel data (demo mode)
        hotspot_positions: Vec::new(),
        intensity_histogram: None,
        spatial_entropy: None,
        max_intensity: 3000.0,
        background_level: 100.0,
        hotspot_count: 2,
        centroid_x: 512.0,
        centroid_y: 512.0,
        velocity_estimate_mps: 1800.0,
        acceleration_estimate: 45.0,
        swir_band_ratio: 1.0,
        thermal_signature: 0.8,
        geolocation: (38.0, 127.0),
    };

    // Spatial entropy is computed internally via extract_ir_features
    // Verify it works via the full ingestion pipeline
    let mut adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();
    let detection = adapter.ingest_ir_frame(1, &frame);

    // Should work without pixel data (backward compatibility)
    assert!(detection.is_ok(), "Metadata-only mode should work");

    // Feature extraction happens internally and uses Tier 4 approximation
    // (We can't directly test private method, but this validates the pipeline works)
}

#[test]
fn test_real_sda_format_ready() {
    // Validate we can accept real SDA sensor format (1024×1024×16-bit)
    let pixels = Array2::<u16>::from_elem((1024, 1024), 150u16);

    let frame = IrSensorFrame::from_pixels(
        42,  // SV-42
        pixels,
        (35.0, 125.0),
        0.0,
        0.0,
    );

    assert!(frame.is_ok(), "Should accept SDA pixel format");

    let frame = frame.unwrap();
    assert_eq!(frame.width, 1024);
    assert_eq!(frame.height, 1024);
    assert!(frame.pixels.is_some(), "Pixels should be stored");
    assert!(frame.spatial_entropy.is_some(), "Entropy should be computed");

    // Platform is ready for real SDA data ✅
}
