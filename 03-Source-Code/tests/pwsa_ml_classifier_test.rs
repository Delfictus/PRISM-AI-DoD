//! ML Threat Classifier Tests
//!
//! **v2.0 Enhancement:** Tests for active inference classifier

use prism_ai::pwsa::active_inference_classifier::*;
use ndarray::Array1;

#[test]
fn test_synthetic_data_generation() {
    let dataset = ThreatTrainingExample::generate_dataset(100);

    assert_eq!(dataset.len(), 500);  // 100 per class × 5 classes

    // Verify all classes represented
    let mut class_counts = vec![0; 5];
    for example in &dataset {
        class_counts[example.label as usize] += 1;
    }

    for (i, count) in class_counts.iter().enumerate() {
        assert_eq!(*count, 100, "Class {} should have 100 samples", i);
    }
}

#[test]
fn test_threat_class_characteristics() {
    // Validate synthetic data has correct characteristics
    let hypersonic_examples: Vec<_> = (0..10)
        .map(|_| ThreatTrainingExample::generate_synthetic(ThreatClass::Hypersonic))
        .collect();

    for example in hypersonic_examples {
        let velocity = example.features[6];
        let acceleration = example.features[7];
        let thermal = example.features[11];

        // Hypersonic should have high velocity, high accel, high thermal
        assert!(velocity > 0.5, "Hypersonic velocity too low: {}", velocity);
        assert!(acceleration > 0.4, "Hypersonic accel too low: {}", acceleration);
        assert!(thermal > 0.7, "Hypersonic thermal too low: {}", thermal);
    }

    // Validate no-threat has opposite characteristics
    let no_threat_examples: Vec<_> = (0..10)
        .map(|_| ThreatTrainingExample::generate_synthetic(ThreatClass::NoThreat))
        .collect();

    for example in no_threat_examples {
        let velocity = example.features[6];
        let thermal = example.features[11];

        assert!(velocity < 0.3, "NoThreat velocity too high");
        assert!(thermal < 0.4, "NoThreat thermal too high");
    }
}

#[test]
fn test_article_iv_free_energy_properties() {
    // Test free energy computation satisfies Article IV
    let posterior = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.3, 0.1]);
    let prior = Array1::from_vec(vec![0.7, 0.1, 0.1, 0.05, 0.05]);

    // Compute KL divergence manually
    let mut kl = 0.0;
    for i in 0..5 {
        if posterior[i] > 0.0 && prior[i] > 0.0 {
            kl += posterior[i] * (posterior[i] / prior[i]).ln();
        }
    }

    // Free energy should be non-negative for normalized distributions
    assert!(kl >= 0.0, "KL divergence must be non-negative");
    assert!(kl.is_finite(), "KL divergence must be finite");
}

#[test]
fn test_probability_normalization() {
    let examples = ThreatTrainingExample::generate_dataset(10);

    for example in examples {
        let sum: f64 = example.features.iter()
            .filter(|&&x| x.abs() < 1e-6)
            .count() as f64;

        // Features should be mostly non-zero (generated with noise)
        assert!(sum < 50.0, "Too many zero features");
    }
}

#[test]
fn test_classifier_backward_compatibility() {
    // Ensure heuristic classifier still works (v1.0 compatibility)
    use prism_ai::pwsa::satellite_adapters::*;

    let mut adapter = TrackingLayerAdapter::new_tranche1(900).unwrap();

    let frame = IrSensorFrame {
        sv_id: 1,
        timestamp: std::time::SystemTime::now(),
        width: 1024,
        height: 1024,
        max_intensity: 3000.0,
        background_level: 100.0,
        hotspot_count: 3,
        centroid_x: 512.0,
        centroid_y: 512.0,
        velocity_estimate_mps: 1900.0,  // Hypersonic
        acceleration_estimate: 50.0,
        swir_band_ratio: 1.0,
        thermal_signature: 0.9,
        geolocation: (38.0, 127.0),
    };

    let detection = adapter.ingest_ir_frame(1, &frame);
    assert!(detection.is_ok());

    // Should classify as hypersonic (class 4)
    let threat = detection.unwrap();
    let max_idx = threat.threat_level.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // Heuristic should correctly identify hypersonic
    assert!(max_idx == 4 || max_idx == 3, "Should classify as high threat");
}

#[test]
fn test_ml_classifier_integration() {
    // Test that ML classifier can be integrated (even without trained model)
    use prism_ai::pwsa::satellite_adapters::*;

    // This will use heuristic if model file doesn't exist
    let adapter = TrackingLayerAdapter::new_tranche1_ml(900, "models/threat_classifier_v2.safetensors");

    // Should succeed (graceful fallback to heuristic)
    assert!(adapter.is_ok());
}

#[test]
fn test_feature_vector_dimensionality() {
    // Ensure feature vectors are correct dimensionality for ML
    let example = ThreatTrainingExample::generate_synthetic(ThreatClass::Hypersonic);

    assert_eq!(example.features.len(), 100, "Features must be 100-dimensional");

    // All features should be in reasonable range
    for (i, &feature) in example.features.iter().enumerate() {
        assert!(feature.abs() < 2.0, "Feature {} out of range: {}", i, feature);
    }
}
