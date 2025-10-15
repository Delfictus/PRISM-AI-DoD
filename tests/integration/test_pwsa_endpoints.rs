// Integration tests for PWSA endpoints

use reqwest;
use serde_json::{json, Value};

const BASE_URL: &str = "http://localhost:8080";
const API_KEY: &str = "test-key";

fn get_client() -> reqwest::Client {
    reqwest::Client::new()
}

#[tokio::test]
async fn test_pwsa_detect_threat() {
    let client = get_client();

    let payload = json!({
        "sv_id": 42,
        "timestamp": 1234567890,
        "ir_frame": {
            "width": 640,
            "height": 480,
            "centroid_x": 320.0,
            "centroid_y": 240.0,
            "hotspot_count": 5
        }
    });

    let response = client
        .post(&format!("{}/api/v1/pwsa/detect", BASE_URL))
        .header("Authorization", format!("Bearer {}", API_KEY))
        .json(&payload)
        .send()
        .await;

    assert!(response.is_ok(), "PWSA detect should be accessible");

    let response = response.unwrap();
    assert_eq!(response.status(), 200, "Should return 200");

    let body: Value = response.json().await.unwrap();
    assert!(body["data"]["threat_id"].is_string(), "Should have threat_id");
    assert!(body["data"]["threat_type"].is_string(), "Should have threat_type");
    assert!(body["data"]["confidence"].is_number(), "Should have confidence");
    assert!(body["data"]["position"].is_array(), "Should have position array");
}

#[tokio::test]
async fn test_pwsa_sensor_fusion() {
    let client = get_client();

    let payload = json!({
        "sv_id": 42,
        "timestamp": 1234567890,
        "sensors": {
            "ir": {
                "frame_id": 1234,
                "targets": [
                    {
                        "id": "target_1",
                        "azimuth": 10.0,
                        "elevation": 5.0,
                        "range": 25000.0,
                        "velocity": 500.0,
                        "ir_signature": 0.85,
                        "radar_cross_section": 1.5
                    }
                ]
            },
            "radar": {
                "scan_id": 5678,
                "targets": []
            }
        }
    });

    let response = client
        .post(&format!("{}/api/v1/pwsa/fuse", BASE_URL))
        .header("Authorization", format!("Bearer {}", API_KEY))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    assert!(body["data"]["num_tracks"].is_number(), "Should have num_tracks");
    assert!(body["data"]["fusion_quality"].is_number(), "Should have fusion_quality");
    assert!(body["data"]["tracks"].is_array(), "Should have tracks array");
}

#[tokio::test]
async fn test_pwsa_trajectory_prediction() {
    let client = get_client();

    let payload = json!({
        "track_id": "threat_001",
        "history": [
            {
                "timestamp": 1234567890,
                "position": [10000.0, 20000.0, 5000.0],
                "velocity": [800.0, -200.0, 0.0]
            },
            {
                "timestamp": 1234567900,
                "position": [18000.0, 18000.0, 5000.0],
                "velocity": [800.0, -200.0, 0.0]
            }
        ],
        "prediction_horizon": 30,
        "model": "kalman_filter"
    });

    let response = client
        .post(&format!("{}/api/v1/pwsa/predict", BASE_URL))
        .header("Authorization", format!("Bearer {}", API_KEY))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    assert!(body["data"]["model"].is_string());
    assert!(body["data"]["confidence"].is_number());
    assert!(body["data"]["predictions"].is_array());
}

#[tokio::test]
async fn test_pwsa_threat_prioritization() {
    let client = get_client();

    let payload = json!({
        "threats": [
            {
                "threat_id": "T001",
                "type": "ballistic_missile",
                "position": [50000.0, 30000.0, 20000.0],
                "velocity": 2500.0,
                "time_to_impact": 180.0,
                "confidence": 0.95
            },
            {
                "threat_id": "T002",
                "type": "cruise_missile",
                "position": [30000.0, 15000.0, 5000.0],
                "velocity": 800.0,
                "time_to_impact": 120.0,
                "confidence": 0.88
            }
        ],
        "prioritization_strategy": "time_weighted_risk",
        "defensive_assets": {
            "interceptors_available": 10,
            "max_engagement_range": 100000.0
        }
    });

    let response = client
        .post(&format!("{}/api/v1/pwsa/prioritize", BASE_URL))
        .header("Authorization", format!("Bearer {}", API_KEY))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    assert!(body["data"]["prioritized_threats"].is_array());
    assert!(body["data"]["strategy_used"].is_string());
    assert!(body["data"]["total_threats"].is_number());
}

#[tokio::test]
async fn test_pwsa_invalid_payload() {
    let client = get_client();

    let invalid_payload = json!({
        "invalid_field": "test"
    });

    let response = client
        .post(&format!("{}/api/v1/pwsa/detect", BASE_URL))
        .header("Authorization", format!("Bearer {}", API_KEY))
        .json(&invalid_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Should return 400 for invalid payload");
}

#[tokio::test]
async fn test_pwsa_rate_limiting() {
    let client = get_client();

    let payload = json!({
        "sv_id": 42,
        "timestamp": 1234567890,
        "ir_frame": {
            "width": 640,
            "height": 480,
            "centroid_x": 320.0,
            "centroid_y": 240.0,
            "hotspot_count": 5
        }
    });

    // Make 101 rapid requests to trigger rate limit (100 req/s)
    let mut rate_limited = false;
    for _ in 0..101 {
        let response = client
            .post(&format!("{}/api/v1/pwsa/detect", BASE_URL))
            .header("Authorization", format!("Bearer {}", API_KEY))
            .json(&payload)
            .send()
            .await
            .unwrap();

        if response.status() == 429 {
            rate_limited = true;
            break;
        }
    }

    assert!(rate_limited, "Rate limiting should trigger after 100 requests");
}
