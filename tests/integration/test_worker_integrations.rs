//! Integration Tests for Worker 1, 3, 7 Integrations
//!
//! Tests real implementations via API endpoints:
//! - Worker 1: Time Series Forecasting (ARIMA, LSTM)
//! - Worker 3: Portfolio Optimization (GPU-accelerated MPT)
//! - Worker 7: Robotics Motion Planning (Active Inference)

use crate::common::{post_authenticated, DEFAULT_API_KEY};
use serde_json::json;

// Helper to make authenticated requests
async fn make_request(
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> Result<reqwest::Response, reqwest::Error> {
    match method {
        "POST" => {
            post_authenticated(path, DEFAULT_API_KEY, &body.unwrap_or(json!({}))).await
        }
        _ => unimplemented!("Method not supported"),
    }
}

#[tokio::test]
async fn test_worker1_time_series_arima() {
    // Test ARIMA forecasting
    let request = json!({
        "historical_data": [100.0, 102.0, 101.0, 105.0, 108.0, 107.0, 110.0, 115.0, 114.0, 118.0],
        "horizon": 5,
        "method": {
            "Arima": {
                "p": 2,
                "d": 1,
                "q": 1
            }
        },
        "include_uncertainty": true
    });

    let response = make_request(
        &client,
        "POST",
        "/api/v1/timeseries/forecast",
        Some(request),
    )
    .await
    .expect("Request failed");

    assert_eq!(response.status(), 200, "Worker 1 ARIMA forecast should succeed");

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");

    assert_eq!(body["status"], "success");
    assert!(body["data"]["predictions"].is_array());
    assert_eq!(body["data"]["predictions"].as_array().unwrap().len(), 5);
    assert!(body["data"]["model_info"]["method"].as_str().unwrap().contains("ARIMA"));

    // Check uncertainty quantification
    if let Some(confidence_intervals) = body["data"]["confidence_intervals"].as_array() {
        assert_eq!(confidence_intervals.len(), 5);
        for interval in confidence_intervals {
            assert!(interval["lower"].is_number());
            assert!(interval["upper"].is_number());
            assert!(interval["lower"].as_f64().unwrap() < interval["upper"].as_f64().unwrap());
        }
    }

    println!("✅ Worker 1 ARIMA forecasting: PASS");
}

#[tokio::test]
async fn test_worker1_time_series_lstm() {
    

    // Test LSTM forecasting
    let request = json!({
        "historical_data": [
            100.0, 102.0, 101.0, 105.0, 108.0, 107.0, 110.0, 115.0, 114.0, 118.0,
            120.0, 122.0, 119.0, 125.0, 128.0, 130.0, 132.0, 135.0, 133.0, 138.0
        ],
        "horizon": 3,
        "method": {
            "Lstm": {
                "hidden_dim": 64,
                "num_layers": 2
            }
        },
        "include_uncertainty": false
    });

    let response = make_request(
        &client,
        "POST",
        "/api/v1/timeseries/forecast",
        Some(request),
    )
    .await
    .expect("Request failed");

    assert_eq!(response.status(), 200, "Worker 1 LSTM forecast should succeed");

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");

    assert_eq!(body["status"], "success");
    assert!(body["data"]["predictions"].is_array());
    assert_eq!(body["data"]["predictions"].as_array().unwrap().len(), 3);
    assert!(body["data"]["model_info"]["method"].as_str().unwrap().contains("LSTM"));
    assert!(body["data"]["computation_time_ms"].as_f64().unwrap() > 0.0);

    println!("✅ Worker 1 LSTM forecasting: PASS");
}

#[tokio::test]
async fn test_worker3_portfolio_optimization() {
    

    // Test GPU-accelerated portfolio optimization
    let request = json!({
        "assets": [
            {
                "symbol": "AAPL",
                "expected_return": 0.12,
                "volatility": 0.20,
                "current_price": 150.0
            },
            {
                "symbol": "GOOGL",
                "expected_return": 0.15,
                "volatility": 0.25,
                "current_price": 2800.0
            },
            {
                "symbol": "MSFT",
                "expected_return": 0.10,
                "volatility": 0.18,
                "current_price": 350.0
            },
            {
                "symbol": "TSLA",
                "expected_return": 0.18,
                "volatility": 0.35,
                "current_price": 250.0
            }
        ],
        "constraints": {
            "max_position_size": 0.4,
            "min_position_size": 0.05
        },
        "objective": "MaximizeSharpe"
    });

    let response = make_request(
        &client,
        "POST",
        "/api/v1/finance/optimize",
        Some(request),
    )
    .await
    .expect("Request failed");

    assert_eq!(response.status(), 200, "Worker 3 portfolio optimization should succeed");

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");

    assert_eq!(body["status"], "success");
    assert!(body["data"]["weights"].is_object());

    let weights = body["data"]["weights"].as_object().unwrap();

    // Verify weights sum to ~1.0
    let weight_sum: f64 = weights.values()
        .filter_map(|v| v.as_f64())
        .sum();
    assert!((weight_sum - 1.0).abs() < 0.01, "Weights should sum to 1.0");

    // Verify all weights within constraints
    for weight in weights.values().filter_map(|v| v.as_f64()) {
        assert!(weight >= 0.04 && weight <= 0.41, "Weights should respect constraints");
    }

    // Check metrics
    assert!(body["data"]["expected_return"].as_f64().unwrap() > 0.0);
    assert!(body["data"]["portfolio_risk"].as_f64().unwrap() > 0.0);
    assert!(body["data"]["sharpe_ratio"].as_f64().unwrap() > 0.0);

    // Check GPU acceleration was used (if available)
    if let Some(gpu_time) = body["data"]["gpu_computation_time_ms"].as_f64() {
        assert!(gpu_time >= 0.0);
    }

    println!("✅ Worker 3 portfolio optimization (GPU-accelerated): PASS");
}

#[tokio::test]
async fn test_worker3_portfolio_risk_parity() {
    

    // Test risk parity strategy
    let request = json!({
        "assets": [
            {
                "symbol": "SPY",
                "expected_return": 0.08,
                "volatility": 0.15,
                "current_price": 450.0
            },
            {
                "symbol": "TLT",
                "expected_return": 0.03,
                "volatility": 0.10,
                "current_price": 95.0
            },
            {
                "symbol": "GLD",
                "expected_return": 0.05,
                "volatility": 0.12,
                "current_price": 180.0
            }
        ],
        "constraints": {
            "max_position_size": 1.0,
            "min_position_size": 0.0
        },
        "objective": {
            "Custom": {
                "strategy": "RiskParity"
            }
        }
    });

    let response = make_request(
        &client,
        "POST",
        "/api/v1/finance/optimize",
        Some(request),
    )
    .await
    .expect("Request failed");

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");
    assert_eq!(body["status"], "success");

    println!("✅ Worker 3 risk parity optimization: PASS");
}

#[tokio::test]
async fn test_worker7_motion_planning() {
    

    // Test Active Inference motion planning
    let request = json!({
        "robot_id": "test_robot_1",
        "start_state": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0]
        },
        "goal_state": {
            "position": [5.0, 3.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0]
        },
        "obstacles": [],
        "constraints": {
            "max_velocity": 2.0,
            "max_acceleration": 1.0
        }
    });

    let response = make_request(
        &client,
        "POST",
        "/api/v1/robotics/plan",
        Some(request),
    )
    .await
    .expect("Request failed");

    assert_eq!(response.status(), 200, "Worker 7 motion planning should succeed");

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");

    assert_eq!(body["status"], "success");
    assert!(body["data"]["trajectory"].is_array());

    let trajectory = body["data"]["trajectory"].as_array().unwrap();
    assert!(trajectory.len() > 0, "Should generate trajectory waypoints");

    // Verify trajectory starts at start position
    let first_waypoint = &trajectory[0];
    assert_eq!(first_waypoint["state"]["position"][0].as_f64().unwrap(), 0.0);
    assert_eq!(first_waypoint["state"]["position"][1].as_f64().unwrap(), 0.0);

    // Verify trajectory ends near goal position
    let last_waypoint = trajectory.last().unwrap();
    let final_x = last_waypoint["state"]["position"][0].as_f64().unwrap();
    let final_y = last_waypoint["state"]["position"][1].as_f64().unwrap();
    assert!((final_x - 5.0).abs() < 1.0, "Should reach near goal X");
    assert!((final_y - 3.0).abs() < 1.0, "Should reach near goal Y");

    // Check metrics
    assert!(body["data"]["total_time"].as_f64().unwrap() > 0.0);
    assert!(body["data"]["total_distance"].as_f64().unwrap() > 0.0);
    assert!(body["data"]["planning_time_ms"].as_f64().unwrap() > 0.0);

    // Verify collision-free status
    assert!(body["data"]["is_collision_free"].is_boolean());

    println!("✅ Worker 7 Active Inference motion planning: PASS");
}

#[tokio::test]
async fn test_worker7_motion_planning_with_obstacles() {
    

    // Test motion planning with obstacle avoidance
    let request = json!({
        "robot_id": "test_robot_2",
        "start_state": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0]
        },
        "goal_state": {
            "position": [10.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0]
        },
        "obstacles": [
            {
                "id": "obstacle_1",
                "position": [5.0, 0.0, 0.0],
                "radius": 1.0,
                "is_static": true
            }
        ],
        "constraints": {
            "max_velocity": 2.0,
            "max_acceleration": 1.0
        }
    });

    let response = make_request(
        &client,
        "POST",
        "/api/v1/robotics/plan",
        Some(request),
    )
    .await
    .expect("Request failed");

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");
    assert_eq!(body["status"], "success");

    let trajectory = body["data"]["trajectory"].as_array().unwrap();
    assert!(trajectory.len() > 0);

    // With obstacles, path should deviate from straight line
    // Check that middle waypoints don't pass through obstacle at (5, 0)
    let mid_idx = trajectory.len() / 2;
    let mid_waypoint = &trajectory[mid_idx];
    let mid_x = mid_waypoint["state"]["position"][0].as_f64().unwrap();
    let mid_y = mid_waypoint["state"]["position"][1].as_f64().unwrap();

    // If near obstacle X position, Y should be non-zero (avoiding)
    if (mid_x - 5.0).abs() < 2.0 {
        assert!(mid_y.abs() > 0.5, "Should deviate to avoid obstacle");
    }

    println!("✅ Worker 7 motion planning with obstacles: PASS");
}

#[tokio::test]
async fn test_cross_worker_integration() {
    

    // Test that all three worker integrations can run in sequence without conflicts

    // 1. Worker 1: Forecast portfolio returns
    let ts_request = json!({
        "historical_data": [0.10, 0.12, 0.08, 0.15, 0.11, 0.13, 0.09, 0.14],
        "horizon": 3,
        "method": {
            "Arima": { "p": 1, "d": 0, "q": 1 }
        },
        "include_uncertainty": false
    });

    let ts_response = make_request("POST", "/api/v1/timeseries/forecast", Some(ts_request))
        .await
        .expect("Time series request failed");
    assert_eq!(ts_response.status(), 200);

    // 2. Worker 3: Optimize portfolio
    let portfolio_request = json!({
        "assets": [
            { "symbol": "A", "expected_return": 0.10, "volatility": 0.20, "current_price": 100.0 },
            { "symbol": "B", "expected_return": 0.12, "volatility": 0.25, "current_price": 150.0 }
        ],
        "constraints": { "max_position_size": 0.8, "min_position_size": 0.2 },
        "objective": "MaximizeSharpe"
    });

    let portfolio_response = make_request("POST", "/api/v1/finance/optimize", Some(portfolio_request))
        .await
        .expect("Portfolio request failed");
    assert_eq!(portfolio_response.status(), 200);

    // 3. Worker 7: Plan robot motion
    let robotics_request = json!({
        "robot_id": "test_robot",
        "start_state": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [],
            "velocity": [0.0, 0.0, 0.0]
        },
        "goal_state": {
            "position": [3.0, 3.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [],
            "velocity": [0.0, 0.0, 0.0]
        },
        "obstacles": [],
        "constraints": { "max_velocity": 2.0, "max_acceleration": 1.0 }
    });

    let robotics_response = make_request("POST", "/api/v1/robotics/plan", Some(robotics_request))
        .await
        .expect("Robotics request failed");
    assert_eq!(robotics_response.status(), 200);

    println!("✅ Cross-worker integration (Workers 1, 3, 7): PASS");
}

#[tokio::test]
async fn test_worker_integration_performance() {
    
    use std::time::Instant;

    // Benchmark Worker 1 ARIMA
    let start = Instant::now();
    let request = json!({
        "historical_data": vec![100.0; 50],
        "horizon": 10,
        "method": { "Arima": { "p": 2, "d": 1, "q": 1 } },
        "include_uncertainty": true
    });
    let _ = make_request("POST", "/api/v1/timeseries/forecast", Some(request))
        .await
        .expect("Worker 1 request failed");
    let worker1_time = start.elapsed();

    // Benchmark Worker 3 Portfolio
    let start = Instant::now();
    let request = json!({
        "assets": vec![
            { "symbol": "S1", "expected_return": 0.10, "volatility": 0.20, "current_price": 100.0 },
            { "symbol": "S2", "expected_return": 0.12, "volatility": 0.22, "current_price": 110.0 },
            { "symbol": "S3", "expected_return": 0.08, "volatility": 0.18, "current_price": 90.0 },
        ],
        "constraints": { "max_position_size": 0.5, "min_position_size": 0.1 },
        "objective": "MaximizeSharpe"
    });
    let _ = make_request("POST", "/api/v1/finance/optimize", Some(request))
        .await
        .expect("Worker 3 request failed");
    let worker3_time = start.elapsed();

    // Benchmark Worker 7 Robotics
    let start = Instant::now();
    let request = json!({
        "robot_id": "perf_test",
        "start_state": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [],
            "velocity": [0.0, 0.0, 0.0]
        },
        "goal_state": {
            "position": [10.0, 10.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": [],
            "velocity": [0.0, 0.0, 0.0]
        },
        "obstacles": [],
        "constraints": { "max_velocity": 2.0, "max_acceleration": 1.0 }
    });
    let _ = make_request("POST", "/api/v1/robotics/plan", Some(request))
        .await
        .expect("Worker 7 request failed");
    let worker7_time = start.elapsed();

    println!("⏱️  Performance Benchmarks:");
    println!("   Worker 1 (ARIMA): {:?}", worker1_time);
    println!("   Worker 3 (Portfolio): {:?}", worker3_time);
    println!("   Worker 7 (Robotics): {:?}", worker7_time);

    // All should complete in reasonable time (< 5 seconds each)
    assert!(worker1_time.as_secs() < 5, "Worker 1 should complete quickly");
    assert!(worker3_time.as_secs() < 5, "Worker 3 should complete quickly");
    assert!(worker7_time.as_secs() < 5, "Worker 7 should complete quickly");

    println!("✅ Worker integration performance: PASS");
}
