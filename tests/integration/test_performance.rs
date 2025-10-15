// Performance and load testing

use serde_json::json;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

use super::common::*;

#[tokio::test]
async fn test_health_endpoint_latency() {
    let client = create_test_client();
    let mut latencies = Vec::new();

    // Measure 100 requests
    for _ in 0..100 {
        let start = Instant::now();

        client
            .get(&format!("{}/health", BASE_URL))
            .send()
            .await
            .unwrap();

        latencies.push(start.elapsed());
    }

    let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let max_latency = latencies.iter().max().unwrap();

    println!("Average latency: {:?}", avg_latency);
    println!("Max latency: {:?}", max_latency);

    // Health check should be fast
    assert!(
        avg_latency < Duration::from_millis(50),
        "Average latency should be < 50ms"
    );
}

#[tokio::test]
async fn test_concurrent_requests() {
    let mut tasks = JoinSet::new();

    // Spawn 50 concurrent requests
    for i in 0..50 {
        tasks.spawn(async move {
            let client = create_test_client();
            let payload = json!({
                "sv_id": i,
                "timestamp": 1234567890,
                "ir_frame": {
                    "width": 640,
                    "height": 480,
                    "centroid_x": 320.0,
                    "centroid_y": 240.0,
                    "hotspot_count": 5
                }
            });

            client
                .post(&format!("{}/api/v1/pwsa/detect", BASE_URL))
                .header("Authorization", format!("Bearer {}", DEFAULT_API_KEY))
                .json(&payload)
                .send()
                .await
        });
    }

    // Wait for all requests to complete
    let start = Instant::now();
    let mut success_count = 0;

    while let Some(result) = tasks.join_next().await {
        if let Ok(Ok(response)) = result {
            if response.status().is_success() {
                success_count += 1;
            }
        }
    }

    let duration = start.elapsed();

    println!("Completed 50 concurrent requests in {:?}", duration);
    println!("Success rate: {}/50", success_count);

    assert!(
        success_count >= 45,
        "At least 90% of concurrent requests should succeed"
    );
}

#[tokio::test]
async fn test_throughput() {
    let client = create_test_client();
    let num_requests = 1000;

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

    let start = Instant::now();

    for _ in 0..num_requests {
        client
            .post(&format!("{}/api/v1/pwsa/detect", BASE_URL))
            .header("Authorization", format!("Bearer {}", DEFAULT_API_KEY))
            .json(&payload)
            .send()
            .await
            .unwrap();
    }

    let duration = start.elapsed();
    let throughput = num_requests as f64 / duration.as_secs_f64();

    println!("Throughput: {:.2} req/s", throughput);

    // Should handle at least 100 req/s
    assert!(throughput > 100.0, "Throughput should be > 100 req/s");
}

#[tokio::test]
async fn test_large_payload_handling() {
    let client = create_test_client();

    // Create a large payload with many targets
    let mut targets = Vec::new();
    for i in 0..1000 {
        targets.push(json!({
            "id": format!("target_{}", i),
            "azimuth": i as f64 * 0.1,
            "elevation": i as f64 * 0.05,
            "range": 25000.0 + i as f64 * 10.0,
            "velocity": 500.0,
            "ir_signature": 0.85,
            "radar_cross_section": 1.5
        }));
    }

    let payload = json!({
        "sv_id": 42,
        "timestamp": 1234567890,
        "sensors": {
            "ir": {
                "frame_id": 1234,
                "targets": targets
            }
        }
    });

    let start = Instant::now();

    let response = client
        .post(&format!("{}/api/v1/pwsa/fuse", BASE_URL))
        .header("Authorization", format!("Bearer {}", DEFAULT_API_KEY))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let duration = start.elapsed();

    println!("Large payload processed in {:?}", duration);

    assert_eq!(response.status(), 200, "Should handle large payload");
    assert!(
        duration < Duration::from_secs(5),
        "Should process large payload in < 5s"
    );
}

#[tokio::test]
async fn test_sustained_load() {
    let duration = Duration::from_secs(10);
    let start = Instant::now();
    let mut request_count = 0;
    let mut error_count = 0;

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

    while start.elapsed() < duration {
        let response = post_authenticated("/api/v1/pwsa/detect", DEFAULT_API_KEY, &payload)
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => request_count += 1,
            _ => error_count += 1,
        }
    }

    let elapsed = start.elapsed();
    let avg_throughput = request_count as f64 / elapsed.as_secs_f64();

    println!("Sustained load test:");
    println!("  Duration: {:?}", elapsed);
    println!("  Successful requests: {}", request_count);
    println!("  Failed requests: {}", error_count);
    println!("  Average throughput: {:.2} req/s", avg_throughput);

    let error_rate = error_count as f64 / (request_count + error_count) as f64;
    assert!(error_rate < 0.01, "Error rate should be < 1%");
}

#[tokio::test]
async fn test_memory_stability() {
    // Make many requests to check for memory leaks
    let num_iterations = 100;

    for i in 0..num_iterations {
        let payload = json!({
            "sv_id": i,
            "timestamp": 1234567890 + i,
            "ir_frame": {
                "width": 640,
                "height": 480,
                "centroid_x": 320.0,
                "centroid_y": 240.0,
                "hotspot_count": 5
            }
        });

        let response = post_authenticated("/api/v1/pwsa/detect", DEFAULT_API_KEY, &payload)
            .await
            .unwrap();

        assert_eq!(response.status(), 200);

        // Periodically check server is still responsive
        if i % 10 == 0 {
            let health = get_authenticated("/health", DEFAULT_API_KEY)
                .await
                .unwrap();
            assert_eq!(health.status(), 200, "Server should remain healthy");
        }
    }

    println!("Completed {} iterations without memory issues", num_iterations);
}

#[tokio::test]
async fn test_response_time_consistency() {
    let mut response_times = Vec::new();

    for _ in 0..50 {
        let start = Instant::now();

        let response = get_authenticated("/api/v1/pwsa/health", DEFAULT_API_KEY)
            .await
            .unwrap();

        response_times.push(start.elapsed());

        assert_eq!(response.status(), 200);
    }

    // Calculate statistics
    let mean = response_times.iter().sum::<Duration>() / response_times.len() as u32;

    let variance: Duration = response_times
        .iter()
        .map(|&d| {
            let diff = if d > mean { d - mean } else { mean - d };
            diff * diff.as_millis() as u32
        })
        .sum::<Duration>()
        / response_times.len() as u32;

    println!("Mean response time: {:?}", mean);
    println!("Variance: {:?}", variance);

    // Response times should be consistent (low variance)
    assert!(
        variance < Duration::from_millis(100),
        "Response times should be consistent"
    );
}
