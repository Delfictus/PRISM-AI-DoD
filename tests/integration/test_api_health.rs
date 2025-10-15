// Integration tests for API health endpoints
// These tests require the API server to be running

use reqwest;
use serde_json::Value;

const BASE_URL: &str = "http://localhost:8080";

#[tokio::test]
async fn test_health_endpoint() {
    let client = reqwest::Client::new();

    let response = client
        .get(&format!("{}/health", BASE_URL))
        .send()
        .await;

    assert!(response.is_ok(), "Health endpoint should be accessible");

    let response = response.unwrap();
    assert_eq!(response.status(), 200, "Health endpoint should return 200");

    let body: Value = response.json().await.unwrap();
    assert_eq!(body["status"], "ok", "Health status should be 'ok'");
}

#[tokio::test]
async fn test_root_endpoint() {
    let client = reqwest::Client::new();

    let response = client
        .get(&format!("{}/", BASE_URL))
        .send()
        .await;

    assert!(response.is_ok(), "Root endpoint should be accessible");

    let response = response.unwrap();
    assert_eq!(response.status(), 200, "Root endpoint should return 200");

    let body: Value = response.json().await.unwrap();
    assert!(body["name"].is_string(), "Should have 'name' field");
    assert!(body["version"].is_string(), "Should have 'version' field");
    assert_eq!(body["status"], "operational", "Status should be 'operational'");
}

#[tokio::test]
async fn test_subsystem_health() {
    let client = reqwest::Client::new();
    let subsystems = vec!["pwsa", "finance", "telecom", "robotics", "llm", "timeseries", "pixels"];

    for subsystem in subsystems {
        let response = client
            .get(&format!("{}/api/v1/{}/health", BASE_URL, subsystem))
            .header("Authorization", "Bearer test-key")
            .send()
            .await;

        assert!(
            response.is_ok(),
            "Health endpoint for {} should be accessible",
            subsystem
        );

        let response = response.unwrap();
        assert_eq!(
            response.status(),
            200,
            "{} health should return 200",
            subsystem
        );

        let body: Value = response.json().await.unwrap();
        assert!(
            body["data"]["status"].is_string(),
            "{} should have status field",
            subsystem
        );
    }
}

#[tokio::test]
async fn test_unauthenticated_access() {
    let client = reqwest::Client::new();

    // Try to access protected endpoint without auth
    let response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        401,
        "Protected endpoints should require authentication"
    );
}
