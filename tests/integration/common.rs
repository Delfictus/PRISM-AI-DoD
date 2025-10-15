// Common test utilities and helpers

use reqwest;
use serde_json::Value;
use std::time::Duration;

pub const BASE_URL: &str = "http://localhost:8080";
pub const DEFAULT_API_KEY: &str = "test-key";
pub const ADMIN_API_KEY: &str = "admin-key";
pub const READ_ONLY_API_KEY: &str = "readonly-key";

/// Create a configured HTTP client for testing
pub fn create_test_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
}

/// Make an authenticated GET request
pub async fn get_authenticated(
    path: &str,
    api_key: &str,
) -> Result<reqwest::Response, reqwest::Error> {
    create_test_client()
        .get(&format!("{}{}", BASE_URL, path))
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
}

/// Make an authenticated POST request with JSON body
pub async fn post_authenticated(
    path: &str,
    api_key: &str,
    body: &Value,
) -> Result<reqwest::Response, reqwest::Error> {
    create_test_client()
        .post(&format!("{}{}", BASE_URL, path))
        .header("Authorization", format!("Bearer {}", api_key))
        .json(body)
        .send()
        .await
}

/// Check if API server is running
pub async fn is_server_running() -> bool {
    let client = create_test_client();
    client
        .get(&format!("{}/health", BASE_URL))
        .send()
        .await
        .is_ok()
}

/// Wait for server to be ready (with timeout)
pub async fn wait_for_server(timeout_secs: u64) -> Result<(), String> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    while start.elapsed() < timeout {
        if is_server_running().await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    Err(format!(
        "Server did not become ready within {} seconds",
        timeout_secs
    ))
}

/// Extract error message from API response
pub async fn extract_error_message(response: reqwest::Response) -> String {
    let body: Value = response.json().await.unwrap_or_default();
    body["error"]
        .as_str()
        .unwrap_or("Unknown error")
        .to_string()
}

/// Verify standard API response structure
pub fn verify_api_response(body: &Value) {
    assert!(body["success"].is_boolean(), "Should have 'success' field");
    assert!(body["data"].is_object() || body["data"].is_array() || body["data"].is_null(), "Should have 'data' field");
}

/// Generate random test data
pub fn generate_test_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("test_{}", timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_client() {
        let client = create_test_client();
        assert!(client.get(BASE_URL).build().is_ok());
    }

    #[test]
    fn test_generate_test_id() {
        let id1 = generate_test_id();
        let id2 = generate_test_id();
        assert_ne!(id1, id2, "Generated IDs should be unique");
        assert!(id1.starts_with("test_"));
    }
}
