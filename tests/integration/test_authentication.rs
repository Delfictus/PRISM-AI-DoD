// Integration tests for authentication and authorization

use reqwest;
use serde_json::json;

use super::common::*;

#[tokio::test]
async fn test_bearer_token_authentication() {
    let response = get_authenticated("/api/v1/pwsa/health", DEFAULT_API_KEY)
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        200,
        "Valid Bearer token should allow access"
    );
}

#[tokio::test]
async fn test_api_key_header_authentication() {
    let client = create_test_client();

    let response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .header("X-API-Key", DEFAULT_API_KEY)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        200,
        "Valid X-API-Key header should allow access"
    );
}

#[tokio::test]
async fn test_missing_authentication() {
    let client = create_test_client();

    let response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        401,
        "Missing authentication should return 401"
    );
}

#[tokio::test]
async fn test_invalid_bearer_token() {
    let response = get_authenticated("/api/v1/pwsa/health", "invalid-token-12345")
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        401,
        "Invalid Bearer token should return 401"
    );
}

#[tokio::test]
async fn test_malformed_authorization_header() {
    let client = create_test_client();

    let response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .header("Authorization", "NotBearer token")
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        401,
        "Malformed Authorization header should return 401"
    );
}

#[tokio::test]
async fn test_admin_role_access() {
    // Admin should be able to access all endpoints
    let response = get_authenticated("/api/v1/pwsa/detect", ADMIN_API_KEY)
        .await
        .unwrap();

    assert!(
        response.status() == 200 || response.status() == 400,
        "Admin should have access (200 or 400 for missing body)"
    );
}

#[tokio::test]
async fn test_readonly_role_restrictions() {
    let client = create_test_client();

    // Read-only should be able to GET
    let get_response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .header("Authorization", format!("Bearer {}", READ_ONLY_API_KEY))
        .send()
        .await
        .unwrap();

    assert_eq!(get_response.status(), 200, "Read-only can GET");

    // Read-only should NOT be able to POST
    let post_response = client
        .post(&format!("{}/api/v1/pwsa/detect", BASE_URL))
        .header("Authorization", format!("Bearer {}", READ_ONLY_API_KEY))
        .json(&json!({
            "sv_id": 42,
            "timestamp": 1234567890,
            "ir_frame": {
                "width": 640,
                "height": 480,
                "centroid_x": 320.0,
                "centroid_y": 240.0,
                "hotspot_count": 5
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        post_response.status(),
        403,
        "Read-only should not be able to POST"
    );
}

#[tokio::test]
async fn test_user_role_access() {
    // Regular user should have read/write access
    let response = post_authenticated(
        "/api/v1/pwsa/detect",
        DEFAULT_API_KEY,
        &json!({
            "sv_id": 42,
            "timestamp": 1234567890,
            "ir_frame": {
                "width": 640,
                "height": 480,
                "centroid_x": 320.0,
                "centroid_y": 240.0,
                "hotspot_count": 5
            }
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), 200, "User can POST");
}

#[tokio::test]
async fn test_token_case_sensitivity() {
    let response = get_authenticated("/api/v1/pwsa/health", "TEST-KEY")
        .await
        .unwrap();

    // Tokens should be case-sensitive
    assert_eq!(
        response.status(),
        401,
        "Tokens should be case-sensitive"
    );
}

#[tokio::test]
async fn test_multiple_auth_methods() {
    let client = create_test_client();

    // Send both Bearer and X-API-Key headers
    // Should use Bearer token (takes precedence)
    let response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .header("Authorization", format!("Bearer {}", DEFAULT_API_KEY))
        .header("X-API-Key", "different-key")
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        200,
        "Should accept valid Bearer token even with X-API-Key present"
    );
}

#[tokio::test]
async fn test_empty_bearer_token() {
    let client = create_test_client();

    let response = client
        .get(&format!("{}/api/v1/pwsa/health", BASE_URL))
        .header("Authorization", "Bearer ")
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        401,
        "Empty Bearer token should return 401"
    );
}

#[tokio::test]
async fn test_special_characters_in_token() {
    let response = get_authenticated("/api/v1/pwsa/health", "token-with-!@#$%^&*()")
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        401,
        "Invalid token with special characters should return 401"
    );
}
