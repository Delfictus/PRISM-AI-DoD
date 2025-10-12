// Integration tests for LLM endpoints

use serde_json::{json, Value};

use super::common::*;

#[tokio::test]
async fn test_llm_simple_query() {
    let payload = json!({
        "prompt": "What is 2+2?",
        "temperature": 0.7,
        "max_tokens": 100
    });

    let response = post_authenticated("/api/v1/llm/query", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["text"].is_string(), "Should have response text");
    assert!(data["model_used"].is_string(), "Should specify model used");
    assert!(data["tokens_used"].is_number(), "Should report tokens used");
    assert!(data["latency_ms"].is_number(), "Should report latency");
}

#[tokio::test]
async fn test_llm_consensus_query() {
    let payload = json!({
        "prompt": "What is the capital of France?",
        "models": [
            {"name": "gpt-4", "weight": 1.0},
            {"name": "gpt-3.5-turbo", "weight": 0.8},
            {"name": "claude-3", "weight": 1.0}
        ],
        "strategy": "majority_vote",
        "temperature": 0.1,
        "max_tokens": 50
    });

    let response = post_authenticated("/api/v1/llm/consensus", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["consensus_text"].is_string(), "Should have consensus text");
    assert!(data["confidence"].is_number(), "Should have confidence score");
    assert!(data["individual_responses"].is_array(), "Should have individual responses");
    assert!(data["strategy"].is_string(), "Should specify strategy used");
}

#[tokio::test]
async fn test_llm_weighted_consensus() {
    let payload = json!({
        "prompt": "Explain quantum entanglement in one sentence.",
        "models": [
            {"name": "gpt-4", "weight": 2.0},  // Higher weight
            {"name": "gpt-3.5-turbo", "weight": 1.0}
        ],
        "strategy": "weighted_average",
        "temperature": 0.3,
        "max_tokens": 200
    });

    let response = post_authenticated("/api/v1/llm/consensus", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    // Verify weighted contributions
    let responses = data["individual_responses"].as_array().unwrap();
    for resp in responses {
        assert!(resp["weight"].is_number(), "Each response should have weight");
        assert!(resp["contribution"].is_number(), "Each should have contribution score");
    }
}

#[tokio::test]
async fn test_llm_batch_query() {
    let payload = json!({
        "queries": [
            {"prompt": "What is 2+2?", "max_tokens": 50},
            {"prompt": "What is 3+3?", "max_tokens": 50},
            {"prompt": "What is 4+4?", "max_tokens": 50}
        ],
        "model": "gpt-3.5-turbo",
        "temperature": 0.1
    });

    let response = post_authenticated("/api/v1/llm/batch", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["results"].is_array(), "Should have results array");
    let results = data["results"].as_array().unwrap();
    assert_eq!(results.len(), 3, "Should have 3 results");

    for result in results {
        assert!(result["text"].is_string(), "Each result should have text");
        assert!(result["tokens_used"].is_number(), "Each should report tokens");
    }
}

#[tokio::test]
async fn test_llm_invalid_model() {
    let payload = json!({
        "prompt": "Test prompt",
        "model": "invalid-model-xyz",
        "temperature": 0.7,
        "max_tokens": 100
    });

    let response = post_authenticated("/api/v1/llm/query", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Should return 400 for invalid model");
}

#[tokio::test]
async fn test_llm_temperature_bounds() {
    // Test temperature > 2.0 (invalid)
    let payload = json!({
        "prompt": "Test",
        "temperature": 2.5,
        "max_tokens": 100
    });

    let response = post_authenticated("/api/v1/llm/query", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Temperature > 2.0 should be invalid");
}

#[tokio::test]
async fn test_llm_max_tokens_limit() {
    let payload = json!({
        "prompt": "Test",
        "temperature": 0.7,
        "max_tokens": 100000  // Excessive
    });

    let response = post_authenticated("/api/v1/llm/query", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Excessive max_tokens should be invalid");
}

#[tokio::test]
async fn test_llm_list_models() {
    let response = get_authenticated("/api/v1/llm/models", DEFAULT_API_KEY)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["models"].is_array(), "Should have models array");
    let models = data["models"].as_array().unwrap();
    assert!(!models.is_empty(), "Should have at least one model");

    for model in models {
        assert!(model["name"].is_string(), "Each model should have name");
        assert!(model["provider"].is_string(), "Each model should have provider");
        assert!(model["max_tokens"].is_number(), "Each model should have max_tokens");
    }
}

#[tokio::test]
async fn test_llm_usage_stats() {
    let response = get_authenticated("/api/v1/llm/usage", DEFAULT_API_KEY)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["total_queries"].is_number(), "Should have total queries");
    assert!(data["total_tokens"].is_number(), "Should have total tokens");
    assert!(data["total_cost_usd"].is_number(), "Should have total cost");
}

#[tokio::test]
async fn test_llm_empty_prompt() {
    let payload = json!({
        "prompt": "",
        "temperature": 0.7,
        "max_tokens": 100
    });

    let response = post_authenticated("/api/v1/llm/query", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Empty prompt should be invalid");
}
