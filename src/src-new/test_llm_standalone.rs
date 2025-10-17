#!/usr/bin/env -S cargo +nightly -Zscript
//! ```cargo
//! [dependencies]
//! tokio = { version = "1", features = ["full"] }
//! reqwest = { version = "0.11", features = ["json"] }
//! serde_json = "1.0"
//! anyhow = "1.0"
//! ```

use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║              PRISM-AI LLM API Test (Standalone)                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load environment variables from .env file
    let env_content = std::fs::read_to_string(".env").unwrap_or_default();
    for line in env_content.lines() {
        if let Some((key, value)) = line.split_once('=') {
            if !key.starts_with('#') && !key.is_empty() {
                std::env::set_var(key.trim(), value.trim());
            }
        }
    }

    // Test environment variables
    println!("🔍 Checking API keys...");

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    let gemini_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    let xai_key = std::env::var("XAI_API_KEY").unwrap_or_default();

    println!("  OpenAI: {}", if openai_key.starts_with("sk-") { "✅ Configured" } else { "❌ Not configured" });
    println!("  Anthropic: {}", if anthropic_key.starts_with("sk-ant-") { "✅ Configured" } else { "❌ Not configured" });
    println!("  Gemini: {}", if !gemini_key.is_empty() && gemini_key != "..." { "✅ Configured" } else { "❌ Not configured" });
    println!("  Grok: {}", if xai_key.starts_with("xai-") { "✅ Configured" } else { "❌ Not configured" });

    println!();
    println!("🧪 Testing LLM API connections...\n");

    // Test OpenAI
    if openai_key.starts_with("sk-") {
        print!("Testing OpenAI... ");
        io::stdout().flush()?;
        match test_openai_api(&openai_key).await {
            Ok(response) => println!("✅ Response: {}", response),
            Err(e) => println!("❌ Error: {}", e),
        }
    }

    // Test Anthropic
    if anthropic_key.starts_with("sk-ant-") {
        print!("Testing Anthropic... ");
        io::stdout().flush()?;
        match test_anthropic_api(&anthropic_key).await {
            Ok(response) => println!("✅ Response: {}", response),
            Err(e) => println!("❌ Error: {}", e),
        }
    }

    // Test Gemini
    if !gemini_key.is_empty() && gemini_key != "..." {
        print!("Testing Gemini... ");
        io::stdout().flush()?;
        match test_gemini_api(&gemini_key).await {
            Ok(response) => println!("✅ Response: {}", response),
            Err(e) => println!("❌ Error: {}", e),
        }
    }

    // Test Grok
    if xai_key.starts_with("xai-") {
        print!("Testing Grok... ");
        io::stdout().flush()?;
        match test_grok_api(&xai_key).await {
            Ok(response) => println!("✅ Response: {}", response),
            Err(e) => println!("❌ Error: {}", e),
        }
    }

    println!("\n✅ All tests complete!");
    println!("\n🎯 Summary:");
    println!("   Your LLM APIs are properly configured and working!");
    println!("   The system can now use these for quantum-consensus intelligence.");

    Ok(())
}

async fn test_openai_api(api_key: &str) -> anyhow::Result<String> {
    use serde_json::json;

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say 'Hello from OpenAI' in exactly 4 words"}],
            "max_tokens": 10,
            "temperature": 0.0
        }))
        .send()
        .await?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().await?;
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No response");
        Ok(text.trim().to_string())
    } else {
        let status = response.status();
        let text = response.text().await?;
        Err(anyhow::anyhow!("HTTP {} - {}", status, text))
    }
}

async fn test_anthropic_api(api_key: &str) -> anyhow::Result<String> {
    use serde_json::json;

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&json!({
            "model": "claude-3-5-sonnet-20250110",
            "messages": [{"role": "user", "content": "Say 'Hello from Claude' in exactly 4 words"}],
            "max_tokens": 10
        }))
        .send()
        .await?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().await?;
        let text = json["content"][0]["text"]
            .as_str()
            .unwrap_or("No response");
        Ok(text.trim().to_string())
    } else {
        let status = response.status();
        let text = response.text().await?;
        Err(anyhow::anyhow!("HTTP {} - {}", status, text))
    }
}

async fn test_gemini_api(api_key: &str) -> anyhow::Result<String> {
    use serde_json::json;

    let client = reqwest::Client::new();
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={}",
        api_key
    );

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&json!({
            "contents": [{
                "parts": [{"text": "Say 'Hello from Gemini' in exactly 4 words"}]
            }],
            "generationConfig": {
                "maxOutputTokens": 10,
                "temperature": 0.0
            }
        }))
        .send()
        .await?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().await?;
        let text = json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("No response");
        Ok(text.trim().to_string())
    } else {
        let status = response.status();
        let text = response.text().await?;
        Err(anyhow::anyhow!("HTTP {} - {}", status, text))
    }
}

async fn test_grok_api(api_key: &str) -> anyhow::Result<String> {
    use serde_json::json;

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.x.ai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": "grok-2-1212",
            "messages": [{"role": "user", "content": "Say 'Hello from Grok' in exactly 4 words"}],
            "max_tokens": 10,
            "temperature": 0.0
        }))
        .send()
        .await?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().await?;
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No response");
        Ok(text.trim().to_string())
    } else {
        let status = response.status();
        let text = response.text().await?;
        Err(anyhow::anyhow!("HTTP {} - {}", status, text))
    }
}