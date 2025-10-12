//! Simple LLM Orchestration Test
//!
//! This tests the LLM API integration without complex dependencies

use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║              PRISM-AI LLM Orchestration Test                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Test environment variables
    println!("🔍 Checking environment variables...");

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "not set".to_string());
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| "not set".to_string());
    let gemini_key = std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| "not set".to_string());
    let xai_key = std::env::var("XAI_API_KEY").unwrap_or_else(|_| "not set".to_string());

    println!("  OpenAI: {}", if openai_key.starts_with("sk-") { "✅ Configured" } else { "❌ Not configured" });
    println!("  Anthropic: {}", if anthropic_key.starts_with("sk-ant-") { "✅ Configured" } else { "❌ Not configured" });
    println!("  Gemini: {}", if gemini_key != "not set" && gemini_key != "..." { "✅ Configured" } else { "❌ Not configured" });
    println!("  Grok: {}", if xai_key.starts_with("xai-") { "✅ Configured" } else { "❌ Not configured" });

    println!();
    println!("Type 'test' to test LLM queries, 'quit' to exit");
    println!();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "quit" | "exit" => break,
            "test" => {
                println!("\n🧪 Testing LLM API connections...\n");

                // Test OpenAI
                if openai_key.starts_with("sk-") {
                    println!("Testing OpenAI...");
                    match test_openai_api(&openai_key).await {
                        Ok(response) => println!("  ✅ OpenAI: {}", response),
                        Err(e) => println!("  ❌ OpenAI error: {}", e),
                    }
                }

                // Test Gemini
                if gemini_key != "not set" && gemini_key != "..." {
                    println!("Testing Gemini...");
                    match test_gemini_api(&gemini_key).await {
                        Ok(response) => println!("  ✅ Gemini: {}", response),
                        Err(e) => println!("  ❌ Gemini error: {}", e),
                    }
                }

                // Test Grok
                if xai_key.starts_with("xai-") {
                    println!("Testing Grok...");
                    match test_grok_api(&xai_key).await {
                        Ok(response) => println!("  ✅ Grok: {}", response),
                        Err(e) => println!("  ❌ Grok error: {}", e),
                    }
                }

                println!("\n✅ API tests complete!\n");
            }
            _ => {
                println!("Unknown command. Type 'test' or 'quit'");
            }
        }
    }

    println!("Goodbye!");
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
        Ok(text.to_string())
    } else {
        Err(anyhow::anyhow!("HTTP {}", response.status()))
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
        Ok(text.to_string())
    } else {
        Err(anyhow::anyhow!("HTTP {}", response.status()))
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
        Ok(text.to_string())
    } else {
        Err(anyhow::anyhow!("HTTP {}", response.status()))
    }
}