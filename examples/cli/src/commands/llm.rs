/*!
 * LLM command implementations
 */

use anyhow::{Context, Result};
use serde_json::{json, Value};
use crate::client::PrismClient;
use crate::output;
use crate::LlmCommands;

pub async fn execute(client: &PrismClient, cmd: LlmCommands, format: &str) -> Result<()> {
    match cmd {
        LlmCommands::Query { prompt, model, temperature, max_tokens } => {
            query_llm(client, &prompt, model.as_deref(), temperature, max_tokens, format).await
        }
        LlmCommands::Consensus { prompt, models, strategy, temperature, max_tokens } => {
            llm_consensus(client, &prompt, &models, &strategy, temperature, max_tokens, format).await
        }
        LlmCommands::Models => {
            list_models(client, format).await
        }
    }
}

async fn query_llm(
    client: &PrismClient,
    prompt: &str,
    model: Option<&str>,
    temperature: f64,
    max_tokens: i32,
    format: &str,
) -> Result<()> {
    let mut body = json!({
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    });

    if let Some(m) = model {
        body["model"] = json!(m);
    }

    let response = client.post("/api/v1/llm/query", &body).await?;

    if format == "table" {
        output::print_success("LLM query complete");
        if let Some(data) = response.get("data") {
            print_llm_response(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn llm_consensus(
    client: &PrismClient,
    prompt: &str,
    models_path: &str,
    strategy: &str,
    temperature: f64,
    max_tokens: i32,
    format: &str,
) -> Result<()> {
    let models: Value = load_json_file(models_path)?;

    let body = json!({
        "prompt": prompt,
        "models": models,
        "strategy": strategy,
        "temperature": temperature,
        "max_tokens": max_tokens,
    });

    let response = client.post("/api/v1/llm/consensus", &body).await?;

    if format == "table" {
        output::print_success("LLM consensus complete");
        if let Some(data) = response.get("data") {
            print_consensus_response(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn list_models(client: &PrismClient, format: &str) -> Result<()> {
    let response = client.get("/api/v1/llm/models").await?;

    if format == "table" {
        if let Some(data) = response.get("data") {
            if let Some(models) = data.get("models").and_then(|v| v.as_array()) {
                output::print_success(&format!("Found {} available models", models.len()));
                for model in models {
                    if let Some(name) = model.get("name").and_then(|v| v.as_str()) {
                        println!("  • {}", name);
                    }
                }
            }
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

fn load_json_file(path: &str) -> Result<Value> {
    let content = std::fs::read_to_string(path)
        .context(format!("Failed to read file: {}", path))?;
    serde_json::from_str(&content)
        .context("Failed to parse JSON")
}

fn print_llm_response(data: &Value) {
    if let Some(text) = data.get("text").and_then(|v| v.as_str()) {
        println!("\n{}\n", text);
    }
    if let Some(model) = data.get("model_used").and_then(|v| v.as_str()) {
        println!("  Model: {}", model);
    }
    if let Some(tokens) = data.get("tokens_used").and_then(|v| v.as_i64()) {
        println!("  Tokens: {}", tokens);
    }
    if let Some(cost) = data.get("cost_usd").and_then(|v| v.as_f64()) {
        println!("  Cost: ${:.4}", cost);
    }
}

fn print_consensus_response(data: &Value) {
    if let Some(text) = data.get("consensus_text").and_then(|v| v.as_str()) {
        println!("\n{}\n", text);
    }
    if let Some(confidence) = data.get("confidence").and_then(|v| v.as_f64()) {
        println!("  Confidence: {:.1}%", confidence * 100.0);
    }
    if let Some(agreement) = data.get("agreement_rate").and_then(|v| v.as_f64()) {
        println!("  Agreement: {:.1}%", agreement * 100.0);
    }
    if let Some(cost) = data.get("total_cost_usd").and_then(|v| v.as_f64()) {
        println!("  Total Cost: ${:.4}", cost);
    }
}
