//! Test PRISM Assistant - Local LLM Integration
//!
//! This example demonstrates the PRISM Assistant with local GPU LLM.
//!
//! Usage:
//! ```bash
//! cargo run --example test_prism_assistant --features cuda
//! ```

use prism_ai::assistant::{PrismAssistant, AssistantMode};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════╗");
    println!("║  PRISM ASSISTANT TEST                    ║");
    println!("║  Local GPU LLM Integration               ║");
    println!("╚══════════════════════════════════════════╝\n");

    // Test 1: Create assistant with local model
    println!("🔧 Test 1: Creating PRISM Assistant in LocalOnly mode...\n");

    let model_path = Some("/home/diddy/Desktop/PRISM-Worker-6/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string());

    let mut assistant = PrismAssistant::new(
        AssistantMode::LocalOnly,
        model_path
    ).await?;

    println!("✅ Assistant created successfully!\n");

    // Test 2: Simple chat
    println!("🔧 Test 2: Testing simple chat...\n");

    let response = assistant.chat("Hello! What is 2 + 2?").await?;

    println!("📥 User: Hello! What is 2 + 2?");
    println!("📤 Assistant: {}", response.text);
    println!("⏱️  Latency: {}ms", response.latency_ms);
    println!("💰 Cost: ${:.4}", response.cost_usd);
    println!("🤖 Model: {}", response.model);
    println!();

    // Test 3: Status check
    println!("🔧 Test 3: Checking assistant status...\n");

    let status = assistant.status();
    println!("Status: {:?}\n", status);

    println!("✅ All tests completed successfully!");

    Ok(())
}
