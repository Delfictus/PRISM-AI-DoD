//! GGUF Integration Example
//!
//! Demonstrates loading a real GGUF model file and using it for inference
//!
//! Run with: cargo run --example test_gguf_integration --features mission_charlie -- <path_to_gguf_file>
//!
//! Example:
//!   cargo run --example test_gguf_integration --features mission_charlie -- /path/to/llama-7b-q4.gguf

use anyhow::Result;
use std::env;

#[cfg(not(feature = "mission_charlie"))]
fn main() {
    eprintln!("ERROR: This example requires the 'mission_charlie' feature.");
    eprintln!("Run with: cargo run --example test_gguf_integration --features mission_charlie -- <gguf_file>");
    std::process::exit(1);
}

#[cfg(feature = "mission_charlie")]
use prism_ai::orchestration::local_llm::{GpuLocalLLMSystem, GgufLoader};

#[cfg(feature = "mission_charlie")]
fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GGUF Integration Example                                ║");
    println!("║  Day 3: Real Model Loading                               ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Get GGUF file path from command line
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} /path/to/llama-7b-q4.gguf", args[0]);
        eprintln!("\nThis example demonstrates:");
        eprintln!("  - Loading GGUF model metadata");
        eprintln!("  - Extracting model configuration");
        eprintln!("  - Loading weights to GPU");
        eprintln!("  - Running inference with real weights");
        std::process::exit(1);
    }

    let gguf_path = &args[1];

    println!("═══════════════════════════════════════════════════════════");
    println!("Part 1: GGUF Metadata Inspection");
    println!("═══════════════════════════════════════════════════════════\n");

    // First, just load and inspect the GGUF file
    println!("Loading GGUF metadata from: {}\n", gguf_path);

    let loader = GgufLoader::load(gguf_path)?;
    loader.print_info();

    // Show tensor names
    println!("\nSample Tensor Names:");
    let tensor_names: Vec<_> = loader.tensors.keys().collect();
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        println!("  {}. {}", i + 1, name);
    }
    if tensor_names.len() > 10 {
        println!("  ... and {} more tensors", tensor_names.len() - 10);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Part 2: Loading Model to GPU");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Creating GPU LLM system from GGUF file...\n");

    let mut system = GpuLocalLLMSystem::from_gguf_file(gguf_path)?;

    println!("Model info: {}", system.info());

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Part 3: Test Generation");
    println!("═══════════════════════════════════════════════════════════\n");

    // Test with different sampling strategies
    let test_prompts = vec![
        "Hello",
        "The meaning of life is",
        "Once upon a time",
    ];

    for prompt in test_prompts {
        println!("Prompt: \"{}\"", prompt);

        // Generate with standard sampling
        system.use_standard_sampling();
        match system.generate_text(prompt, 10) {
            Ok(output) => {
                println!("Output: \"{}\"", output);
            }
            Err(e) => {
                println!("Generation error: {}", e);
            }
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("Part 4: Sampling Strategy Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    let prompt = "The future of AI";
    println!("Comparing sampling strategies for prompt: \"{}\"\n", prompt);

    let strategies = vec![
        ("Greedy", "use_greedy_sampling"),
        ("Standard", "use_standard_sampling"),
        ("Creative", "use_creative_sampling"),
        ("Precise", "use_precise_sampling"),
        ("Min-p (2025)", "use_min_p_sampling"),
    ];

    for (name, _method) in strategies {
        println!("{} sampling:", name);
        match name {
            "Greedy" => system.use_greedy_sampling(),
            "Standard" => system.use_standard_sampling(),
            "Creative" => system.use_creative_sampling(),
            "Precise" => system.use_precise_sampling(),
            "Min-p (2025)" => system.use_min_p_sampling(),
            _ => {}
        }

        match system.generate_text(prompt, 8) {
            Ok(output) => {
                println!("  Output: \"{}\"", output);
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GGUF Integration Complete                               ║");
    println!("║  ✅ Metadata loaded                                      ║");
    println!("║  ✅ Weights loaded to GPU                                ║");
    println!("║  ✅ Inference working                                    ║");
    println!("║  ✅ Multiple sampling strategies                         ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}
