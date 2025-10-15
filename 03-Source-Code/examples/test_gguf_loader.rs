//! GGUF Model Loader Test
//!
//! Demonstrates loading and inspecting GGUF model files
//!
//! Usage:
//!   cargo run --example test_gguf_loader --features cuda -- <path_to_gguf_file>

use anyhow::Result;
use prism_ai::orchestration::local_llm::{GgufLoader, GgufGpuLoader};
use std::env;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════╗");
    println!("║  GGUF MODEL LOADER TEST                  ║");
    println!("╚══════════════════════════════════════════╝\n");

    // Get GGUF file path from command line
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path_to_gguf_file>", args[0]);
        println!("\nExample:");
        println!("  cargo run --example test_gguf_loader --features cuda -- /path/to/model.gguf");
        println!("\nYou can download GGUF models from:");
        println!("  https://huggingface.co/TheBloke/Llama-2-7B-GGUF");
        println!("  https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF");
        return Ok(());
    }

    let model_path = &args[1];
    println!("Loading model from: {}\n", model_path);

    // Test 1: Load and inspect model metadata
    println!("═══ TEST 1: Load Model Metadata ═══\n");
    let loader = GgufLoader::load(model_path)?;
    loader.print_info();

    // Test 2: List some tensors
    println!("\n═══ TEST 2: Tensor Inspection ═══\n");
    println!("First 10 tensors:");
    for (i, (name, tensor)) in loader.tensors.iter().take(10).enumerate() {
        println!("  {}. {} - {:?} - {} elements ({} bytes)",
            i + 1,
            name,
            tensor.data_type,
            tensor.element_count(),
            tensor.size_bytes()
        );
    }

    if loader.tensors.len() > 10 {
        println!("  ... and {} more tensors", loader.tensors.len() - 10);
    }

    // Test 3: Try to load a tensor to GPU
    println!("\n═══ TEST 3: GPU Weight Loading ═══\n");

    // Find the token embedding tensor (common names)
    let embedding_tensor_names = vec![
        "token_embd.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    ];

    let mut found_tensor = None;
    for name in &embedding_tensor_names {
        if loader.tensors.contains_key(*name) {
            found_tensor = Some(*name);
            break;
        }
    }

    if let Some(tensor_name) = found_tensor {
        println!("Found embedding tensor: {}", tensor_name);

        // Try to load to GPU
        match GgufGpuLoader::new(model_path, 0) {
            Ok(mut gpu_loader) => {
                println!("✓ GPU loader created");

                match gpu_loader.load_tensor_to_gpu(tensor_name) {
                    Ok(gpu_tensor) => {
                        println!("✓ Tensor loaded to GPU");
                        println!("  GPU memory size: {} MB",
                            gpu_tensor.len() * 4 / 1024 / 1024);
                        println!("\n✅ GPU loading test successful!");
                    }
                    Err(e) => {
                        println!("⚠️  Could not load tensor to GPU: {}", e);
                        println!("   This is OK if quantization type is not yet supported");
                    }
                }
            }
            Err(e) => {
                println!("⚠️  Could not initialize GPU: {}", e);
                println!("   Make sure you have CUDA and compatible GPU");
            }
        }
    } else {
        println!("⚠️  Could not find embedding tensor with common names");
        println!("   Available tensors: {:?}",
            loader.tensors.keys().take(5).collect::<Vec<_>>());
    }

    // Test 4: Metadata extraction
    println!("\n═══ TEST 4: Model Configuration ═══\n");

    if let Some(arch) = loader.architecture() {
        println!("Architecture: {}", arch);
    }

    if let Some(vocab) = loader.vocab_size() {
        println!("Vocabulary size: {}", vocab);
    }

    if let Some(dim) = loader.embedding_dim() {
        println!("Embedding dimension: {}", dim);
    }

    if let Some(layers) = loader.layer_count() {
        println!("Number of layers: {}", layers);
    }

    if let Some(heads) = loader.head_count() {
        println!("Attention heads: {}", heads);
    }

    if let Some(ctx) = loader.context_length() {
        println!("Context length: {}", ctx);
    }

    println!("\n╔══════════════════════════════════════════╗");
    println!("║  ALL TESTS COMPLETED                     ║");
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}
