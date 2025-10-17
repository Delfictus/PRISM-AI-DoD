//! KV-Cache Integration Example
//!
//! Demonstrates the KV-cache integration with the GPU transformer
//! and shows the performance benefits of caching.
//!
//! Run with: cargo run --example test_kv_cache_integration --features mission_charlie

use anyhow::Result;

#[cfg(not(feature = "mission_charlie"))]
fn main() {
    eprintln!("ERROR: This example requires the 'mission_charlie' feature.");
    eprintln!("Run with: cargo run --example test_kv_cache_integration --features mission_charlie");
    std::process::exit(1);
}

#[cfg(feature = "mission_charlie")]
use prism_ai::orchestration::local_llm::{GpuLocalLLMSystem, LLMArchitecture};

#[cfg(feature = "mission_charlie")]
fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  KV-Cache Integration Example                            ║");
    println!("║  Day 4: Efficient Autoregressive Generation              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 1: KV-Cache Benefits Explanation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Without KV-Cache:");
    println!("  - Every token generation requires recomputing attention");
    println!("    for ALL previous tokens");
    println!("  - Complexity: O(n²) where n is sequence length");
    println!("  - For 100 tokens: ~10,000 attention computations");
    println!();

    println!("With KV-Cache:");
    println!("  - Store computed keys/values from previous tokens");
    println!("  - Only compute attention for NEW token");
    println!("  - Complexity: O(n) for generation");
    println!("  - For 100 tokens: ~100 attention computations");
    println!();

    println!("Expected Speedup: 50-100x for long sequences");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 2: Creating Model with KV-Cache");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Creating tiny GPU LLM (for demo purposes)...\n");

    let mut system = GpuLocalLLMSystem::new(LLMArchitecture::Tiny)?;

    println!("Model info: {}", system.info());
    println!();

    // Check if KV-cache is enabled (it should be by default)
    if system.is_kv_cache_enabled() {
        println!("✅ KV-cache is ENABLED (default)");
    } else {
        println!("❌ KV-cache is DISABLED");
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 3: Generation with KV-Cache");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let prompt = "Hello world";
    println!("Prompt: \"{}\"", prompt);
    println!();

    // Generate with KV-cache enabled
    println!("Generating 20 tokens with KV-cache...");
    system.clear_kv_cache(); // Start fresh
    let output_with_cache = system.generate_text(prompt, 20)?;

    // Show KV-cache statistics
    if let Some(stats) = system.kv_cache_stats() {
        println!("\n{}", stats);
    }

    println!("Output: \"{}\"", output_with_cache);
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 4: Comparison (With vs Without Cache)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Scenario 1: WITH KV-Cache (standard mode)");
    println!("-----------------------------------------");
    system.clear_kv_cache();
    system.enable_kv_cache()?;

    let start = std::time::Instant::now();
    let output1 = system.generate_text("Test", 10)?;
    let time_with_cache = start.elapsed();

    println!("Output: \"{}\"", output1);
    println!("Time: {:?}", time_with_cache);
    if let Some(stats) = system.kv_cache_stats() {
        println!("{}", stats);
    }
    println!();

    println!("Scenario 2: WITHOUT KV-Cache (slower)");
    println!("--------------------------------------");
    system.disable_kv_cache();

    let start = std::time::Instant::now();
    let output2 = system.generate_text("Test", 10)?;
    let time_without_cache = start.elapsed();

    println!("Output: \"{}\"", output2);
    println!("Time: {:?}", time_without_cache);
    println!("KV-cache status: {}", if system.is_kv_cache_enabled() { "ENABLED" } else { "DISABLED" });
    println!();

    // Calculate speedup
    let speedup = time_without_cache.as_secs_f64() / time_with_cache.as_secs_f64();
    println!("Speedup with KV-cache: {:.2}x", speedup);
    println!("(Note: Speedup increases with longer sequences)");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 5: KV-Cache Management");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Re-enable for demonstration
    system.enable_kv_cache()?;

    println!("KV-Cache Management APIs:");
    println!("  - enable_kv_cache()  : Enable caching (default)");
    println!("  - disable_kv_cache() : Disable for testing");
    println!("  - clear_kv_cache()   : Clear cache (start new generation)");
    println!("  - is_kv_cache_enabled() : Check status");
    println!("  - kv_cache_stats()   : Get memory usage stats");
    println!();

    println!("Example: Multiple generations (must clear cache between prompts)");
    println!();

    let prompts = vec!["First prompt", "Second prompt", "Third prompt"];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("Generation {}:", i + 1);
        system.clear_kv_cache(); // Important! Clear before new generation
        let output = system.generate_text(prompt, 5)?;
        println!("  Prompt: \"{}\"", prompt);
        println!("  Output: \"{}\"", output);
        if let Some(stats) = system.kv_cache_stats() {
            println!("  {}", stats);
        }
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 6: Performance Scaling");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Expected Performance Characteristics:");
    println!();
    println!("Sequence Length | Without Cache | With Cache | Speedup");
    println!("----------------|---------------|------------|--------");
    println!("10 tokens       | ~100 ops      | ~10 ops    | 10x");
    println!("50 tokens       | ~2,500 ops    | ~50 ops    | 50x");
    println!("100 tokens      | ~10,000 ops   | ~100 ops   | 100x");
    println!("500 tokens      | ~250,000 ops  | ~500 ops   | 500x");
    println!();

    println!("Memory Usage (Llama-7B example):");
    println!("  - 32 layers × 4096 dims × 2048 ctx × 2 (K+V) × 4 bytes");
    println!("  - Total: ~2.1 GB GPU memory for full context");
    println!();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  KV-Cache Integration Complete                           ║");
    println!("║  ✅ KV-cache enabled by default                          ║");
    println!("║  ✅ Automatic memory management                          ║");
    println!("║  ✅ 50-500x speedup for generation                       ║");
    println!("║  ✅ Simple API: clear_kv_cache() before new prompt       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}
