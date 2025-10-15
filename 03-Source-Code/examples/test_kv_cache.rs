//! KV-Cache Demonstration
//!
//! Shows how KV-cache improves transformer inference performance
//!
//! Usage:
//!   cargo run --example test_kv_cache --features cuda

use anyhow::Result;
use prism_ai::orchestration::local_llm::TransformerKVCache;
use std::sync::Arc;
use cudarc::driver::CudaContext;
use std::time::Instant;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════╗");
    println!("║  KV-CACHE DEMONSTRATION                  ║");
    println!("╚══════════════════════════════════════════╝\n");

    // Initialize CUDA
    let context = Arc::new(CudaContext::new(0)?);
    println!("✓ CUDA initialized\n");

    // Test 1: Create cache for small model
    println!("═══ TEST 1: Small Model Cache ═══\n");
    test_small_model_cache(&context)?;

    // Test 2: Create cache for Llama-7B-like model
    println!("\n═══ TEST 2: Llama-7B-like Model Cache ═══\n");
    test_large_model_cache(&context)?;

    // Test 3: Cache append operations
    println!("\n═══ TEST 3: Cache Operations ═══\n");
    test_cache_operations(&context)?;

    // Test 4: Memory usage tracking
    println!("\n═══ TEST 4: Memory Usage ═══\n");
    test_memory_usage(&context)?;

    // Test 5: Performance simulation
    println!("\n═══ TEST 5: Performance Simulation ═══\n");
    simulate_generation_performance(&context)?;

    println!("\n╔══════════════════════════════════════════╗");
    println!("║  ALL TESTS COMPLETED                     ║");
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

fn test_small_model_cache(context: &Arc<CudaContext>) -> Result<()> {
    // Create cache for small model (similar to GPT-2 small)
    let cache = TransformerKVCache::new(
        12,    // 12 layers
        1,     // batch size
        512,   // max sequence length
        768,   // d_model
        context.clone(),
    )?;

    println!("Created cache for small model:");
    println!("  Layers: {}", cache.num_layers());
    println!("  Max sequence: 512 tokens");
    println!("  Dimension: 768");

    cache.print_stats();

    println!("✓ Small model cache created successfully");

    Ok(())
}

fn test_large_model_cache(context: &Arc<CudaContext>) -> Result<()> {
    // Create cache for Llama-7B-like model
    let cache = TransformerKVCache::new(
        32,    // 32 layers
        1,     // batch size
        2048,  // max sequence length
        4096,  // d_model
        context.clone(),
    )?;

    println!("Created cache for Llama-7B-like model:");
    println!("  Layers: {}", cache.num_layers());
    println!("  Max sequence: 2048 tokens");
    println!("  Dimension: 4096");

    cache.print_stats();

    println!("✓ Large model cache created successfully");

    Ok(())
}

fn test_cache_operations(context: &Arc<CudaContext>) -> Result<()> {
    let mut cache = TransformerKVCache::new(4, 1, 512, 768, context.clone())?;

    println!("Initial state:");
    println!("  Sequence length: {}", cache.len());
    println!("  Is empty: {}", cache.is_empty());

    // Simulate adding 10 tokens to layer 0
    let layer0 = cache.layer(0)?;
    let new_tokens = 10;
    let new_keys = vec![1.0f32; new_tokens * 768];
    let new_values = vec![2.0f32; new_tokens * 768];

    println!("\nAppending 10 tokens to layer 0...");
    layer0.append(&new_keys, &new_values, new_tokens, context)?;

    println!("After append:");
    println!("  Sequence length: {}", cache.len());
    println!("  Remaining capacity: {}", layer0.remaining_capacity());

    // Simulate adding more tokens
    let new_tokens2 = 20;
    let new_keys2 = vec![3.0f32; new_tokens2 * 768];
    let new_values2 = vec![4.0f32; new_tokens2 * 768];

    println!("\nAppending 20 more tokens...");
    layer0.append(&new_keys2, &new_values2, new_tokens2, context)?;

    println!("After second append:");
    println!("  Sequence length: {}", cache.len());
    println!("  Remaining capacity: {}", layer0.remaining_capacity());

    // Clear cache
    println!("\nClearing cache...");
    cache.clear_all();

    println!("After clear:");
    println!("  Sequence length: {}", cache.len());
    println!("  Is empty: {}", cache.is_empty());

    println!("\n✓ Cache operations successful");

    Ok(())
}

fn test_memory_usage(context: &Arc<CudaContext>) -> Result<()> {
    let mut cache = TransformerKVCache::new(12, 1, 2048, 768, context.clone())?;

    println!("Simulating generation with 100 tokens:");

    // Simulate generating 100 tokens, adding to cache incrementally
    for step in 1..=100 {
        let layer0 = cache.layer(0)?;
        let new_keys = vec![1.0f32; 768];
        let new_values = vec![2.0f32; 768];
        layer0.append(&new_keys, &new_values, 1, context)?;

        if step % 20 == 0 {
            let stats = cache.stats();
            println!("  Step {}: {:.1}% full, {:.2} MB",
                step,
                stats.utilization * 100.0,
                stats.memory_bytes as f64 / 1024.0 / 1024.0
            );
        }
    }

    cache.print_stats();

    println!("\n✓ Memory tracking successful");

    Ok(())
}

fn simulate_generation_performance(context: &Arc<CudaContext>) -> Result<()> {
    println!("Comparing generation with and without KV-cache:\n");

    // Without cache: O(n²) - need to recompute all previous tokens
    println!("WITHOUT KV-cache:");
    let start = Instant::now();
    let mut total_ops_no_cache = 0u64;
    for token_idx in 1..=100 {
        // Each token requires processing ALL previous tokens
        total_ops_no_cache += token_idx;
    }
    let time_no_cache = start.elapsed();
    println!("  Total operations: {}", total_ops_no_cache);
    println!("  Complexity: O(n²)");
    println!("  Simulated time: {:?}", time_no_cache);

    // With cache: O(n) - only process new token
    println!("\nWITH KV-cache:");
    let start = Instant::now();
    let mut cache = TransformerKVCache::new(12, 1, 512, 768, context.clone())?;
    let mut total_ops_with_cache = 0u64;
    for _token_idx in 1..=100 {
        // Each token only requires 1 operation (new token)
        total_ops_with_cache += 1;

        // Simulate cache update
        let layer0 = cache.layer(0)?;
        let new_keys = vec![1.0f32; 768];
        let new_values = vec![2.0f32; 768];
        layer0.append(&new_keys, &new_values, 1, context)?;
    }
    let time_with_cache = start.elapsed();
    println!("  Total operations: {}", total_ops_with_cache);
    println!("  Complexity: O(n)");
    println!("  Simulated time: {:?}", time_with_cache);

    // Calculate speedup
    let speedup = total_ops_no_cache as f64 / total_ops_with_cache as f64;
    println!("\nSpeedup: {:.1}x faster with KV-cache!", speedup);
    println!("Operations reduced by {:.1}%", (1.0 - 1.0/speedup) * 100.0);

    println!("\n✓ Performance simulation complete");
    println!("\nConclusion:");
    println!("  For 100 token generation:");
    println!("    Without cache: 5,050 operations");
    println!("    With cache: 100 operations");
    println!("    Speedup: 50.5x");

    Ok(())
}
