//! Comprehensive tests for KV-cache implementation

use prism_ai::orchestration::local_llm::{
    LayerKVCache, TransformerKVCache, KVCacheStats,
};
use std::sync::Arc;
use cudarc::driver::CudaContext;
use anyhow::Result;

#[test]
fn test_layer_kv_cache_creation() {
    let cache = LayerKVCache::new(1, 512, 768);
    assert_eq!(cache.batch_size, 1);
    assert_eq!(cache.max_seq_len, 512);
    assert_eq!(cache.d_model, 768);
    assert_eq!(cache.seq_len, 0);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.remaining_capacity(), 512);
}

#[test]
fn test_layer_kv_cache_initialization() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 512, 768);

    assert!(cache.keys.is_none());
    assert!(cache.values.is_none());

    cache.initialize(&context)?;

    assert!(cache.keys.is_some());
    assert!(cache.values.is_some());
    assert_eq!(cache.seq_len, 0);

    Ok(())
}

#[test]
fn test_layer_kv_cache_append() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 512, 768);
    cache.initialize(&context)?;

    // Create dummy keys and values for 10 tokens
    let new_tokens = 10;
    let new_keys = vec![1.0f32; new_tokens * 768];
    let new_values = vec![2.0f32; new_tokens * 768];

    // Append to cache
    let (_keys, _values) = cache.append(&new_keys, &new_values, new_tokens, &context)?;

    assert_eq!(cache.len(), 10);
    assert_eq!(cache.remaining_capacity(), 502);
    assert!(!cache.is_empty());

    // Append more tokens
    let new_tokens2 = 5;
    let new_keys2 = vec![3.0f32; new_tokens2 * 768];
    let new_values2 = vec![4.0f32; new_tokens2 * 768];

    cache.append(&new_keys2, &new_values2, new_tokens2, &context)?;

    assert_eq!(cache.len(), 15);
    assert_eq!(cache.remaining_capacity(), 497);

    Ok(())
}

#[test]
fn test_layer_kv_cache_overflow() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 10, 768); // Small cache
    cache.initialize(&context)?;

    // Try to add more tokens than capacity
    let new_tokens = 15; // More than max_seq_len of 10
    let new_keys = vec![1.0f32; new_tokens * 768];
    let new_values = vec![2.0f32; new_tokens * 768];

    let result = cache.append(&new_keys, &new_values, new_tokens, &context);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("overflow"));

    Ok(())
}

#[test]
fn test_layer_kv_cache_clear() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 512, 768);
    cache.initialize(&context)?;

    // Add some data
    let new_tokens = 10;
    let new_keys = vec![1.0f32; new_tokens * 768];
    let new_values = vec![2.0f32; new_tokens * 768];
    cache.append(&new_keys, &new_values, new_tokens, &context)?;

    assert_eq!(cache.len(), 10);

    // Clear cache
    cache.clear();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.remaining_capacity(), 512);

    Ok(())
}

#[test]
fn test_transformer_kv_cache_creation() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let cache = TransformerKVCache::new(12, 1, 512, 768, context)?;

    assert_eq!(cache.num_layers(), 12);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    Ok(())
}

#[test]
fn test_transformer_kv_cache_layer_access() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(4, 1, 512, 768, context)?;

    // Test valid layer access
    for i in 0..4 {
        let layer = cache.layer(i)?;
        assert_eq!(layer.max_seq_len, 512);
        assert_eq!(layer.d_model, 768);
    }

    // Test invalid layer access
    assert!(cache.layer(4).is_err());
    assert!(cache.layer(100).is_err());

    Ok(())
}

#[test]
fn test_transformer_kv_cache_clear_all() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(4, 1, 512, 768, context)?;

    // Simulate adding data to first layer
    let layer0 = cache.layer(0)?;
    let new_tokens = 10;
    let new_keys = vec![1.0f32; new_tokens * 768];
    let new_values = vec![2.0f32; new_tokens * 768];
    layer0.append(&new_keys, &new_values, new_tokens, cache.context())?;

    assert_eq!(cache.len(), 10);

    // Clear all caches
    cache.clear_all();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    Ok(())
}

#[test]
fn test_kv_cache_stats() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let cache = TransformerKVCache::new(12, 1, 2048, 4096, context)?;

    let stats = cache.stats();

    assert_eq!(stats.n_layers, 12);
    assert_eq!(stats.seq_len, 0);
    assert_eq!(stats.max_seq_len, 2048);
    assert_eq!(stats.utilization, 0.0);
    assert_eq!(stats.memory_bytes, 0); // No data cached yet

    Ok(())
}

#[test]
fn test_kv_cache_memory_calculation() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(2, 1, 512, 768, context.clone())?;

    // Add 100 tokens to layer 0
    let layer0 = cache.layer(0)?;
    let new_tokens = 100;
    let new_keys = vec![1.0f32; new_tokens * 768];
    let new_values = vec![2.0f32; new_tokens * 768];
    layer0.append(&new_keys, &new_values, new_tokens, &context)?;

    let stats = cache.stats();

    // Memory: 2 tensors (K+V) * 2 layers * 1 batch * 100 seq * 768 dim * 4 bytes
    let expected_memory = 2 * 2 * 1 * 100 * 768 * 4;
    assert_eq!(stats.memory_bytes, expected_memory);

    assert_eq!(stats.seq_len, 100);
    assert!((stats.utilization - 100.0 / 512.0).abs() < 0.001);

    Ok(())
}

#[test]
fn test_large_model_cache() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);

    // Llama-7B-like config: 32 layers, 4096 dims, 2048 context
    let cache = TransformerKVCache::new(32, 1, 2048, 4096, context)?;

    assert_eq!(cache.num_layers(), 32);

    let stats = cache.stats();
    assert_eq!(stats.max_seq_len, 2048);

    // Verify all layers initialized correctly
    for i in 0..32 {
        let layer = cache.get_layer(i)?;
        assert!(layer.keys.is_some());
        assert!(layer.values.is_some());
    }

    Ok(())
}

#[test]
fn test_cache_remaining_capacity() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 100, 768);
    cache.initialize(&context)?;

    assert_eq!(cache.remaining_capacity(), 100);

    // Add 30 tokens
    let new_keys = vec![1.0f32; 30 * 768];
    let new_values = vec![2.0f32; 30 * 768];
    cache.append(&new_keys, &new_values, 30, &context)?;

    assert_eq!(cache.remaining_capacity(), 70);

    // Add 50 more
    let new_keys2 = vec![3.0f32; 50 * 768];
    let new_values2 = vec![4.0f32; 50 * 768];
    cache.append(&new_keys2, &new_values2, 50, &context)?;

    assert_eq!(cache.remaining_capacity(), 20);

    Ok(())
}

#[test]
fn test_multi_batch_cache() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);

    // Batch size of 4
    let mut cache = LayerKVCache::new(4, 512, 768);
    cache.initialize(&context)?;

    assert_eq!(cache.batch_size, 4);

    // Add 10 tokens per batch (total 40 values)
    let new_tokens = 10;
    let new_keys = vec![1.0f32; new_tokens * 768 * 4]; // 4 batches
    let new_values = vec![2.0f32; new_tokens * 768 * 4];
    cache.append(&new_keys, &new_values, new_tokens, &context)?;

    assert_eq!(cache.len(), 10); // Sequence length, not total elements

    Ok(())
}

#[test]
fn test_cache_clone() {
    let cache1 = LayerKVCache::new(1, 512, 768);
    let cache2 = cache1.clone();

    assert_eq!(cache1.batch_size, cache2.batch_size);
    assert_eq!(cache1.max_seq_len, cache2.max_seq_len);
    assert_eq!(cache1.d_model, cache2.d_model);
    assert_eq!(cache1.seq_len, cache2.seq_len);
}

// ===== Additional Edge Case Tests for Day 5 =====

#[test]
fn test_zero_dimension_cache() {
    // Test edge case: zero dimensions (should handle gracefully)
    let cache_zero_seq = LayerKVCache::new(1, 0, 768);
    assert_eq!(cache_zero_seq.max_seq_len, 0);
    assert_eq!(cache_zero_seq.remaining_capacity(), 0);

    let cache_zero_dim = LayerKVCache::new(1, 512, 0);
    assert_eq!(cache_zero_dim.d_model, 0);
}

#[test]
fn test_cache_full_detection() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 10, 768);
    cache.initialize(&context)?;

    // Fill exactly to capacity
    let new_keys = vec![1.0f32; 10 * 768];
    let new_values = vec![2.0f32; 10 * 768];
    cache.append(&new_keys, &new_values, 10, &context)?;

    assert_eq!(cache.len(), 10);
    assert_eq!(cache.remaining_capacity(), 0);

    // Try to add one more token (should fail)
    let extra_keys = vec![3.0f32; 1 * 768];
    let extra_values = vec![4.0f32; 1 * 768];
    let result = cache.append(&extra_keys, &extra_values, 1, &context);

    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_transformer_cache_partial_fill() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(4, 1, 100, 768, context.clone())?;

    // Fill only first two layers
    for layer_idx in 0..2 {
        let layer = cache.layer(layer_idx)?;
        let new_keys = vec![1.0f32; 20 * 768];
        let new_values = vec![2.0f32; 20 * 768];
        layer.append(&new_keys, &new_values, 20, &context)?;
    }

    // Check that only filled layers have data
    assert_eq!(cache.layer(0)?.len(), 20);
    assert_eq!(cache.layer(1)?.len(), 20);
    assert_eq!(cache.layer(2)?.len(), 0);
    assert_eq!(cache.layer(3)?.len(), 0);

    Ok(())
}

#[test]
fn test_cache_stats_display() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let cache = TransformerKVCache::new(32, 1, 2048, 4096, context)?;

    let stats = cache.stats();

    // Verify stats formatting
    let display = format!("{}", stats);
    assert!(display.contains("32 layers"));
    assert!(display.contains("2048"));
    assert!(display.contains("0.0%"));

    Ok(())
}

#[test]
fn test_cache_stats_with_data() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(2, 1, 100, 768, context.clone())?;

    // Add data to first layer
    let layer = cache.layer(0)?;
    let new_keys = vec![1.0f32; 50 * 768];
    let new_values = vec![2.0f32; 50 * 768];
    layer.append(&new_keys, &new_values, 50, &context)?;

    let stats = cache.stats();

    assert_eq!(stats.seq_len, 50);
    assert_eq!(stats.max_seq_len, 100);
    assert!((stats.utilization - 0.5).abs() < 0.001); // 50%

    // Memory: 2 tensors (K+V) * 2 layers * 1 batch * 50 seq * 768 dim * 4 bytes
    let expected_memory = 2 * 2 * 1 * 50 * 768 * 4;
    assert_eq!(stats.memory_bytes, expected_memory);

    Ok(())
}

#[test]
fn test_very_large_cache() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);

    // Test with very large dimensions (8B model scale)
    // 64 layers, 8192 dims, 4096 context
    let result = TransformerKVCache::new(64, 1, 4096, 8192, context);

    // Should succeed (assuming sufficient GPU memory)
    // If it fails, it should be due to OOM, not logic error
    match result {
        Ok(cache) => {
            assert_eq!(cache.num_layers(), 64);
            let stats = cache.stats();
            assert_eq!(stats.max_seq_len, 4096);
        }
        Err(e) => {
            // Acceptable if OOM
            assert!(
                e.to_string().contains("memory") || e.to_string().contains("allocation"),
                "Should fail with memory error, got: {}",
                e
            );
        }
    }

    Ok(())
}

#[test]
fn test_cache_clear_and_refill() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 100, 768);
    cache.initialize(&context)?;

    // First fill
    let keys1 = vec![1.0f32; 30 * 768];
    let values1 = vec![1.0f32; 30 * 768];
    cache.append(&keys1, &values1, 30, &context)?;
    assert_eq!(cache.len(), 30);

    // Clear
    cache.clear();
    assert_eq!(cache.len(), 0);

    // Second fill (should work)
    let keys2 = vec![2.0f32; 40 * 768];
    let values2 = vec![2.0f32; 40 * 768];
    cache.append(&keys2, &values2, 40, &context)?;
    assert_eq!(cache.len(), 40);

    Ok(())
}

#[test]
fn test_transformer_cache_layer_independence() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(3, 1, 100, 768, context.clone())?;

    // Add different amounts to each layer
    for layer_idx in 0..3 {
        let layer = cache.layer(layer_idx)?;
        let num_tokens = (layer_idx + 1) * 10; // 10, 20, 30 tokens
        let keys = vec![(layer_idx as f32); num_tokens * 768];
        let values = vec![(layer_idx as f32 * 10.0); num_tokens * 768];
        layer.append(&keys, &values, num_tokens, &context)?;
    }

    // Verify each layer has correct length
    assert_eq!(cache.layer(0)?.len(), 10);
    assert_eq!(cache.layer(1)?.len(), 20);
    assert_eq!(cache.layer(2)?.len(), 30);

    // Clear layer 1 only
    cache.layer(1)?.clear();

    // Verify only layer 1 cleared
    assert_eq!(cache.layer(0)?.len(), 10);
    assert_eq!(cache.layer(1)?.len(), 0);
    assert_eq!(cache.layer(2)?.len(), 30);

    Ok(())
}

#[test]
fn test_cache_memory_efficiency() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);

    // Compare memory for different configurations
    let small_cache = TransformerKVCache::new(12, 1, 512, 768, context.clone())?;
    let large_cache = TransformerKVCache::new(32, 1, 2048, 4096, context)?;

    let small_stats = small_cache.stats();
    let large_stats = large_cache.stats();

    // Large cache should report significantly more potential memory
    assert!(large_stats.max_seq_len > small_stats.max_seq_len);
    assert_eq!(large_stats.n_layers, 32);
    assert_eq!(small_stats.n_layers, 12);

    Ok(())
}

#[test]
fn test_incremental_append() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 100, 768);
    cache.initialize(&context)?;

    // Append tokens one at a time (simulating autoregressive generation)
    for i in 0..10 {
        let keys = vec![(i as f32); 768];
        let values = vec![(i as f32 * 10.0); 768];
        cache.append(&keys, &values, 1, &context)?;
        assert_eq!(cache.len(), i + 1);
    }

    assert_eq!(cache.len(), 10);
    assert_eq!(cache.remaining_capacity(), 90);

    Ok(())
}

#[test]
fn test_cache_invalid_append_size() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = LayerKVCache::new(1, 100, 768);
    cache.initialize(&context)?;

    // Try to append with mismatched dimensions
    let wrong_keys = vec![1.0f32; 10 * 512]; // Wrong d_model (512 instead of 768)
    let wrong_values = vec![2.0f32; 10 * 768];

    let result = cache.append(&wrong_keys, &wrong_values, 10, &context);

    // Should fail due to dimension mismatch
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_stats_utilization_calculation() -> Result<()> {
    let context = Arc::new(CudaContext::new(0)?);
    let mut cache = TransformerKVCache::new(1, 1, 1000, 768, context.clone())?;

    // Fill to 25%
    let layer = cache.layer(0)?;
    let keys = vec![1.0f32; 250 * 768];
    let values = vec![2.0f32; 250 * 768];
    layer.append(&keys, &values, 250, &context)?;

    let stats = cache.stats();
    assert_eq!(stats.seq_len, 250);
    assert!((stats.utilization - 0.25).abs() < 0.001);

    // Fill to 75%
    let keys2 = vec![3.0f32; 500 * 768];
    let values2 = vec![4.0f32; 500 * 768];
    layer.append(&keys2, &values2, 500, &context)?;

    let stats2 = cache.stats();
    assert_eq!(stats2.seq_len, 750);
    assert!((stats2.utilization - 0.75).abs() < 0.001);

    Ok(())
}
