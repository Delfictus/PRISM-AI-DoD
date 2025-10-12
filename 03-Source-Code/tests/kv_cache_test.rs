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
