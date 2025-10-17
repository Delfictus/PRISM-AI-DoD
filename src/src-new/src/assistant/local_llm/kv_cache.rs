//! KV-Cache for Efficient Transformer Inference
//!
//! Implements key-value caching for autoregressive generation
//! to avoid recomputing attention keys and values for previous tokens.
//!
//! Performance improvement: O(n²) -> O(n) for generation
//! where n is sequence length

use anyhow::Result;
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice};

/// KV-Cache for a single transformer layer
///
/// Stores key and value tensors on GPU for efficient autoregressive generation
#[derive(Clone)]
pub struct LayerKVCache {
    /// Cached key tensor: [batch_size, seq_len, d_model]
    /// Stored on GPU
    pub keys: Option<CudaSlice<f32>>,

    /// Cached value tensor: [batch_size, seq_len, d_model]
    /// Stored on GPU
    pub values: Option<CudaSlice<f32>>,

    /// Current sequence length in cache
    pub seq_len: usize,

    /// Maximum sequence length supported
    pub max_seq_len: usize,

    /// Model dimension
    pub d_model: usize,

    /// Batch size
    pub batch_size: usize,
}

impl LayerKVCache {
    /// Create new empty KV cache
    pub fn new(batch_size: usize, max_seq_len: usize, d_model: usize) -> Self {
        Self {
            keys: None,
            values: None,
            seq_len: 0,
            max_seq_len,
            d_model,
            batch_size,
        }
    }

    /// Initialize cache with empty GPU buffers
    pub fn initialize(&mut self, context: &Arc<CudaContext>) -> Result<()> {
        let stream = context.default_stream();

        // Allocate GPU memory for maximum sequence length
        let cache_size = self.batch_size * self.max_seq_len * self.d_model;

        self.keys = Some(stream.alloc_zeros::<f32>(cache_size)?);
        self.values = Some(stream.alloc_zeros::<f32>(cache_size)?);

        self.seq_len = 0;

        Ok(())
    }

    /// Append new keys and values to the cache
    ///
    /// # Arguments
    /// * `new_keys` - New key tensor to append [batch_size, new_tokens, d_model]
    /// * `new_values` - New value tensor to append [batch_size, new_tokens, d_model]
    /// * `context` - CUDA context for GPU operations
    ///
    /// Returns the full cached keys and values including the new tokens
    pub fn append(
        &mut self,
        new_keys: &[f32],
        new_values: &[f32],
        new_tokens: usize,
        context: &Arc<CudaContext>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        // Check if cache is initialized
        if self.keys.is_none() {
            self.initialize(context)?;
        }

        // Check if we have space
        if self.seq_len + new_tokens > self.max_seq_len {
            anyhow::bail!(
                "KV-cache overflow: current {} + new {} > max {}",
                self.seq_len,
                new_tokens,
                self.max_seq_len
            );
        }

        let stream = context.default_stream();

        // Get mutable references to cache
        let keys_cache = self.keys.as_ref().unwrap();
        let values_cache = self.values.as_ref().unwrap();

        // Copy cached data back to CPU (temporary - in production would do this on GPU)
        let mut keys_cpu = stream.memcpy_dtov(keys_cache)?;
        let mut values_cpu = stream.memcpy_dtov(values_cache)?;

        // Append new keys and values at the current position
        let start_idx = self.seq_len * self.d_model * self.batch_size;
        let new_size = new_tokens * self.d_model * self.batch_size;

        keys_cpu[start_idx..start_idx + new_size].copy_from_slice(&new_keys[..new_size]);
        values_cpu[start_idx..start_idx + new_size].copy_from_slice(&new_values[..new_size]);

        // Upload back to GPU
        let updated_keys = stream.memcpy_stod(&keys_cpu)?;
        let updated_values = stream.memcpy_stod(&values_cpu)?;

        self.keys = Some(updated_keys.clone());
        self.values = Some(updated_values.clone());

        // Update sequence length
        self.seq_len += new_tokens;

        // Return slices containing only the valid data (up to seq_len)
        let valid_size = self.seq_len * self.d_model * self.batch_size;
        let keys_view = stream.memcpy_stod(&keys_cpu[..valid_size])?;
        let values_view = stream.memcpy_stod(&values_cpu[..valid_size])?;

        Ok((keys_view, values_view))
    }

    /// Get current cached keys and values
    ///
    /// Returns slices containing only valid cached data (up to seq_len)
    pub fn get(&self, context: &Arc<CudaContext>) -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>)>> {
        if self.keys.is_none() || self.seq_len == 0 {
            return Ok(None);
        }

        let stream = context.default_stream();

        let keys = self.keys.as_ref().unwrap();
        let values = self.values.as_ref().unwrap();

        // Get only the valid portion of the cache
        let valid_size = self.seq_len * self.d_model * self.batch_size;
        let keys_cpu = stream.memcpy_dtov(keys)?;
        let values_cpu = stream.memcpy_dtov(values)?;

        let keys_valid = stream.memcpy_stod(&keys_cpu[..valid_size])?;
        let values_valid = stream.memcpy_stod(&values_cpu[..valid_size])?;

        Ok(Some((keys_valid, values_valid)))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.seq_len
    }
}

/// Multi-layer KV-Cache
///
/// Manages KV-cache for all transformer layers
pub struct TransformerKVCache {
    /// Per-layer caches
    layer_caches: Vec<LayerKVCache>,

    /// Number of layers
    n_layers: usize,

    /// CUDA context
    context: Arc<CudaContext>,
}

impl TransformerKVCache {
    /// Create new multi-layer KV-cache
    pub fn new(
        n_layers: usize,
        batch_size: usize,
        max_seq_len: usize,
        d_model: usize,
        context: Arc<CudaContext>,
    ) -> Result<Self> {
        let mut layer_caches = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let mut cache = LayerKVCache::new(batch_size, max_seq_len, d_model);
            cache.initialize(&context)?;
            layer_caches.push(cache);
        }

        Ok(Self {
            layer_caches,
            n_layers,
            context,
        })
    }

    /// Get cache for a specific layer
    pub fn layer(&mut self, layer_idx: usize) -> Result<&mut LayerKVCache> {
        if layer_idx >= self.n_layers {
            anyhow::bail!("Layer index {} out of range (max {})", layer_idx, self.n_layers);
        }
        Ok(&mut self.layer_caches[layer_idx])
    }

    /// Get immutable reference to layer cache
    pub fn get_layer(&self, layer_idx: usize) -> Result<&LayerKVCache> {
        if layer_idx >= self.n_layers {
            anyhow::bail!("Layer index {} out of range (max {})", layer_idx, self.n_layers);
        }
        Ok(&self.layer_caches[layer_idx])
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        for cache in &mut self.layer_caches {
            cache.clear();
        }
    }

    /// Get current sequence length (from first layer)
    pub fn len(&self) -> usize {
        self.layer_caches[0].len()
    }

    /// Check if all caches are empty
    pub fn is_empty(&self) -> bool {
        self.layer_caches[0].is_empty()
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.n_layers
    }

    /// Get CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

/// KV-Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    /// Current total memory usage in bytes
    pub memory_bytes: usize,

    /// Current sequence length
    pub seq_len: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Cache utilization (0.0 to 1.0)
    pub utilization: f64,

    /// Number of layers
    pub n_layers: usize,
}

impl TransformerKVCache {
    /// Get cache statistics
    pub fn stats(&self) -> KVCacheStats {
        let seq_len = self.len();
        let max_seq_len = self.layer_caches[0].max_seq_len;
        let d_model = self.layer_caches[0].d_model;
        let batch_size = self.layer_caches[0].batch_size;

        // Memory: 2 tensors (keys + values) per layer * layers * batch * seq * d_model * 4 bytes (f32)
        let memory_bytes = 2 * self.n_layers * batch_size * seq_len * d_model * 4;

        let utilization = if max_seq_len > 0 {
            seq_len as f64 / max_seq_len as f64
        } else {
            0.0
        };

        KVCacheStats {
            memory_bytes,
            seq_len,
            max_seq_len,
            utilization,
            n_layers: self.n_layers,
        }
    }

    /// Print cache statistics
    pub fn print_stats(&self) {
        let stats = self.stats();
        println!("╔══════════════════════════════════════════╗");
        println!("║  KV-CACHE STATISTICS                     ║");
        println!("╚══════════════════════════════════════════╝");
        println!("Layers: {}", stats.n_layers);
        println!("Sequence length: {}/{}", stats.seq_len, stats.max_seq_len);
        println!("Utilization: {:.1}%", stats.utilization * 100.0);
        println!("Memory usage: {:.2} MB", stats.memory_bytes as f64 / 1024.0 / 1024.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache_creation() {
        let cache = LayerKVCache::new(1, 512, 768);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.max_seq_len, 512);
        assert_eq!(cache.d_model, 768);
        assert!(cache.is_empty());
        assert_eq!(cache.remaining_capacity(), 512);
    }

    #[test]
    fn test_kv_cache_stats() {
        let context = CudaContext::new(0).unwrap();
        let cache = TransformerKVCache::new(12, 1, 512, 768, context).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.n_layers, 12);
        assert_eq!(stats.seq_len, 0);
        assert_eq!(stats.max_seq_len, 512);
        assert_eq!(stats.utilization, 0.0);
    }

    #[test]
    fn test_multi_layer_cache_access() {
        let context = CudaContext::new(0).unwrap();
        let cache = TransformerKVCache::new(4, 1, 512, 768, context).unwrap();

        assert_eq!(cache.num_layers(), 4);

        // Test valid layer access
        for i in 0..4 {
            assert!(cache.get_layer(i).is_ok());
        }

        // Test invalid layer access
        assert!(cache.get_layer(4).is_err());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = LayerKVCache::new(1, 512, 768);
        cache.seq_len = 10; // Simulate some cached data

        assert_eq!(cache.len(), 10);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
