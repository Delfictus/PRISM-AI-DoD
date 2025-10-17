//! GGUF to GPU Weight Loader
//!
//! Handles loading GGUF model weights directly to GPU memory
//! Supports dequantization and format conversion

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice};
use super::gguf_loader::{GgufLoader, GgufType, TensorInfo};

/// GGUF to GPU loader
pub struct GgufGpuLoader {
    loader: GgufLoader,
    context: Arc<CudaContext>,
}

impl GgufGpuLoader {
    /// Create new GPU loader from GGUF file
    pub fn new<P: AsRef<std::path::Path>>(path: P, device_id: usize) -> Result<Self> {
        let loader = GgufLoader::load(path)?;
        let context = CudaContext::new(device_id)?;

        Ok(Self { loader, context })
    }

    /// Load tensor to GPU memory as f32
    pub fn load_tensor_to_gpu(&mut self, name: &str) -> Result<CudaSlice<f32>> {
        let tensor = self.loader.tensors.get(name)
            .context(format!("Tensor not found: {}", name))?
            .clone();

        // Read raw tensor data
        let raw_data = self.loader.read_tensor(name)?;

        // Dequantize/convert to f32
        let f32_data = self.dequantize_tensor(&raw_data, &tensor)?;

        // Upload to GPU
        let stream = self.context.default_stream();
        let gpu_slice = stream.memcpy_stod(&f32_data)?;

        Ok(gpu_slice)
    }

    /// Dequantize tensor data to f32
    fn dequantize_tensor(&self, data: &[u8], tensor: &TensorInfo) -> Result<Vec<f32>> {
        let elem_count = tensor.element_count() as usize;

        match tensor.data_type {
            GgufType::F32 => {
                // Already F32, just convert bytes
                Ok(self.bytes_to_f32(data))
            }
            GgufType::F16 => {
                // Convert F16 to F32
                self.f16_to_f32(data)
            }
            GgufType::Q4_0 => {
                // Dequantize Q4_0
                self.dequantize_q4_0(data, elem_count)
            }
            GgufType::Q4_1 => {
                self.dequantize_q4_1(data, elem_count)
            }
            GgufType::Q8_0 => {
                self.dequantize_q8_0(data, elem_count)
            }
            _ => {
                // For other quantization types, return zeros for now
                // Full dequantization can be added as needed
                println!("⚠️  Quantization type {:?} not fully implemented, using zeros", tensor.data_type);
                Ok(vec![0.0f32; elem_count])
            }
        }
    }

    /// Convert raw bytes to f32 slice
    fn bytes_to_f32(&self, data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Convert F16 to F32
    fn f16_to_f32(&self, data: &[u8]) -> Result<Vec<f32>> {
        let mut result = Vec::new();

        for chunk in data.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let f = half::f16::from_bits(bits).to_f32();
            result.push(f);
        }

        Ok(result)
    }

    /// Dequantize Q4_0 format (4-bit quantization, 32 elements per block)
    ///
    /// Block format:
    /// - delta (f16): scale factor
    /// - qs[16]: 16 bytes containing 32 4-bit values
    fn dequantize_q4_0(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = 32;
        const BYTES_PER_BLOCK: usize = 18; // 2 (delta) + 16 (qs)

        let num_blocks = (elem_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * BYTES_PER_BLOCK;
            if block_start + BYTES_PER_BLOCK > data.len() {
                break;
            }

            // Read delta (scale factor) as f16
            let delta_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();

            // Read and dequantize 32 4-bit values
            let qs = &data[block_start + 2..block_start + 18];

            for i in 0..BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    qs[byte_idx] & 0x0F
                } else {
                    qs[byte_idx] >> 4
                };

                // Convert 4-bit value to float: value = (nibble - 8) * delta
                let val = ((nibble as i8) - 8) as f32 * delta;
                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q4_1 format (4-bit quantization with min, 32 elements per block)
    ///
    /// Block format:
    /// - delta (f16): scale factor
    /// - min (f16): minimum value
    /// - qs[16]: 16 bytes containing 32 4-bit values
    fn dequantize_q4_1(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = 32;
        const BYTES_PER_BLOCK: usize = 20; // 2 (delta) + 2 (min) + 16 (qs)

        let num_blocks = (elem_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * BYTES_PER_BLOCK;
            if block_start + BYTES_PER_BLOCK > data.len() {
                break;
            }

            // Read delta and min as f16
            let delta_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();

            let min_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
            let min = half::f16::from_bits(min_bits).to_f32();

            // Read and dequantize 32 4-bit values
            let qs = &data[block_start + 4..block_start + 20];

            for i in 0..BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    qs[byte_idx] & 0x0F
                } else {
                    qs[byte_idx] >> 4
                };

                // Convert 4-bit value to float: value = nibble * delta + min
                let val = (nibble as f32) * delta + min;
                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q8_0 format (8-bit quantization, 32 elements per block)
    ///
    /// Block format:
    /// - delta (f16): scale factor
    /// - qs[32]: 32 bytes containing 32 8-bit values
    fn dequantize_q8_0(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = 32;
        const BYTES_PER_BLOCK: usize = 34; // 2 (delta) + 32 (qs)

        let num_blocks = (elem_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * BYTES_PER_BLOCK;
            if block_start + BYTES_PER_BLOCK > data.len() {
                break;
            }

            // Read delta (scale factor) as f16
            let delta_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
            let delta = half::f16::from_bits(delta_bits).to_f32();

            // Read and dequantize 32 8-bit values
            let qs = &data[block_start + 2..block_start + 34];

            for &byte in qs.iter().take(BLOCK_SIZE) {
                if result.len() >= elem_count {
                    break;
                }

                // Convert 8-bit value to float: value = byte * delta
                let val = (byte as i8) as f32 * delta;
                result.push(val);
            }
        }

        Ok(result)
    }

    /// Get reference to loader
    pub fn loader(&self) -> &GgufLoader {
        &self.loader
    }

    /// Get reference to CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// List all available tensors
    pub fn list_tensors(&self) -> Vec<String> {
        self.loader.tensors.keys().cloned().collect()
    }

    /// Get tensor info
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.loader.tensors.get(name)
    }
}

#[cfg(test)]
mod tests {
    // Tests disabled - require GgufLoader public fields or factory methods

    // TODO: Add integration tests when GgufLoader provides public API
    // for test construction
}
