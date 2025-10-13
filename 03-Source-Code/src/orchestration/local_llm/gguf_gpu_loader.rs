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
        let context = Arc::new(CudaContext::new(device_id)?);

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
            GgufType::Q2_K => {
                self.dequantize_q2_k(data, elem_count)
            }
            GgufType::Q3_K => {
                self.dequantize_q3_k(data, elem_count)
            }
            GgufType::Q4_K => {
                self.dequantize_q4_k(data, elem_count)
            }
            GgufType::Q5_K => {
                self.dequantize_q5_k(data, elem_count)
            }
            GgufType::Q6_K => {
                self.dequantize_q6_k(data, elem_count)
            }
            GgufType::Q8_K => {
                self.dequantize_q8_k(data, elem_count)
            }
            _ => {
                // For other quantization types, return zeros for now
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

    /// Dequantize Q2_K format (2-bit K-quant, 256 elements per super-block)
    ///
    /// Super-block format (256 elements, 82 bytes):
    /// - scales[16]: f16 scale factors for 16 sub-blocks
    /// - bsums[2]: f16 sums
    /// - qs[64]: 64 bytes containing 256 2-bit values
    fn dequantize_q2_k(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const SUPER_BLOCK_SIZE: usize = 256;
        const BYTES_PER_SUPER_BLOCK: usize = 82;

        let num_super_blocks = (elem_count + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * BYTES_PER_SUPER_BLOCK;
            if sb_start + BYTES_PER_SUPER_BLOCK > data.len() {
                break;
            }

            // Read scales (16 x f16 = 32 bytes)
            let mut scales = [0.0f32; 16];
            for i in 0..16 {
                let offset = sb_start + i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                scales[i] = half::f16::from_bits(bits).to_f32();
            }

            // Skip bsums (2 x f16 = 4 bytes at offset 32)
            // Read qs (64 bytes starting at offset 36, but actual start is after scales and bsums)
            let qs_start = sb_start + 36;

            // Dequantize 256 2-bit values (4 values per byte)
            for i in 0..SUPER_BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let byte_idx = i / 4;
                let shift = (i % 4) * 2;
                let val_2bit = (data[qs_start + byte_idx] >> shift) & 0x03;

                // Determine which scale to use (16 sub-blocks of 16 elements each)
                let scale_idx = i / 16;
                let val = ((val_2bit as i8) - 2) as f32 * scales[scale_idx];

                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q3_K format (3-bit K-quant, 256 elements per super-block)
    ///
    /// Super-block format (256 elements, 110 bytes):
    /// - scales and mins
    /// - qs[96]: bytes containing 256 3-bit values
    fn dequantize_q3_k(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const SUPER_BLOCK_SIZE: usize = 256;
        const BYTES_PER_SUPER_BLOCK: usize = 110;

        let num_super_blocks = (elem_count + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * BYTES_PER_SUPER_BLOCK;
            if sb_start + BYTES_PER_SUPER_BLOCK > data.len() {
                break;
            }

            // Simplified Q3_K dequantization
            // Read scales (first 32 bytes contain scales and high bits)
            let mut scales = [0.0f32; 16];
            for i in 0..16 {
                let offset = sb_start + i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                scales[i] = half::f16::from_bits(bits).to_f32().max(0.001);
            }

            // qs start at offset 32
            let qs_start = sb_start + 32;

            // Dequantize 256 3-bit values (approximate - 3 bits is complex packing)
            for i in 0..SUPER_BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                // Simplified: treat as 4-bit for now (actual Q3_K is more complex)
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    data[qs_start + byte_idx] & 0x0F
                } else {
                    data[qs_start + byte_idx] >> 4
                };

                let scale_idx = i / 16;
                let val = ((nibble as i8) - 4) as f32 * scales[scale_idx];

                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q4_K format (4-bit K-quant, 256 elements per super-block)
    ///
    /// Super-block format (256 elements, 144 bytes):
    /// - scales and mins: 32 bytes
    /// - qs[128]: 128 bytes containing 256 4-bit values
    fn dequantize_q4_k(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const SUPER_BLOCK_SIZE: usize = 256;
        const BYTES_PER_SUPER_BLOCK: usize = 144;

        let num_super_blocks = (elem_count + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * BYTES_PER_SUPER_BLOCK;
            if sb_start + BYTES_PER_SUPER_BLOCK > data.len() {
                break;
            }

            // Read d (super-scale) and dmin
            let d_bits = u16::from_le_bytes([data[sb_start], data[sb_start + 1]]);
            let d = half::f16::from_bits(d_bits).to_f32();

            let dmin_bits = u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]);
            let dmin = half::f16::from_bits(dmin_bits).to_f32();

            // Read scales (12 bytes starting at offset 4, packed as 6-bit values)
            let scales_start = sb_start + 4;
            let mut scales = [0u8; 8];
            for i in 0..8 {
                // Simplified: use every other byte as scale
                scales[i] = data[scales_start + i];
            }

            // qs start at offset 16
            let qs_start = sb_start + 16;

            // Dequantize 256 4-bit values
            for i in 0..SUPER_BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    data[qs_start + byte_idx] & 0x0F
                } else {
                    data[qs_start + byte_idx] >> 4
                };

                // Get scale for this sub-block (32 sub-blocks of 8 elements)
                let scale_idx = i / 32;
                let scale = if scale_idx < 8 { scales[scale_idx] } else { 64 };

                let val = ((nibble as f32) * d * (scale as f32 / 64.0)) - dmin;
                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q5_K format (5-bit K-quant, 256 elements per super-block)
    ///
    /// Super-block format (256 elements, 176 bytes):
    /// - scales and mins: 32 bytes
    /// - qs[128]: 128 bytes for low 4 bits
    /// - qh[32]: 32 bytes for high bits
    fn dequantize_q5_k(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const SUPER_BLOCK_SIZE: usize = 256;
        const BYTES_PER_SUPER_BLOCK: usize = 176;

        let num_super_blocks = (elem_count + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * BYTES_PER_SUPER_BLOCK;
            if sb_start + BYTES_PER_SUPER_BLOCK > data.len() {
                break;
            }

            // Read d (super-scale)
            let d_bits = u16::from_le_bytes([data[sb_start], data[sb_start + 1]]);
            let d = half::f16::from_bits(d_bits).to_f32();

            let dmin_bits = u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]);
            let dmin = half::f16::from_bits(dmin_bits).to_f32();

            // Read scales
            let scales_start = sb_start + 4;
            let mut scales = [0u8; 12];
            for i in 0..12 {
                scales[i] = data[scales_start + i];
            }

            // qs (low 4 bits) start at offset 16
            let qs_start = sb_start + 16;
            // qh (high bits) start at offset 144
            let qh_start = sb_start + 144;

            // Dequantize 256 5-bit values
            for i in 0..SUPER_BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let byte_idx = i / 2;
                let low_4bits = if i % 2 == 0 {
                    data[qs_start + byte_idx] & 0x0F
                } else {
                    data[qs_start + byte_idx] >> 4
                };

                // Get high bit from qh
                let qh_byte_idx = i / 8;
                let qh_bit_idx = i % 8;
                let high_bit = (data[qh_start + qh_byte_idx] >> qh_bit_idx) & 0x01;

                let val_5bit = (high_bit << 4) | low_4bits;

                let scale_idx = i / 32;
                let scale = if scale_idx < 12 { scales[scale_idx] } else { 64 };

                let val = ((val_5bit as f32) * d * (scale as f32 / 64.0)) - dmin;
                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q6_K format (6-bit K-quant, 256 elements per super-block)
    ///
    /// Super-block format (256 elements, 210 bytes):
    /// - scales: 16 x f16 = 32 bytes
    /// - qs[128]: 128 bytes for low 4 bits
    /// - qh[64]: 64 bytes for high 2 bits
    fn dequantize_q6_k(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const SUPER_BLOCK_SIZE: usize = 256;
        const BYTES_PER_SUPER_BLOCK: usize = 210;

        let num_super_blocks = (elem_count + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * BYTES_PER_SUPER_BLOCK;
            if sb_start + BYTES_PER_SUPER_BLOCK > data.len() {
                break;
            }

            // Read scales (16 x f16 = 32 bytes)
            let mut scales = [0.0f32; 16];
            for i in 0..16 {
                let offset = sb_start + i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                scales[i] = half::f16::from_bits(bits).to_f32();
            }

            // qs (low 4 bits) start at offset 32
            let qs_start = sb_start + 32;
            // qh (high 2 bits) start at offset 160
            let qh_start = sb_start + 160;

            // Dequantize 256 6-bit values
            for i in 0..SUPER_BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let byte_idx = i / 2;
                let low_4bits = if i % 2 == 0 {
                    data[qs_start + byte_idx] & 0x0F
                } else {
                    data[qs_start + byte_idx] >> 4
                };

                // Get high 2 bits from qh
                let qh_byte_idx = i / 4;
                let qh_shift = (i % 4) * 2;
                let high_2bits = (data[qh_start + qh_byte_idx] >> qh_shift) & 0x03;

                let val_6bit = (high_2bits << 4) | low_4bits;

                let scale_idx = i / 16;
                let val = ((val_6bit as i8) - 32) as f32 * scales[scale_idx];

                result.push(val);
            }
        }

        Ok(result)
    }

    /// Dequantize Q8_K format (8-bit K-quant, 256 elements per super-block)
    ///
    /// Super-block format (256 elements, 292 bytes):
    /// - d: f32 super-scale (4 bytes)
    /// - scales[32]: 32 bytes
    /// - qs[256]: 256 bytes containing 256 8-bit values
    fn dequantize_q8_k(&self, data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        const SUPER_BLOCK_SIZE: usize = 256;
        const BYTES_PER_SUPER_BLOCK: usize = 292;

        let num_super_blocks = (elem_count + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let mut result = Vec::with_capacity(elem_count);

        for sb_idx in 0..num_super_blocks {
            let sb_start = sb_idx * BYTES_PER_SUPER_BLOCK;
            if sb_start + BYTES_PER_SUPER_BLOCK > data.len() {
                break;
            }

            // Read d (super-scale) as f32
            let d = f32::from_le_bytes([
                data[sb_start],
                data[sb_start + 1],
                data[sb_start + 2],
                data[sb_start + 3],
            ]);

            // Read scales (32 bytes starting at offset 4)
            let scales_start = sb_start + 4;
            let scales = &data[scales_start..scales_start + 32];

            // qs start at offset 36
            let qs_start = sb_start + 36;

            // Dequantize 256 8-bit values
            for i in 0..SUPER_BLOCK_SIZE {
                if result.len() >= elem_count {
                    break;
                }

                let q = data[qs_start + i] as i8;
                let scale_idx = i / 8;
                let scale = scales[scale_idx] as f32;

                let val = (q as f32) * d * scale / 127.0;
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
    use super::*;

    #[test]
    fn test_f32_conversion() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        let data = vec![0u8, 0, 128, 63]; // 1.0 in f32 little-endian
        let result = loader.bytes_to_f32(&data);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_q4_0_dequantization() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Create Q4_0 block with known values
        // Block: delta (f16) + 16 bytes of 4-bit values
        let delta = 2.0f32;
        let delta_bits = half::f16::from_f32(delta).to_bits();
        let mut data = vec![
            (delta_bits & 0xFF) as u8,
            (delta_bits >> 8) as u8,
        ];

        // Add 16 bytes of 4-bit values (32 values total)
        // Each byte contains two 4-bit values
        for _ in 0..16 {
            data.push(0x48); // 4 in lower nibble, 8 in upper = values -4, 0 after offset
        }

        let result = loader.dequantize_q4_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
        // Each value should be (nibble - 8) * delta
        // 0x4 = 4, so (4 - 8) * 2.0 = -8.0
        // 0x8 = 8, so (8 - 8) * 2.0 = 0.0
        assert!((result[0] - (-8.0)).abs() < 0.1);
        assert!((result[1] - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_q8_0_dequantization() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Create Q8_0 block with known values
        let delta = 0.5f32;
        let delta_bits = half::f16::from_f32(delta).to_bits();
        let mut data = vec![
            (delta_bits & 0xFF) as u8,
            (delta_bits >> 8) as u8,
        ];

        // Add 32 8-bit values
        for i in 0..32 {
            data.push((i as i8) as u8);
        }

        let result = loader.dequantize_q8_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);

        // Values should be byte * delta
        assert!((result[0] - 0.0).abs() < 0.1);
        assert!((result[10] - 5.0).abs() < 0.1); // 10 * 0.5 = 5.0
    }

    #[test]
    fn test_q2_k_dequantization() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Create Q2_K super-block (82 bytes)
        let mut data = vec![0u8; 82];

        // Set scales (16 x f16 = 32 bytes)
        for i in 0..16 {
            let scale = 1.0f32;
            let bits = half::f16::from_f32(scale).to_bits();
            data[i * 2] = (bits & 0xFF) as u8;
            data[i * 2 + 1] = (bits >> 8) as u8;
        }

        // Set bsums (2 x f16 = 4 bytes at offset 32) - not used in dequant
        data[32] = 0;
        data[33] = 0;
        data[34] = 0;
        data[35] = 0;

        // Set qs (64 bytes starting at offset 36)
        // Each byte contains 4 2-bit values
        for i in 36..82 {
            data[i] = 0b10_01_00_11; // values: 3, 0, 1, 2 in 2-bit
        }

        let result = loader.dequantize_q2_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);

        // Values should be (2bit_val - 2) * scale
        // First 4 values from byte 0b10_01_00_11 = 3,0,1,2
        assert!((result[0] - 1.0).abs() < 0.1);  // (3 - 2) * 1.0
        assert!((result[1] - (-2.0)).abs() < 0.1); // (0 - 2) * 1.0
    }

    #[test]
    fn test_q4_k_dequantization() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Create Q4_K super-block (144 bytes)
        let mut data = vec![0u8; 144];

        // Set d (super-scale) as f16
        let d = 2.0f32;
        let d_bits = half::f16::from_f32(d).to_bits();
        data[0] = (d_bits & 0xFF) as u8;
        data[1] = (d_bits >> 8) as u8;

        // Set dmin as f16
        let dmin = 0.5f32;
        let dmin_bits = half::f16::from_f32(dmin).to_bits();
        data[2] = (dmin_bits & 0xFF) as u8;
        data[3] = (dmin_bits >> 8) as u8;

        // Set scales (8 bytes starting at offset 4)
        for i in 0..8 {
            data[4 + i] = 64; // scale = 64 means 1.0x
        }

        // Set qs (128 bytes starting at offset 16)
        for i in 16..144 {
            data[i] = 0x55; // 5 in both nibbles
        }

        let result = loader.dequantize_q4_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);

        // Values should be (nibble * d * scale/64) - dmin
        // 5 * 2.0 * (64/64) - 0.5 = 10.0 - 0.5 = 9.5
        assert!((result[0] - 9.5).abs() < 0.5);
    }

    #[test]
    fn test_q6_k_dequantization() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Create Q6_K super-block (210 bytes)
        let mut data = vec![0u8; 210];

        // Set scales (16 x f16 = 32 bytes)
        for i in 0..16 {
            let scale = 0.5f32;
            let bits = half::f16::from_f32(scale).to_bits();
            data[i * 2] = (bits & 0xFF) as u8;
            data[i * 2 + 1] = (bits >> 8) as u8;
        }

        // Set qs (low 4 bits, 128 bytes starting at offset 32)
        for i in 32..160 {
            data[i] = 0x88; // 8 in both nibbles
        }

        // Set qh (high 2 bits, 64 bytes starting at offset 160)
        for i in 160..210 {
            data[i] = 0b01_01_01_01; // 1 in all 2-bit positions
        }

        let result = loader.dequantize_q6_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);

        // Values should be ((high_2bits << 4 | low_4bits) - 32) * scale
        // (0b01_1000 - 32) * 0.5 = (24 - 32) * 0.5 = -4.0
        assert!((result[0] - (-4.0)).abs() < 0.5);
    }

    #[test]
    fn test_q8_k_dequantization() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Create Q8_K super-block (292 bytes)
        let mut data = vec![0u8; 292];

        // Set d (super-scale) as f32
        let d = 1.0f32;
        let d_bytes = d.to_le_bytes();
        data[0] = d_bytes[0];
        data[1] = d_bytes[1];
        data[2] = d_bytes[2];
        data[3] = d_bytes[3];

        // Set scales (32 bytes starting at offset 4)
        for i in 0..32 {
            data[4 + i] = 127; // max scale
        }

        // Set qs (256 bytes starting at offset 36)
        for i in 36..292 {
            data[i] = 64; // value of 64
        }

        let result = loader.dequantize_q8_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);

        // Values should be q * d * scale / 127.0
        // 64 * 1.0 * 127 / 127.0 = 64.0
        assert!((result[0] - 64.0).abs() < 1.0);
    }

    #[test]
    fn test_multiple_blocks() {
        let loader = GgufGpuLoader {
            loader: GgufLoader {
                reader: todo!(),
                metadata: Default::default(),
                tensors: Default::default(),
                data_offset: 0,
                alignment: 32,
            },
            context: Arc::new(CudaContext::new(0).unwrap()),
        };

        // Test with 2 Q4_0 blocks (64 elements)
        let delta = 1.0f32;
        let delta_bits = half::f16::from_f32(delta).to_bits();
        let mut data = vec![];

        // First block
        data.push((delta_bits & 0xFF) as u8);
        data.push((delta_bits >> 8) as u8);
        for _ in 0..16 {
            data.push(0x88); // All 8s
        }

        // Second block
        data.push((delta_bits & 0xFF) as u8);
        data.push((delta_bits >> 8) as u8);
        for _ in 0..16 {
            data.push(0x00); // All 0s
        }

        let result = loader.dequantize_q4_0(&data, 64).unwrap();
        assert_eq!(result.len(), 64);

        // First block: (8 - 8) * 1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 0.1);
        // Second block: (0 - 8) * 1.0 = -8.0
        assert!((result[32] - (-8.0)).abs() < 0.1);
    }
}
