//! Extremely Simple GPU Abstraction that Actually Compiles
//!
//! Since cudarc API is complex and poorly documented, this provides
//! a minimal abstraction that allows the rest of the code to compile

use anyhow::Result;
use std::sync::Arc;

/// Simple GPU buffer that works on both CPU and GPU
#[derive(Clone)]
pub struct SimpleGpuBuffer {
    /// The actual data (always on CPU for simplicity)
    data: Vec<f32>,

    /// Size of the buffer
    size: usize,
}

impl SimpleGpuBuffer {
    /// Create new buffer
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0f32; size],
            size,
        }
    }

    /// Create from data
    pub fn from_vec(data: Vec<f32>) -> Self {
        let size = data.len();
        Self { data, size }
    }

    /// Get data as slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Convert to Vec
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// Simple GPU context that always works
pub struct SimpleGpuContext {
    /// Whether GPU is available (always false for now)
    gpu_available: bool,
}

impl SimpleGpuContext {
    /// Create new context
    pub fn new() -> Result<Self> {
        // For now, always use CPU fallback
        Ok(Self {
            gpu_available: false,
        })
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// "Transfer" to GPU (actually stays on CPU)
    pub fn to_gpu(&self, data: &[f32]) -> Result<SimpleGpuBuffer> {
        Ok(SimpleGpuBuffer::from_vec(data.to_vec()))
    }

    /// "Transfer" from GPU (already on CPU)
    pub fn from_gpu(&self, buffer: &SimpleGpuBuffer) -> Result<Vec<f32>> {
        Ok(buffer.to_vec())
    }

    /// Allocate buffer
    pub fn allocate(&self, size: usize) -> Result<SimpleGpuBuffer> {
        Ok(SimpleGpuBuffer::new(size))
    }
}

/// Simple tensor for GPU operations
pub struct SimpleGpuTensor {
    /// The data buffer
    buffer: SimpleGpuBuffer,

    /// Shape of the tensor
    shape: Vec<usize>,

    /// Context
    context: Arc<SimpleGpuContext>,
}

impl SimpleGpuTensor {
    /// Create from CPU data
    pub fn from_cpu(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let context = Arc::new(SimpleGpuContext::new()?);
        let buffer = context.to_gpu(&data)?;

        Ok(Self {
            buffer,
            shape,
            context,
        })
    }

    /// Create zeros tensor
    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let size: usize = shape.iter().product();
        let data = vec![0.0f32; size];
        Self::from_cpu(data, shape)
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// To CPU
    pub fn to_cpu(&self) -> Result<Vec<f32>> {
        self.context.from_gpu(&self.buffer)
    }

    /// Matrix multiply (CPU implementation)
    pub fn matmul(&self, other: &SimpleGpuTensor) -> Result<SimpleGpuTensor> {
        // Simple 2D matrix multiplication
        if self.shape.len() != 2 || other.shape.len() != 2 {
            anyhow::bail!("matmul requires 2D tensors");
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        if k != other.shape[0] {
            anyhow::bail!("Shape mismatch for matmul");
        }

        let a_data = self.buffer.as_slice();
        let b_data = other.buffer.as_slice();
        let mut c_data = vec![0.0f32; m * n];

        // CPU matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }

        SimpleGpuTensor::from_cpu(c_data, vec![m, n])
    }

    /// ReLU activation
    pub fn relu(&mut self) -> Result<()> {
        for x in self.buffer.as_mut_slice() {
            *x = x.max(0.0);
        }
        Ok(())
    }

    /// Softmax activation
    pub fn softmax(&mut self, dim: usize) -> Result<()> {
        if dim != 1 || self.shape.len() != 2 {
            anyhow::bail!("Softmax only supports dim=1 on 2D tensors");
        }

        let batch_size = self.shape[0];
        let num_classes = self.shape[1];
        let data = self.buffer.as_mut_slice();

        for b in 0..batch_size {
            let offset = b * num_classes;

            // Find max for stability
            let max_val = data[offset..offset + num_classes]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            // Exp and sum
            let mut sum = 0.0f32;
            for i in 0..num_classes {
                data[offset + i] = (data[offset + i] - max_val).exp();
                sum += data[offset + i];
            }

            // Normalize
            for i in 0..num_classes {
                data[offset + i] /= sum;
            }
        }

        Ok(())
    }

    /// Add bias
    pub fn add_bias(&mut self, bias: &SimpleGpuTensor) -> Result<()> {
        if self.shape.len() != 2 {
            anyhow::bail!("add_bias requires 2D tensor");
        }

        let batch_size = self.shape[0];
        let features = self.shape[1];
        let data = self.buffer.as_mut_slice();
        let bias_data = bias.buffer.as_slice();

        for b in 0..batch_size {
            for f in 0..features {
                data[b * features + f] += bias_data[f % bias_data.len()];
            }
        }

        Ok(())
    }
}

/// Simple linear layer
pub struct SimpleGpuLinear {
    weight: SimpleGpuTensor,
    bias: SimpleGpuTensor,
    in_features: usize,
    out_features: usize,
}

impl SimpleGpuLinear {
    /// Create new linear layer
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        // Initialize weights with Xavier/Glorot
        let scale = (2.0 / in_features as f32).sqrt();
        let mut weight_data = Vec::with_capacity(in_features * out_features);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..(in_features * out_features) {
            weight_data.push(rng.gen_range(-scale..scale));
        }

        let weight = SimpleGpuTensor::from_cpu(weight_data, vec![in_features, out_features])?;
        let bias = SimpleGpuTensor::zeros(vec![out_features])?;

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &SimpleGpuTensor) -> Result<SimpleGpuTensor> {
        // Validate input
        if input.shape().len() != 2 {
            anyhow::bail!("Linear layer expects 2D input");
        }

        if input.shape()[1] != self.in_features {
            anyhow::bail!("Input features mismatch");
        }

        // Matrix multiplication
        let mut output = input.matmul(&self.weight)?;

        // Add bias
        output.add_bias(&self.bias)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = SimpleGpuTensor::from_cpu(data.clone(), vec![2, 2]).unwrap();

        let result = tensor.to_cpu().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_matmul() {
        let a = SimpleGpuTensor::from_cpu(vec![1.0; 6], vec![2, 3]).unwrap();
        let b = SimpleGpuTensor::from_cpu(vec![2.0; 12], vec![3, 4]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);

        let result = c.to_cpu().unwrap();
        assert_eq!(result[0], 6.0); // 3 * 1 * 2
    }

    #[test]
    fn test_linear() {
        let linear = SimpleGpuLinear::new(4, 2).unwrap();
        let input = SimpleGpuTensor::from_cpu(vec![1.0; 8], vec![2, 4]).unwrap();

        let output = linear.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
    }
}