// GPU-Accelerated Activation Functions for GNN
// Integrates Worker 2's activation GPU kernels (Phase 4)
// Constitution: GNN Solver + Production GPU Optimization

use anyhow::{Result, Context};
use ndarray::{Array1, Array2};

#[cfg(feature = "cuda")]
use crate::gpu::kernel_executor::get_global_executor;

/// GPU-accelerated activation functions for Graph Neural Networks
///
/// Leverages Worker 2's GPU kernels for:
/// - relu_inplace: 10-30x speedup for ReLU activation
/// - sigmoid_inplace: 10-30x speedup for Sigmoid activation
/// - tanh_inplace: 10-30x speedup for Tanh activation
/// - softmax: 10-30x speedup for attention weights normalization
///
/// # Use Cases
/// - GNN forward pass acceleration
/// - Attention mechanism computation
/// - Graph coloring neural network training
pub struct GpuActivations {
    /// Use GPU if available
    pub use_gpu: bool,
}

impl Default for GpuActivations {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

impl GpuActivations {
    /// Create new GPU activations handler
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable GPU (for testing/comparison)
    pub fn cpu_only() -> Self {
        Self { use_gpu: false }
    }

    /// ReLU activation: f(x) = max(0, x) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `input` - Input array (modified in-place for GPU efficiency)
    ///
    /// # Returns
    /// Activated array
    ///
    /// # Use Cases
    /// - GNN hidden layer activation
    /// - Non-linear transformation in graph convolutions
    /// - Standard deep learning activation
    pub fn relu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.relu_gpu(input) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(input.mapv(|x| x.max(0.0)))
    }

    /// GPU implementation using Worker 2's relu_inplace kernel
    #[cfg(feature = "cuda")]
    fn relu_gpu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU (mutable for in-place operation)
        let mut data_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();

        // Use Worker 2's relu_inplace kernel (modifies in-place)
        executor.relu_inplace(&mut data_f32)
            .context("GPU relu_inplace failed")?;

        // Convert back to f64
        let result = Array1::from_vec(data_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// ReLU activation for 2D arrays (batched processing)
    pub fn relu_2d(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let (rows, cols) = input.dim();
        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            let row = input.row(i).to_owned();
            let activated = self.relu(&row)?;
            result.row_mut(i).assign(&activated);
        }

        Ok(result)
    }

    /// Sigmoid activation: f(x) = 1 / (1 + exp(-x)) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `input` - Input array
    ///
    /// # Returns
    /// Activated array in range (0, 1)
    ///
    /// # Use Cases
    /// - Binary classification in GNN nodes
    /// - Gate mechanisms in graph attention
    /// - Probability estimation
    pub fn sigmoid(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.sigmoid_gpu(input) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(input.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }

    /// GPU implementation using Worker 2's sigmoid_inplace kernel
    #[cfg(feature = "cuda")]
    fn sigmoid_gpu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        let mut data_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();

        executor.sigmoid_inplace(&mut data_f32)
            .context("GPU sigmoid_inplace failed")?;

        let result = Array1::from_vec(data_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// Tanh activation: f(x) = tanh(x) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `input` - Input array
    ///
    /// # Returns
    /// Activated array in range (-1, 1)
    ///
    /// # Use Cases
    /// - GNN hidden layer activation (alternative to ReLU)
    /// - Signed activation for graph features
    /// - Normalized nonlinear transformation
    pub fn tanh(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.tanh_gpu(input) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(input.mapv(|x| x.tanh()))
    }

    /// GPU implementation using Worker 2's tanh_inplace kernel
    #[cfg(feature = "cuda")]
    fn tanh_gpu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        let mut data_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();

        executor.tanh_inplace(&mut data_f32)
            .context("GPU tanh_inplace failed")?;

        let result = Array1::from_vec(data_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// Softmax activation: f(x_i) = exp(x_i) / sum(exp(x_j)) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `input` - Input array (logits)
    ///
    /// # Returns
    /// Probability distribution (sums to 1.0)
    ///
    /// # Use Cases
    /// - Graph attention weights normalization
    /// - Multi-class classification in GNN
    /// - Attention mechanism in GAT
    pub fn softmax(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.softmax_gpu(input) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        self.softmax_cpu(input)
    }

    /// GPU implementation using Worker 2's softmax kernel
    #[cfg(feature = "cuda")]
    fn softmax_gpu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        let data_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();

        // Use Worker 2's softmax kernel
        let result_f32 = executor.softmax(&data_f32)
            .context("GPU softmax failed")?;

        let result = Array1::from_vec(result_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// CPU softmax with numerical stability
    fn softmax_cpu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // Numerical stability: subtract max before exp
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Array1<f64> = input.mapv(|x| (x - max_val).exp());
        let sum_exp: f64 = exp_values.sum();

        if sum_exp < 1e-10 {
            anyhow::bail!("Softmax denominator too small (numerical instability)");
        }

        Ok(exp_values / sum_exp)
    }

    /// Softmax for 2D arrays (batch processing for attention heads)
    ///
    /// # Arguments
    /// * `input` - Input array (batch_size x num_classes)
    ///
    /// # Returns
    /// Probability distributions (each row sums to 1.0)
    pub fn softmax_2d(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let (rows, cols) = input.dim();
        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            let row = input.row(i).to_owned();
            let softmax_row = self.softmax(&row)?;
            result.row_mut(i).assign(&softmax_row);
        }

        Ok(result)
    }

    /// Leaky ReLU: f(x) = max(alpha * x, x) for negative values
    ///
    /// # Arguments
    /// * `input` - Input array
    /// * `alpha` - Slope for negative values (typically 0.01)
    ///
    /// # Returns
    /// Activated array
    ///
    /// # Use Cases
    /// - Alternative to ReLU to prevent dead neurons
    /// - Gradient flow for negative inputs
    pub fn leaky_relu(&self, input: &Array1<f64>, alpha: f64) -> Result<Array1<f64>> {
        // Note: Worker 2 doesn't have leaky_relu kernel, so CPU only for now
        Ok(input.mapv(|x| if x > 0.0 { x } else { alpha * x }))
    }

    /// ELU (Exponential Linear Unit): f(x) = x if x > 0, alpha * (exp(x) - 1) otherwise
    ///
    /// # Arguments
    /// * `input` - Input array
    /// * `alpha` - Scale for negative values (typically 1.0)
    ///
    /// # Returns
    /// Activated array
    pub fn elu(&self, input: &Array1<f64>, alpha: f64) -> Result<Array1<f64>> {
        Ok(input.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }))
    }

    /// GELU (Gaussian Error Linear Unit): smooth approximation of ReLU
    ///
    /// # Arguments
    /// * `input` - Input array
    ///
    /// # Returns
    /// Activated array
    ///
    /// # Use Cases
    /// - Modern transformer architectures
    /// - Smooth alternative to ReLU
    pub fn gelu(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2/π)

        Ok(input.mapv(|x| {
            let inner = SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        }))
    }
}

/// Activation function type for GNN layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(u32), // alpha encoded as u32 bits (0.01 -> 1036831949)
    ELU(u32),
    GELU,
}

impl ActivationType {
    /// Apply this activation to an array
    pub fn apply(&self, activations: &GpuActivations, input: &Array1<f64>) -> Result<Array1<f64>> {
        match self {
            ActivationType::ReLU => activations.relu(input),
            ActivationType::Sigmoid => activations.sigmoid(input),
            ActivationType::Tanh => activations.tanh(input),
            ActivationType::LeakyReLU(alpha_bits) => {
                let alpha = f64::from_bits((*alpha_bits as u64) << 32);
                activations.leaky_relu(input, alpha)
            }
            ActivationType::ELU(alpha_bits) => {
                let alpha = f64::from_bits((*alpha_bits as u64) << 32);
                activations.elu(input, alpha)
            }
            ActivationType::GELU => activations.gelu(input),
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &str {
        match self {
            ActivationType::ReLU => "ReLU",
            ActivationType::Sigmoid => "Sigmoid",
            ActivationType::Tanh => "Tanh",
            ActivationType::LeakyReLU(_) => "LeakyReLU",
            ActivationType::ELU(_) => "ELU",
            ActivationType::GELU => "GELU",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_relu_positive() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = activations.relu(&input).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_negative() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-1.0, -2.0, -3.0]);
        let result = activations.relu(&input).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_mixed() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-2.0, 0.0, 3.0]);
        let result = activations.relu(&input).unwrap();

        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let result = activations.sigmoid(&input).unwrap();

        // sigmoid(0) = 0.5
        assert!((result[0] - 0.5).abs() < 1e-6);

        // sigmoid(1) ≈ 0.731
        assert!((result[1] - 0.7310585786300049).abs() < 1e-6);

        // sigmoid(-1) ≈ 0.269
        assert!((result[2] - 0.2689414213699951).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_range() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-10.0, 0.0, 10.0]);
        let result = activations.sigmoid(&input).unwrap();

        // All values should be in (0, 1)
        for &val in result.iter() {
            assert!(val > 0.0 && val < 1.0);
        }

        // Large negative -> close to 0
        assert!(result[0] < 0.01);

        // Large positive -> close to 1
        assert!(result[2] > 0.99);
    }

    #[test]
    fn test_tanh() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let result = activations.tanh(&input).unwrap();

        // tanh(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-6);

        // tanh(1) ≈ 0.7616
        assert!((result[1] - 0.7615941559557649).abs() < 1e-6);

        // tanh(-1) ≈ -0.7616
        assert!((result[2] + 0.7615941559557649).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_range() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-10.0, 0.0, 10.0]);
        let result = activations.tanh(&input).unwrap();

        // All values should be in (-1, 1)
        for &val in result.iter() {
            assert!(val > -1.0 && val < 1.0);
        }

        // Large negative -> close to -1
        assert!(result[0] < -0.99);

        // Large positive -> close to 1
        assert!(result[2] > 0.99);
    }

    #[test]
    fn test_softmax() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = activations.softmax(&input).unwrap();

        // Should sum to 1.0
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        for &val in result.iter() {
            assert!(val > 0.0);
        }

        // Larger inputs should have larger probabilities
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let result = activations.softmax(&input).unwrap();

        // Uniform input -> uniform output (1/3 each)
        for &val in result.iter() {
            assert!((val - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let activations = GpuActivations::cpu_only();

        // Large values that could cause overflow
        let input = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let result = activations.softmax(&input).unwrap();

        // Should still work due to max subtraction
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-2.0, 0.0, 2.0]);
        let alpha = 0.01;
        let result = activations.leaky_relu(&input, alpha).unwrap();

        // Negative: alpha * x
        assert!((result[0] - (-0.02)).abs() < 1e-6);

        // Zero stays zero
        assert!((result[1] - 0.0).abs() < 1e-6);

        // Positive unchanged
        assert!((result[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_elu() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let alpha = 1.0;
        let result = activations.elu(&input, alpha).unwrap();

        // Positive unchanged
        assert!((result[2] - 1.0).abs() < 1e-6);

        // Zero stays zero
        assert!((result[1] - 0.0).abs() < 1e-6);

        // Negative: alpha * (exp(x) - 1)
        let expected_neg = alpha * ((-1.0_f64).exp() - 1.0);
        assert!((result[0] - expected_neg).abs() < 1e-6);
    }

    #[test]
    fn test_gelu() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-2.0, 0.0, 2.0]);
        let result = activations.gelu(&input).unwrap();

        // GELU(0) ≈ 0
        assert!((result[1] - 0.0).abs() < 0.01);

        // GELU is smooth, so negative values not exactly 0
        assert!(result[0].abs() > 0.0 && result[0] < 0.0);

        // GELU(2) ≈ 2 (for large positive)
        assert!((result[2] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_activation_type_apply() {
        let activations = GpuActivations::cpu_only();
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);

        let relu_result = ActivationType::ReLU.apply(&activations, &input).unwrap();
        assert!((relu_result[0] - 0.0).abs() < 1e-6);
        assert!((relu_result[2] - 1.0).abs() < 1e-6);

        let sigmoid_result = ActivationType::Sigmoid.apply(&activations, &input).unwrap();
        assert!(sigmoid_result[1] > 0.49 && sigmoid_result[1] < 0.51);

        let tanh_result = ActivationType::Tanh.apply(&activations, &input).unwrap();
        assert!((tanh_result[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_2d() {
        let activations = GpuActivations::cpu_only();
        let input = Array2::from_shape_vec((2, 3), vec![
            -1.0, 0.0, 1.0,
            -2.0, 3.0, -4.0,
        ]).unwrap();

        let result = activations.relu_2d(&input).unwrap();

        assert_eq!(result.dim(), (2, 3));
        assert!((result[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((result[[0, 2]] - 1.0).abs() < 1e-6);
        assert!((result[[1, 1]] - 3.0).abs() < 1e-6);
        assert!((result[[1, 2]] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_2d() {
        let activations = GpuActivations::cpu_only();
        let input = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            3.0, 2.0, 1.0,
        ]).unwrap();

        let result = activations.softmax_2d(&input).unwrap();

        // Each row should sum to 1
        for i in 0..2 {
            let row_sum: f64 = result.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
}
