// GPU-Accelerated Linear Algebra for Financial Computing
// Integrates Worker 2's linear algebra GPU kernels
// Constitution: Financial Application + Production GPU Optimization

use anyhow::{Result, Context};
use ndarray::{Array1, Array2};

#[cfg(feature = "cuda")]
use crate::gpu::kernel_executor::get_global_executor;

/// GPU-accelerated vector operations for financial calculations
///
/// Leverages Worker 2's GPU kernels for:
/// - dot_product: 5-10x speedup for portfolio returns
/// - elementwise_multiply: 5-10x speedup for weight adjustments
/// - elementwise_exp: 5-10x speedup for log-returns conversion
/// - reduce_sum: 5-10x speedup for portfolio aggregation
/// - normalize_inplace: 5-10x speedup for weight normalization
pub struct GpuVectorOps {
    /// Use GPU if available
    pub use_gpu: bool,
}

impl Default for GpuVectorOps {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

impl GpuVectorOps {
    /// Create new GPU vector operations handler
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate dot product: a · b with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// Dot product scalar value
    ///
    /// # Use Cases
    /// - Portfolio expected return: weights · expected_returns
    /// - Risk calculation: weights · (covariance @ weights)
    /// - Correlation analysis
    pub fn dot_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> Result<f64> {
        if a.len() != b.len() {
            anyhow::bail!("Vectors must have same length for dot product");
        }

        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.dot_product_gpu(a, b) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(a.dot(b))
    }

    /// GPU implementation using Worker 2's dot_product kernel
    #[cfg(feature = "cuda")]
    fn dot_product_gpu(&self, a: &Array1<f64>, b: &Array1<f64>) -> Result<f64> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

        // Use Worker 2's dot_product kernel
        let result = executor.dot_product(&a_f32, &b_f32)
            .context("GPU dot_product failed")?;

        Ok(result as f64)
    }

    /// Element-wise multiplication: a ⊙ b with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// Element-wise product vector
    ///
    /// # Use Cases
    /// - Asset weight adjustments: weights ⊙ adjustment_factors
    /// - Return scaling: returns ⊙ scaling_factors
    /// - Masking operations
    pub fn elementwise_multiply(&self, a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        if a.len() != b.len() {
            anyhow::bail!("Vectors must have same length for elementwise multiply");
        }

        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.elementwise_multiply_gpu(a, b) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(a * b)
    }

    /// GPU implementation using Worker 2's elementwise_multiply kernel
    #[cfg(feature = "cuda")]
    fn elementwise_multiply_gpu(&self, a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

        // Use Worker 2's elementwise_multiply kernel
        let result_f32 = executor.elementwise_multiply(&a_f32, &b_f32)
            .context("GPU elementwise_multiply failed")?;

        // Convert back to f64
        let result = Array1::from_vec(result_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// Element-wise exponential: exp(a) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `a` - Input vector
    ///
    /// # Returns
    /// Element-wise exponential vector
    ///
    /// # Use Cases
    /// - Log-returns to returns conversion
    /// - Exponential growth modeling
    /// - Softmax computation (with normalization)
    pub fn elementwise_exp(&self, a: &Array1<f64>) -> Result<Array1<f64>> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.elementwise_exp_gpu(a) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(a.mapv(|x| x.exp()))
    }

    /// GPU implementation using Worker 2's elementwise_exp kernel
    #[cfg(feature = "cuda")]
    fn elementwise_exp_gpu(&self, a: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

        // Use Worker 2's elementwise_exp kernel
        let result_f32 = executor.elementwise_exp(&a_f32)
            .context("GPU elementwise_exp failed")?;

        // Convert back to f64
        let result = Array1::from_vec(result_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// Reduce sum: Σ(a) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `a` - Input vector
    ///
    /// # Returns
    /// Sum of all elements
    ///
    /// # Use Cases
    /// - Portfolio total value
    /// - Weight sum verification
    /// - Aggregation operations
    pub fn reduce_sum(&self, a: &Array1<f64>) -> Result<f64> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.reduce_sum_gpu(a) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(a.sum())
    }

    /// GPU implementation using Worker 2's reduce_sum kernel
    #[cfg(feature = "cuda")]
    fn reduce_sum_gpu(&self, a: &Array1<f64>) -> Result<f64> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

        // Use Worker 2's reduce_sum kernel
        let result = executor.reduce_sum(&a_f32)
            .context("GPU reduce_sum failed")?;

        Ok(result as f64)
    }

    /// Normalize vector in-place: a / ||a||₁ with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `a` - Input vector (will be normalized)
    ///
    /// # Returns
    /// Normalized vector (L1 norm = 1)
    ///
    /// # Use Cases
    /// - Portfolio weight normalization (ensure weights sum to 1)
    /// - Probability distribution normalization
    /// - Feature scaling
    pub fn normalize(&self, a: &Array1<f64>) -> Result<Array1<f64>> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.normalize_gpu(a) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        let sum = a.sum();
        if sum.abs() < 1e-10 {
            anyhow::bail!("Cannot normalize zero vector");
        }
        Ok(a / sum)
    }

    /// GPU implementation using Worker 2's normalize_inplace kernel
    #[cfg(feature = "cuda")]
    fn normalize_gpu(&self, a: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to f32 for GPU
        let mut a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

        // Use Worker 2's normalize_inplace kernel (modifies in-place)
        executor.normalize_inplace(&mut a_f32)
            .context("GPU normalize_inplace failed")?;

        // Convert back to f64
        let result = Array1::from_vec(a_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }

    /// Calculate portfolio expected return: weights · expected_returns
    ///
    /// # Arguments
    /// * `weights` - Portfolio weights (must sum to 1.0)
    /// * `expected_returns` - Expected returns for each asset
    ///
    /// # Returns
    /// Expected portfolio return
    pub fn portfolio_return(&self, weights: &Array1<f64>, expected_returns: &Array1<f64>) -> Result<f64> {
        self.dot_product(weights, expected_returns)
    }

    /// Calculate weighted sum with normalization
    ///
    /// # Arguments
    /// * `values` - Values to weight
    /// * `weights` - Weights (will be normalized automatically)
    ///
    /// # Returns
    /// Weighted sum
    pub fn weighted_sum(&self, values: &Array1<f64>, weights: &Array1<f64>) -> Result<f64> {
        let normalized_weights = self.normalize(weights)?;
        self.dot_product(values, &normalized_weights)
    }
}

/// GPU-accelerated matrix operations for financial computing
pub struct GpuMatrixOps {
    pub use_gpu: bool,
}

impl Default for GpuMatrixOps {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

impl GpuMatrixOps {
    pub fn new() -> Self {
        Self::default()
    }

    /// Matrix-vector multiplication: A @ x with automatic GPU/CPU fallback
    ///
    /// # Use Cases
    /// - Covariance risk: covariance_matrix @ weights
    /// - Linear transformations
    /// - Portfolio analytics
    pub fn matvec(&self, matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>> {
        if matrix.ncols() != vector.len() {
            anyhow::bail!("Matrix columns must match vector length");
        }

        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(result) = self.matvec_gpu(matrix, vector) {
                    return Ok(result);
                }
            }
        }

        // Fall back to CPU
        Ok(matrix.dot(vector))
    }

    #[cfg(feature = "cuda")]
    fn matvec_gpu(&self, matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        let m = matrix.nrows();
        let n = matrix.ncols();

        // Convert to f32 for GPU (row-major layout)
        let matrix_f32: Vec<f32> = matrix.iter().map(|&x| x as f32).collect();
        let vector_f32: Vec<f32> = vector.iter().map(|&x| x as f32).collect();

        // Reshape vector to matrix for matrix_multiply kernel
        let vector_as_matrix: Vec<f32> = vector_f32.clone();

        // Use Worker 2's matrix_multiply kernel: result = matrix @ vector (as column)
        let result_f32 = executor.matrix_multiply(&matrix_f32, &vector_as_matrix, m, n, 1)
            .context("GPU matrix_multiply failed")?;

        // Convert back to f64
        let result = Array1::from_vec(result_f32.iter().map(|&x| x as f64).collect());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_dot_product() {
        let ops = GpuVectorOps::new();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let result = ops.dot_product(&a, &b).unwrap();

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_multiply() {
        let ops = GpuVectorOps::new();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0]);

        let result = ops.elementwise_multiply(&a, &b).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
        assert!((result[2] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_exp() {
        let ops = GpuVectorOps::new();

        let a = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let result = ops.elementwise_exp(&a).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - std::f64::consts::E).abs() < 1e-5);
        assert!((result[2] - std::f64::consts::E.powi(2)).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_sum() {
        let ops = GpuVectorOps::new();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ops.reduce_sum(&a).unwrap();

        assert!((result - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let ops = GpuVectorOps::new();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = ops.normalize(&a).unwrap();

        // Sum should be 1.0
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Proportions should be maintained
        assert!((result[0] - 0.1).abs() < 1e-6); // 1/10
        assert!((result[1] - 0.2).abs() < 1e-6); // 2/10
        assert!((result[2] - 0.3).abs() < 1e-6); // 3/10
        assert!((result[3] - 0.4).abs() < 1e-6); // 4/10
    }

    #[test]
    fn test_portfolio_return() {
        let ops = GpuVectorOps::new();

        // 3-asset portfolio
        let weights = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let expected_returns = Array1::from_vec(vec![0.10, 0.15, 0.08]);

        let portfolio_return = ops.portfolio_return(&weights, &expected_returns).unwrap();

        // 0.5*0.10 + 0.3*0.15 + 0.2*0.08 = 0.05 + 0.045 + 0.016 = 0.111
        assert!((portfolio_return - 0.111).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_sum() {
        let ops = GpuVectorOps::new();

        let values = Array1::from_vec(vec![100.0, 200.0, 300.0]);
        let weights = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Will be normalized

        let result = ops.weighted_sum(&values, &weights).unwrap();

        // Normalized weights: [1/6, 2/6, 3/6]
        // 100*(1/6) + 200*(2/6) + 300*(3/6) = 16.67 + 66.67 + 150 = 233.33
        assert!((result - 233.333).abs() < 0.01);
    }

    #[test]
    fn test_matvec() {
        let ops = GpuMatrixOps::new();

        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();

        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = ops.matvec(&matrix, &vector).unwrap();

        // [1,2,3] @ [1,2,3] = 14
        // [4,5,6] @ [1,2,3] = 32
        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-6);
        assert!((result[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_ops_consistency() {
        let ops = GpuVectorOps::new();

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        // Multiple calls should give same result
        let dot1 = ops.dot_product(&a, &b).unwrap();
        let dot2 = ops.dot_product(&a, &b).unwrap();

        assert!((dot1 - dot2).abs() < 1e-10);
    }
}
