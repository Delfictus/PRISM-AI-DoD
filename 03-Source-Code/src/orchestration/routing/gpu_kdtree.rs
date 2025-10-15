//! GPU-accelerated k-Nearest Neighbors for Transfer Entropy
//!
//! Implements efficient k-NN search on GPU for KSG transfer entropy estimation.
//! Uses parallel distance computation and GPU-based top-k selection.
//!
//! For KSG algorithm, we need to find k nearest neighbors in:
//! 1. Joint space (X, Y past, Y present)
//! 2. Marginal spaces (X alone, Y past alone, etc.)

use anyhow::{Result, Context as AnyhowContext};
use cudarc::driver::LaunchConfig;
use ndarray::{Array1, Array2};

use crate::gpu::kernel_executor::get_global_executor;

/// Distance metric for k-NN search
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance: sqrt(sum((x_i - y_i)^2))
    Euclidean,
    /// Manhattan distance: sum(|x_i - y_i|)
    Manhattan,
    /// Chebyshev distance: max(|x_i - y_i|)
    Chebyshev,
    /// Maximum norm (for KSG algorithm)
    MaxNorm,
}

/// GPU-accelerated k-Nearest Neighbors search
///
/// Optimized for high-dimensional spaces typical in transfer entropy calculations.
///
/// # Example
/// ```no_run
/// use prism_ai::orchestration::routing::gpu_kdtree::{GpuNearestNeighbors, DistanceMetric};
/// use ndarray::Array2;
///
/// let knn = GpuNearestNeighbors::new()?;
///
/// // Dataset: 100 points in 3D space
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect())?;
///
/// // Query: find 5 nearest neighbors for first point
/// let query = data.row(0).to_owned();
/// let (indices, distances) = knn.find_k_nearest(&data, &query, 5, DistanceMetric::Euclidean)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct GpuNearestNeighbors;

impl GpuNearestNeighbors {
    /// Create new GPU k-NN instance
    pub fn new() -> Result<Self> {
        // Ensure global executor is initialized
        let _ = get_global_executor()
            .context("Failed to initialize GPU executor")?;

        Ok(Self)
    }

    /// Find k nearest neighbors for a single query point
    ///
    /// # Arguments
    /// * `dataset` - Reference dataset (n_points, n_dims)
    /// * `query` - Query point (n_dims,)
    /// * `k` - Number of neighbors to find
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    /// (indices, distances) - Arrays of k nearest neighbor indices and their distances
    pub fn find_k_nearest(
        &self,
        dataset: &Array2<f64>,
        query: &Array1<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let (n_points, n_dims) = dataset.dim();

        anyhow::ensure!(
            query.len() == n_dims,
            "Query dimension {} doesn't match dataset dimension {}",
            query.len(),
            n_dims
        );

        anyhow::ensure!(
            k > 0 && k <= n_points,
            "k must be between 1 and {}, got {}",
            n_points,
            k
        );

        // Compute all distances on GPU
        let distances = self.compute_distances_gpu(dataset, query, metric)?;

        // Find top-k smallest distances
        let (indices, top_distances) = self.select_top_k(&distances, k)?;

        Ok((indices, top_distances))
    }

    /// Find k nearest neighbors for multiple query points (batch mode)
    ///
    /// # Arguments
    /// * `dataset` - Reference dataset (n_points, n_dims)
    /// * `queries` - Multiple query points (n_queries, n_dims)
    /// * `k` - Number of neighbors to find per query
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    /// Vector of (indices, distances) for each query
    pub fn find_k_nearest_batch(
        &self,
        dataset: &Array2<f64>,
        queries: &Array2<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<(Vec<usize>, Vec<f64>)>> {
        let (n_queries, query_dims) = queries.dim();
        let (_, data_dims) = dataset.dim();

        anyhow::ensure!(
            query_dims == data_dims,
            "Query dimensions {} don't match dataset dimensions {}",
            query_dims,
            data_dims
        );

        let mut results = Vec::with_capacity(n_queries);

        for i in 0..n_queries {
            let query = queries.row(i).to_owned();
            let result = self.find_k_nearest(dataset, &query, k, metric)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compute distances from query to all points in dataset using GPU
    ///
    /// Uses parallel computation on GPU for efficiency
    fn compute_distances_gpu(
        &self,
        dataset: &Array2<f64>,
        query: &Array1<f64>,
        metric: DistanceMetric,
    ) -> Result<Vec<f64>> {
        let (n_points, n_dims) = dataset.dim();

        // Convert to f32 for GPU
        let dataset_f32: Vec<f32> = dataset.iter().map(|&x| x as f32).collect();
        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();

        // Get executor
        let executor = get_global_executor()?;
        let executor_lock = executor.lock().unwrap();
        let context = executor_lock.context();

        // Prepare GPU memory
        let stream = context.default_stream();
        let dataset_dev = stream.memcpy_stod(&dataset_f32)?;
        let query_dev = stream.memcpy_stod(&query_f32)?;
        let mut distances_dev = stream.alloc_zeros::<f32>(n_points)?;

        // Compile and launch distance kernel based on metric
        let kernel_code = self.generate_distance_kernel(metric);

        // Register kernel if not already present
        drop(executor_lock);
        {
            let mut exec_mut = executor.lock().unwrap();
            exec_mut.register_kernel("compute_distances", &kernel_code)?;
        }
        let executor_lock = executor.lock().unwrap();

        let kernel = executor_lock.get_kernel("compute_distances")?;

        // Launch kernel
        let block_size = 256;
        let grid_size = (n_points as u32 + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    &dataset_dev,
                    &query_dev,
                    &mut distances_dev,
                    n_points as i32,
                    n_dims as i32,
                ),
            )?;
        }

        // Synchronize and download
        context.synchronize()?;
        let distances_f32 = stream.memcpy_dtov(&distances_dev)?;

        // Convert back to f64
        let distances_f64: Vec<f64> = distances_f32.iter().map(|&x| x as f64).collect();

        Ok(distances_f64)
    }

    /// Generate CUDA kernel code for distance computation
    fn generate_distance_kernel(&self, metric: DistanceMetric) -> String {
        match metric {
            DistanceMetric::Euclidean => r#"
            extern "C" __global__ void compute_distances(
                float* dataset, float* query, float* distances,
                int n_points, int n_dims
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_points) return;

                float sum = 0.0f;
                for (int d = 0; d < n_dims; d++) {
                    float diff = dataset[idx * n_dims + d] - query[d];
                    sum += diff * diff;
                }
                distances[idx] = sqrtf(sum);
            }
            "#.to_string(),

            DistanceMetric::Manhattan => r#"
            extern "C" __global__ void compute_distances(
                float* dataset, float* query, float* distances,
                int n_points, int n_dims
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_points) return;

                float sum = 0.0f;
                for (int d = 0; d < n_dims; d++) {
                    float diff = dataset[idx * n_dims + d] - query[d];
                    sum += fabsf(diff);
                }
                distances[idx] = sum;
            }
            "#.to_string(),

            DistanceMetric::Chebyshev => r#"
            extern "C" __global__ void compute_distances(
                float* dataset, float* query, float* distances,
                int n_points, int n_dims
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_points) return;

                float max_diff = 0.0f;
                for (int d = 0; d < n_dims; d++) {
                    float diff = fabsf(dataset[idx * n_dims + d] - query[d]);
                    max_diff = fmaxf(max_diff, diff);
                }
                distances[idx] = max_diff;
            }
            "#.to_string(),

            DistanceMetric::MaxNorm => r#"
            extern "C" __global__ void compute_distances(
                float* dataset, float* query, float* distances,
                int n_points, int n_dims
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_points) return;

                // Max norm (same as Chebyshev for KSG)
                float max_diff = 0.0f;
                for (int d = 0; d < n_dims; d++) {
                    float diff = fabsf(dataset[idx * n_dims + d] - query[d]);
                    max_diff = fmaxf(max_diff, diff);
                }
                distances[idx] = max_diff;
            }
            "#.to_string(),
        }
    }

    /// Select top-k smallest values and their indices
    ///
    /// Uses CPU-based selection for now (GPU top-k for large k could be added)
    fn select_top_k(&self, distances: &[f64], k: usize) -> Result<(Vec<usize>, Vec<f64>)> {
        // Create index-distance pairs
        let mut indexed: Vec<(usize, f64)> = distances.iter()
            .enumerate()
            .map(|(i, &d)| (i, d))
            .collect();

        // Partial sort to get k smallest
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take first k elements and sort them
        indexed.truncate(k);
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let distances: Vec<f64> = indexed.iter().map(|(_, d)| *d).collect();

        Ok((indices, distances))
    }

    /// Count neighbors within a given radius (for KSG algorithm)
    ///
    /// # Arguments
    /// * `dataset` - Reference dataset
    /// * `query` - Query point
    /// * `radius` - Search radius
    /// * `metric` - Distance metric
    ///
    /// # Returns
    /// Number of points within radius
    pub fn count_within_radius(
        &self,
        dataset: &Array2<f64>,
        query: &Array1<f64>,
        radius: f64,
        metric: DistanceMetric,
    ) -> Result<usize> {
        let distances = self.compute_distances_gpu(dataset, query, metric)?;
        let count = distances.iter().filter(|&&d| d < radius).count();
        Ok(count)
    }

    /// Find distance to k-th nearest neighbor (for KSG algorithm)
    ///
    /// This is useful for adaptive radius searches in KSG
    pub fn find_kth_distance(
        &self,
        dataset: &Array2<f64>,
        query: &Array1<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Result<f64> {
        let (_, distances) = self.find_k_nearest(dataset, query, k, metric)?;
        Ok(distances[k - 1])
    }
}

impl Default for GpuNearestNeighbors {
    fn default() -> Self {
        Self::new().expect("Failed to create GpuNearestNeighbors")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_knn_simple_2d() -> Result<()> {
        let knn = GpuNearestNeighbors::new()?;

        // Simple 2D dataset
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0,  // Point 0
                1.0, 0.0,  // Point 1
                0.0, 1.0,  // Point 2
                2.0, 0.0,  // Point 3
                0.0, 2.0,  // Point 4
            ],
        )?;

        // Query at origin - should find itself and then (1,0) and (0,1)
        let query = array![0.0, 0.0];
        let (indices, distances) = knn.find_k_nearest(&data, &query, 3, DistanceMetric::Euclidean)?;

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);

        // First neighbor should be itself (distance 0)
        assert_eq!(indices[0], 0);
        assert!(distances[0].abs() < 1e-6);

        // Second and third neighbors should be at distance 1.0
        assert!(distances[1] - 1.0 < 1e-5);
        assert!(distances[2] - 1.0 < 1e-5);

        Ok(())
    }

    #[test]
    fn test_knn_different_metrics() -> Result<()> {
        let knn = GpuNearestNeighbors::new()?;

        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0,
                1.0, 1.0,
                2.0, 0.0,
                0.0, 2.0,
            ],
        )?;

        let query = array![0.5, 0.5];

        // Euclidean distance
        let (_, dist_euc) = knn.find_k_nearest(&data, &query, 2, DistanceMetric::Euclidean)?;

        // Manhattan distance
        let (_, dist_man) = knn.find_k_nearest(&data, &query, 2, DistanceMetric::Manhattan)?;

        // Distances should be different for different metrics
        assert!(dist_euc[0] != dist_man[0]);

        Ok(())
    }

    #[test]
    fn test_knn_batch() -> Result<()> {
        let knn = GpuNearestNeighbors::new()?;

        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0,
                1.0, 0.0,
                0.0, 1.0,
                2.0, 0.0,
                0.0, 2.0,
            ],
        )?;

        let queries = Array2::from_shape_vec(
            (2, 2),
            vec![
                0.0, 0.0,  // Query 1
                1.0, 1.0,  // Query 2
            ],
        )?;

        let results = knn.find_k_nearest_batch(&data, &queries, 2, DistanceMetric::Euclidean)?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.len(), 2); // 2 neighbors for query 1
        assert_eq!(results[1].0.len(), 2); // 2 neighbors for query 2

        Ok(())
    }

    #[test]
    fn test_count_within_radius() -> Result<()> {
        let knn = GpuNearestNeighbors::new()?;

        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0,
                0.5, 0.0,
                1.0, 0.0,
                1.5, 0.0,
                2.0, 0.0,
            ],
        )?;

        let query = array![0.0, 0.0];

        // Count points within radius 1.1
        let count = knn.count_within_radius(&data, &query, 1.1, DistanceMetric::Euclidean)?;

        // Should find 3 points: (0,0), (0.5,0), and (1.0,0)
        assert_eq!(count, 3);

        Ok(())
    }

    #[test]
    fn test_kth_distance() -> Result<()> {
        let knn = GpuNearestNeighbors::new()?;

        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0,
                1.0, 0.0,
                2.0, 0.0,
                3.0, 0.0,
                4.0, 0.0,
            ],
        )?;

        let query = array![0.0, 0.0];

        // Distance to 3rd nearest neighbor
        let dist = knn.find_kth_distance(&data, &query, 3, DistanceMetric::Euclidean)?;

        // Should be distance to (2.0, 0.0) = 2.0
        assert!((dist - 2.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_invalid_k() {
        let knn = GpuNearestNeighbors::new().unwrap();

        let data = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        ).unwrap();

        let query = array![0.0, 0.0];

        // k=0 should fail
        let result = knn.find_k_nearest(&data, &query, 0, DistanceMetric::Euclidean);
        assert!(result.is_err());

        // k > n_points should fail
        let result = knn.find_k_nearest(&data, &query, 10, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }
}
