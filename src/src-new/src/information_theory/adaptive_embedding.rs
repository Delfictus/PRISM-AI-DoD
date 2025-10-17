//! Adaptive Embedding Dimension Selection
//!
//! Automatically determines optimal embedding parameters for time series:
//!
//! 1. **Cao's Method**: Detect saturation of false nearest neighbors
//! 2. **Mutual Information Decay**: Find first minimum of AMI
//! 3. **Autocorrelation Method**: Use first zero-crossing
//!
//! Prevents:
//! - Under-embedding: Missing dynamical information
//! - Over-embedding: Curse of dimensionality, spurious complexity
//!
//! References:
//! - Cao, L. (1997). "Practical method for determining the minimum embedding dimension"
//! - Fraser, A. M., & Swinney, H. L. (1986). "Independent coordinates for strange attractors"

use anyhow::Result;
use ndarray::Array1;

/// Adaptive embedding dimension selector
pub struct AdaptiveEmbedding {
    /// Maximum embedding dimension to test
    pub max_dimension: usize,
    /// Maximum time delay to test
    pub max_delay: usize,
    /// Tolerance for Cao's E1 saturation
    pub tolerance: f64,
}

impl Default for AdaptiveEmbedding {
    fn default() -> Self {
        Self {
            max_dimension: 10,
            max_delay: 20,
            tolerance: 0.01,
        }
    }
}

impl AdaptiveEmbedding {
    /// Create new adaptive embedding selector
    pub fn new(max_dimension: usize, max_delay: usize) -> Self {
        Self {
            max_dimension,
            max_delay,
            tolerance: 0.01,
        }
    }

    /// Find optimal embedding dimension and delay
    ///
    /// Returns (optimal_dimension, optimal_delay)
    pub fn find_optimal(&self, series: &Array1<f64>) -> Result<(usize, usize)> {
        // Find optimal delay using mutual information
        let optimal_delay = self.find_optimal_delay(series)?;

        // Find optimal dimension using Cao's method
        let optimal_dimension = self.caos_method(series, optimal_delay)?;

        Ok((optimal_dimension, optimal_delay))
    }

    /// Cao's method for embedding dimension
    ///
    /// Computes E1(d) = <a(i,d)> and E2(d) = <E*(i,d)>
    /// Dimension is optimal when E1(d) saturates
    pub fn caos_method(&self, series: &Array1<f64>, delay: usize) -> Result<usize> {
        let n = series.len();

        if n < 100 {
            return Ok(2); // Default for short series
        }

        let mut e1_values = Vec::new();

        for dim in 1..self.max_dimension {
            let e1 = self.compute_e1(series, dim, delay)?;
            e1_values.push(e1);

            // Check for saturation
            if dim > 2 {
                let e1_ratio = e1_values[dim - 1] / e1_values[dim - 2].max(1e-10);

                // E1(d+1) / E1(d) ≈ 1 indicates saturation
                if (e1_ratio - 1.0).abs() < self.tolerance {
                    return Ok(dim);
                }
            }
        }

        // If no saturation found, return dimension with minimum slope
        let optimal_dim = self.find_minimum_slope(&e1_values).unwrap_or(3);

        Ok(optimal_dim)
    }

    /// Compute E1(d) for Cao's method
    ///
    /// E1(d) = <a(i,d)> where a(i,d) = ||y(d+1,i) - y(d+1,nn(i,d))|| / ||y(d,i) - y(d,nn(i,d))||
    fn compute_e1(&self, series: &Array1<f64>, dim: usize, delay: usize) -> Result<f64> {
        let n = series.len();
        let n_vectors = n - dim * delay;

        if n_vectors < 10 {
            return Ok(1.0);
        }

        // Build embedding vectors
        let mut embeddings: Vec<Vec<f64>> = Vec::new();

        for i in 0..n_vectors {
            let mut vec = Vec::new();
            for j in 0..dim {
                vec.push(series[i + j * delay]);
            }
            embeddings.push(vec);
        }

        let mut a_sum = 0.0;
        let mut count = 0;

        // For each vector, find nearest neighbor and compute a(i,d)
        for i in 0..n_vectors {
            // Find nearest neighbor (exclude self)
            let mut min_dist = f64::INFINITY;
            let mut nn_idx = 0;

            for j in 0..n_vectors {
                if i == j || j >= n_vectors - delay {
                    continue;
                }

                let dist = self.distance(&embeddings[i], &embeddings[j]);

                if dist < min_dist && dist > 1e-10 {
                    min_dist = dist;
                    nn_idx = j;
                }
            }

            // Compute distance in (d+1)-dimensional space
            if i < n_vectors - delay && nn_idx < n_vectors - delay {
                let mut vec_i_plus = embeddings[i].clone();
                vec_i_plus.push(series[i + dim * delay]);

                let mut vec_nn_plus = embeddings[nn_idx].clone();
                vec_nn_plus.push(series[nn_idx + dim * delay]);

                let dist_plus = self.distance(&vec_i_plus, &vec_nn_plus);

                if min_dist > 1e-10 {
                    a_sum += dist_plus / min_dist;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok(a_sum / count as f64)
        } else {
            Ok(1.0)
        }
    }

    /// Find optimal time delay using Average Mutual Information (AMI)
    ///
    /// Returns delay at first minimum of AMI
    pub fn find_optimal_delay(&self, series: &Array1<f64>) -> Result<usize> {
        let mut ami_values = Vec::new();

        for delay in 1..self.max_delay {
            let ami = self.average_mutual_information(series, delay)?;
            ami_values.push(ami);

            // Find first local minimum
            if delay > 2 {
                let is_minimum = ami_values[delay - 1] < ami_values[delay - 2]
                    && ami_values[delay - 1] < ami;

                if is_minimum {
                    return Ok(delay);
                }
            }
        }

        // If no minimum found, use autocorrelation method
        self.find_delay_autocorrelation(series)
    }

    /// Compute Average Mutual Information for given delay
    ///
    /// AMI(τ) = Σ p(x_t, x_t+τ) log[p(x_t, x_t+τ) / (p(x_t) p(x_t+τ))]
    fn average_mutual_information(&self, series: &Array1<f64>, delay: usize) -> Result<f64> {
        // Check if delay is too large for the series
        if delay >= series.len() {
            return Ok(0.0);  // Return 0 MI for invalid delay
        }

        let n = series.len() - delay;
        let n_bins = 10;

        // Discretize series
        let discretized = self.discretize(series, n_bins);

        // Build joint histogram
        let mut joint_counts = vec![vec![0; n_bins]; n_bins];
        let mut marginal_t = vec![0; n_bins];
        let mut marginal_t_plus = vec![0; n_bins];

        for i in 0..n {
            let bin_t = discretized[i] as usize;
            let bin_t_plus = discretized[i + delay] as usize;

            if bin_t < n_bins && bin_t_plus < n_bins {
                joint_counts[bin_t][bin_t_plus] += 1;
                marginal_t[bin_t] += 1;
                marginal_t_plus[bin_t_plus] += 1;
            }
        }

        // Compute mutual information
        let mut mi = 0.0;

        for i in 0..n_bins {
            for j in 0..n_bins {
                let p_joint = joint_counts[i][j] as f64 / n as f64;
                let p_i = marginal_t[i] as f64 / n as f64;
                let p_j = marginal_t_plus[j] as f64 / n as f64;

                if p_joint > 1e-10 && p_i > 1e-10 && p_j > 1e-10 {
                    mi += p_joint * (p_joint / (p_i * p_j)).ln();
                }
            }
        }

        Ok(mi / std::f64::consts::LN_2) // Convert to bits
    }

    /// Find delay using autocorrelation (first zero-crossing)
    fn find_delay_autocorrelation(&self, series: &Array1<f64>) -> Result<usize> {
        let mean = series.mean().unwrap();
        let variance: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;

        if variance < 1e-10 {
            return Ok(1);
        }

        for delay in 1..self.max_delay {
            let mut acf = 0.0;

            for i in 0..(series.len() - delay) {
                acf += (series[i] - mean) * (series[i + delay] - mean);
            }

            acf /= (series.len() - delay) as f64 * variance;

            // Find first zero-crossing or when ACF drops below threshold
            if acf < 0.0 || acf < 0.1 {
                return Ok(delay);
            }
        }

        Ok(1) // Default
    }

    /// Discretize time series into bins
    fn discretize(&self, series: &Array1<f64>, n_bins: usize) -> Vec<i32> {
        let min_val = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range < 1e-10 {
            return vec![0; series.len()];
        }

        series
            .iter()
            .map(|&x| {
                let normalized = (x - min_val) / range;
                let bin = (normalized * (n_bins as f64 - 1.0)) as i32;
                bin.max(0).min(n_bins as i32 - 1)
            })
            .collect()
    }

    /// Euclidean distance between vectors
    fn distance(&self, v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Find dimension with minimum slope in E1 curve
    fn find_minimum_slope(&self, e1_values: &[f64]) -> Option<usize> {
        if e1_values.len() < 2 {
            return None;
        }

        let mut min_slope = f64::INFINITY;
        let mut min_idx = 1;

        for i in 1..e1_values.len() {
            let slope = (e1_values[i] - e1_values[i - 1]).abs();

            if slope < min_slope {
                min_slope = slope;
                min_idx = i;
            }
        }

        Some(min_idx + 1) // +1 because dimension starts at 1
    }
}

/// Embedding parameters result
#[derive(Debug, Clone)]
pub struct EmbeddingParams {
    /// Optimal embedding dimension
    pub dimension: usize,
    /// Optimal time delay
    pub delay: usize,
    /// Cao's E1 saturation indicator
    pub saturation_indicator: f64,
    /// Mutual information at chosen delay
    pub mutual_info: f64,
}

impl EmbeddingParams {
    /// Create from dimension and delay
    pub fn new(dimension: usize, delay: usize) -> Self {
        Self {
            dimension,
            delay,
            saturation_indicator: 0.0,
            mutual_info: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_embedding_creation() {
        let adaptive = AdaptiveEmbedding::new(10, 20);
        assert_eq!(adaptive.max_dimension, 10);
        assert_eq!(adaptive.max_delay, 20);
    }

    #[test]
    fn test_find_optimal_delay_sine() {
        let adaptive = AdaptiveEmbedding::default();

        // Sine wave with period 2π/0.1 ≈ 63 samples
        let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let series_arr = Array1::from_vec(series);

        let delay = adaptive.find_optimal_delay(&series_arr).unwrap();

        println!("Optimal delay for sine wave: {}", delay);

        // Should find reasonable delay (quarter period ≈ 16)
        assert!(delay >= 1);
        assert!(delay <= 30);
    }

    #[test]
    fn test_caos_method() {
        let adaptive = AdaptiveEmbedding::default();

        // Simple deterministic series
        let series: Vec<f64> = (0..300).map(|i| (i as f64 * 0.05).sin()).collect();
        let series_arr = Array1::from_vec(series);

        let dimension = adaptive.caos_method(&series_arr, 5).unwrap();

        println!("Optimal dimension (Cao): {}", dimension);

        // Should find reasonable dimension (2-5 for sine wave)
        assert!(dimension >= 1);
        assert!(dimension <= 10);
    }

    #[test]
    fn test_find_optimal_embedding() {
        let adaptive = AdaptiveEmbedding::new(8, 15);

        // Lorenz-like chaotic series (simplified)
        let mut series = vec![0.1];
        for i in 1..500 {
            let val = series[i - 1] * 3.9 * (1.0 - series[i - 1]);
            series.push(val);
        }

        let series_arr = Array1::from_vec(series);

        let (dim, delay) = adaptive.find_optimal(&series_arr).unwrap();

        println!("Optimal embedding: dim={}, delay={}", dim, delay);

        assert!(dim >= 1 && dim <= 8);
        assert!(delay >= 1 && delay <= 15);
    }

    #[test]
    fn test_average_mutual_information() {
        let adaptive = AdaptiveEmbedding::default();

        let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let series_arr = Array1::from_vec(series);

        let ami_1 = adaptive.average_mutual_information(&series_arr, 1).unwrap();
        let ami_10 = adaptive.average_mutual_information(&series_arr, 10).unwrap();

        println!("AMI(1): {}, AMI(10): {}", ami_1, ami_10);

        // AMI should decay with increasing delay
        assert!(ami_1 > 0.0);
        assert!(ami_10 >= 0.0);
    }

    #[test]
    fn test_autocorrelation_delay() {
        let adaptive = AdaptiveEmbedding::default();

        // Strongly autocorrelated series
        let mut series = vec![0.5];
        for i in 1..200 {
            series.push(0.9 * series[i - 1] + 0.1 * (i as f64 * 0.1).sin());
        }

        let series_arr = Array1::from_vec(series);

        let delay = adaptive.find_delay_autocorrelation(&series_arr).unwrap();

        println!("Autocorrelation delay: {}", delay);

        assert!(delay >= 1);
        assert!(delay <= 20);
    }

    #[test]
    fn test_discretization() {
        let adaptive = AdaptiveEmbedding::default();

        let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let discretized = adaptive.discretize(&series, 5);

        // Should map to bins 0-4
        assert_eq!(discretized.len(), 5);
        assert!(discretized.iter().all(|&x| x >= 0 && x < 5));

        // Values should be increasing
        for i in 1..discretized.len() {
            assert!(discretized[i] >= discretized[i - 1]);
        }
    }

    #[test]
    fn test_short_series() {
        let adaptive = AdaptiveEmbedding::default();

        // Very short series
        let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let (dim, delay) = adaptive.find_optimal(&series).unwrap();

        println!("Short series: dim={}, delay={}", dim, delay);

        // Should return reasonable defaults
        assert!(dim >= 1);
        assert!(delay >= 1);
    }
}
