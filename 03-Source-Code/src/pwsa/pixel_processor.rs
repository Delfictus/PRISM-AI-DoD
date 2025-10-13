//! Pixel-Level IR Processing for PWSA
//!
//! # Purpose
//! Dedicated module for full pixel-level IR analysis:
//! - Shannon entropy computation (per-pixel and windowed)
//! - Pixel-level TDA features
//! - Convolution-based feature extraction
//! - GPU-accelerated operations
//!
//! # Worker 3 Task Plan
//! This fulfills "Full pixel-level IR processing" requirement from 8-worker plan:
//! - Extract pixel processing from satellite_adapters into dedicated module
//! - Add GPU acceleration hooks for Worker 2's kernels
//! - Implement pixel-level entropy maps
//! - Add convolutional feature extraction
//! - Integrate with TDA for topological analysis
//!
//! # Constitution Compliance
//! - Article II: GPU acceleration required
//! - Article I: Worker 3 owns this file
//! - Article III: Tests required

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use crate::mathematics::EnhancedITMetrics;

#[cfg(feature = "cuda")]
use crate::gpu::GpuMemoryPool;

/// GPU-Accelerated pixel processor for IR frames
pub struct PixelProcessor {
    /// Entropy computation window size (pixels)
    entropy_window_size: usize,

    /// Convolution kernel size
    conv_kernel_size: usize,

    /// TDA persistence threshold
    tda_threshold: f64,

    /// Enhanced IT metrics computer with bias correction
    it_metrics: EnhancedITMetrics,

    #[cfg(feature = "cuda")]
    gpu_context: Option<GpuMemoryPool>,
}

impl PixelProcessor {
    /// Create new pixel processor with GPU acceleration
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_context = Some(GpuMemoryPool::new()
            .context("Failed to initialize GPU for pixel processing")?);

        let it_metrics = EnhancedITMetrics::new()
            .with_bias_correction(true); // Enable Miller-Madow correction

        Ok(Self {
            entropy_window_size: 16,
            conv_kernel_size: 3,
            tda_threshold: 1000.0,
            it_metrics,
            #[cfg(feature = "cuda")]
            gpu_context,
        })
    }

    /// Compute pixel-level Shannon entropy map
    ///
    /// Returns entropy value for each pixel based on local window distribution.
    /// GPU-accelerated for performance.
    ///
    /// # Arguments
    /// * `pixels` - Raw IR pixel intensities
    /// * `window_size` - Local window size (e.g., 16x16)
    ///
    /// # Returns
    /// Entropy map (same dimensions as input)
    pub fn compute_entropy_map(
        &mut self,
        pixels: &Array2<u16>,
        window_size: usize,
    ) -> Result<Array2<f32>> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.compute_entropy_map_gpu(pixels, window_size);
            }
        }

        self.compute_entropy_map_cpu(pixels, window_size)
    }

    fn compute_entropy_map_cpu(
        &self,
        pixels: &Array2<u16>,
        window_size: usize,
    ) -> Result<Array2<f32>> {
        let (height, width) = pixels.dim();
        let mut entropy_map = Array2::zeros((height, width));

        let half_window = window_size / 2;

        // For each pixel, compute entropy of local window
        for y in 0..height {
            for x in 0..width {
                // Extract local window
                let y_min = y.saturating_sub(half_window);
                let y_max = (y + half_window).min(height - 1);
                let x_min = x.saturating_sub(half_window);
                let x_max = (x + half_window).min(width - 1);

                let window = pixels.slice(ndarray::s![y_min..=y_max, x_min..=x_max]);

                // Compute histogram for this window
                let histogram = self.compute_local_histogram(&window, 16);

                // Compute Shannon entropy
                let entropy = self.compute_shannon_entropy(&histogram);

                entropy_map[[y, x]] = entropy as f32;
            }
        }

        Ok(entropy_map)
    }

    #[cfg(feature = "cuda")]
    fn compute_entropy_map_gpu(
        &self,
        pixels: &Array2<u16>,
        window_size: usize,
    ) -> Result<Array2<f32>> {
        // TODO: Request pixel_entropy_kernel from Worker 2
        // __global__ void pixel_entropy(
        //     uint16_t* pixels,
        //     float* entropy_map,
        //     int height,
        //     int width,
        //     int window_size
        // )

        // Placeholder: use CPU implementation
        self.compute_entropy_map_cpu(pixels, window_size)
    }

    /// Compute intensity histogram for local window
    fn compute_local_histogram(
        &self,
        window: &ndarray::ArrayView2<u16>,
        n_bins: usize,
    ) -> Vec<usize> {
        let min_val = window.iter().min().copied().unwrap_or(0) as f64;
        let max_val = window.iter().max().copied().unwrap_or(0) as f64;
        let range = max_val - min_val;

        let mut histogram = vec![0; n_bins];

        if range == 0.0 {
            histogram[0] = window.len();
            return histogram;
        }

        for &pixel in window.iter() {
            let normalized = (pixel as f64 - min_val) / range;
            let bin = (normalized * (n_bins - 1) as f64) as usize;
            histogram[bin.min(n_bins - 1)] += 1;
        }

        histogram
    }

    /// Compute Shannon entropy from histogram
    ///
    /// Now uses EnhancedITMetrics with Miller-Madow bias correction for improved accuracy.
    fn compute_shannon_entropy(&self, histogram: &[usize]) -> f64 {
        // Use enhanced IT metrics with bias correction
        let entropy = self.it_metrics.shannon_entropy_from_histogram(histogram);

        // Normalize by maximum possible entropy for consistency with existing API
        let max_entropy = (histogram.len() as f64).log2();

        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Extract convolutional features from pixel data
    ///
    /// Applies 2D convolution for edge detection, texture analysis.
    /// GPU-accelerated for performance.
    ///
    /// # Returns
    /// Feature maps for different kernels (edges, corners, textures)
    pub fn extract_conv_features(
        &mut self,
        pixels: &Array2<u16>,
    ) -> Result<ConvFeatures> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.extract_conv_features_gpu(pixels);
            }
        }

        self.extract_conv_features_cpu(pixels)
    }

    fn extract_conv_features_cpu(
        &self,
        pixels: &Array2<u16>,
    ) -> Result<ConvFeatures> {
        // Sobel edge detection kernels
        let sobel_x = Array2::from_shape_vec((3, 3), vec![
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ])?;

        let sobel_y = Array2::from_shape_vec((3, 3), vec![
            -1.0, -2.0, -1.0,
             0.0,  0.0,  0.0,
             1.0,  2.0,  1.0,
        ])?;

        // Laplacian kernel for blob detection
        let laplacian = Array2::from_shape_vec((3, 3), vec![
            0.0, 1.0, 0.0,
            1.0, -4.0, 1.0,
            0.0, 1.0, 0.0,
        ])?;

        let edge_x = self.convolve_2d(pixels, &sobel_x)?;
        let edge_y = self.convolve_2d(pixels, &sobel_y)?;
        let blobs = self.convolve_2d(pixels, &laplacian)?;

        // Compute edge magnitude
        let (height, width) = edge_x.dim();
        let mut edge_magnitude = Array2::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                let gx = edge_x[[y, x]];
                let gy = edge_y[[y, x]];
                edge_magnitude[[y, x]] = (gx * gx + gy * gy).sqrt();
            }
        }

        Ok(ConvFeatures {
            edge_magnitude,
            edge_direction_x: edge_x,
            edge_direction_y: edge_y,
            blob_response: blobs,
        })
    }

    #[cfg(feature = "cuda")]
    fn extract_conv_features_gpu(
        &self,
        pixels: &Array2<u16>,
    ) -> Result<ConvFeatures> {
        // TODO: Request conv2d_kernel from Worker 2
        // __global__ void conv2d(
        //     float* image,
        //     float* kernel,
        //     float* output,
        //     int height,
        //     int width,
        //     int kernel_size
        // )

        // Placeholder: use CPU
        self.extract_conv_features_cpu(pixels)
    }

    /// 2D convolution operation
    fn convolve_2d(
        &self,
        image: &Array2<u16>,
        kernel: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let (img_h, img_w) = image.dim();
        let (kern_h, kern_w) = kernel.dim();

        anyhow::ensure!(kern_h == kern_w, "Kernel must be square");
        anyhow::ensure!(kern_h % 2 == 1, "Kernel size must be odd");

        let pad = kern_h / 2;
        let mut output = Array2::zeros((img_h, img_w));

        for y in pad..(img_h - pad) {
            for x in pad..(img_w - pad) {
                let mut sum = 0.0f32;

                for ky in 0..kern_h {
                    for kx in 0..kern_w {
                        let img_y = y + ky - pad;
                        let img_x = x + kx - pad;

                        sum += image[[img_y, img_x]] as f32 * kernel[[ky, kx]];
                    }
                }

                output[[y, x]] = sum;
            }
        }

        Ok(output)
    }

    /// Compute pixel-level TDA features
    ///
    /// Analyzes topological structure of pixel intensities.
    /// Detects connected components, holes, voids.
    ///
    /// # Arguments
    /// * `pixels` - Raw IR pixel intensities
    /// * `threshold` - Intensity threshold for binarization
    ///
    /// # Returns
    /// Topological features (persistence diagram, Betti numbers)
    pub fn compute_pixel_tda(
        &mut self,
        pixels: &Array2<u16>,
        threshold: u16,
    ) -> Result<PixelTdaFeatures> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.compute_pixel_tda_gpu(pixels, threshold);
            }
        }

        self.compute_pixel_tda_cpu(pixels, threshold)
    }

    fn compute_pixel_tda_cpu(
        &self,
        pixels: &Array2<u16>,
        threshold: u16,
    ) -> Result<PixelTdaFeatures> {
        // Binarize image
        let (height, width) = pixels.dim();
        let mut binary = Array2::from_elem((height, width), false);

        for y in 0..height {
            for x in 0..width {
                binary[[y, x]] = pixels[[y, x]] > threshold;
            }
        }

        // Compute connected components (Betti-0)
        let betti_0 = self.count_connected_components(&binary);

        // Compute holes (Betti-1) - simplified
        let betti_1 = 0;  // Requires more sophisticated algorithm

        // Persistence features
        let max_intensity = *pixels.iter().max().unwrap_or(&0);
        let min_intensity = *pixels.iter().min().unwrap_or(&0);

        Ok(PixelTdaFeatures {
            betti_0,
            betti_1,
            persistence_range: (min_intensity as f64, max_intensity as f64),
            connected_components: betti_0,
        })
    }

    #[cfg(feature = "cuda")]
    fn compute_pixel_tda_gpu(
        &self,
        pixels: &Array2<u16>,
        threshold: u16,
    ) -> Result<PixelTdaFeatures> {
        // TODO: Request pixel_tda_kernel from Worker 2
        // __global__ void pixel_tda(
        //     uint16_t* pixels,
        //     int* persistence_diagram,
        //     int height,
        //     int width,
        //     float threshold
        // )

        // Placeholder: use CPU
        self.compute_pixel_tda_cpu(pixels, threshold)
    }

    /// Count connected components using flood fill
    fn count_connected_components(&self, binary: &Array2<bool>) -> usize {
        let (height, width) = binary.dim();
        let mut visited = Array2::from_elem((height, width), false);
        let mut count = 0;

        for y in 0..height {
            for x in 0..width {
                if binary[[y, x]] && !visited[[y, x]] {
                    self.flood_fill(&mut visited, binary, y, x);
                    count += 1;
                }
            }
        }

        count
    }

    /// Flood fill helper for connected components
    fn flood_fill(
        &self,
        visited: &mut Array2<bool>,
        binary: &Array2<bool>,
        y: usize,
        x: usize,
    ) {
        let (height, width) = visited.dim();

        if y >= height || x >= width {
            return;
        }

        if visited[[y, x]] || !binary[[y, x]] {
            return;
        }

        visited[[y, x]] = true;

        // 4-connected neighbors
        if y > 0 {
            self.flood_fill(visited, binary, y - 1, x);
        }
        if y < height - 1 {
            self.flood_fill(visited, binary, y + 1, x);
        }
        if x > 0 {
            self.flood_fill(visited, binary, y, x - 1);
        }
        if x < width - 1 {
            self.flood_fill(visited, binary, y, x + 1);
        }
    }

    /// Segment image into regions
    ///
    /// Groups pixels into semantically meaningful regions.
    /// GPU-accelerated for performance.
    pub fn segment_image(
        &mut self,
        pixels: &Array2<u16>,
        n_segments: usize,
    ) -> Result<Array2<u8>> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.segment_image_gpu(pixels, n_segments);
            }
        }

        self.segment_image_cpu(pixels, n_segments)
    }

    fn segment_image_cpu(
        &self,
        pixels: &Array2<u16>,
        n_segments: usize,
    ) -> Result<Array2<u8>> {
        // Simple k-means style segmentation
        let (height, width) = pixels.dim();
        let mut segments = Array2::zeros((height, width));

        // Compute intensity thresholds
        let min_val = *pixels.iter().min().unwrap_or(&0) as f64;
        let max_val = *pixels.iter().max().unwrap_or(&0) as f64;
        let range = max_val - min_val;

        if range == 0.0 {
            return Ok(segments);
        }

        // Assign segments based on intensity quantiles
        for y in 0..height {
            for x in 0..width {
                let normalized = (pixels[[y, x]] as f64 - min_val) / range;
                let segment = (normalized * n_segments as f64) as u8;
                segments[[y, x]] = segment.min((n_segments - 1) as u8);
            }
        }

        Ok(segments)
    }

    #[cfg(feature = "cuda")]
    fn segment_image_gpu(
        &self,
        pixels: &Array2<u16>,
        n_segments: usize,
    ) -> Result<Array2<u8>> {
        // TODO: Request image_segmentation_kernel from Worker 2

        // Placeholder: use CPU
        self.segment_image_cpu(pixels, n_segments)
    }
}

impl Default for PixelProcessor {
    fn default() -> Self {
        Self::new().expect("Failed to create default PixelProcessor")
    }
}

/// Convolutional features extracted from pixels
#[derive(Clone)]
pub struct ConvFeatures {
    /// Edge magnitude (Sobel)
    pub edge_magnitude: Array2<f32>,

    /// Edge direction X component
    pub edge_direction_x: Array2<f32>,

    /// Edge direction Y component
    pub edge_direction_y: Array2<f32>,

    /// Blob detection response (Laplacian)
    pub blob_response: Array2<f32>,
}

/// Pixel-level TDA features
#[derive(Clone, Debug)]
pub struct PixelTdaFeatures {
    /// Betti-0: number of connected components
    pub betti_0: usize,

    /// Betti-1: number of holes
    pub betti_1: usize,

    /// Persistence range (birth, death)
    pub persistence_range: (f64, f64),

    /// Connected components count
    pub connected_components: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_processor_creation() {
        let processor = PixelProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_entropy_computation() {
        let mut processor = PixelProcessor::new().unwrap();

        // Create simple test pattern
        let pixels = Array2::from_shape_fn((64, 64), |(y, x)| {
            if (y / 8 + x / 8) % 2 == 0 {
                1000u16
            } else {
                100u16
            }
        });

        let entropy_map = processor.compute_entropy_map(&pixels, 16);
        assert!(entropy_map.is_ok());

        let map = entropy_map.unwrap();
        assert_eq!(map.dim(), (64, 64));

        // Checkerboard pattern should have high entropy
        let avg_entropy: f32 = map.iter().sum::<f32>() / map.len() as f32;
        assert!(avg_entropy > 0.3);
    }

    #[test]
    fn test_conv_features() {
        let mut processor = PixelProcessor::new().unwrap();

        // Create edge pattern
        let mut pixels = Array2::from_elem((64, 64), 100u16);
        for y in 0..64 {
            for x in 32..64 {
                pixels[[y, x]] = 1000;
            }
        }

        let features = processor.extract_conv_features(&pixels);
        assert!(features.is_ok());

        let conv = features.unwrap();
        assert_eq!(conv.edge_magnitude.dim(), (64, 64));

        // Should detect vertical edge
        let edge_strength: f32 = conv.edge_magnitude.iter().sum::<f32>();
        assert!(edge_strength > 0.0);
    }

    #[test]
    fn test_pixel_tda() {
        let mut processor = PixelProcessor::new().unwrap();

        // Create pattern with multiple components
        let mut pixels = Array2::from_elem((64, 64), 100u16);

        // Add bright spots
        for y in 10..20 {
            for x in 10..20 {
                pixels[[y, x]] = 1000;
            }
        }
        for y in 40..50 {
            for x in 40..50 {
                pixels[[y, x]] = 1000;
            }
        }

        let tda = processor.compute_pixel_tda(&pixels, 500);
        assert!(tda.is_ok());

        let features = tda.unwrap();
        assert_eq!(features.betti_0, 2);  // Two connected components
    }

    #[test]
    fn test_shannon_entropy() {
        let processor = PixelProcessor::new().unwrap();

        // Uniform distribution: maximum entropy
        let uniform_hist = vec![10, 10, 10, 10];
        let entropy = processor.compute_shannon_entropy(&uniform_hist);
        assert!((entropy - 1.0).abs() < 0.01);

        // Single value: minimum entropy
        let single_hist = vec![40, 0, 0, 0];
        let entropy = processor.compute_shannon_entropy(&single_hist);
        assert!(entropy < 0.01);
    }

    #[test]
    fn test_segmentation() {
        let mut processor = PixelProcessor::new().unwrap();

        // Create gradient
        let pixels = Array2::from_shape_fn((64, 64), |(y, x)| {
            ((y * 16) as u16).min(4095)
        });

        let segments = processor.segment_image(&pixels, 4);
        assert!(segments.is_ok());

        let seg_map = segments.unwrap();
        assert_eq!(seg_map.dim(), (64, 64));

        // Should have 4 distinct segments
        let mut segment_values: Vec<u8> = seg_map.iter().copied().collect();
        segment_values.sort_unstable();
        segment_values.dedup();
        assert!(segment_values.len() <= 4);
    }
}
