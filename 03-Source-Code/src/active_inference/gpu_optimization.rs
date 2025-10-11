// GPU Optimization for Active Inference
// Replaces CPU fallback with actual GPU computation

use anyhow::Result;
use ndarray::Array1;

use super::{HierarchicalModel, VariationalInference};

/// Extension trait to add GPU acceleration to Active Inference
pub trait ActiveInferenceGpuExt {
    /// Run inference using GPU if available
    fn infer_auto(&mut self, model: &mut HierarchicalModel, observations: &Array1<f64>) -> Result<f64>;
}

impl ActiveInferenceGpuExt for VariationalInference {
    fn infer_auto(&mut self, model: &mut HierarchicalModel, observations: &Array1<f64>) -> Result<f64> {
        #[cfg(feature = "cuda")]
        {
            // Try GPU acceleration
            use std::sync::Arc;
            use cudarc::driver::CudaContext;
            use super::gpu::ActiveInferenceGpu;

            if let Ok(context) = CudaContext::new(0) {
                if let Ok(gpu_engine) = ActiveInferenceGpu::new(Arc::new(context), self.clone()) {
                    // Actually use GPU free energy computation
                    let free_energy = gpu_engine.compute_free_energy_gpu(
                        observations,
                        &model.level1.belief.mean,
                        &model.level1.belief.variance,
                        &model.level1.prior.mean,
                        &model.level1.prior.variance,
                        &self.observation_model.noise_precision,
                    )?;

                    // Update beliefs on GPU
                    for _ in 0..self.max_iterations.min(10) {
                        gpu_engine.update_beliefs_gpu(
                            &mut model.level1.belief.mean,
                            &self.observation_model.jacobian,
                            observations,
                            &self.observation_model.noise_precision,
                        )?;
                    }

                    return Ok(free_energy);
                }
            }
        }

        // Fall back to CPU
        self.infer(model, observations)
    }
}