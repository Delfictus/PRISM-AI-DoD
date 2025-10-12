//! GPU-Accelerated Active Inference Classifier v2
//!
//! Enhanced version with automatic GPU/CPU fallback and performance tracking

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use std::collections::VecDeque;
use crate::gpu::gpu_executor::{GpuExecutor, Backend, matmul_auto, relu_auto, softmax_auto};

/// Threat classification result
#[derive(Debug, Clone)]
pub struct ThreatClassification {
    pub class_probabilities: Array1<f64>,
    pub free_energy: f64,
    pub confidence: f64,
    pub expected_class: ThreatClass,
    pub backend_used: String,
    pub inference_time_us: u64,
}

/// Threat class enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatClass {
    NoThreat = 0,
    Aircraft = 1,
    CruiseMissile = 2,
    BallisticMissile = 3,
    Hypersonic = 4,
}

impl ThreatClass {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => ThreatClass::NoThreat,
            1 => ThreatClass::Aircraft,
            2 => ThreatClass::CruiseMissile,
            3 => ThreatClass::BallisticMissile,
            4 => ThreatClass::Hypersonic,
            _ => ThreatClass::NoThreat,
        }
    }
}

/// Enhanced recognition network with GPU/CPU fallback
pub struct RecognitionNetworkV2 {
    weights1: Vec<f32>,  // 100 -> 64
    weights2: Vec<f32>,  // 64 -> 32
    weights3: Vec<f32>,  // 32 -> 16
    weights4: Vec<f32>,  // 16 -> 5
    bias1: Vec<f32>,
    bias2: Vec<f32>,
    bias3: Vec<f32>,
    bias4: Vec<f32>,
    executor: GpuExecutor,
}

impl RecognitionNetworkV2 {
    pub fn new(backend: Backend) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale1 = (2.0 / 100.0_f32).sqrt();
        let scale2 = (2.0 / 64.0_f32).sqrt();
        let scale3 = (2.0 / 32.0_f32).sqrt();
        let scale4 = (2.0 / 16.0_f32).sqrt();

        Ok(Self {
            weights1: (0..100*64).map(|_| rng.gen_range(-scale1..scale1)).collect(),
            weights2: (0..64*32).map(|_| rng.gen_range(-scale2..scale2)).collect(),
            weights3: (0..32*16).map(|_| rng.gen_range(-scale3..scale3)).collect(),
            weights4: (0..16*5).map(|_| rng.gen_range(-scale4..scale4)).collect(),
            bias1: vec![0.0; 64],
            bias2: vec![0.0; 32],
            bias3: vec![0.0; 16],
            bias4: vec![0.0; 5],
            executor: GpuExecutor::new(backend)?,
        })
    }

    pub fn forward(&self, features: &[f32]) -> Result<(Vec<f32>, String)> {
        let batch_size = features.len() / 100;
        let mut backend_used = String::new();

        // Layer 1: 100 -> 64
        let x1 = self.executor.execute(
            "pwsa_layer1",
            || {
                let mut result = matmul_auto(features, &self.weights1, batch_size, 100, 64)?;
                for i in 0..batch_size {
                    for j in 0..64 {
                        result[i * 64 + j] += self.bias1[j];
                    }
                }
                backend_used.push_str("GPU:");
                Ok(relu_auto(&result)?)
            },
            || {
                let mut result = vec![0.0; batch_size * 64];
                // CPU matmul
                for b in 0..batch_size {
                    for j in 0..64 {
                        let mut sum = self.bias1[j];
                        for i in 0..100 {
                            sum += features[b * 100 + i] * self.weights1[i * 64 + j];
                        }
                        result[b * 64 + j] = sum.max(0.0); // ReLU
                    }
                }
                backend_used.push_str("CPU:");
                Ok(result)
            }
        )?;

        // Layer 2: 64 -> 32
        let x2 = self.executor.execute(
            "pwsa_layer2",
            || {
                let mut result = matmul_auto(&x1, &self.weights2, batch_size, 64, 32)?;
                for i in 0..batch_size {
                    for j in 0..32 {
                        result[i * 32 + j] += self.bias2[j];
                    }
                }
                backend_used.push_str("GPU:");
                Ok(relu_auto(&result)?)
            },
            || {
                let mut result = vec![0.0; batch_size * 32];
                for b in 0..batch_size {
                    for j in 0..32 {
                        let mut sum = self.bias2[j];
                        for i in 0..64 {
                            sum += x1[b * 64 + i] * self.weights2[i * 32 + j];
                        }
                        result[b * 32 + j] = sum.max(0.0);
                    }
                }
                backend_used.push_str("CPU:");
                Ok(result)
            }
        )?;

        // Layer 3: 32 -> 16
        let x3 = self.executor.execute(
            "pwsa_layer3",
            || {
                let mut result = matmul_auto(&x2, &self.weights3, batch_size, 32, 16)?;
                for i in 0..batch_size {
                    for j in 0..16 {
                        result[i * 16 + j] += self.bias3[j];
                    }
                }
                backend_used.push_str("GPU:");
                Ok(relu_auto(&result)?)
            },
            || {
                let mut result = vec![0.0; batch_size * 16];
                for b in 0..batch_size {
                    for j in 0..16 {
                        let mut sum = self.bias3[j];
                        for i in 0..32 {
                            sum += x2[b * 32 + i] * self.weights3[i * 16 + j];
                        }
                        result[b * 16 + j] = sum.max(0.0);
                    }
                }
                backend_used.push_str("CPU:");
                Ok(result)
            }
        )?;

        // Layer 4: 16 -> 5 (logits)
        let logits = self.executor.execute(
            "pwsa_layer4",
            || {
                let mut result = matmul_auto(&x3, &self.weights4, batch_size, 16, 5)?;
                for i in 0..batch_size {
                    for j in 0..5 {
                        result[i * 5 + j] += self.bias4[j];
                    }
                }
                backend_used.push_str("GPU");
                Ok(result)
            },
            || {
                let mut result = vec![0.0; batch_size * 5];
                for b in 0..batch_size {
                    for j in 0..5 {
                        let mut sum = self.bias4[j];
                        for i in 0..16 {
                            sum += x3[b * 16 + i] * self.weights4[i * 5 + j];
                        }
                        result[b * 5 + j] = sum;
                    }
                }
                backend_used.push_str("CPU");
                Ok(result)
            }
        )?;

        Ok((logits, backend_used))
    }

    pub fn get_performance_report(&self) -> String {
        self.executor.get_performance_report()
    }
}

/// Enhanced Active Inference Classifier with GPU/CPU fallback
pub struct ActiveInferenceClassifierV2 {
    recognition_network: RecognitionNetworkV2,
    prior_beliefs: Array1<f64>,
    free_energy_history: VecDeque<f64>,
    backend: Backend,
}

impl ActiveInferenceClassifierV2 {
    pub fn new(backend: Backend) -> Result<Self> {
        let recognition_network = RecognitionNetworkV2::new(backend)?;
        let prior_beliefs = Array1::from_vec(vec![0.7, 0.1, 0.1, 0.05, 0.05]);

        Ok(Self {
            recognition_network,
            prior_beliefs,
            free_energy_history: VecDeque::with_capacity(100),
            backend,
        })
    }

    pub fn classify(&mut self, features: &Array1<f64>) -> Result<ThreatClassification> {
        let start = std::time::Instant::now();

        // Convert to f32 and forward pass
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let (logits, backend_used) = self.recognition_network.forward(&features_f32)?;

        // Apply softmax
        let probs = softmax_auto(&logits, 5)?;

        // Convert to Array1<f64>
        let posterior = Array1::from_vec(probs.iter().map(|&x| x as f64).collect());

        // Compute free energy
        let free_energy = self.compute_free_energy(&posterior)?;

        // Update beliefs
        let beliefs = self.update_beliefs(&posterior)?;

        // Track free energy
        self.free_energy_history.push_back(free_energy);
        if self.free_energy_history.len() > 100 {
            self.free_energy_history.pop_front();
        }

        // Get expected class
        let expected_idx = beliefs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let confidence = beliefs.iter().cloned().fold(0.0_f64, f64::max);
        let inference_time_us = start.elapsed().as_micros() as u64;

        Ok(ThreatClassification {
            class_probabilities: beliefs,
            free_energy,
            confidence,
            expected_class: ThreatClass::from_index(expected_idx),
            backend_used,
            inference_time_us,
        })
    }

    pub fn classify_batch(&mut self, features_batch: &[Array1<f64>]) -> Result<Vec<ThreatClassification>> {
        let start = std::time::Instant::now();
        let batch_size = features_batch.len();

        // Prepare batch data
        let mut batch_data = Vec::with_capacity(batch_size * 100);
        for features in features_batch {
            for &val in features.iter() {
                batch_data.push(val as f32);
            }
        }

        // Forward pass for entire batch
        let (logits, backend_used) = self.recognition_network.forward(&batch_data)?;

        // Apply softmax to entire batch
        let probs = softmax_auto(&logits, 5)?;

        // Process each result
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start_idx = i * 5;
            let end_idx = start_idx + 5;

            let posterior = Array1::from_vec(
                probs[start_idx..end_idx].iter()
                    .map(|&x| x as f64)
                    .collect()
            );

            let free_energy = self.compute_free_energy(&posterior)?;
            let beliefs = self.update_beliefs(&posterior)?;

            let expected_idx = beliefs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let confidence = beliefs.iter().cloned().fold(0.0_f64, f64::max);

            results.push(ThreatClassification {
                class_probabilities: beliefs,
                free_energy,
                confidence,
                expected_class: ThreatClass::from_index(expected_idx),
                backend_used: backend_used.clone(),
                inference_time_us: 0, // Will be set after
            });
        }

        // Calculate average inference time
        let total_time_us = start.elapsed().as_micros() as u64;
        let avg_time_us = total_time_us / batch_size as u64;

        for result in &mut results {
            result.inference_time_us = avg_time_us;
        }

        Ok(results)
    }

    fn compute_free_energy(&self, posterior: &Array1<f64>) -> Result<f64> {
        let mut kl_divergence = 0.0;

        for i in 0..posterior.len() {
            let q = posterior[i];
            let p = self.prior_beliefs[i];

            if q > 1e-10 && p > 1e-10 {
                kl_divergence += q * (q / p).ln();
            }
        }

        Ok(kl_divergence)
    }

    fn update_beliefs(&self, posterior: &Array1<f64>) -> Result<Array1<f64>> {
        let mut beliefs = Array1::zeros(posterior.len());

        for i in 0..posterior.len() {
            beliefs[i] = posterior[i] * self.prior_beliefs[i];
        }

        let sum: f64 = beliefs.iter().sum();
        if sum > 0.0 {
            beliefs.mapv_inplace(|p| p / sum);
        }

        Ok(beliefs)
    }

    pub fn get_performance_report(&self) -> String {
        self.recognition_network.get_performance_report()
    }
}

/// Benchmark enhanced classifier
pub fn benchmark_v2(num_samples: usize) -> Result<()> {
    println!("\n=== Enhanced GPU Classifier Benchmark ===\n");

    // Generate test data
    let mut test_features = Vec::new();
    for i in 0..num_samples {
        let pattern = match i % 5 {
            0 => vec![0.1; 100],
            1 => {
                let mut v = vec![0.1; 100];
                v[0..20].iter_mut().for_each(|x| *x = 0.9);
                v
            },
            _ => vec![0.5; 100],
        };
        test_features.push(Array1::from_vec(pattern));
    }

    // Test different backends
    let backends = vec![
        ("CPU-Only", Backend::CpuOnly),
        ("GPU-Preferred", Backend::PreferGpu),
        ("Auto-Select", Backend::Auto),
    ];

    for (name, backend) in backends {
        println!("Testing backend: {}", name);

        let mut classifier = ActiveInferenceClassifierV2::new(backend)?;

        // Warmup
        for i in 0..10 {
            classifier.classify(&test_features[i % test_features.len()]).ok();
        }

        // Measure performance
        let start = std::time::Instant::now();

        for features in &test_features {
            classifier.classify(features)?;
        }

        let elapsed = start.elapsed();
        let throughput = num_samples as f64 / elapsed.as_secs_f64();

        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Throughput: {:.0} samples/sec", throughput);

        // Test batch processing
        let batch_start = std::time::Instant::now();
        classifier.classify_batch(&test_features)?;
        let batch_elapsed = batch_start.elapsed();
        let batch_throughput = num_samples as f64 / batch_elapsed.as_secs_f64();

        println!("  Batch time: {:.2}ms", batch_elapsed.as_secs_f64() * 1000.0);
        println!("  Batch throughput: {:.0} samples/sec", batch_throughput);

        // Print performance report
        println!("{}", classifier.get_performance_report());
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_classifier() {
        let mut classifier = ActiveInferenceClassifierV2::new(Backend::Auto).unwrap();
        let features = Array1::from_vec(vec![0.5; 100]);

        let result = classifier.classify(&features).unwrap();

        assert!(result.free_energy.is_finite());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

        let sum: f64 = result.class_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_classification_v2() {
        let mut classifier = ActiveInferenceClassifierV2::new(Backend::Auto).unwrap();

        let batch = vec![
            Array1::from_vec(vec![0.3; 100]),
            Array1::from_vec(vec![0.5; 100]),
            Array1::from_vec(vec![0.7; 100]),
        ];

        let results = classifier.classify_batch(&batch).unwrap();

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.free_energy.is_finite());
        }
    }
}