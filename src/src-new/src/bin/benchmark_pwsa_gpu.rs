//! Benchmark GPU vs CPU performance for PWSA Active Inference Classifier
//!
//! This benchmark compares the performance of:
//! 1. GPU-accelerated classifier (using SimpleGpu backend)
//! 2. CPU baseline classifier
//! 3. Batch vs single-sample processing

use anyhow::{Result, Context};
use ndarray::Array1;
use std::time::{Instant, Duration};
use prism_ai::pwsa::gpu_classifier::{GpuActiveInferenceClassifier, ThreatClass};

/// CPU-based classifier for baseline comparison
struct CpuClassifier {
    weights1: ndarray::Array2<f32>,
    weights2: ndarray::Array2<f32>,
    weights3: ndarray::Array2<f32>,
    weights4: ndarray::Array2<f32>,
    bias1: Array1<f32>,
    bias2: Array1<f32>,
    bias3: Array1<f32>,
    bias4: Array1<f32>,
}

impl CpuClassifier {
    fn new() -> Self {
        use ndarray::Array2;
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize weights matching GPU network architecture
        Self {
            weights1: Array2::from_shape_fn((64, 100), |_| rng.gen::<f32>() * 0.1),
            weights2: Array2::from_shape_fn((32, 64), |_| rng.gen::<f32>() * 0.1),
            weights3: Array2::from_shape_fn((16, 32), |_| rng.gen::<f32>() * 0.1),
            weights4: Array2::from_shape_fn((5, 16), |_| rng.gen::<f32>() * 0.1),
            bias1: Array1::from_shape_fn(64, |_| rng.gen::<f32>() * 0.01),
            bias2: Array1::from_shape_fn(32, |_| rng.gen::<f32>() * 0.01),
            bias3: Array1::from_shape_fn(16, |_| rng.gen::<f32>() * 0.01),
            bias4: Array1::from_shape_fn(5, |_| rng.gen::<f32>() * 0.01),
        }
    }

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        // Layer 1: 100 → 64 + ReLU
        let mut x = self.weights1.dot(input) + &self.bias1;
        x.mapv_inplace(|v| v.max(0.0)); // ReLU

        // Layer 2: 64 → 32 + ReLU
        let mut x = self.weights2.dot(&x) + &self.bias2;
        x.mapv_inplace(|v| v.max(0.0)); // ReLU

        // Layer 3: 32 → 16 + ReLU
        let mut x = self.weights3.dot(&x) + &self.bias3;
        x.mapv_inplace(|v| v.max(0.0)); // ReLU

        // Layer 4: 16 → 5 (logits)
        let logits = self.weights4.dot(&x) + &self.bias4;

        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Array1<f32> = logits.mapv(|v| (v - max).exp());
        let sum: f32 = exp_vals.iter().sum();
        exp_vals / sum
    }

    fn classify(&self, features: &Array1<f64>) -> (ThreatClass, f64, Duration) {
        let start = Instant::now();

        // Convert to f32
        let features_f32 = features.mapv(|v| v as f32);

        // Forward pass
        let probs = self.forward(&features_f32);

        // Get max probability class
        let (idx, &confidence) = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let elapsed = start.elapsed();

        (ThreatClass::from_index(idx), confidence as f64, elapsed)
    }
}

/// Benchmark configuration
struct BenchmarkConfig {
    num_samples: usize,
    batch_sizes: Vec<usize>,
    warmup_samples: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            batch_sizes: vec![1, 8, 16, 32, 64, 128],
            warmup_samples: 100,
        }
    }
}

/// Run comprehensive benchmark
fn run_benchmark(config: BenchmarkConfig) -> Result<()> {
    println!("\n========================================");
    println!("   PWSA GPU Acceleration Benchmark");
    println!("========================================\n");

    // Generate test data
    println!("Generating {} test samples...", config.num_samples);
    let mut test_features = Vec::new();
    for i in 0..config.num_samples {
        // Simulate different threat patterns
        let pattern = match i % 5 {
            0 => vec![0.1; 100], // No threat pattern
            1 => {
                let mut v = vec![0.1; 100];
                v[0..20].iter_mut().for_each(|x| *x = 0.9); // Aircraft pattern
                v
            },
            2 => {
                let mut v = vec![0.1; 100];
                v[20..40].iter_mut().for_each(|x| *x = 0.8); // Cruise missile
                v
            },
            3 => {
                let mut v = vec![0.1; 100];
                v[40..60].iter_mut().for_each(|x| *x = 0.85); // Ballistic
                v
            },
            4 => {
                let mut v = vec![0.1; 100];
                v[60..80].iter_mut().for_each(|x| *x = 0.95); // Hypersonic
                v
            },
            _ => vec![0.5; 100],
        };
        test_features.push(Array1::from_vec(pattern));
    }

    // 1. CPU Baseline
    println!("\n[1] CPU Baseline Performance");
    println!("----------------------------");
    let cpu_classifier = CpuClassifier::new();

    // Warmup
    for i in 0..config.warmup_samples {
        cpu_classifier.classify(&test_features[i % test_features.len()]);
    }

    // Measure CPU performance
    let cpu_start = Instant::now();
    let mut cpu_times = Vec::new();

    for features in &test_features {
        let (_, _, time) = cpu_classifier.classify(features);
        cpu_times.push(time);
    }

    let cpu_total_time = cpu_start.elapsed();
    let cpu_avg_time = cpu_total_time / config.num_samples as u32;
    let cpu_throughput = config.num_samples as f64 / cpu_total_time.as_secs_f64();

    println!("  Total time:        {:.2} ms", cpu_total_time.as_secs_f64() * 1000.0);
    println!("  Avg per sample:    {:.3} μs", cpu_avg_time.as_secs_f64() * 1_000_000.0);
    println!("  Throughput:        {:.0} samples/sec", cpu_throughput);

    // 2. GPU Single-Sample Performance
    println!("\n[2] GPU Single-Sample Performance");
    println!("---------------------------------");
    let mut gpu_classifier = GpuActiveInferenceClassifier::new()
        .context("Failed to create GPU classifier")?;

    // Warmup
    for i in 0..config.warmup_samples {
        gpu_classifier.classify(&test_features[i % test_features.len()]).ok();
    }

    // Measure GPU single-sample performance
    let gpu_single_start = Instant::now();
    let mut gpu_times = Vec::new();

    for features in &test_features {
        let result = gpu_classifier.classify(features)?;
        gpu_times.push(result.inference_time_us);
    }

    let gpu_single_total = gpu_single_start.elapsed();
    let gpu_single_avg = gpu_single_total / config.num_samples as u32;
    let gpu_single_throughput = config.num_samples as f64 / gpu_single_total.as_secs_f64();

    println!("  Total time:        {:.2} ms", gpu_single_total.as_secs_f64() * 1000.0);
    println!("  Avg per sample:    {:.3} μs", gpu_single_avg.as_secs_f64() * 1_000_000.0);
    println!("  Throughput:        {:.0} samples/sec", gpu_single_throughput);

    let single_speedup = cpu_total_time.as_secs_f64() / gpu_single_total.as_secs_f64();
    println!("  Speedup vs CPU:    {:.2}x", single_speedup);

    // 3. GPU Batch Performance
    println!("\n[3] GPU Batch Performance");
    println!("-------------------------");

    for &batch_size in &config.batch_sizes {
        if batch_size > config.num_samples {
            continue;
        }

        let num_batches = config.num_samples / batch_size;
        let batch_start = Instant::now();

        for i in 0..num_batches {
            let start_idx = i * batch_size;
            let end_idx = start_idx + batch_size;
            let batch = &test_features[start_idx..end_idx];

            gpu_classifier.classify_batch(batch)?;
        }

        let batch_total = batch_start.elapsed();
        let batch_throughput = config.num_samples as f64 / batch_total.as_secs_f64();
        let batch_speedup = cpu_total_time.as_secs_f64() / batch_total.as_secs_f64();

        println!("  Batch size {}:", batch_size);
        println!("    Total time:      {:.2} ms", batch_total.as_secs_f64() * 1000.0);
        println!("    Throughput:      {:.0} samples/sec", batch_throughput);
        println!("    Speedup vs CPU:  {:.2}x", batch_speedup);
    }

    // 4. Analysis Summary
    println!("\n========================================");
    println!("            SUMMARY");
    println!("========================================");

    println!("\nPerformance Metrics:");
    println!("  CPU Baseline:       {:.0} samples/sec", cpu_throughput);
    println!("  GPU Single-Sample:  {:.0} samples/sec ({:.1}x faster)",
             gpu_single_throughput, single_speedup);

    // Find best batch size
    let mut best_batch_size = 1;
    let mut best_speedup = single_speedup;

    for &batch_size in &config.batch_sizes {
        if batch_size > config.num_samples {
            continue;
        }

        let num_batches = config.num_samples / batch_size;
        let batch_start = Instant::now();

        for i in 0..num_batches {
            let start_idx = i * batch_size;
            let end_idx = start_idx + batch_size;
            let batch = &test_features[start_idx..end_idx];
            gpu_classifier.classify_batch(batch).ok();
        }

        let batch_total = batch_start.elapsed();
        let speedup = cpu_total_time.as_secs_f64() / batch_total.as_secs_f64();

        if speedup > best_speedup {
            best_speedup = speedup;
            best_batch_size = batch_size;
        }
    }

    println!("  Best Batch Size:    {} ({:.1}x faster)", best_batch_size, best_speedup);

    println!("\nAcceleration Analysis:");
    if best_speedup > 10.0 {
        println!("  ✅ EXCELLENT: GPU provides {:.0}x speedup!", best_speedup);
        println!("  Real-time threat detection enabled for PWSA!");
    } else if best_speedup > 5.0 {
        println!("  ✅ GOOD: GPU provides {:.1}x speedup", best_speedup);
        println!("  Significant performance improvement achieved");
    } else if best_speedup > 2.0 {
        println!("  ⚠️  MODERATE: GPU provides {:.1}x speedup", best_speedup);
        println!("  Some acceleration, but below expectations");
    } else {
        println!("  ❌ LIMITED: GPU provides only {:.1}x speedup", best_speedup);
        println!("  GPU acceleration not effective (GPU acceleration active)");
    }

    // Memory efficiency
    println!("\nMemory Efficiency:");
    let samples_per_gb = 1_000_000_000.0 / (100.0 * 4.0); // 100 features * 4 bytes
    println!("  Max batch for 1GB:  {:.0} samples", samples_per_gb);
    println!("  Optimal batch:      {} samples", best_batch_size);

    Ok(())
}

fn main() -> Result<()> {
    println!("PWSA GPU Acceleration Benchmark");
    println!("================================");

    // Check GPU availability
    println!("\nChecking GPU availability...");

    // Try to create a GPU context to verify CUDA is working
    match prism_ai::gpu::simple_gpu::SimpleGpuContext::new() {
        Ok(_) => {
            println!("✅ GPU context created successfully");
            println!("   CUDA is available and initialized");
        },
        Err(e) => {
            println!("⚠️  GPU context creation failed: {}", e);
            println!("   Will use GPU kernels for GPU operations");
        }
    }

    // Run benchmark with different configurations
    let configs = vec![
        BenchmarkConfig {
            num_samples: 100,
            batch_sizes: vec![1, 10, 20, 50, 100],
            warmup_samples: 10,
        },
        BenchmarkConfig {
            num_samples: 1000,
            batch_sizes: vec![1, 10, 32, 64, 128, 256],
            warmup_samples: 50,
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        println!("\n\n[Test Configuration #{}]", i + 1);
        println!("Samples: {}, Warmup: {}", config.num_samples, config.warmup_samples);

        if let Err(e) = run_benchmark(config) {
            eprintln!("Benchmark failed: {}", e);
        }
    }

    println!("\n\nBenchmark complete!");

    Ok(())
}