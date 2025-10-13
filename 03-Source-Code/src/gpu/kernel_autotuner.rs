//! Kernel Auto-Tuning System
//!
//! Automatically selects optimal kernel launch configurations (block size, grid size)
//! based on workload characteristics and GPU architecture. Uses empirical tuning
//! to find best-performing configurations.

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Kernel launch configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaunchConfig {
    /// Block size (threads per block)
    pub block_size: usize,

    /// Grid size (blocks per grid)
    pub grid_size: usize,
}

impl LaunchConfig {
    /// Create new launch configuration
    pub fn new(block_size: usize, grid_size: usize) -> Self {
        Self { block_size, grid_size }
    }

    /// Calculate total threads
    pub fn total_threads(&self) -> usize {
        self.block_size * self.grid_size
    }

    /// Create default config for given problem size
    pub fn for_size(n: usize) -> Self {
        // Default: 256 threads per block, enough blocks to cover n
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        Self { block_size, grid_size }
    }
}

/// Performance measurement for a configuration
#[derive(Debug, Clone)]
struct ConfigPerformance {
    config: LaunchConfig,
    avg_time_us: f64,
    num_samples: usize,
    last_updated: Instant,
}

impl ConfigPerformance {
    fn new(config: LaunchConfig, time_us: f64) -> Self {
        Self {
            config,
            avg_time_us: time_us,
            num_samples: 1,
            last_updated: Instant::now(),
        }
    }

    fn update(&mut self, time_us: f64) {
        // Exponential moving average
        let alpha = 0.3;
        self.avg_time_us = alpha * time_us + (1.0 - alpha) * self.avg_time_us;
        self.num_samples += 1;
        self.last_updated = Instant::now();
    }
}

/// Kernel identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelId {
    pub name: String,
    pub problem_size: usize,
}

impl KernelId {
    pub fn new(name: impl Into<String>, problem_size: usize) -> Self {
        Self {
            name: name.into(),
            problem_size,
        }
    }

    /// Get size bucket (for generalization across similar sizes)
    pub fn size_bucket(&self) -> usize {
        // Bucket sizes by order of magnitude
        if self.problem_size < 1000 {
            self.problem_size / 100 * 100 // 0, 100, 200, ...
        } else if self.problem_size < 10000 {
            self.problem_size / 1000 * 1000 // 1k, 2k, ...
        } else if self.problem_size < 100000 {
            self.problem_size / 10000 * 10000 // 10k, 20k, ...
        } else {
            self.problem_size / 100000 * 100000 // 100k, 200k, ...
        }
    }
}

/// Auto-tuner configuration
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Enable auto-tuning
    pub enabled: bool,

    /// Number of configurations to try during tuning
    pub num_configs_to_try: usize,

    /// Minimum samples before considering a config tuned
    pub min_samples_for_tuning: usize,

    /// Re-tune after this many executions
    pub retune_interval: usize,

    /// Block size options to try
    pub block_size_options: Vec<usize>,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_configs_to_try: 5,
            min_samples_for_tuning: 3,
            retune_interval: 1000,
            block_size_options: vec![64, 128, 256, 512, 1024],
        }
    }
}

/// Kernel auto-tuner
pub struct KernelAutoTuner {
    config: AutoTunerConfig,
    /// Best configurations found: KernelId -> LaunchConfig
    best_configs: Arc<Mutex<HashMap<KernelId, LaunchConfig>>>,
    /// Performance history: (KernelId, LaunchConfig) -> ConfigPerformance
    performance_history: Arc<Mutex<HashMap<(KernelId, LaunchConfig), ConfigPerformance>>>,
    /// Execution counts: KernelId -> count
    execution_counts: Arc<Mutex<HashMap<KernelId, usize>>>,
}

impl KernelAutoTuner {
    /// Create new auto-tuner
    pub fn new(config: AutoTunerConfig) -> Self {
        Self {
            config,
            best_configs: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            execution_counts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(AutoTunerConfig::default())
    }

    /// Get optimal configuration for a kernel
    pub fn get_config(&self, kernel_id: &KernelId) -> LaunchConfig {
        if !self.config.enabled {
            return LaunchConfig::for_size(kernel_id.problem_size);
        }

        let best_configs = self.best_configs.lock().unwrap();

        // Check if we have a tuned config
        if let Some(config) = best_configs.get(kernel_id) {
            return *config;
        }

        // Check if we have a config for similar size (bucketed)
        let bucketed_id = KernelId {
            name: kernel_id.name.clone(),
            problem_size: kernel_id.size_bucket(),
        };
        if let Some(config) = best_configs.get(&bucketed_id) {
            return *config;
        }

        // Default config
        LaunchConfig::for_size(kernel_id.problem_size)
    }

    /// Record kernel execution time for tuning
    pub fn record_execution(&self, kernel_id: KernelId, config: LaunchConfig, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        let time_us = duration.as_micros() as f64;

        // Update performance history
        {
            let mut history = self.performance_history.lock().unwrap();
            let key = (kernel_id.clone(), config);

            if let Some(perf) = history.get_mut(&key) {
                perf.update(time_us);
            } else {
                history.insert(key, ConfigPerformance::new(config, time_us));
            }
        }

        // Update execution count
        {
            let mut counts = self.execution_counts.lock().unwrap();
            *counts.entry(kernel_id.clone()).or_insert(0) += 1;
        }

        // Check if we should update best config
        self.update_best_config(kernel_id);
    }

    /// Update best configuration based on performance history
    fn update_best_config(&self, kernel_id: KernelId) {
        let history = self.performance_history.lock().unwrap();

        // Find all configs for this kernel with enough samples
        let mut configs: Vec<_> = history
            .iter()
            .filter(|((kid, _), perf)| {
                kid == &kernel_id && perf.num_samples >= self.config.min_samples_for_tuning
            })
            .map(|((_, config), perf)| (*config, perf.avg_time_us))
            .collect();

        if configs.is_empty() {
            return; // Not enough data yet
        }

        // Find best config (lowest avg time)
        configs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let best_config = configs[0].0;

        // Update best config
        let mut best_configs = self.best_configs.lock().unwrap();
        best_configs.insert(kernel_id, best_config);
    }

    /// Get configurations to try for tuning
    pub fn get_tuning_configs(&self, kernel_id: &KernelId) -> Vec<LaunchConfig> {
        let mut configs = Vec::new();

        for &block_size in &self.config.block_size_options {
            if block_size > kernel_id.problem_size {
                continue; // Skip if block size > problem size
            }

            let grid_size = (kernel_id.problem_size + block_size - 1) / block_size;
            configs.push(LaunchConfig::new(block_size, grid_size));

            if configs.len() >= self.config.num_configs_to_try {
                break;
            }
        }

        // Ensure at least one config
        if configs.is_empty() {
            configs.push(LaunchConfig::for_size(kernel_id.problem_size));
        }

        configs
    }

    /// Check if kernel needs tuning
    pub fn needs_tuning(&self, kernel_id: &KernelId) -> bool {
        if !self.config.enabled {
            return false;
        }

        let best_configs = self.best_configs.lock().unwrap();

        // Needs tuning if no config found
        if !best_configs.contains_key(kernel_id) {
            return true;
        }

        // Check if we should re-tune
        let counts = self.execution_counts.lock().unwrap();
        if let Some(&count) = counts.get(kernel_id) {
            return count % self.config.retune_interval == 0;
        }

        false
    }

    /// Get tuning statistics
    pub fn get_stats(&self) -> AutoTunerStats {
        let best_configs = self.best_configs.lock().unwrap();
        let history = self.performance_history.lock().unwrap();
        let counts = self.execution_counts.lock().unwrap();

        let total_kernels = best_configs.len();
        let total_measurements = history.len();
        let total_executions: usize = counts.values().sum();

        // Calculate average speedup
        let mut speedups = Vec::new();
        for (kernel_id, best_config) in best_configs.iter() {
            let default_config = LaunchConfig::for_size(kernel_id.problem_size);

            if let Some(best_perf) = history.get(&(kernel_id.clone(), *best_config)) {
                if let Some(default_perf) = history.get(&(kernel_id.clone(), default_config)) {
                    let speedup = default_perf.avg_time_us / best_perf.avg_time_us;
                    speedups.push(speedup);
                }
            }
        }

        let avg_speedup = if speedups.is_empty() {
            1.0
        } else {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        };

        AutoTunerStats {
            enabled: self.config.enabled,
            total_kernels_tuned: total_kernels,
            total_measurements: total_measurements,
            total_executions,
            avg_speedup,
        }
    }

    /// Get tuning report
    pub fn get_report(&self) -> String {
        let stats = self.get_stats();
        let best_configs = self.best_configs.lock().unwrap();

        let mut report = format!(
            "Kernel Auto-Tuner Report:\n\
             ══════════════════════════════════════════\n\
             Status:       {}\n\
             Tuned Kernels: {}\n\
             Measurements:  {}\n\
             Executions:    {}\n\
             Avg Speedup:   {:.2}x\n\
             \n",
            if stats.enabled { "✅ Enabled" } else { "❌ Disabled" },
            stats.total_kernels_tuned,
            stats.total_measurements,
            stats.total_executions,
            stats.avg_speedup
        );

        if !best_configs.is_empty() {
            report.push_str("Top Tuned Kernels:\n");
            let mut configs: Vec<_> = best_configs.iter().collect();
            configs.sort_by_key(|(kid, _)| &kid.name);

            for (kernel_id, config) in configs.iter().take(10) {
                report.push_str(&format!(
                    "  • {:<30} (n={:<8}): block={}, grid={}\n",
                    kernel_id.name,
                    kernel_id.problem_size,
                    config.block_size,
                    config.grid_size
                ));
            }
        }

        report.push_str("══════════════════════════════════════════");

        report
    }
}

/// Auto-tuner statistics
#[derive(Debug, Clone)]
pub struct AutoTunerStats {
    pub enabled: bool,
    pub total_kernels_tuned: usize,
    pub total_measurements: usize,
    pub total_executions: usize,
    pub avg_speedup: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::new(256, 4);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.grid_size, 4);
        assert_eq!(config.total_threads(), 1024);
    }

    #[test]
    fn test_launch_config_for_size() {
        let config = LaunchConfig::for_size(1000);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.grid_size, 4); // ceil(1000 / 256) = 4
    }

    #[test]
    fn test_kernel_id_bucketing() {
        let kid1 = KernelId::new("matmul", 150);
        assert_eq!(kid1.size_bucket(), 100);

        let kid2 = KernelId::new("matmul", 2500);
        assert_eq!(kid2.size_bucket(), 2000);

        let kid3 = KernelId::new("matmul", 25000);
        assert_eq!(kid3.size_bucket(), 20000);
    }

    #[test]
    fn test_autotuner_creation() {
        let tuner = KernelAutoTuner::with_default_config();
        let stats = tuner.get_stats();
        assert!(stats.enabled);
        assert_eq!(stats.total_kernels_tuned, 0);
    }

    #[test]
    fn test_autotuner_get_config() {
        let tuner = KernelAutoTuner::with_default_config();
        let kernel_id = KernelId::new("test_kernel", 1024);

        let config = tuner.get_config(&kernel_id);
        assert_eq!(config.block_size, 256);
        assert!(config.grid_size > 0);
    }

    #[test]
    fn test_autotuner_record_execution() {
        let tuner = KernelAutoTuner::with_default_config();
        let kernel_id = KernelId::new("test_kernel", 1024);
        let config = LaunchConfig::new(256, 4);

        tuner.record_execution(kernel_id.clone(), config, Duration::from_micros(100));
        tuner.record_execution(kernel_id.clone(), config, Duration::from_micros(110));
        tuner.record_execution(kernel_id.clone(), config, Duration::from_micros(90));

        let stats = tuner.get_stats();
        assert_eq!(stats.total_executions, 3);
    }

    #[test]
    fn test_autotuner_needs_tuning() {
        let tuner = KernelAutoTuner::with_default_config();
        let kernel_id = KernelId::new("test_kernel", 1024);

        // Should need tuning initially
        assert!(tuner.needs_tuning(&kernel_id));

        // Record some executions
        let config = LaunchConfig::new(256, 4);
        for _ in 0..5 {
            tuner.record_execution(kernel_id.clone(), config, Duration::from_micros(100));
        }

        // After tuning, shouldn't need it for a while
        // (until retune_interval is reached)
    }
}
