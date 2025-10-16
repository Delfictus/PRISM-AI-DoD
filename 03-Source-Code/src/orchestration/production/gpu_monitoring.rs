//! GPU Monitoring and Profiling Infrastructure
//!
//! Provides real-time GPU utilization tracking, performance profiling,
//! and resource monitoring for production deployments.
//!
//! Features:
//! - GPU utilization tracking (per-kernel)
//! - Memory usage monitoring
//! - Kernel execution timing
//! - Performance metrics aggregation
//! - Alert generation for anomalies

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

/// GPU performance metrics for a single kernel execution
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub kernel_name: String,
    pub execution_time_us: u64,
    pub memory_allocated_bytes: usize,
    pub timestamp: Instant,
    pub success: bool,
}

/// Aggregated GPU utilization statistics
#[derive(Debug, Clone)]
pub struct GpuUtilizationStats {
    /// Total number of kernel executions
    pub total_kernel_calls: u64,

    /// Number of successful kernel executions
    pub successful_calls: u64,

    /// Number of failed kernel executions
    pub failed_calls: u64,

    /// Total GPU time spent (microseconds)
    pub total_gpu_time_us: u64,

    /// Average execution time per kernel (microseconds)
    pub avg_execution_time_us: u64,

    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,

    /// Current memory usage (bytes)
    pub current_memory_bytes: usize,

    /// Utilization percentage (0-100)
    pub utilization_percentage: f32,

    /// Per-kernel statistics
    pub per_kernel_stats: HashMap<String, KernelStats>,
}

/// Statistics for a specific kernel
#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelStats {
    pub call_count: u64,
    pub total_time_us: u64,
    pub avg_time_us: u64,
    pub min_time_us: u64,
    pub max_time_us: u64,
    pub success_rate: f32,
}

/// GPU monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable detailed per-kernel profiling
    pub enable_profiling: bool,

    /// Maximum number of metrics to store in memory
    pub max_metrics_buffer: usize,

    /// Alert threshold for GPU utilization (percentage)
    pub high_utilization_threshold: f32,

    /// Alert threshold for memory usage (percentage)
    pub high_memory_threshold: f32,

    /// Enable automatic alerts
    pub enable_alerts: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            max_metrics_buffer: 10000,
            high_utilization_threshold: 90.0,
            high_memory_threshold: 85.0,
            enable_alerts: true,
        }
    }
}

/// GPU monitoring system
pub struct GpuMonitor {
    config: MonitoringConfig,
    metrics_buffer: Arc<Mutex<Vec<KernelMetrics>>>,
    start_time: Instant,

    #[cfg(feature = "cuda")]
    device: Option<Arc<CudaContext>>,
}

impl GpuMonitor {
    /// Create a new GPU monitor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(MonitoringConfig::default())
    }

    /// Create a new GPU monitor with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let device = CudaContext::new(0).ok();

        Ok(Self {
            config,
            metrics_buffer: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),

            #[cfg(feature = "cuda")]
            device,
        })
    }

    /// Record a kernel execution
    pub fn record_kernel_execution(
        &self,
        kernel_name: String,
        execution_time: Duration,
        memory_allocated: usize,
        success: bool,
    ) -> Result<()> {
        if !self.config.enable_profiling {
            return Ok(());
        }

        let metrics = KernelMetrics {
            kernel_name: kernel_name.clone(),
            execution_time_us: execution_time.as_micros() as u64,
            memory_allocated_bytes: memory_allocated,
            timestamp: Instant::now(),
            success,
        };

        let mut buffer = self.metrics_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock metrics buffer in record: {}", e))?;

        // Maintain buffer size limit
        if buffer.len() >= self.config.max_metrics_buffer {
            buffer.remove(0); // Remove oldest metric
        }

        buffer.push(metrics);

        // Check for alerts
        if self.config.enable_alerts {
            self.check_alerts(&buffer)?;
        }

        Ok(())
    }

    /// Get current GPU utilization statistics
    pub fn get_utilization_stats(&self) -> Result<GpuUtilizationStats> {
        let buffer = self.metrics_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock metrics buffer: {}", e))?;

        if buffer.is_empty() {
            return Ok(GpuUtilizationStats {
                total_kernel_calls: 0,
                successful_calls: 0,
                failed_calls: 0,
                total_gpu_time_us: 0,
                avg_execution_time_us: 0,
                peak_memory_bytes: 0,
                current_memory_bytes: 0,
                utilization_percentage: 0.0,
                per_kernel_stats: HashMap::new(),
            });
        }

        // Aggregate statistics
        let total_kernel_calls = buffer.len() as u64;
        let successful_calls = buffer.iter().filter(|m| m.success).count() as u64;
        let failed_calls = total_kernel_calls - successful_calls;

        let total_gpu_time_us: u64 = buffer.iter()
            .map(|m| m.execution_time_us)
            .sum();

        let avg_execution_time_us = if total_kernel_calls > 0 {
            total_gpu_time_us / total_kernel_calls
        } else {
            0
        };

        let peak_memory_bytes = buffer.iter()
            .map(|m| m.memory_allocated_bytes)
            .max()
            .unwrap_or(0);

        let current_memory_bytes = buffer.last()
            .map(|m| m.memory_allocated_bytes)
            .unwrap_or(0);

        // Calculate utilization percentage
        let elapsed_time_us = self.start_time.elapsed().as_micros() as u64;
        let utilization_percentage = if elapsed_time_us > 0 {
            ((total_gpu_time_us as f64 / elapsed_time_us as f64) * 100.0) as f32
        } else {
            0.0
        };

        // Per-kernel statistics
        let mut per_kernel_stats = HashMap::new();
        let mut kernel_data: HashMap<String, Vec<&KernelMetrics>> = HashMap::new();

        for metric in buffer.iter() {
            kernel_data.entry(metric.kernel_name.clone())
                .or_insert_with(Vec::new)
                .push(metric);
        }

        for (kernel_name, metrics) in kernel_data {
            let call_count = metrics.len() as u64;
            let successful = metrics.iter().filter(|m| m.success).count() as u64;
            let total_time_us: u64 = metrics.iter().map(|m| m.execution_time_us).sum();
            let avg_time_us = total_time_us / call_count;
            let min_time_us = metrics.iter().map(|m| m.execution_time_us).min().unwrap_or(0);
            let max_time_us = metrics.iter().map(|m| m.execution_time_us).max().unwrap_or(0);
            let success_rate = (successful as f32 / call_count as f32) * 100.0;

            per_kernel_stats.insert(kernel_name, KernelStats {
                call_count,
                total_time_us,
                avg_time_us,
                min_time_us,
                max_time_us,
                success_rate,
            });
        }

        Ok(GpuUtilizationStats {
            total_kernel_calls,
            successful_calls,
            failed_calls,
            total_gpu_time_us,
            avg_execution_time_us,
            peak_memory_bytes,
            current_memory_bytes,
            utilization_percentage,
            per_kernel_stats,
        })
    }

    /// Get detailed report as formatted string
    pub fn get_report(&self) -> Result<String> {
        let stats = self.get_utilization_stats()?;

        let mut report = String::new();
        report.push_str("╔════════════════════════════════════════════════════════════╗\n");
        report.push_str("║  GPU Utilization Report - Worker 2 Infrastructure          ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════╝\n");
        report.push_str("\n");

        report.push_str(&format!("📊 Overall Statistics:\n"));
        report.push_str(&format!("  • Total Kernel Calls: {}\n", stats.total_kernel_calls));
        report.push_str(&format!("  • Successful Calls: {} ({:.1}%)\n",
            stats.successful_calls,
            (stats.successful_calls as f32 / stats.total_kernel_calls as f32) * 100.0
        ));
        report.push_str(&format!("  • Failed Calls: {}\n", stats.failed_calls));
        report.push_str(&format!("  • GPU Utilization: {:.1}%\n", stats.utilization_percentage));
        report.push_str(&format!("  • Average Execution Time: {:.2} ms\n",
            stats.avg_execution_time_us as f64 / 1000.0
        ));
        report.push_str(&format!("  • Peak Memory Usage: {:.2} MB\n",
            stats.peak_memory_bytes as f64 / (1024.0 * 1024.0)
        ));
        report.push_str(&format!("  • Current Memory Usage: {:.2} MB\n",
            stats.current_memory_bytes as f64 / (1024.0 * 1024.0)
        ));
        report.push_str("\n");

        if !stats.per_kernel_stats.is_empty() {
            report.push_str("🔍 Per-Kernel Statistics:\n");

            let mut kernel_names: Vec<_> = stats.per_kernel_stats.keys().collect();
            kernel_names.sort();

            for kernel_name in kernel_names {
                let stats = &stats.per_kernel_stats[kernel_name];
                report.push_str(&format!("\n  {} (calls: {})\n", kernel_name, stats.call_count));
                report.push_str(&format!("    Avg: {:.2} ms  |  Min: {:.2} ms  |  Max: {:.2} ms\n",
                    stats.avg_time_us as f64 / 1000.0,
                    stats.min_time_us as f64 / 1000.0,
                    stats.max_time_us as f64 / 1000.0
                ));
                report.push_str(&format!("    Success Rate: {:.1}%\n", stats.success_rate));
            }
        }

        Ok(report)
    }

    /// Check for alert conditions
    fn check_alerts(&self, buffer: &[KernelMetrics]) -> Result<()> {
        let stats = self.get_utilization_stats()?;

        // High utilization alert
        if stats.utilization_percentage > self.config.high_utilization_threshold {
            log::warn!(
                "⚠️  HIGH GPU UTILIZATION: {:.1}% (threshold: {:.1}%)",
                stats.utilization_percentage,
                self.config.high_utilization_threshold
            );
        }

        // High memory usage alert
        let memory_usage_percent = (stats.current_memory_bytes as f64 / stats.peak_memory_bytes as f64) * 100.0;
        if memory_usage_percent > self.config.high_memory_threshold as f64 {
            log::warn!(
                "⚠️  HIGH MEMORY USAGE: {:.1}% (threshold: {:.1}%)",
                memory_usage_percent,
                self.config.high_memory_threshold
            );
        }

        // Kernel failure alert
        let recent_failures = buffer.iter()
            .rev()
            .take(100)
            .filter(|m| !m.success)
            .count();

        if recent_failures > 10 {
            log::error!(
                "⚠️  HIGH FAILURE RATE: {} failures in last 100 calls",
                recent_failures
            );
        }

        Ok(())
    }

    /// Reset all metrics
    pub fn reset(&self) -> Result<()> {
        let mut buffer = self.metrics_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock metrics buffer in reset: {}", e))?;
        buffer.clear();
        Ok(())
    }

    /// Export metrics to JSON
    pub fn export_json(&self) -> Result<String> {
        let stats = self.get_utilization_stats()?;
        serde_json::to_string_pretty(&serde_json::json!({
            "total_kernel_calls": stats.total_kernel_calls,
            "successful_calls": stats.successful_calls,
            "failed_calls": stats.failed_calls,
            "utilization_percentage": stats.utilization_percentage,
            "avg_execution_time_us": stats.avg_execution_time_us,
            "peak_memory_mb": stats.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            "current_memory_mb": stats.current_memory_bytes as f64 / (1024.0 * 1024.0),
            "per_kernel_stats": stats.per_kernel_stats,
        })).context("Failed to serialize metrics to JSON")
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new().expect("Failed to create default GPU monitor")
    }
}

/// Global GPU monitor instance
static GLOBAL_MONITOR: once_cell::sync::Lazy<Arc<Mutex<GpuMonitor>>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(Mutex::new(GpuMonitor::new().expect("Failed to initialize GPU monitor")))
    });

/// Get the global GPU monitor
pub fn get_global_monitor() -> Arc<Mutex<GpuMonitor>> {
    Arc::clone(&GLOBAL_MONITOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = GpuMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_record_kernel_execution() {
        let monitor = GpuMonitor::new().unwrap();

        let result = monitor.record_kernel_execution(
            "test_kernel".to_string(),
            Duration::from_micros(1000),
            1024 * 1024,
            true,
        );

        assert!(result.is_ok());

        let stats = monitor.get_utilization_stats().unwrap();
        assert_eq!(stats.total_kernel_calls, 1);
        assert_eq!(stats.successful_calls, 1);
    }

    #[test]
    fn test_per_kernel_stats() {
        let monitor = GpuMonitor::new().unwrap();

        // Record multiple executions
        for i in 0..10 {
            monitor.record_kernel_execution(
                "test_kernel".to_string(),
                Duration::from_micros(1000 + i * 100),
                1024 * 1024,
                true,
            ).unwrap();
        }

        let stats = monitor.get_utilization_stats().unwrap();
        assert_eq!(stats.total_kernel_calls, 10);

        let kernel_stats = stats.per_kernel_stats.get("test_kernel").unwrap();
        assert_eq!(kernel_stats.call_count, 10);
        assert_eq!(kernel_stats.success_rate, 100.0);
    }
}
