//! API Performance Profiling and Optimization
//!
//! Provides performance monitoring, profiling, and optimization recommendations
//! for all API endpoints with focus on Workers 1, 3, 7 integrations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance profile for an API endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointProfile {
    pub endpoint: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub p50_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub min_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub bottlenecks: Vec<Bottleneck>,
    pub optimization_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub avg_time_ms: f64,
    pub percentage_of_total: f64,
    pub severity: BottleneckSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance optimizer
pub struct PerformanceOptimizer {
    profiles: HashMap<String, Vec<Duration>>,
    start_time: Instant,
}

impl PerformanceOptimizer {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    /// Record a request duration
    pub fn record(&mut self, endpoint: &str, duration: Duration) {
        self.profiles
            .entry(endpoint.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Get profile for an endpoint
    pub fn get_profile(&self, endpoint: &str) -> Option<EndpointProfile> {
        let durations = self.profiles.get(endpoint)?;
        if durations.is_empty() {
            return None;
        }

        let mut sorted_durations: Vec<f64> = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total = durations.len() as u64;
        let successful = total; // Simplified
        let failed = 0;

        let sum: f64 = sorted_durations.iter().sum();
        let avg = sum / sorted_durations.len() as f64;

        let p50 = percentile(&sorted_durations, 50.0);
        let p95 = percentile(&sorted_durations, 95.0);
        let p99 = percentile(&sorted_durations, 99.0);
        let min = sorted_durations.first().copied().unwrap_or(0.0);
        let max = sorted_durations.last().copied().unwrap_or(0.0);

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rps = if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        };

        let error_rate = if total > 0 {
            (failed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let bottlenecks = identify_bottlenecks(endpoint, avg);
        let recommendations = generate_recommendations(endpoint, avg, p95, rps);

        Some(EndpointProfile {
            endpoint: endpoint.to_string(),
            total_requests: total,
            successful_requests: successful,
            failed_requests: failed,
            avg_response_time_ms: avg,
            p50_response_time_ms: p50,
            p95_response_time_ms: p95,
            p99_response_time_ms: p99,
            min_response_time_ms: min,
            max_response_time_ms: max,
            requests_per_second: rps,
            error_rate,
            bottlenecks,
            optimization_recommendations: recommendations,
        })
    }

    /// Get all profiles
    pub fn get_all_profiles(&self) -> Vec<EndpointProfile> {
        self.profiles
            .keys()
            .filter_map(|endpoint| self.get_profile(endpoint))
            .collect()
    }
}

impl Default for PerformanceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let index = (p / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

/// Identify bottlenecks for an endpoint
fn identify_bottlenecks(endpoint: &str, avg_time_ms: f64) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    // Worker-specific bottleneck analysis
    if endpoint.contains("timeseries") {
        // Worker 1: Time Series Forecasting
        if avg_time_ms > 100.0 {
            bottlenecks.push(Bottleneck {
                component: "ARIMA/LSTM Model Training".to_string(),
                avg_time_ms: avg_time_ms * 0.6,
                percentage_of_total: 60.0,
                severity: if avg_time_ms > 500.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
            });
        }

        if avg_time_ms > 50.0 {
            bottlenecks.push(Bottleneck {
                component: "Data Preprocessing".to_string(),
                avg_time_ms: avg_time_ms * 0.2,
                percentage_of_total: 20.0,
                severity: BottleneckSeverity::Low,
            });
        }
    } else if endpoint.contains("finance") {
        // Worker 3: Portfolio Optimization
        if avg_time_ms > 150.0 {
            bottlenecks.push(Bottleneck {
                component: "Covariance Matrix Computation".to_string(),
                avg_time_ms: avg_time_ms * 0.5,
                percentage_of_total: 50.0,
                severity: if avg_time_ms > 1000.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::High
                },
            });
        }

        if avg_time_ms > 100.0 {
            bottlenecks.push(Bottleneck {
                component: "Quadratic Programming Solver".to_string(),
                avg_time_ms: avg_time_ms * 0.3,
                percentage_of_total: 30.0,
                severity: BottleneckSeverity::Medium,
            });
        }
    } else if endpoint.contains("robotics") {
        // Worker 7: Robotics Motion Planning
        if avg_time_ms > 200.0 {
            bottlenecks.push(Bottleneck {
                component: "Active Inference Policy Search".to_string(),
                avg_time_ms: avg_time_ms * 0.7,
                percentage_of_total: 70.0,
                severity: if avg_time_ms > 1000.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
            });
        }

        if avg_time_ms > 50.0 {
            bottlenecks.push(Bottleneck {
                component: "Obstacle Avoidance Computation".to_string(),
                avg_time_ms: avg_time_ms * 0.2,
                percentage_of_total: 20.0,
                severity: BottleneckSeverity::Low,
            });
        }
    }

    // General bottlenecks
    if avg_time_ms > 20.0 {
        bottlenecks.push(Bottleneck {
            component: "Request Serialization/Deserialization".to_string(),
            avg_time_ms: avg_time_ms * 0.1,
            percentage_of_total: 10.0,
            severity: BottleneckSeverity::Low,
        });
    }

    bottlenecks
}

/// Generate optimization recommendations
fn generate_recommendations(
    endpoint: &str,
    avg_ms: f64,
    p95_ms: f64,
    rps: f64,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Worker-specific recommendations
    if endpoint.contains("timeseries") {
        if avg_ms > 100.0 {
            recommendations.push(
                "Consider caching pre-trained LSTM/ARIMA models for repeated forecasts".to_string(),
            );
        }
        if p95_ms > 500.0 {
            recommendations.push(
                "Enable GPU acceleration for LSTM forward passes (Worker 2 integration)".to_string(),
            );
        }
        if rps < 10.0 {
            recommendations.push(
                "Implement batch forecasting API to process multiple time series in parallel".to_string(),
            );
        }
    } else if endpoint.contains("finance") {
        if avg_ms > 200.0 {
            recommendations.push(
                "Use Worker 2 GPU kernels for covariance matrix computation".to_string(),
            );
        }
        if p95_ms > 1000.0 {
            recommendations.push(
                "Enable Tensor Core optimization for large portfolio optimizations (100+ assets)".to_string(),
            );
        }
        if avg_ms > 150.0 {
            recommendations.push(
                "Cache historical price data and reuse covariance matrices for similar requests".to_string(),
            );
        }
    } else if endpoint.contains("robotics") {
        if avg_ms > 300.0 {
            recommendations.push(
                "Use GPU-accelerated policy search from Worker 7's Active Inference planner".to_string(),
            );
        }
        if p95_ms > 1000.0 {
            recommendations.push(
                "Reduce planning horizon or increase time step dt for faster planning".to_string(),
            );
        }
        if avg_ms > 100.0 {
            recommendations.push(
                "Cache motion primitives for common start/goal configurations".to_string(),
            );
        }
    }

    // General recommendations
    if avg_ms > 50.0 {
        recommendations.push(
            "Consider adding response compression (gzip) for large JSON payloads".to_string(),
        );
    }

    if rps > 100.0 {
        recommendations.push(
            "High load detected - consider horizontal scaling or request rate limiting".to_string(),
        );
    }

    if p95_ms > avg_ms * 3.0 {
        recommendations.push(
            "High variance in response times - investigate outliers and cold start performance".to_string(),
        );
    }

    // Add GPU recommendation if not GPU-accelerated
    if !endpoint.contains("gpu") && avg_ms > 100.0 {
        recommendations.push(
            "Verify GPU acceleration is enabled - check /api/v1/gpu/status endpoint".to_string(),
        );
    }

    recommendations
}

/// Performance summary across all endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_requests: u64,
    pub total_endpoints: usize,
    pub avg_response_time_ms: f64,
    pub overall_rps: f64,
    pub slowest_endpoints: Vec<SlowEndpoint>,
    pub fastest_endpoints: Vec<FastEndpoint>,
    pub critical_bottlenecks: Vec<String>,
    pub top_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowEndpoint {
    pub endpoint: String,
    pub avg_time_ms: f64,
    pub p95_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastEndpoint {
    pub endpoint: String,
    pub avg_time_ms: f64,
}

impl PerformanceOptimizer {
    /// Get performance summary
    pub fn get_summary(&self) -> PerformanceSummary {
        let profiles = self.get_all_profiles();

        let total_requests: u64 = profiles.iter().map(|p| p.total_requests).sum();
        let total_endpoints = profiles.len();

        let avg_response_time_ms = if !profiles.is_empty() {
            profiles.iter().map(|p| p.avg_response_time_ms).sum::<f64>() / profiles.len() as f64
        } else {
            0.0
        };

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let overall_rps = if elapsed > 0.0 {
            total_requests as f64 / elapsed
        } else {
            0.0
        };

        // Find slowest endpoints
        let mut sorted_by_time = profiles.clone();
        sorted_by_time.sort_by(|a, b| {
            b.avg_response_time_ms
                .partial_cmp(&a.avg_response_time_ms)
                .unwrap()
        });

        let slowest_endpoints: Vec<SlowEndpoint> = sorted_by_time
            .iter()
            .take(5)
            .map(|p| SlowEndpoint {
                endpoint: p.endpoint.clone(),
                avg_time_ms: p.avg_response_time_ms,
                p95_time_ms: p.p95_response_time_ms,
            })
            .collect();

        // Find fastest endpoints
        sorted_by_time.reverse();
        let fastest_endpoints: Vec<FastEndpoint> = sorted_by_time
            .iter()
            .take(5)
            .map(|p| FastEndpoint {
                endpoint: p.endpoint.clone(),
                avg_time_ms: p.avg_response_time_ms,
            })
            .collect();

        // Critical bottlenecks
        let critical_bottlenecks: Vec<String> = profiles
            .iter()
            .flat_map(|p| &p.bottlenecks)
            .filter(|b| matches!(b.severity, BottleneckSeverity::Critical | BottleneckSeverity::High))
            .map(|b| format!("{} ({}ms, {}%)", b.component, b.avg_time_ms, b.percentage_of_total))
            .collect();

        // Top recommendations
        let mut all_recommendations: Vec<String> = profiles
            .iter()
            .flat_map(|p| p.optimization_recommendations.clone())
            .collect();
        all_recommendations.dedup();
        let top_recommendations = all_recommendations.into_iter().take(10).collect();

        PerformanceSummary {
            total_requests,
            total_endpoints,
            avg_response_time_ms,
            overall_rps,
            slowest_endpoints,
            fastest_endpoints,
            critical_bottlenecks,
            top_recommendations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&values, 50.0), 5.0);
        assert_eq!(percentile(&values, 95.0), 10.0);
    }

    #[test]
    fn test_performance_optimizer() {
        let mut optimizer = PerformanceOptimizer::new();
        optimizer.record("/api/v1/timeseries/forecast", Duration::from_millis(100));
        optimizer.record("/api/v1/timeseries/forecast", Duration::from_millis(150));

        let profile = optimizer.get_profile("/api/v1/timeseries/forecast").unwrap();
        assert_eq!(profile.total_requests, 2);
        assert!(profile.avg_response_time_ms > 100.0);
        assert!(profile.avg_response_time_ms < 150.0);
    }

    #[test]
    fn test_bottleneck_identification() {
        let bottlenecks = identify_bottlenecks("/api/v1/finance/optimize", 200.0);
        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks.iter().any(|b| b.component.contains("Covariance")));
    }

    #[test]
    fn test_recommendations() {
        let recs = generate_recommendations("/api/v1/timeseries/forecast", 150.0, 500.0, 5.0);
        assert!(!recs.is_empty());
        assert!(recs.iter().any(|r| r.contains("GPU") || r.contains("cache")));
    }
}
