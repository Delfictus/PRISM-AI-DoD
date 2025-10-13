//! Production features

pub mod error_handling;
pub mod logging;
pub mod config;
pub mod gpu_monitoring;

pub use error_handling::{ProductionErrorHandler, ProductionMonitoring, RecoveryAction, CircuitBreakerStatus};
pub use logging::{ProductionLogger, LogConfig, LogLevel, LogFormat, LLMOperationLogger, IntegrationLogger};
pub use config::{MissionCharlieConfig, ConfigBuilder, LLMConfiguration, CacheConfiguration,
                 ConsensusConfiguration, ErrorConfiguration, LoggingConfiguration, PerformanceConfiguration};
pub use gpu_monitoring::{GpuMonitor, GpuUtilizationStats, KernelMetrics, KernelStats, MonitoringConfig, get_global_monitor};
