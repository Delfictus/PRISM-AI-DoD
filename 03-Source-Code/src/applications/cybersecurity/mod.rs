//! Cybersecurity Threat Detection Module
//!
//! GPU-accelerated cybersecurity threat detection and response:
//! - Network intrusion detection (IDS/IPS)
//! - Anomaly detection and behavioral analysis
//! - Threat classification and risk scoring
//! - Incident response prioritization
//! - Multi-stage attack pattern recognition
//! - **NEW**: Threat trajectory forecasting with time series analysis
//! - Active Inference for adaptive security
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated pattern matching
//! - Article III: Comprehensive testing required
//! - Article IV: Active Inference for adaptive defense
//! - Defensive security only - no offensive capabilities

pub mod detector;
pub mod threat_forecaster;

// Re-export main types
pub use detector::{
    ThreatDetector,
    NetworkEvent,
    EventType,
    ThreatAssessment,
    ThreatLevel,
    AttackType,
    SecurityConfig,
    DetectionStrategy,
    IncidentResponse,
};

pub use threat_forecaster::{
    ThreatForecaster,
    ForecastConfig,
    SecurityMetricPoint,
    ThreatTrajectory,
    ThreatWarning,
    MitigationImpact,
    ImpactCategory,
};
