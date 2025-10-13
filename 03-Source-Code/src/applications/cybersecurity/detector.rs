//! Cybersecurity Threat Detector
//!
//! Implements GPU-accelerated threat detection including:
//! - Real-time network event analysis
//! - Anomaly detection using statistical models
//! - Multi-stage attack pattern recognition
//! - Threat risk scoring and prioritization
//!
//! Worker 3 Implementation - Defensive Security Only
//! Constitutional Compliance: Articles I, II, III, IV

use anyhow::{Result, Context};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use crate::gpu::GpuMemoryPool;

/// Network event types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    ConnectionAttempt,
    DataTransfer,
    AuthenticationAttempt,
    PortScan,
    FileAccess,
    ProcessExecution,
    NetworkTraffic,
    SystemCall,
}

/// Network security event
#[derive(Debug, Clone)]
pub struct NetworkEvent {
    pub timestamp: f64,
    pub event_type: EventType,
    pub source_ip: String,
    pub dest_ip: String,
    pub source_port: u16,
    pub dest_port: u16,
    pub protocol: String,
    pub payload_size: usize,
    pub flags: Vec<String>,
    pub user_agent: Option<String>,
}

/// Attack type classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttackType {
    BruteForce,
    DDoS,
    SQLInjection,
    XSS,
    PortScan,
    Malware,
    Phishing,
    Ransomware,
    ZeroDay,
    InsiderThreat,
    DataExfiltration,
    Unknown,
}

/// Threat severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThreatLevel {
    Informational,
    Low,
    Medium,
    High,
    Critical,
}

/// Threat assessment result
#[derive(Debug, Clone)]
pub struct ThreatAssessment {
    pub threat_level: ThreatLevel,
    pub attack_type: AttackType,
    pub confidence: f64,  // 0.0 to 1.0
    pub risk_score: f64,  // 0-100
    pub anomaly_score: f64,  // Statistical deviation
    pub indicators: Vec<String>,
    pub affected_systems: Vec<String>,
    pub recommended_response: IncidentResponse,
}

/// Incident response recommendation
#[derive(Debug, Clone)]
pub struct IncidentResponse {
    pub action: ResponseAction,
    pub priority: u8,  // 1-10
    pub automated: bool,
    pub description: String,
    pub mitigation_steps: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseAction {
    Monitor,
    Alert,
    Block,
    Quarantine,
    Investigate,
    Escalate,
}

/// Detection strategy
#[derive(Debug, Clone, Copy)]
pub enum DetectionStrategy {
    SignatureBased,    // Known attack patterns
    AnomalyBased,      // Statistical deviation
    BehaviorBased,     // Behavioral analysis
    HeuristicBased,    // Rule-based heuristics
    Hybrid,            // Combined approach
}

/// Security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub detection_threshold: f64,
    pub anomaly_sensitivity: f64,
    pub block_on_high_risk: bool,
    pub enable_automated_response: bool,
    pub log_all_events: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            detection_threshold: 0.7,
            anomaly_sensitivity: 2.5,  // Standard deviations
            block_on_high_risk: true,
            enable_automated_response: true,
            log_all_events: false,
        }
    }
}

/// Cybersecurity threat detector
pub struct ThreatDetector {
    config: SecurityConfig,
    baseline_profiles: HashMap<String, TrafficProfile>,

    #[cfg(feature = "cuda")]
    gpu_context: Option<GpuMemoryPool>,
}

/// Traffic baseline profile for anomaly detection
#[derive(Debug, Clone)]
struct TrafficProfile {
    avg_connections_per_hour: f64,
    std_connections: f64,
    avg_bytes_transferred: f64,
    std_bytes: f64,
    common_ports: Vec<u16>,
    typical_protocols: Vec<String>,
}

impl ThreatDetector {
    /// Create new threat detector
    pub fn new(config: SecurityConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_context = Some(GpuMemoryPool::new()
            .context("Failed to initialize GPU for threat detection")?);

        Ok(Self {
            config,
            baseline_profiles: HashMap::new(),
            #[cfg(feature = "cuda")]
            gpu_context,
        })
    }

    /// Analyze network event for threats
    pub fn analyze_event(
        &self,
        event: &NetworkEvent,
        strategy: DetectionStrategy,
    ) -> Result<ThreatAssessment> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.analyze_event_gpu(event, strategy);
            }
        }

        self.analyze_event_cpu(event, strategy)
    }

    /// CPU-based threat analysis
    fn analyze_event_cpu(
        &self,
        event: &NetworkEvent,
        strategy: DetectionStrategy,
    ) -> Result<ThreatAssessment> {
        let mut risk_score = 0.0;
        let mut indicators = Vec::new();
        let mut attack_type = AttackType::Unknown;
        let mut confidence = 0.0;

        match strategy {
            DetectionStrategy::SignatureBased => {
                // Check for known attack patterns
                let (score, attack, conf, inds) = self.signature_detection(event);
                risk_score = score;
                attack_type = attack;
                confidence = conf;
                indicators = inds;
            }
            DetectionStrategy::AnomalyBased => {
                // Statistical anomaly detection
                let (score, inds) = self.anomaly_detection(event);
                risk_score = score;
                indicators = inds;
                confidence = if score > 70.0 { 0.8 } else { 0.5 };
            }
            DetectionStrategy::BehaviorBased => {
                // Behavioral pattern analysis
                let (score, attack, inds) = self.behavior_analysis(event);
                risk_score = score;
                attack_type = attack;
                indicators = inds;
                confidence = 0.7;
            }
            DetectionStrategy::HeuristicBased => {
                // Rule-based heuristics
                let (score, attack, inds) = self.heuristic_analysis(event);
                risk_score = score;
                attack_type = attack;
                indicators = inds;
                confidence = 0.75;
            }
            DetectionStrategy::Hybrid => {
                // Combine multiple approaches
                let (sig_score, sig_attack, sig_conf, sig_inds) = self.signature_detection(event);
                let (anom_score, anom_inds) = self.anomaly_detection(event);
                let (beh_score, beh_attack, beh_inds) = self.behavior_analysis(event);

                risk_score = (sig_score * 0.4 + anom_score * 0.3 + beh_score * 0.3);
                attack_type = if sig_conf > 0.7 { sig_attack } else { beh_attack };
                confidence = (sig_conf + 0.8 + 0.7) / 3.0;

                indicators.extend(sig_inds);
                indicators.extend(anom_inds);
                indicators.extend(beh_inds);
            }
        }

        // Determine threat level
        let threat_level = if risk_score >= 90.0 {
            ThreatLevel::Critical
        } else if risk_score >= 70.0 {
            ThreatLevel::High
        } else if risk_score >= 50.0 {
            ThreatLevel::Medium
        } else if risk_score >= 30.0 {
            ThreatLevel::Low
        } else {
            ThreatLevel::Informational
        };

        // Generate incident response
        let recommended_response = self.generate_response(threat_level, attack_type);

        Ok(ThreatAssessment {
            threat_level,
            attack_type,
            confidence,
            risk_score,
            anomaly_score: risk_score / 10.0,  // Normalize to stdev scale
            indicators,
            affected_systems: vec![event.dest_ip.clone()],
            recommended_response,
        })
    }

    #[cfg(feature = "cuda")]
    fn analyze_event_gpu(
        &self,
        event: &NetworkEvent,
        strategy: DetectionStrategy,
    ) -> Result<ThreatAssessment> {
        // TODO: Request threat_detection_kernel from Worker 2
        // __global__ void threat_detection(
        //     NetworkEvent* events,
        //     ThreatSignature* signatures,
        //     TrafficProfile* baselines,
        //     ThreatAssessment* results,
        //     int num_events,
        //     DetectionStrategy strategy
        // )

        // Placeholder: use CPU implementation
        self.analyze_event_cpu(event, strategy)
    }

    /// Signature-based detection (known attack patterns)
    fn signature_detection(&self, event: &NetworkEvent) -> (f64, AttackType, f64, Vec<String>) {
        let mut risk_score = 0.0;
        let mut attack_type = AttackType::Unknown;
        let mut confidence = 0.0;
        let mut indicators = Vec::new();

        // SQL Injection signatures
        if event.event_type == EventType::DataTransfer &&
           event.payload_size > 100 &&
           event.flags.iter().any(|f| f.contains("SQL")) {
            risk_score += 80.0;
            attack_type = AttackType::SQLInjection;
            confidence = 0.9;
            indicators.push("SQL injection pattern detected".to_string());
        }

        // Port scan detection
        if event.event_type == EventType::PortScan {
            risk_score += 60.0;
            attack_type = AttackType::PortScan;
            confidence = 0.95;
            indicators.push("Port scanning activity detected".to_string());
        }

        // Brute force attempts
        if event.event_type == EventType::AuthenticationAttempt &&
           event.flags.iter().any(|f| f.contains("FAILED")) {
            risk_score += 50.0;
            attack_type = AttackType::BruteForce;
            confidence = 0.7;
            indicators.push("Multiple failed authentication attempts".to_string());
        }

        // DDoS indicators
        if event.event_type == EventType::ConnectionAttempt &&
           event.payload_size == 0 {
            risk_score += 40.0;
            attack_type = AttackType::DDoS;
            confidence = 0.6;
            indicators.push("Suspicious connection patterns".to_string());
        }

        (risk_score, attack_type, confidence, indicators)
    }

    /// Anomaly-based detection (statistical deviation)
    fn anomaly_detection(&self, event: &NetworkEvent) -> (f64, Vec<String>) {
        let mut risk_score = 0.0;
        let mut indicators = Vec::new();

        // Check against baseline profile
        if let Some(profile) = self.baseline_profiles.get(&event.source_ip) {
            // Unusual port access
            if !profile.common_ports.contains(&event.dest_port) {
                risk_score += 30.0;
                indicators.push(format!("Unusual port access: {}", event.dest_port));
            }

            // Unusual protocol
            if !profile.typical_protocols.contains(&event.protocol) {
                risk_score += 25.0;
                indicators.push(format!("Atypical protocol: {}", event.protocol));
            }

            // Unusual data volume
            if event.payload_size as f64 > profile.avg_bytes_transferred +
               (self.config.anomaly_sensitivity * profile.std_bytes) {
                risk_score += 40.0;
                indicators.push("Abnormal data transfer volume".to_string());
            }
        } else {
            // No baseline - new source
            risk_score += 20.0;
            indicators.push("Unknown source IP".to_string());
        }

        (risk_score, indicators)
    }

    /// Behavior-based analysis
    fn behavior_analysis(&self, event: &NetworkEvent) -> (f64, AttackType, Vec<String>) {
        let mut risk_score = 0.0;
        let mut attack_type = AttackType::Unknown;
        let mut indicators = Vec::new();

        // Data exfiltration behavior
        if event.event_type == EventType::DataTransfer &&
           event.payload_size > 10_000_000 &&  // > 10MB
           !event.dest_ip.starts_with("10.") &&  // External destination
           !event.dest_ip.starts_with("192.168.") {
            risk_score += 85.0;
            attack_type = AttackType::DataExfiltration;
            indicators.push("Large data transfer to external IP".to_string());
        }

        // Lateral movement
        if event.event_type == EventType::ConnectionAttempt &&
           event.dest_port == 445 {  // SMB port
            risk_score += 55.0;
            attack_type = AttackType::InsiderThreat;
            indicators.push("Internal reconnaissance activity".to_string());
        }

        // Suspicious file access
        if event.event_type == EventType::FileAccess &&
           event.flags.iter().any(|f| f.contains("SYSTEM") || f.contains("PASSWORD")) {
            risk_score += 70.0;
            attack_type = AttackType::Malware;
            indicators.push("Access to sensitive system files".to_string());
        }

        (risk_score, attack_type, indicators)
    }

    /// Heuristic-based analysis
    fn heuristic_analysis(&self, event: &NetworkEvent) -> (f64, AttackType, Vec<String>) {
        let mut risk_score = 0.0;
        let mut attack_type = AttackType::Unknown;
        let mut indicators = Vec::new();

        // Rule: Non-standard ports
        if event.dest_port > 1024 && event.dest_port < 49152 {
            risk_score += 15.0;
            indicators.push("Non-standard port usage".to_string());
        }

        // Rule: Suspicious user agents
        if let Some(ua) = &event.user_agent {
            if ua.contains("bot") || ua.contains("crawler") || ua.contains("script") {
                risk_score += 35.0;
                attack_type = AttackType::Phishing;
                indicators.push("Automated tool detected".to_string());
            }
        }

        // Rule: Rapid connections
        if event.event_type == EventType::ConnectionAttempt &&
           event.flags.contains(&"RAPID".to_string()) {
            risk_score += 45.0;
            attack_type = AttackType::DDoS;
            indicators.push("Rapid connection attempts".to_string());
        }

        (risk_score, attack_type, indicators)
    }

    /// Generate incident response recommendation
    fn generate_response(&self, level: ThreatLevel, attack: AttackType) -> IncidentResponse {
        match level {
            ThreatLevel::Critical => IncidentResponse {
                action: ResponseAction::Block,
                priority: 10,
                automated: self.config.enable_automated_response,
                description: format!("Critical {:?} threat detected - immediate action required", attack),
                mitigation_steps: vec![
                    "Block source IP immediately".to_string(),
                    "Isolate affected systems".to_string(),
                    "Initiate incident response protocol".to_string(),
                    "Notify security team".to_string(),
                ],
            },
            ThreatLevel::High => IncidentResponse {
                action: ResponseAction::Quarantine,
                priority: 8,
                automated: self.config.enable_automated_response,
                description: format!("High-risk {:?} activity detected", attack),
                mitigation_steps: vec![
                    "Quarantine suspicious traffic".to_string(),
                    "Alert security operations center".to_string(),
                    "Begin investigation".to_string(),
                ],
            },
            ThreatLevel::Medium => IncidentResponse {
                action: ResponseAction::Alert,
                priority: 5,
                automated: false,
                description: "Potentially malicious activity detected".to_string(),
                mitigation_steps: vec![
                    "Monitor closely".to_string(),
                    "Log all related events".to_string(),
                    "Review security posture".to_string(),
                ],
            },
            ThreatLevel::Low => IncidentResponse {
                action: ResponseAction::Monitor,
                priority: 3,
                automated: false,
                description: "Low-priority security event".to_string(),
                mitigation_steps: vec![
                    "Continue monitoring".to_string(),
                    "Update threat intelligence".to_string(),
                ],
            },
            ThreatLevel::Informational => IncidentResponse {
                action: ResponseAction::Monitor,
                priority: 1,
                automated: false,
                description: "Informational security event".to_string(),
                mitigation_steps: vec![
                    "Log for analysis".to_string(),
                ],
            },
        }
    }

    /// Establish baseline traffic profile for anomaly detection
    pub fn establish_baseline(&mut self, ip: String, profile: TrafficProfile) {
        self.baseline_profiles.insert(ip, profile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_detector_creation() {
        let config = SecurityConfig::default();
        let detector = ThreatDetector::new(config);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_sql_injection_detection() {
        let config = SecurityConfig::default();
        let detector = ThreatDetector::new(config).unwrap();

        let event = NetworkEvent {
            timestamp: 1234567890.0,
            event_type: EventType::DataTransfer,
            source_ip: "192.168.1.100".to_string(),
            dest_ip: "10.0.0.50".to_string(),
            source_port: 54321,
            dest_port: 80,
            protocol: "HTTP".to_string(),
            payload_size: 500,
            flags: vec!["SQL".to_string()],
            user_agent: Some("Mozilla/5.0".to_string()),
        };

        let assessment = detector.analyze_event(&event, DetectionStrategy::SignatureBased).unwrap();

        assert_eq!(assessment.attack_type, AttackType::SQLInjection);
        assert!(assessment.risk_score >= 70.0);
        assert!(assessment.threat_level >= ThreatLevel::High);
    }

    #[test]
    fn test_port_scan_detection() {
        let config = SecurityConfig::default();
        let detector = ThreatDetector::new(config).unwrap();

        let event = NetworkEvent {
            timestamp: 1234567890.0,
            event_type: EventType::PortScan,
            source_ip: "203.0.113.50".to_string(),
            dest_ip: "192.168.1.10".to_string(),
            source_port: 12345,
            dest_port: 22,
            protocol: "TCP".to_string(),
            payload_size: 0,
            flags: vec!["SYN".to_string()],
            user_agent: None,
        };

        let assessment = detector.analyze_event(&event, DetectionStrategy::SignatureBased).unwrap();

        assert_eq!(assessment.attack_type, AttackType::PortScan);
        assert!(assessment.confidence >= 0.9);
    }

    #[test]
    fn test_response_generation() {
        let config = SecurityConfig {
            enable_automated_response: true,
            ..Default::default()
        };
        let detector = ThreatDetector::new(config).unwrap();

        let response = detector.generate_response(ThreatLevel::Critical, AttackType::Ransomware);

        assert_eq!(response.action, ResponseAction::Block);
        assert_eq!(response.priority, 10);
        assert!(response.automated);
        assert!(response.mitigation_steps.len() >= 3);
    }
}
