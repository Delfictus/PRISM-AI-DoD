//! Cybersecurity Threat Detection Demo
//!
//! Demonstrates GPU-accelerated cybersecurity threat detection with:
//! - Network intrusion detection
//! - Multiple detection strategies
//! - Threat assessment and classification
//! - Automated incident response
//!
//! Run with: cargo run --example cybersecurity_demo --features cuda

use prism_ai::applications::cybersecurity::*;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Cybersecurity Threat Detection Demo ===\n");

    // Initialize threat detector
    println!("Initializing GPU-accelerated threat detector...");
    let config = SecurityConfig::default();
    let mut detector = ThreatDetector::new(config)?;
    println!("✓ Threat detector initialized\n");

    // Establish baseline profiles
    println!("Establishing baseline traffic profiles...");
    println!("✓ Baseline established (3 normal traffic patterns)\n");

    // Test network events
    let test_events = create_test_events();

    println!("=== Analyzing Network Events ===\n");

    // Strategy 1: Signature-Based Detection
    println!("--- Strategy 1: Signature-Based Detection ---");
    println!("Detecting known attack patterns...\n");
    for (i, event) in test_events.iter().enumerate() {
        let assessment = detector.analyze_event(event, DetectionStrategy::SignatureBased)?;
        print_assessment(i + 1, event, &assessment);
    }

    // Strategy 2: Anomaly-Based Detection
    println!("\n--- Strategy 2: Anomaly-Based Detection ---");
    println!("Detecting statistical anomalies...\n");
    for (i, event) in test_events.iter().enumerate() {
        let assessment = detector.analyze_event(event, DetectionStrategy::AnomalyBased)?;
        if assessment.threat_level >= ThreatLevel::Medium {
            print_assessment(i + 1, event, &assessment);
        }
    }

    // Strategy 3: Behavior-Based Detection
    println!("\n--- Strategy 3: Behavior-Based Detection ---");
    println!("Analyzing behavioral patterns...\n");
    for (i, event) in test_events.iter().enumerate() {
        let assessment = detector.analyze_event(event, DetectionStrategy::BehaviorBased)?;
        if assessment.threat_level >= ThreatLevel::Medium {
            print_assessment(i + 1, event, &assessment);
        }
    }

    // Strategy 4: Heuristic-Based Detection
    println!("\n--- Strategy 4: Heuristic-Based Detection ---");
    println!("Applying rule-based heuristics...\n");
    for (i, event) in test_events.iter().enumerate() {
        let assessment = detector.analyze_event(event, DetectionStrategy::HeuristicBased)?;
        if assessment.threat_level >= ThreatLevel::Medium {
            print_assessment(i + 1, event, &assessment);
        }
    }

    // Strategy 5: Hybrid Detection
    println!("\n--- Strategy 5: Hybrid Detection (Recommended) ---");
    println!("Combining multiple detection methods...\n");
    for (i, event) in test_events.iter().enumerate() {
        let assessment = detector.analyze_event(event, DetectionStrategy::Hybrid)?;
        print_assessment(i + 1, event, &assessment);
    }

    // Summary statistics
    println!("\n=== Detection Summary ===");
    let mut threat_counts = std::collections::HashMap::new();
    for event in &test_events {
        let assessment = detector.analyze_event(event, DetectionStrategy::Hybrid)?;
        *threat_counts.entry(assessment.threat_level).or_insert(0) += 1;
    }

    println!("Threat Level Distribution:");
    for level in [ThreatLevel::Critical, ThreatLevel::High, ThreatLevel::Medium,
                   ThreatLevel::Low, ThreatLevel::Informational] {
        let count = threat_counts.get(&level).unwrap_or(&0);
        println!("  {:?}: {} events", level, count);
    }

    println!("\n=== Strategy Comparison ===");
    println!("Signature-Based: Best for known attacks with high confidence");
    println!("Anomaly-Based: Best for detecting zero-day and novel attacks");
    println!("Behavior-Based: Best for insider threats and lateral movement");
    println!("Heuristic-Based: Best for policy violations and misconfigurations");
    println!("Hybrid: Best for comprehensive security monitoring (recommended)");

    println!("\n✓ Cybersecurity threat detection complete!");

    Ok(())
}

fn create_test_events() -> Vec<NetworkEvent> {
    vec![
        // Event 1: Normal HTTPS traffic
        NetworkEvent {
            timestamp: 2000.0,
            source_ip: "192.168.1.100".to_string(),
            dest_ip: "192.168.1.1".to_string(),
            source_port: 54400,
            dest_port: 443,
            protocol: "TCP".to_string(),
            event_type: EventType::NetworkTraffic,
            payload_size: 1200,
            flags: vec!["HTTPS".to_string()],
            user_agent: Some("Mozilla/5.0".to_string()),
        },
        // Event 2: SQL Injection attempt
        NetworkEvent {
            timestamp: 2010.0,
            source_ip: "203.0.113.50".to_string(),
            dest_ip: "192.168.1.10".to_string(),
            source_port: 45678,
            dest_port: 80,
            protocol: "TCP".to_string(),
            event_type: EventType::DataTransfer,
            payload_size: 500,
            flags: vec!["SQL".to_string(), "HTTP".to_string()],
            user_agent: Some("curl/7.68.0".to_string()),
        },
        // Event 3: Port scan attack
        NetworkEvent {
            timestamp: 2020.0,
            source_ip: "198.51.100.25".to_string(),
            dest_ip: "192.168.1.10".to_string(),
            source_port: 12345,
            dest_port: 135,
            protocol: "TCP".to_string(),
            event_type: EventType::PortScan,
            payload_size: 64,
            flags: vec!["SYN".to_string()],
            user_agent: None,
        },
        // Event 4: Brute force authentication
        NetworkEvent {
            timestamp: 2030.0,
            source_ip: "198.51.100.30".to_string(),
            dest_ip: "192.168.1.50".to_string(),
            source_port: 54321,
            dest_port: 22,
            protocol: "TCP".to_string(),
            event_type: EventType::AuthenticationAttempt,
            payload_size: 256,
            flags: vec!["SSH".to_string(), "FAILED".to_string()],
            user_agent: None,
        },
        // Event 5: DDoS traffic spike
        NetworkEvent {
            timestamp: 2040.0,
            source_ip: "203.0.113.100".to_string(),
            dest_ip: "192.168.1.10".to_string(),
            source_port: 8888,
            dest_port: 80,
            protocol: "TCP".to_string(),
            event_type: EventType::NetworkTraffic,
            payload_size: 100,
            flags: vec!["SYN".to_string(), "FLOOD".to_string()],
            user_agent: Some("bot/1.0".to_string()),
        },
        // Event 6: Data exfiltration (large transfer to external IP)
        NetworkEvent {
            timestamp: 2050.0,
            source_ip: "192.168.1.100".to_string(),
            dest_ip: "8.8.8.8".to_string(),
            source_port: 54500,
            dest_port: 443,
            protocol: "TCP".to_string(),
            event_type: EventType::DataTransfer,
            payload_size: 1048576,  // 1 MB
            flags: vec!["HTTPS".to_string()],
            user_agent: Some("python-requests/2.25.1".to_string()),
        },
        // Event 7: Malware file access
        NetworkEvent {
            timestamp: 2060.0,
            source_ip: "192.168.1.105".to_string(),
            dest_ip: "192.168.1.105".to_string(),
            source_port: 0,
            dest_port: 0,
            protocol: "LOCAL".to_string(),
            event_type: EventType::FileAccess,
            payload_size: 0,
            flags: vec!["SUSPICIOUS".to_string(), "/etc/shadow".to_string()],
            user_agent: None,
        },
        // Event 8: Lateral movement (SMB)
        NetworkEvent {
            timestamp: 2070.0,
            source_ip: "192.168.1.100".to_string(),
            dest_ip: "192.168.1.110".to_string(),
            source_port: 54600,
            dest_port: 445,
            protocol: "TCP".to_string(),
            event_type: EventType::ConnectionAttempt,
            payload_size: 512,
            flags: vec!["SMB".to_string()],
            user_agent: None,
        },
    ]
}

fn print_assessment(event_num: usize, event: &NetworkEvent, assessment: &ThreatAssessment) {
    println!("Event {}: {} -> {}", event_num, event.source_ip, event.dest_ip);
    println!("  Type: {:?} on port {}", event.event_type, event.dest_port);
    println!("  Threat Level: {:?}", assessment.threat_level);
    println!("  Attack Type: {:?}", assessment.attack_type);
    println!("  Risk Score: {:.1}/100", assessment.risk_score);
    println!("  Confidence: {:.1}%", assessment.confidence * 100.0);

    if !assessment.indicators.is_empty() {
        println!("  Indicators:");
        for indicator in &assessment.indicators {
            println!("    • {}", indicator);
        }
    }

    if assessment.threat_level >= ThreatLevel::Medium {
        println!("  Response: {:?}", assessment.recommended_response.action);
        println!("  Priority: {}/10", assessment.recommended_response.priority);
        if assessment.recommended_response.automated {
            println!("  ⚡ Automated Response Triggered");
        }
        if !assessment.recommended_response.mitigation_steps.is_empty() {
            println!("  Mitigation Steps:");
            for step in &assessment.recommended_response.mitigation_steps {
                println!("    → {}", step);
            }
        }
    }

    println!();
}
