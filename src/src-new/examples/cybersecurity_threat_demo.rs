//! Cybersecurity Threat Forecasting Demo
//!
//! Demonstrates threat prediction with time series forecasting

use prism_ai::applications::cybersecurity::{
    ThreatForecaster, ForecastConfig, SecurityMetricPoint,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Cybersecurity Threat Forecasting Demo ===\n");

    // Create threat forecaster
    let mut forecast_config = ForecastConfig::default();
    forecast_config.horizon_hours = 24;
    forecast_config.early_warning_hours = 6;
    forecast_config.anomaly_threshold = 1.5;  // 1.5 std devs

    let mut forecaster = ThreatForecaster::new(forecast_config);

    // Simulate escalating threat over 12 hours
    println!("Simulating network security metrics (12-hour history)...\n");

    let mut historical_metrics = Vec::new();

    for hour in 0..12 {
        let t = hour as f64;

        // Gradual escalation + some variation
        let escalation_factor = 1.0 + t * 0.08;  // 8% per hour
        let variation = (t * 0.7).sin() * 0.1;

        let metrics = SecurityMetricPoint {
            time_hours: t,
            event_count: (50.0 + t * 5.0) * (escalation_factor + variation),
            avg_threat_level: (0.2 + t * 0.04).min(1.0),
            traffic_volume: (1.0 + t * 0.15) * (escalation_factor + variation * 0.5),
            failed_logins: (10.0 + t * 3.0) * escalation_factor,
        };

        println!("T-{:02}h: Events={:.1}/h, ThreatLevel={:.2}, Traffic={:.2}x, FailedLogins={:.0}",
            12 - hour,
            metrics.event_count,
            metrics.avg_threat_level,
            metrics.traffic_volume,
            metrics.failed_logins
        );

        historical_metrics.push(metrics);
    }

    // Forecast 24-hour threat trajectory
    println!("\n--- Forecasting Threat Trajectory (24-hour horizon) ---");

    let trajectory = forecaster.forecast_trajectory(&historical_metrics)?;

    // Print trajectory summary
    trajectory.print_summary();

    // Check for critical warnings
    let critical_warnings: Vec<_> = trajectory.warnings.iter()
        .filter(|w| w.hours_ahead < 4)
        .collect();

    if !critical_warnings.is_empty() {
        println!("\n🚨 IMMEDIATE ACTION REQUIRED ({} critical warnings):", critical_warnings.len());
        for warning in critical_warnings {
            println!("  - {} (confidence: {:.0}%)", warning.message, warning.confidence * 100.0);
        }
    }

    // Assess mitigation strategies
    println!("\n--- Mitigation Strategy Assessment ---");

    // Scenario 1: Firewall rules update (50% effectiveness)
    println!("\n1. Firewall Rules Update (50% threat reduction):");
    let impact_1 = forecaster.assess_mitigation_impact(&trajectory, 0.5)?;
    println!("   Impact category: {:?}", impact_1.impact_category);
    println!("   Event reduction: {:.1} events", impact_1.event_reduction);
    println!("   Reduction: {:.1}%", impact_1.reduction_percentage);

    // Scenario 2: IDS/IPS deployment (80% effectiveness)
    println!("\n2. Advanced IDS/IPS Deployment (80% threat reduction):");
    let impact_2 = forecaster.assess_mitigation_impact(&trajectory, 0.8)?;
    println!("   Impact category: {:?}", impact_2.impact_category);
    println!("   Event reduction: {:.1} events", impact_2.event_reduction);
    println!("   Reduction: {:.1}%", impact_2.reduction_percentage);

    // Scenario 3: Rate limiting only (20% effectiveness)
    println!("\n3. Rate Limiting Only (20% threat reduction):");
    let impact_3 = forecaster.assess_mitigation_impact(&trajectory, 0.2)?;
    println!("   Impact category: {:?}", impact_3.impact_category);
    println!("   Event reduction: {:.1} events", impact_3.event_reduction);
    println!("   Reduction: {:.1}%", impact_3.reduction_percentage);

    // Recommendation
    println!("\n--- Recommended Action ---");
    if trajectory.warnings.len() >= 3 {
        println!("⚠️  Multiple threats detected. Recommend: Advanced IDS/IPS deployment");
        println!("   Expected outcome: {:?} threat reduction", impact_2.impact_category);
    } else if !trajectory.warnings.is_empty() {
        println!("⚠️  Moderate threat detected. Recommend: Firewall rules update");
        println!("   Expected outcome: {:?} threat reduction", impact_1.impact_category);
    } else {
        println!("✅ Threat level acceptable. Continue monitoring.");
    }

    println!("\n✅ Cybersecurity threat forecasting demo complete!");

    Ok(())
}
