//! Healthcare Risk Trajectory Forecasting Demo
//!
//! Demonstrates patient risk prediction with time series forecasting

use prism_ai::applications::healthcare::{
    HealthcareRiskPredictor, HealthcareConfig,
    PatientProfile, VitalSigns, LabResults, MedicalHistory,
    RiskTrajectoryForecaster, TrajectoryConfig, RiskTimePoint,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Healthcare Risk Trajectory Forecasting Demo ===\n");

    // Create healthcare risk predictor
    let config = HealthcareConfig::default();
    let mut predictor = HealthcareRiskPredictor::new(config)?;

    // Simulate patient deteriorating over 12 hours (4 assessments, 4 hours apart)
    println!("Simulating patient with gradual deterioration...\n");

    let mut historical_risk = Vec::new();

    // Assessment 1: Initial admission (4 hours ago)
    let patient_t0 = create_patient_history(0);
    let assessment_t0 = predictor.assess_risk(&patient_t0)?;
    println!("T-12h: Mortality={:.1}%, Sepsis={:.1}%, Category={:?}",
             assessment_t0.mortality_risk * 100.0,
             assessment_t0.sepsis_risk * 100.0,
             assessment_t0.risk_category);

    historical_risk.push(RiskTimePoint {
        time_hours: 0.0,
        mortality_risk: assessment_t0.mortality_risk,
        sepsis_risk: assessment_t0.sepsis_risk,
        severity_score: assessment_t0.severity_score,
        risk_category: assessment_t0.risk_category,
    });

    // Assessment 2: 4 hours later
    let patient_t1 = create_patient_history(4);
    let assessment_t1 = predictor.assess_risk(&patient_t1)?;
    println!("T-8h:  Mortality={:.1}%, Sepsis={:.1}%, Category={:?}",
             assessment_t1.mortality_risk * 100.0,
             assessment_t1.sepsis_risk * 100.0,
             assessment_t1.risk_category);

    historical_risk.push(RiskTimePoint {
        time_hours: 4.0,
        mortality_risk: assessment_t1.mortality_risk,
        sepsis_risk: assessment_t1.sepsis_risk,
        severity_score: assessment_t1.severity_score,
        risk_category: assessment_t1.risk_category,
    });

    // Assessment 3: 8 hours from start
    let patient_t2 = create_patient_history(8);
    let assessment_t2 = predictor.assess_risk(&patient_t2)?;
    println!("T-4h:  Mortality={:.1}%, Sepsis={:.1}%, Category={:?}",
             assessment_t2.mortality_risk * 100.0,
             assessment_t2.sepsis_risk * 100.0,
             assessment_t2.risk_category);

    historical_risk.push(RiskTimePoint {
        time_hours: 8.0,
        mortality_risk: assessment_t2.mortality_risk,
        sepsis_risk: assessment_t2.sepsis_risk,
        severity_score: assessment_t2.severity_score,
        risk_category: assessment_t2.risk_category,
    });

    // Assessment 4: 12 hours (current)
    let patient_t3 = create_patient_history(12);
    let assessment_t3 = predictor.assess_risk(&patient_t3)?;
    println!("T-0h:  Mortality={:.1}%, Sepsis={:.1}%, Category={:?}",
             assessment_t3.mortality_risk * 100.0,
             assessment_t3.sepsis_risk * 100.0,
             assessment_t3.risk_category);

    historical_risk.push(RiskTimePoint {
        time_hours: 12.0,
        mortality_risk: assessment_t3.mortality_risk,
        sepsis_risk: assessment_t3.sepsis_risk,
        severity_score: assessment_t3.severity_score,
        risk_category: assessment_t3.risk_category,
    });

    // Forecast risk trajectory for next 24 hours
    println!("\n--- Forecasting Risk Trajectory (24-hour horizon) ---");

    let mut trajectory_config = TrajectoryConfig::default();
    // Use simpler ARIMA model for this demo (fewer parameters)
    trajectory_config.arima_config.p = 1;
    trajectory_config.arima_config.d = 0;  // No differencing (already trend-stationary)
    trajectory_config.arima_config.q = 0;  // Simple AR model

    let mut trajectory_forecaster = RiskTrajectoryForecaster::new(trajectory_config);

    let trajectory = trajectory_forecaster.forecast_trajectory(&historical_risk)?;

    // Print trajectory summary
    trajectory.print_summary();

    // Generate treatment recommendations based on forecast
    println!("\n--- Treatment Recommendations ---");

    let treatment = predictor.recommend_treatment(&assessment_t3, &patient_t3)?;

    println!("Monitoring frequency: Every {:.1} hours", treatment.monitoring_frequency_hours);
    println!("\nInterventions:");
    for (i, intervention) in treatment.interventions.iter().enumerate() {
        println!("  {}. {}", i + 1, intervention);
    }

    println!("\nEscalation criteria:");
    for criterion in &treatment.escalation_criteria {
        println!("  - {}", criterion);
    }

    println!("\nExpected success: {:.1}%", treatment.expected_success_probability * 100.0);

    // Simulate treatment impact assessment
    println!("\n--- Treatment Impact Assessment ---");
    println!("Administering sepsis bundle...");

    // Assume treatment reduces risk by 15%
    let post_treatment_risk = assessment_t3.mortality_risk - 0.15;
    let impact = trajectory_forecaster.assess_treatment_impact(&trajectory, post_treatment_risk)?;

    println!("Impact category: {:?}", impact.impact_category);
    println!("Risk reduction: {:.1}%", impact.risk_reduction * 100.0);
    println!("Baseline forecast (1h): {:.1}%", impact.baseline_forecast[0] * 100.0);
    println!("Post-treatment risk: {:.1}%", impact.post_treatment_risk * 100.0);

    println!("\n✅ Healthcare trajectory forecasting demo complete!");

    Ok(())
}

/// Create patient history at specific time point (simulating deterioration)
fn create_patient_history(time_hours: u32) -> MedicalHistory {
    // Patient gradually deteriorates over time with some variation
    let deterioration_factor = time_hours as f64 / 12.0;  // 0.0 to 1.0 over 12 hours
    // Add some non-linear variation to avoid singular matrix
    let variation = (time_hours as f64 * 0.5).sin() * 0.05;

    MedicalHistory {
        profile: PatientProfile {
            age: 68,
            gender: 1,
            bmi: 29.0,
            chronic_conditions: vec!["Diabetes".to_string(), "Hypertension".to_string()],
            medications: vec!["Metformin".to_string(), "Lisinopril".to_string(), "Aspirin".to_string()],
            hospitalizations_last_year: 1,
            smoking_status: 1,  // Former smoker
        },
        vitals: VitalSigns {
            // Heart rate increases with deterioration + variation
            heart_rate_bpm: 85.0 + deterioration_factor * 30.0 + variation * 5.0,

            // Blood pressure drops + variation
            systolic_bp: 120.0 - deterioration_factor * 30.0 + variation * 3.0,
            diastolic_bp: 78.0 - deterioration_factor * 20.0 + variation * 2.0,

            // Respiratory rate increases + variation
            respiratory_rate: 18.0 + deterioration_factor * 10.0 + variation * 2.0,

            // Temperature rises (fever) + variation
            temperature_c: 37.2 + deterioration_factor * 1.5 + variation * 0.3,

            // Oxygen saturation drops + variation
            spo2_percent: 96.0 - deterioration_factor * 7.0 + variation * 1.5,

            // Mental status declines
            gcs_score: 15 - (deterioration_factor * 2.0) as u8,
        },
        labs: LabResults {
            // WBC increases (infection) + variation
            wbc_count: 9.5 + deterioration_factor * 8.0 + variation * 1.5,

            // Hemoglobin stable
            hemoglobin: 13.5,

            // Platelet count drops + variation
            platelet_count: 220.0 - deterioration_factor * 110.0 + variation * 10.0,

            // Kidney function worsens + variation
            creatinine: 1.1 + deterioration_factor * 1.0 + variation * 0.1,

            // BUN increases + variation
            bun: 18.0 + deterioration_factor * 22.0 + variation * 2.0,

            // Lactate increases (hypoperfusion) + variation
            lactate: 1.5 + deterioration_factor * 2.8 + variation * 0.3,

            // pH drops (acidosis) + variation
            ph: 7.38 - deterioration_factor * 0.09 + variation * 0.01,
        },
    }
}
