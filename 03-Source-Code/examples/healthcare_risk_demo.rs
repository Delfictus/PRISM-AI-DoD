//! Healthcare Patient Risk Prediction Demo
//!
//! Demonstrates GPU-accelerated patient risk assessment and clinical decision support.
//!
//! Run with: cargo run --example healthcare_risk_demo --features cuda

use prism_ai::applications::healthcare::{
    HealthcareRiskPredictor, PatientProfile, VitalSigns, LabResults,
    MedicalHistory, RiskCategory, HealthcareConfig,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Healthcare Risk Prediction Demo ===\n");

    // Initialize GPU-accelerated risk predictor
    println!("Initializing GPU-accelerated risk predictor...");
    let config = HealthcareConfig::default();
    let mut predictor = HealthcareRiskPredictor::new(config)?;
    println!("✓ GPU initialization successful\n");

    // Demo 1: Stable patient (routine care)
    println!("--- Case 1: Stable Patient (Routine Care) ---");
    let stable_patient = create_stable_patient();
    print_patient_info(&stable_patient, "45yo male, hypertension, routine check");

    let assessment = predictor.assess_risk(&stable_patient)?;
    print_risk_assessment(&assessment);

    let treatment = predictor.recommend_treatment(&assessment, &stable_patient)?;
    print_treatment_recommendations(&treatment);

    // Demo 2: High-risk patient (sepsis concern)
    println!("\n--- Case 2: High-Risk Patient (Sepsis Concern) ---");
    let septic_patient = create_septic_patient();
    print_patient_info(&septic_patient, "72yo female, diabetes + COPD, acute illness");

    let assessment = predictor.assess_risk(&septic_patient)?;
    print_risk_assessment(&assessment);

    let treatment = predictor.recommend_treatment(&assessment, &septic_patient)?;
    print_treatment_recommendations(&treatment);

    // Demo 3: Critical patient (multiple organ dysfunction)
    println!("\n--- Case 3: Critical Patient (Multiple Organ Dysfunction) ---");
    let critical_patient = create_critical_patient();
    print_patient_info(&critical_patient, "68yo male, CHF + CKD, cardiogenic shock");

    let assessment = predictor.assess_risk(&critical_patient)?;
    print_risk_assessment(&assessment);

    let treatment = predictor.recommend_treatment(&assessment, &critical_patient)?;
    print_treatment_recommendations(&treatment);

    // Summary
    println!("\n=== Clinical Decision Support Summary ===");
    println!("Risk Assessment Components:");
    println!("  • Mortality risk: Age, vital signs, organ function");
    println!("  • Sepsis risk: SIRS criteria, lactate, hypotension");
    println!("  • ICU admission: Respiratory failure, shock, AMS");
    println!("  • Readmission: Comorbidities, prior admissions");
    println!("  • Severity score: APACHE II-style (0-71 scale)");
    println!("\nTreatment Recommendations:");
    println!("  • Risk-stratified interventions");
    println!("  • Monitoring frequency adjusted by risk");
    println!("  • Early warning for clinical deterioration");
    println!("  • Sepsis bundle activation when indicated");

    println!("\n✓ Healthcare risk assessment complete!");

    Ok(())
}

fn create_stable_patient() -> MedicalHistory {
    MedicalHistory {
        profile: PatientProfile {
            age: 45,
            gender: 1,
            bmi: 24.5,
            chronic_conditions: vec!["Hypertension".to_string()],
            medications: vec!["Lisinopril 10mg daily".to_string()],
            hospitalizations_last_year: 0,
            smoking_status: 0,
        },
        vitals: VitalSigns {
            heart_rate_bpm: 75.0,
            systolic_bp: 125.0,
            diastolic_bp: 82.0,
            respiratory_rate: 16.0,
            temperature_c: 37.0,
            spo2_percent: 98.0,
            gcs_score: 15,
        },
        labs: LabResults {
            wbc_count: 7.5,
            hemoglobin: 14.2,
            platelet_count: 250.0,
            creatinine: 1.0,
            bun: 15.0,
            lactate: 1.2,
            ph: 7.40,
        },
    }
}

fn create_septic_patient() -> MedicalHistory {
    MedicalHistory {
        profile: PatientProfile {
            age: 72,
            gender: 0,
            bmi: 28.0,
            chronic_conditions: vec![
                "Type 2 Diabetes".to_string(),
                "COPD".to_string(),
                "Chronic Kidney Disease Stage 3".to_string(),
            ],
            medications: vec![
                "Metformin 1000mg BID".to_string(),
                "Albuterol inhaler PRN".to_string(),
                "Lisinopril 20mg daily".to_string(),
            ],
            hospitalizations_last_year: 2,
            smoking_status: 1,  // Former smoker
        },
        vitals: VitalSigns {
            heart_rate_bpm: 115.0,  // Tachycardia
            systolic_bp: 85.0,       // Hypotension
            diastolic_bp: 55.0,
            respiratory_rate: 28.0,  // Tachypnea
            temperature_c: 38.9,     // Fever
            spo2_percent: 88.0,      // Hypoxia
            gcs_score: 13,           // Altered mental status
        },
        labs: LabResults {
            wbc_count: 18.5,         // Leukocytosis
            hemoglobin: 11.2,        // Anemia
            platelet_count: 95.0,    // Thrombocytopenia
            creatinine: 2.3,         // Acute kidney injury
            bun: 45.0,
            lactate: 4.5,            // Severe lactic acidosis
            ph: 7.28,                // Metabolic acidosis
        },
    }
}

fn create_critical_patient() -> MedicalHistory {
    MedicalHistory {
        profile: PatientProfile {
            age: 68,
            gender: 1,
            bmi: 32.5,
            chronic_conditions: vec![
                "Congestive Heart Failure (EF 25%)".to_string(),
                "Chronic Kidney Disease Stage 4".to_string(),
                "Atrial Fibrillation".to_string(),
                "Diabetes Type 2".to_string(),
            ],
            medications: vec![
                "Furosemide 80mg BID".to_string(),
                "Carvedilol 25mg BID".to_string(),
                "Warfarin 5mg daily".to_string(),
                "Insulin glargine 30u qHS".to_string(),
            ],
            hospitalizations_last_year: 4,
            smoking_status: 2,  // Current smoker
        },
        vitals: VitalSigns {
            heart_rate_bpm: 125.0,  // Severe tachycardia
            systolic_bp: 70.0,       // Severe hypotension (shock)
            diastolic_bp: 45.0,
            respiratory_rate: 32.0,  // Severe tachypnea
            temperature_c: 36.2,     // Hypothermia (poor perfusion)
            spo2_percent: 82.0,      // Severe hypoxia
            gcs_score: 11,           // Moderate altered mental status
        },
        labs: LabResults {
            wbc_count: 22.0,         // Severe leukocytosis
            hemoglobin: 9.5,         // Significant anemia
            platelet_count: 75.0,    // Severe thrombocytopenia
            creatinine: 3.8,         // Severe acute kidney injury
            bun: 85.0,               // Azotemia
            lactate: 8.2,            // Critical lactic acidosis
            ph: 7.18,                // Severe metabolic acidosis
        },
    }
}

fn print_patient_info(history: &MedicalHistory, description: &str) {
    println!("Patient: {}", description);
    println!("  Age: {} years, BMI: {:.1}", history.profile.age, history.profile.bmi);
    println!("  Comorbidities: {}", history.profile.chronic_conditions.join(", "));
    println!("\nVital Signs:");
    println!("  HR: {:.0} bpm, BP: {:.0}/{:.0} mmHg, RR: {:.0}/min",
        history.vitals.heart_rate_bpm,
        history.vitals.systolic_bp,
        history.vitals.diastolic_bp,
        history.vitals.respiratory_rate
    );
    println!("  Temp: {:.1}°C, SpO2: {:.0}%, GCS: {}",
        history.vitals.temperature_c,
        history.vitals.spo2_percent,
        history.vitals.gcs_score
    );
    println!("\nLaboratory Results:");
    println!("  WBC: {:.1} K/µL, Hgb: {:.1} g/dL, Plt: {:.0} K/µL",
        history.labs.wbc_count,
        history.labs.hemoglobin,
        history.labs.platelet_count
    );
    println!("  Cr: {:.1} mg/dL, BUN: {:.0} mg/dL, Lactate: {:.1} mmol/L, pH: {:.2}",
        history.labs.creatinine,
        history.labs.bun,
        history.labs.lactate,
        history.labs.ph
    );
}

fn print_risk_assessment(assessment: &prism_ai::applications::healthcare::RiskAssessment) {
    println!("\n=== Risk Assessment ===");

    let category_str = match assessment.risk_category {
        RiskCategory::Low => "LOW (routine monitoring)",
        RiskCategory::Moderate => "MODERATE (increased monitoring)",
        RiskCategory::High => "HIGH (intensive care consideration)",
        RiskCategory::Critical => "⚠️  CRITICAL (immediate intervention required)",
    };
    println!("Overall Risk Category: {}", category_str);

    println!("\nRisk Scores:");
    println!("  Mortality Risk:     {:.1}%", assessment.mortality_risk * 100.0);
    println!("  Sepsis Risk:        {:.1}%", assessment.sepsis_risk * 100.0);
    println!("  ICU Admission Risk: {:.1}%", assessment.icu_admission_risk * 100.0);
    println!("  30-Day Readmission: {:.1}%", assessment.readmission_risk * 100.0);
    println!("  Severity Score:     {:.1}/71 (APACHE II-like)", assessment.severity_score);

    if !assessment.risk_factors.is_empty() {
        println!("\nKey Risk Factors:");
        for factor in &assessment.risk_factors {
            println!("  • {}", factor);
        }
    }
}

fn print_treatment_recommendations(treatment: &prism_ai::applications::healthcare::TreatmentRecommendation) {
    println!("\n=== Treatment Recommendations ===");

    println!("Recommended Interventions:");
    for (i, intervention) in treatment.interventions.iter().enumerate() {
        if intervention.contains("IMMEDIATE") || intervention.contains("SEPSIS") {
            println!("  {}. ⚠️  {}", i + 1, intervention);
        } else {
            println!("  {}. {}", i + 1, intervention);
        }
    }

    println!("\nMonitoring:");
    if treatment.monitoring_frequency_hours < 1.0 {
        println!("  Frequency: Every {:.0} minutes", treatment.monitoring_frequency_hours * 60.0);
    } else {
        println!("  Frequency: Every {:.1} hours", treatment.monitoring_frequency_hours);
    }

    if !treatment.escalation_criteria.is_empty() {
        println!("\nEscalation Criteria:");
        for criterion in &treatment.escalation_criteria {
            println!("  • {}", criterion);
        }
    }

    println!("\nExpected Treatment Success: {:.1}%", treatment.expected_success_probability * 100.0);
}
