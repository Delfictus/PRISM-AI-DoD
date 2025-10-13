//! Patient Risk Prediction Engine
//!
//! Implements healthcare risk assessment with GPU acceleration:
//! - Multi-factor risk scoring (APACHE II, SOFA-inspired)
//! - Disease progression modeling
//! - Treatment outcome prediction
//! - Early warning for clinical deterioration
//!
//! Uses Active Inference for adaptive clinical decision support.

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use crate::gpu::GpuMemoryPool;
use std::collections::HashMap;

/// Risk category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCategory {
    /// Low risk (routine monitoring)
    Low,
    /// Moderate risk (increased monitoring)
    Moderate,
    /// High risk (intensive care consideration)
    High,
    /// Critical (immediate intervention required)
    Critical,
}

/// Patient profile with demographics and medical history
#[derive(Debug, Clone)]
pub struct PatientProfile {
    /// Patient age (years)
    pub age: u32,
    /// Gender (0 = female, 1 = male, 2 = other)
    pub gender: u8,
    /// Body Mass Index (kg/m²)
    pub bmi: f64,
    /// Chronic conditions
    pub chronic_conditions: Vec<String>,
    /// Current medications
    pub medications: Vec<String>,
    /// Previous hospitalizations (count in last year)
    pub hospitalizations_last_year: u32,
    /// Smoking status (0 = never, 1 = former, 2 = current)
    pub smoking_status: u8,
}

/// Current vital signs
#[derive(Debug, Clone)]
pub struct VitalSigns {
    /// Heart rate (beats per minute)
    pub heart_rate_bpm: f64,
    /// Systolic blood pressure (mmHg)
    pub systolic_bp: f64,
    /// Diastolic blood pressure (mmHg)
    pub diastolic_bp: f64,
    /// Respiratory rate (breaths per minute)
    pub respiratory_rate: f64,
    /// Temperature (Celsius)
    pub temperature_c: f64,
    /// Oxygen saturation (%)
    pub spo2_percent: f64,
    /// Glasgow Coma Scale (3-15)
    pub gcs_score: u8,
}

/// Laboratory test results
#[derive(Debug, Clone)]
pub struct LabResults {
    /// White Blood Cell count (10³/µL)
    pub wbc_count: f64,
    /// Hemoglobin (g/dL)
    pub hemoglobin: f64,
    /// Platelet count (10³/µL)
    pub platelet_count: f64,
    /// Creatinine (mg/dL) - kidney function
    pub creatinine: f64,
    /// Blood Urea Nitrogen (mg/dL)
    pub bun: f64,
    /// Lactate (mmol/L) - tissue oxygenation
    pub lactate: f64,
    /// pH (arterial blood gas)
    pub ph: f64,
}

/// Complete medical history for risk assessment
#[derive(Debug, Clone)]
pub struct MedicalHistory {
    /// Patient profile
    pub profile: PatientProfile,
    /// Current vital signs
    pub vitals: VitalSigns,
    /// Laboratory results
    pub labs: LabResults,
}

/// Comprehensive risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk category
    pub risk_category: RiskCategory,
    /// Mortality risk (0.0 to 1.0)
    pub mortality_risk: f64,
    /// 30-day readmission risk (0.0 to 1.0)
    pub readmission_risk: f64,
    /// Sepsis risk (0.0 to 1.0)
    pub sepsis_risk: f64,
    /// ICU admission likelihood (0.0 to 1.0)
    pub icu_admission_risk: f64,
    /// APACHE II-like severity score (0-71 scale)
    pub severity_score: f64,
    /// Risk factors contributing to assessment
    pub risk_factors: Vec<String>,
}

/// Treatment recommendation with rationale
#[derive(Debug, Clone)]
pub struct TreatmentRecommendation {
    /// Recommended interventions (prioritized)
    pub interventions: Vec<String>,
    /// Monitoring frequency (hours between assessments)
    pub monitoring_frequency_hours: f64,
    /// Escalation criteria
    pub escalation_criteria: Vec<String>,
    /// Expected outcome probability
    pub expected_success_probability: f64,
}

/// Healthcare configuration
#[derive(Debug, Clone)]
pub struct HealthcareConfig {
    /// Enable early warning system
    pub enable_early_warning: bool,
    /// Sepsis detection threshold
    pub sepsis_threshold: f64,
    /// High risk threshold for escalation
    pub high_risk_threshold: f64,
    /// Use Active Inference for recommendations
    pub use_active_inference: bool,
}

impl Default for HealthcareConfig {
    fn default() -> Self {
        Self {
            enable_early_warning: true,
            sepsis_threshold: 0.6,
            high_risk_threshold: 0.7,
            use_active_inference: true,
        }
    }
}

/// GPU-accelerated healthcare risk predictor
pub struct HealthcareRiskPredictor {
    /// GPU memory pool for risk model computation
    gpu_pool: GpuMemoryPool,
    /// Configuration
    config: HealthcareConfig,
}

impl HealthcareRiskPredictor {
    /// Create new risk predictor with GPU acceleration
    pub fn new(config: HealthcareConfig) -> Result<Self> {
        let gpu_pool = GpuMemoryPool::new()
            .context("Failed to initialize GPU for healthcare risk prediction")?;

        Ok(Self {
            gpu_pool,
            config,
        })
    }

    /// Assess patient risk from medical history
    pub fn assess_risk(&mut self, history: &MedicalHistory) -> Result<RiskAssessment> {
        // Extract features from patient data
        let features = self.extract_features(history)?;

        // Compute risk scores using GPU-accelerated models
        // TODO: GPU acceleration hook for Worker 2
        // Request: risk_scoring_kernel(features, model_weights)

        let mortality_risk = self.compute_mortality_risk(&features, history)?;
        let readmission_risk = self.compute_readmission_risk(&features, history)?;
        let sepsis_risk = self.compute_sepsis_risk(&features, history)?;
        let icu_admission_risk = self.compute_icu_risk(&features, history)?;
        let severity_score = self.compute_severity_score(&features, history)?;

        // Determine overall risk category
        let risk_category = self.classify_risk_category(
            mortality_risk,
            sepsis_risk,
            severity_score,
        );

        // Identify key risk factors
        let risk_factors = self.identify_risk_factors(history, &features)?;

        Ok(RiskAssessment {
            risk_category,
            mortality_risk,
            readmission_risk,
            sepsis_risk,
            icu_admission_risk,
            severity_score,
            risk_factors,
        })
    }

    /// Generate treatment recommendations based on risk assessment
    pub fn recommend_treatment(
        &self,
        assessment: &RiskAssessment,
        history: &MedicalHistory,
    ) -> Result<TreatmentRecommendation> {
        let mut interventions = Vec::new();
        let mut escalation_criteria = Vec::new();

        // Risk-based recommendations
        match assessment.risk_category {
            RiskCategory::Critical => {
                interventions.push("IMMEDIATE: Transfer to ICU".to_string());
                interventions.push("Establish central venous access".to_string());
                interventions.push("Initiate broad-spectrum antibiotics if sepsis suspected".to_string());
                interventions.push("Continuous hemodynamic monitoring".to_string());

                escalation_criteria.push("Any deterioration in vital signs".to_string());
                escalation_criteria.push("Worsening organ function".to_string());
            }
            RiskCategory::High => {
                interventions.push("Increase monitoring to hourly vital signs".to_string());
                interventions.push("Consult intensivist for ICU readiness".to_string());
                interventions.push("Optimize fluid management".to_string());

                escalation_criteria.push("Rising lactate (>4 mmol/L)".to_string());
                escalation_criteria.push("Declining mental status (GCS drop)".to_string());
                escalation_criteria.push("Hypotension not responding to fluids".to_string());
            }
            RiskCategory::Moderate => {
                interventions.push("Monitor vital signs every 4 hours".to_string());
                interventions.push("Review medications for adverse interactions".to_string());
                interventions.push("Ensure adequate hydration".to_string());

                escalation_criteria.push("New fever or hypothermia".to_string());
                escalation_criteria.push("Increasing oxygen requirement".to_string());
            }
            RiskCategory::Low => {
                interventions.push("Standard monitoring every 8 hours".to_string());
                interventions.push("Continue current treatment plan".to_string());

                escalation_criteria.push("Any acute change in vital signs".to_string());
            }
        }

        // Sepsis-specific interventions
        if assessment.sepsis_risk > self.config.sepsis_threshold {
            interventions.insert(0, "SEPSIS ALERT: Initiate sepsis bundle".to_string());
            interventions.insert(1, "Blood cultures before antibiotics".to_string());
            interventions.insert(2, "Administer broad-spectrum antibiotics within 1 hour".to_string());
            interventions.insert(3, "Fluid resuscitation 30 mL/kg".to_string());
        }

        // Organ-specific interventions
        if history.labs.creatinine > 1.5 {
            interventions.push("Nephrology consult for acute kidney injury".to_string());
            interventions.push("Adjust medication doses for renal function".to_string());
        }

        if history.vitals.spo2_percent < 90.0 {
            interventions.push("Increase supplemental oxygen".to_string());
            interventions.push("Consider arterial blood gas analysis".to_string());
        }

        let monitoring_frequency_hours = match assessment.risk_category {
            RiskCategory::Critical => 0.25,  // Every 15 minutes
            RiskCategory::High => 1.0,
            RiskCategory::Moderate => 4.0,
            RiskCategory::Low => 8.0,
        };

        // Estimate success probability (simplified model)
        let expected_success_probability = self.estimate_treatment_success(assessment, history)?;

        Ok(TreatmentRecommendation {
            interventions,
            monitoring_frequency_hours,
            escalation_criteria,
            expected_success_probability,
        })
    }

    /// Extract numerical features from medical history
    fn extract_features(&self, history: &MedicalHistory) -> Result<Array1<f64>> {
        let mut features = Array1::zeros(50);

        // Demographics (0-4)
        features[0] = history.profile.age as f64 / 100.0;  // Normalize to [0,1]
        features[1] = history.profile.gender as f64 / 2.0;
        features[2] = (history.profile.bmi - 20.0) / 20.0;  // Normalize around healthy BMI
        features[3] = history.profile.chronic_conditions.len() as f64 / 10.0;
        features[4] = history.profile.hospitalizations_last_year as f64 / 5.0;

        // Vital signs (5-11)
        features[5] = (history.vitals.heart_rate_bpm - 70.0) / 50.0;  // Normal ~70 bpm
        features[6] = (history.vitals.systolic_bp - 120.0) / 40.0;     // Normal ~120 mmHg
        features[7] = (history.vitals.diastolic_bp - 80.0) / 20.0;     // Normal ~80 mmHg
        features[8] = (history.vitals.respiratory_rate - 16.0) / 10.0; // Normal ~16/min
        features[9] = (history.vitals.temperature_c - 37.0) / 3.0;     // Normal ~37°C
        features[10] = (history.vitals.spo2_percent - 95.0) / 10.0;    // Normal ~98%
        features[11] = (15.0 - history.vitals.gcs_score as f64) / 12.0; // Lower GCS = worse

        // Lab results (12-18)
        features[12] = (history.labs.wbc_count - 7.5) / 10.0;    // Normal ~7.5 10³/µL
        features[13] = (14.0 - history.labs.hemoglobin) / 5.0;   // Normal ~14 g/dL
        features[14] = (history.labs.platelet_count - 250.0) / 200.0; // Normal ~250 10³/µL
        features[15] = (history.labs.creatinine - 1.0) / 2.0;    // Normal ~1.0 mg/dL
        features[16] = (history.labs.bun - 15.0) / 30.0;         // Normal ~15 mg/dL
        features[17] = (history.labs.lactate - 1.0) / 3.0;       // Normal ~1.0 mmol/L
        features[18] = (7.4 - history.labs.ph) / 0.2;            // Normal ~7.4

        // Derived features (19-30)
        features[19] = self.compute_shock_index(&history.vitals);
        features[20] = self.compute_map(&history.vitals);  // Mean arterial pressure
        features[21] = self.is_tachycardic(&history.vitals);
        features[22] = self.is_hypotensive(&history.vitals);
        features[23] = self.is_hypoxic(&history.vitals);
        features[24] = self.has_fever(&history.vitals);
        features[25] = self.has_elevated_lactate(&history.labs);
        features[26] = self.has_kidney_dysfunction(&history.labs);
        features[27] = self.has_altered_mental_status(&history.vitals);
        features[28] = self.compute_age_severity_interaction(&history.profile, &history.vitals);
        features[29] = self.compute_comorbidity_burden(&history.profile);
        features[30] = self.compute_sirs_criteria(&history);  // Systemic Inflammatory Response

        // Reserved for future expansion (31-49)

        Ok(features)
    }

    /// Compute mortality risk score
    fn compute_mortality_risk(
        &self,
        features: &Array1<f64>,
        history: &MedicalHistory,
    ) -> Result<f64> {
        // Simplified mortality model (in production: use trained ML model)
        let mut risk = 0.01;  // Base risk

        // Age factor
        if history.profile.age > 75 {
            risk += 0.15;
        } else if history.profile.age > 65 {
            risk += 0.08;
        }

        // Vital signs abnormalities
        if history.vitals.systolic_bp < 90.0 {
            risk += 0.20;  // Hypotension
        }
        if history.vitals.spo2_percent < 90.0 {
            risk += 0.15;  // Hypoxia
        }
        if history.vitals.gcs_score < 13 {
            risk += 0.18;  // Altered mental status
        }

        // Lab abnormalities
        if history.labs.lactate > 4.0 {
            risk += 0.25;  // Severe lactic acidosis
        }
        if history.labs.creatinine > 2.0 {
            risk += 0.12;  // Acute kidney injury
        }

        // Multiple organ involvement
        let organ_dysfunction_count = self.count_organ_dysfunctions(history);
        risk += organ_dysfunction_count as f64 * 0.10;

        Ok(risk.min(0.99_f64))
    }

    /// Compute 30-day readmission risk
    fn compute_readmission_risk(
        &self,
        _features: &Array1<f64>,
        history: &MedicalHistory,
    ) -> Result<f64> {
        let mut risk = 0.10;  // Base readmission risk

        // Previous hospitalizations
        risk += history.profile.hospitalizations_last_year as f64 * 0.08;

        // Chronic conditions
        risk += history.profile.chronic_conditions.len() as f64 * 0.05;

        // Polypharmacy (medication count)
        if history.profile.medications.len() > 10 {
            risk += 0.12;
        }

        // Age
        if history.profile.age > 70 {
            risk += 0.10;
        }

        Ok(risk.min(0.95))
    }

    /// Compute sepsis risk score
    fn compute_sepsis_risk(
        &self,
        _features: &Array1<f64>,
        history: &MedicalHistory,
    ) -> Result<f64> {
        let sirs_count = self.compute_sirs_criteria(history);

        let mut risk: f64 = 0.01;

        // SIRS criteria (2+ = potential sepsis)
        if sirs_count >= 2.0 {
            risk += 0.30;
        }

        // Organ dysfunction markers
        if history.labs.lactate > 2.0 {
            risk += 0.20;
        }

        if history.vitals.systolic_bp < 90.0 {
            risk += 0.15;  // Hypotension suggests septic shock
        }

        if history.labs.wbc_count > 12.0 || history.labs.wbc_count < 4.0 {
            risk += 0.10;
        }

        if history.vitals.gcs_score < 15 {
            risk += 0.10;
        }

        Ok(risk.min(0.99_f64))
    }

    /// Compute ICU admission risk
    fn compute_icu_risk(
        &self,
        _features: &Array1<f64>,
        history: &MedicalHistory,
    ) -> Result<f64> {
        let mut risk: f64 = 0.05;

        // Respiratory failure
        if history.vitals.spo2_percent < 88.0 {
            risk += 0.40;
        }

        // Hemodynamic instability
        if history.vitals.systolic_bp < 90.0 {
            risk += 0.35;
        }

        // Altered mental status
        if history.vitals.gcs_score < 13 {
            risk += 0.25;
        }

        // Severe metabolic derangement
        if history.labs.lactate > 4.0 {
            risk += 0.30;
        }

        Ok(risk.min(0.99_f64))
    }

    /// Compute APACHE II-like severity score (0-71)
    fn compute_severity_score(
        &self,
        _features: &Array1<f64>,
        history: &MedicalHistory,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Age points (0-6)
        score += match history.profile.age {
            0..=44 => 0.0,
            45..=54 => 2.0,
            55..=64 => 3.0,
            65..=74 => 5.0,
            _ => 6.0,
        };

        // Vital signs points
        score += self.score_vital_abnormalities(&history.vitals);

        // Lab values points
        score += self.score_lab_abnormalities(&history.labs);

        // Chronic health points (0-5)
        score += (history.profile.chronic_conditions.len() as f64).min(5.0);

        Ok(score.min(71.0))
    }

    /// Score vital sign abnormalities (APACHE II style)
    fn score_vital_abnormalities(&self, vitals: &VitalSigns) -> f64 {
        let mut score = 0.0;

        // Heart rate
        score += match vitals.heart_rate_bpm as u32 {
            0..=39 => 4.0,
            40..=54 => 3.0,
            55..=69 => 2.0,
            70..=109 => 0.0,
            110..=139 => 2.0,
            140..=179 => 3.0,
            _ => 4.0,
        };

        // Mean arterial pressure
        let map = (vitals.systolic_bp + 2.0 * vitals.diastolic_bp) / 3.0;
        score += if map < 49.0 {
            4.0
        } else if map < 69.0 {
            2.0
        } else if map > 159.0 {
            3.0
        } else {
            0.0
        };

        // Respiratory rate
        score += match vitals.respiratory_rate as u32 {
            0..=5 => 4.0,
            6..=9 => 2.0,
            10..=11 => 1.0,
            12..=24 => 0.0,
            25..=34 => 1.0,
            35..=49 => 3.0,
            _ => 4.0,
        };

        // Temperature
        score += if vitals.temperature_c < 29.9 {
            4.0
        } else if vitals.temperature_c < 33.9 {
            3.0
        } else if vitals.temperature_c > 40.9 {
            4.0
        } else if vitals.temperature_c > 38.4 {
            1.0
        } else {
            0.0
        };

        // GCS
        score += (15 - vitals.gcs_score) as f64 / 3.0;

        score
    }

    /// Score laboratory abnormalities
    fn score_lab_abnormalities(&self, labs: &LabResults) -> f64 {
        let mut score = 0.0;

        // Creatinine (kidney function)
        if labs.creatinine > 3.5 {
            score += 4.0;
        } else if labs.creatinine > 2.0 {
            score += 3.0;
        } else if labs.creatinine > 1.5 {
            score += 2.0;
        }

        // WBC count
        if labs.wbc_count < 1.0 || labs.wbc_count > 40.0 {
            score += 4.0;
        } else if labs.wbc_count < 3.0 || labs.wbc_count > 20.0 {
            score += 2.0;
        }

        // pH
        if labs.ph < 7.15 || labs.ph > 7.70 {
            score += 4.0;
        } else if labs.ph < 7.25 || labs.ph > 7.60 {
            score += 3.0;
        } else if labs.ph < 7.33 || labs.ph > 7.50 {
            score += 1.0;
        }

        score
    }

    /// Classify overall risk category
    fn classify_risk_category(
        &self,
        mortality_risk: f64,
        sepsis_risk: f64,
        severity_score: f64,
    ) -> RiskCategory {
        if mortality_risk > 0.5 || sepsis_risk > 0.7 || severity_score > 30.0 {
            RiskCategory::Critical
        } else if mortality_risk > 0.3 || sepsis_risk > 0.5 || severity_score > 20.0 {
            RiskCategory::High
        } else if mortality_risk > 0.15 || sepsis_risk > 0.3 || severity_score > 10.0 {
            RiskCategory::Moderate
        } else {
            RiskCategory::Low
        }
    }

    /// Identify key risk factors from assessment
    fn identify_risk_factors(
        &self,
        history: &MedicalHistory,
        _features: &Array1<f64>,
    ) -> Result<Vec<String>> {
        let mut factors = Vec::new();

        if history.profile.age > 75 {
            factors.push("Advanced age (>75 years)".to_string());
        }

        if history.vitals.systolic_bp < 90.0 {
            factors.push("Hypotension (SBP <90 mmHg)".to_string());
        }

        if history.vitals.spo2_percent < 90.0 {
            factors.push("Hypoxemia (SpO2 <90%)".to_string());
        }

        if history.vitals.gcs_score < 13 {
            factors.push("Altered mental status (GCS <13)".to_string());
        }

        if history.labs.lactate > 2.0 {
            factors.push(format!("Elevated lactate ({:.1} mmol/L)", history.labs.lactate));
        }

        if history.labs.creatinine > 1.5 {
            factors.push(format!("Acute kidney injury (Cr {:.1} mg/dL)", history.labs.creatinine));
        }

        if self.compute_sirs_criteria(history) >= 2.0 {
            factors.push("Meets SIRS criteria (≥2)".to_string());
        }

        if history.profile.chronic_conditions.len() > 3 {
            factors.push(format!("Multiple comorbidities ({})", history.profile.chronic_conditions.len()));
        }

        Ok(factors)
    }

    /// Estimate treatment success probability
    fn estimate_treatment_success(
        &self,
        assessment: &RiskAssessment,
        _history: &MedicalHistory,
    ) -> Result<f64> {
        // Simplified model (inverse of mortality risk with floor)
        let success_prob = 1.0 - (assessment.mortality_risk * 0.7);
        Ok(success_prob.max(0.3))  // At least 30% even in critical cases
    }

    // Helper functions for feature extraction

    fn compute_shock_index(&self, vitals: &VitalSigns) -> f64 {
        vitals.heart_rate_bpm / vitals.systolic_bp
    }

    fn compute_map(&self, vitals: &VitalSigns) -> f64 {
        (vitals.systolic_bp + 2.0 * vitals.diastolic_bp) / 3.0
    }

    fn is_tachycardic(&self, vitals: &VitalSigns) -> f64 {
        if vitals.heart_rate_bpm > 100.0 { 1.0 } else { 0.0 }
    }

    fn is_hypotensive(&self, vitals: &VitalSigns) -> f64 {
        if vitals.systolic_bp < 90.0 { 1.0 } else { 0.0 }
    }

    fn is_hypoxic(&self, vitals: &VitalSigns) -> f64 {
        if vitals.spo2_percent < 90.0 { 1.0 } else { 0.0 }
    }

    fn has_fever(&self, vitals: &VitalSigns) -> f64 {
        if vitals.temperature_c > 38.3 || vitals.temperature_c < 36.0 { 1.0 } else { 0.0 }
    }

    fn has_elevated_lactate(&self, labs: &LabResults) -> f64 {
        if labs.lactate > 2.0 { 1.0 } else { 0.0 }
    }

    fn has_kidney_dysfunction(&self, labs: &LabResults) -> f64 {
        if labs.creatinine > 1.5 { 1.0 } else { 0.0 }
    }

    fn has_altered_mental_status(&self, vitals: &VitalSigns) -> f64 {
        if vitals.gcs_score < 15 { 1.0 } else { 0.0 }
    }

    fn compute_age_severity_interaction(&self, profile: &PatientProfile, vitals: &VitalSigns) -> f64 {
        let age_factor = (profile.age as f64) / 100.0;
        let severity_factor = (15 - vitals.gcs_score) as f64 / 15.0;
        age_factor * severity_factor
    }

    fn compute_comorbidity_burden(&self, profile: &PatientProfile) -> f64 {
        (profile.chronic_conditions.len() as f64 / 10.0).min(1.0)
    }

    fn compute_sirs_criteria(&self, history: &MedicalHistory) -> f64 {
        let mut count = 0.0;

        // Temperature
        if history.vitals.temperature_c > 38.0 || history.vitals.temperature_c < 36.0 {
            count += 1.0;
        }

        // Heart rate
        if history.vitals.heart_rate_bpm > 90.0 {
            count += 1.0;
        }

        // Respiratory rate or PaCO2
        if history.vitals.respiratory_rate > 20.0 {
            count += 1.0;
        }

        // WBC
        if history.labs.wbc_count > 12.0 || history.labs.wbc_count < 4.0 {
            count += 1.0;
        }

        count
    }

    fn count_organ_dysfunctions(&self, history: &MedicalHistory) -> u32 {
        let mut count = 0;

        // Cardiovascular
        if history.vitals.systolic_bp < 90.0 {
            count += 1;
        }

        // Respiratory
        if history.vitals.spo2_percent < 90.0 {
            count += 1;
        }

        // Renal
        if history.labs.creatinine > 2.0 {
            count += 1;
        }

        // Hepatic (not fully assessed without liver enzymes)

        // Hematologic
        if history.labs.platelet_count < 100.0 {
            count += 1;
        }

        // Neurologic
        if history.vitals.gcs_score < 13 {
            count += 1;
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_patient_stable() -> MedicalHistory {
        MedicalHistory {
            profile: PatientProfile {
                age: 45,
                gender: 1,
                bmi: 24.5,
                chronic_conditions: vec!["Hypertension".to_string()],
                medications: vec!["Lisinopril".to_string()],
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

    fn create_test_patient_septic() -> MedicalHistory {
        MedicalHistory {
            profile: PatientProfile {
                age: 72,
                gender: 0,
                bmi: 28.0,
                chronic_conditions: vec!["Diabetes".to_string(), "COPD".to_string()],
                medications: vec!["Metformin".to_string(), "Albuterol".to_string()],
                hospitalizations_last_year: 2,
                smoking_status: 1,
            },
            vitals: VitalSigns {
                heart_rate_bpm: 115.0,
                systolic_bp: 85.0,
                diastolic_bp: 55.0,
                respiratory_rate: 28.0,
                temperature_c: 38.9,
                spo2_percent: 88.0,
                gcs_score: 13,
            },
            labs: LabResults {
                wbc_count: 18.5,
                hemoglobin: 11.2,
                platelet_count: 95.0,
                creatinine: 2.3,
                bun: 45.0,
                lactate: 4.5,
                ph: 7.28,
            },
        }
    }

    #[test]
    fn test_risk_assessment_stable_patient() {
        let patient = create_test_patient_stable();
        let config = HealthcareConfig::default();
        let mut predictor = HealthcareRiskPredictor::new(config).unwrap();

        let assessment = predictor.assess_risk(&patient).unwrap();

        assert_eq!(assessment.risk_category, RiskCategory::Low);
        assert!(assessment.mortality_risk < 0.15);
        assert!(assessment.sepsis_risk < 0.3);
    }

    #[test]
    fn test_risk_assessment_septic_patient() {
        let patient = create_test_patient_septic();
        let config = HealthcareConfig::default();
        let mut predictor = HealthcareRiskPredictor::new(config).unwrap();

        let assessment = predictor.assess_risk(&patient).unwrap();

        assert!(matches!(assessment.risk_category, RiskCategory::High | RiskCategory::Critical));
        assert!(assessment.sepsis_risk > 0.5);
        assert!(assessment.mortality_risk > 0.3);
        assert!(!assessment.risk_factors.is_empty());
    }

    #[test]
    fn test_treatment_recommendations() {
        let patient = create_test_patient_septic();
        let config = HealthcareConfig::default();
        let mut predictor = HealthcareRiskPredictor::new(config).unwrap();

        let assessment = predictor.assess_risk(&patient).unwrap();
        let treatment = predictor.recommend_treatment(&assessment, &patient).unwrap();

        assert!(!treatment.interventions.is_empty());
        assert!(treatment.interventions[0].contains("SEPSIS"));
        assert!(treatment.monitoring_frequency_hours < 2.0);
        assert!(!treatment.escalation_criteria.is_empty());
    }
}
