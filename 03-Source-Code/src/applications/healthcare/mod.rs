//! Healthcare Risk Prediction Module
//!
//! GPU-accelerated patient risk assessment and clinical decision support:
//! - Patient risk scoring (mortality, readmission, adverse events)
//! - Disease progression prediction
//! - Treatment outcome prediction
//! - Early warning systems (sepsis, deterioration)
//! - Active Inference for adaptive clinical recommendations
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated risk models
//! - Article III: Comprehensive testing with synthetic patient data
//! - Article IV: Active Inference for treatment optimization

pub mod risk_predictor;

// Re-export main types
pub use risk_predictor::{
    HealthcareRiskPredictor,
    PatientProfile,
    VitalSigns,
    LabResults,
    MedicalHistory,
    RiskAssessment,
    RiskCategory,
    TreatmentRecommendation,
    HealthcareConfig,
};
