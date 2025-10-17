//! Agriculture Optimization Module
//!
//! GPU-accelerated precision agriculture optimization including:
//! - Crop yield prediction
//! - Irrigation scheduling
//! - Fertilizer optimization
//! - Pest management
//! - Resource allocation
//! - Climate adaptation
//!
//! Worker 3 Implementation
//! Constitutional Compliance: Articles I, II, III, IV

pub mod optimizer;

// Re-export main types
pub use optimizer::{
    AgricultureOptimizer,
    Field,
    CropType,
    SoilConditions,
    SoilTexture,
    WeatherForecast,
    IrrigationSchedule,
    IrrigationApplication,
    IrrigationMethod,
    FertilizerPlan,
    FertilizerApplication,
    YieldPrediction,
    YieldFactors,
    AgricultureConfig,
    OptimizationObjective,
    OptimizationResult,
};
