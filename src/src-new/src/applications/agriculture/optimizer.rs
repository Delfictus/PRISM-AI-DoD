//! Agriculture Optimizer
//!
//! Implements GPU-accelerated precision agriculture optimization including:
//! - Crop yield prediction using environmental factors
//! - Irrigation scheduling with water efficiency
//! - Fertilizer optimization (NPK ratios)
//! - Multi-objective optimization (yield, cost, sustainability)
//!
//! Worker 3 Implementation
//! Constitutional Compliance: Articles I, II, III, IV

use anyhow::{Result, Context};

#[cfg(feature = "cuda")]
use crate::gpu::GpuMemoryPool;

/// Crop types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CropType {
    Corn,
    Wheat,
    Soybeans,
    Cotton,
    Rice,
    Vegetables,
    Fruits,
}

/// Field definition
#[derive(Debug, Clone)]
pub struct Field {
    pub id: usize,
    pub area_hectares: f64,
    pub crop: CropType,
    pub soil_conditions: SoilConditions,
    pub elevation: f64,        // meters
    pub slope: f64,            // degrees
    pub latitude: f64,
    pub longitude: f64,
}

/// Soil conditions
#[derive(Debug, Clone)]
pub struct SoilConditions {
    pub ph: f64,                    // 0-14 scale
    pub nitrogen_ppm: f64,          // parts per million
    pub phosphorus_ppm: f64,
    pub potassium_ppm: f64,
    pub organic_matter: f64,        // percentage
    pub moisture: f64,              // percentage
    pub texture: SoilTexture,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoilTexture {
    Clay,
    SiltyClay,
    ClayLoam,
    SiltyClayLoam,
    Loam,
    SandyLoam,
    Sand,
}

/// Weather forecast
#[derive(Debug, Clone)]
pub struct WeatherForecast {
    pub days_ahead: usize,
    pub temperature_celsius: Vec<f64>,    // Daily average
    pub precipitation_mm: Vec<f64>,       // Daily total
    pub humidity: Vec<f64>,               // Daily average (0-1)
    pub solar_radiation: Vec<f64>,        // MJ/m²/day
    pub wind_speed: Vec<f64>,             // m/s
}

/// Irrigation schedule
#[derive(Debug, Clone)]
pub struct IrrigationSchedule {
    pub field_id: usize,
    pub total_water_mm: f64,
    pub applications: Vec<IrrigationApplication>,
    pub efficiency: f64,                   // 0-1 (amount utilized by crop)
}

#[derive(Debug, Clone)]
pub struct IrrigationApplication {
    pub day: usize,
    pub amount_mm: f64,
    pub method: IrrigationMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IrrigationMethod {
    Sprinkler,
    DripIrrigation,
    FloodIrrigation,
    CenterPivot,
}

/// Fertilizer plan
#[derive(Debug, Clone)]
pub struct FertilizerPlan {
    pub field_id: usize,
    pub nitrogen_kg_per_ha: f64,
    pub phosphorus_kg_per_ha: f64,
    pub potassium_kg_per_ha: f64,
    pub applications: Vec<FertilizerApplication>,
    pub cost_per_ha: f64,
}

#[derive(Debug, Clone)]
pub struct FertilizerApplication {
    pub day: usize,
    pub npk_ratio: (f64, f64, f64),       // N-P-K
    pub amount_kg_per_ha: f64,
}

/// Yield prediction
#[derive(Debug, Clone)]
pub struct YieldPrediction {
    pub field_id: usize,
    pub predicted_yield_kg_per_ha: f64,
    pub confidence: f64,                   // 0-1
    pub factors: YieldFactors,
}

#[derive(Debug, Clone)]
pub struct YieldFactors {
    pub soil_quality: f64,                 // 0-1 contribution
    pub water_availability: f64,
    pub nutrient_balance: f64,
    pub climate_suitability: f64,
    pub pest_pressure: f64,
}

/// Optimization objective
#[derive(Debug, Clone, Copy)]
pub enum OptimizationObjective {
    MaximizeYield,              // Maximize crop yield
    MinimizeCost,               // Minimize input costs
    MinimizeWaterUse,           // Water conservation
    MaximizeSustainability,     // Environmental impact
    Balanced,                   // Balance all objectives
}

/// Agriculture configuration
#[derive(Debug, Clone)]
pub struct AgricultureConfig {
    pub planning_horizon_days: usize,
    pub water_price_per_mm_per_ha: f64,
    pub fertilizer_price_per_kg: f64,
    pub yield_weight: f64,
    pub cost_weight: f64,
    pub sustainability_weight: f64,
}

impl Default for AgricultureConfig {
    fn default() -> Self {
        Self {
            planning_horizon_days: 120,        // Growing season
            water_price_per_mm_per_ha: 5.0,    // $5 per mm per hectare
            fertilizer_price_per_kg: 2.0,      // $2 per kg
            yield_weight: 0.5,
            cost_weight: 0.3,
            sustainability_weight: 0.2,
        }
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub irrigation_schedules: Vec<IrrigationSchedule>,
    pub fertilizer_plans: Vec<FertilizerPlan>,
    pub yield_predictions: Vec<YieldPrediction>,
    pub total_cost: f64,
    pub total_yield_kg: f64,
    pub water_use_efficiency: f64,
    pub carbon_footprint: f64,             // kg CO2 equivalent
}

/// Agriculture optimizer
pub struct AgricultureOptimizer {
    config: AgricultureConfig,

    #[cfg(feature = "cuda")]
    gpu_context: Option<GpuMemoryPool>,
}

impl AgricultureOptimizer {
    /// Create new agriculture optimizer
    pub fn new(config: AgricultureConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_context = Some(GpuMemoryPool::new()
            .context("Failed to initialize GPU for agriculture optimization")?);

        Ok(Self {
            config,
            #[cfg(feature = "cuda")]
            gpu_context,
        })
    }

    /// Optimize agricultural operations
    pub fn optimize(
        &mut self,
        fields: &[Field],
        weather: &WeatherForecast,
        objective: OptimizationObjective,
    ) -> Result<OptimizationResult> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.optimize_gpu(fields, weather, objective);
            }
        }

        self.optimize_cpu(fields, weather, objective)
    }

    /// CPU-based optimization
    fn optimize_cpu(
        &self,
        fields: &[Field],
        weather: &WeatherForecast,
        objective: OptimizationObjective,
    ) -> Result<OptimizationResult> {
        let mut irrigation_schedules = Vec::new();
        let mut fertilizer_plans = Vec::new();
        let mut yield_predictions = Vec::new();
        let mut total_cost = 0.0;
        let mut total_yield_kg = 0.0;
        let mut total_water_mm = 0.0;
        let mut total_area = 0.0;

        for field in fields {
            // Calculate water requirements
            let water_requirement = self.calculate_water_requirement(field, weather);

            // Create irrigation schedule
            let irrigation = self.create_irrigation_schedule(
                field,
                weather,
                water_requirement,
                objective,
            );

            // Calculate fertilizer needs
            let fertilizer = self.create_fertilizer_plan(field, objective);

            // Predict yield
            let yield_pred = self.predict_yield(field, &irrigation, &fertilizer, weather);

            // Update totals
            total_cost += irrigation.total_water_mm * field.area_hectares *
                          self.config.water_price_per_mm_per_ha;
            total_cost += fertilizer.cost_per_ha * field.area_hectares;
            total_yield_kg += yield_pred.predicted_yield_kg_per_ha * field.area_hectares;
            total_water_mm += irrigation.total_water_mm * field.area_hectares;
            total_area += field.area_hectares;

            irrigation_schedules.push(irrigation);
            fertilizer_plans.push(fertilizer);
            yield_predictions.push(yield_pred);
        }

        let water_use_efficiency = if total_water_mm > 0.0 {
            total_yield_kg / total_water_mm
        } else {
            0.0
        };

        // Estimate carbon footprint (simplified)
        let carbon_footprint = fertilizer_plans.iter()
            .map(|f| {
                (f.nitrogen_kg_per_ha * 2.5 +  // N has high carbon footprint
                 f.phosphorus_kg_per_ha * 0.5 +
                 f.potassium_kg_per_ha * 0.3) * fields.iter()
                    .find(|field| field.id == f.field_id)
                    .map(|field| field.area_hectares)
                    .unwrap_or(0.0)
            })
            .sum();

        Ok(OptimizationResult {
            irrigation_schedules,
            fertilizer_plans,
            yield_predictions,
            total_cost,
            total_yield_kg,
            water_use_efficiency,
            carbon_footprint,
        })
    }

    #[cfg(feature = "cuda")]
    fn optimize_gpu(
        &self,
        fields: &[Field],
        weather: &WeatherForecast,
        objective: OptimizationObjective,
    ) -> Result<OptimizationResult> {
        // TODO: Request agriculture_optimization_kernel from Worker 2
        // __global__ void agriculture_optimization(
        //     Field* fields,
        //     WeatherForecast* weather,
        //     OptimizationResult* results,
        //     int num_fields,
        //     OptimizationObjective objective
        // )

        // Placeholder: use CPU implementation
        self.optimize_cpu(fields, weather, objective)
    }

    /// Calculate water requirement for a field
    fn calculate_water_requirement(
        &self,
        field: &Field,
        weather: &WeatherForecast,
    ) -> f64 {
        // Simplified FAO Penman-Monteith evapotranspiration
        let mut total_et = 0.0;

        for day in 0..weather.days_ahead.min(self.config.planning_horizon_days) {
            let temp = weather.temperature_celsius[day];
            let humidity = weather.humidity[day];
            let radiation = weather.solar_radiation[day];
            let wind = weather.wind_speed[day];

            // Simplified ET calculation (mm/day)
            let et0 = 0.408 * radiation * (temp + 273.0) / (temp + 273.0 + 237.3) *
                     (1.0 + 0.34 * wind) * (1.0 - 0.01 * humidity * 100.0);

            // Crop coefficient (varies by growth stage and crop type)
            let kc = self.get_crop_coefficient(field.crop, day);

            total_et += et0 * kc;
        }

        // Adjust for rainfall
        let total_rainfall: f64 = weather.precipitation_mm.iter()
            .take(self.config.planning_horizon_days)
            .sum();

        // Adjust for soil moisture and efficiency
        let soil_water = field.soil_conditions.moisture * 100.0;  // mm
        let required_water = (total_et - total_rainfall - soil_water).max(0.0);

        required_water
    }

    /// Get crop coefficient for growth stage
    fn get_crop_coefficient(&self, crop: CropType, day: usize) -> f64 {
        // Simplified crop coefficients (actual values vary by growth stage)
        match crop {
            CropType::Corn => {
                if day < 30 { 0.4 }
                else if day < 60 { 0.8 }
                else if day < 90 { 1.2 }
                else { 0.6 }
            }
            CropType::Wheat => {
                if day < 20 { 0.3 }
                else if day < 50 { 0.7 }
                else if day < 80 { 1.15 }
                else { 0.4 }
            }
            CropType::Soybeans => {
                if day < 20 { 0.4 }
                else if day < 60 { 1.0 }
                else if day < 90 { 1.15 }
                else { 0.5 }
            }
            CropType::Cotton => {
                if day < 30 { 0.35 }
                else if day < 70 { 1.0 }
                else if day < 100 { 1.25 }
                else { 0.7 }
            }
            CropType::Rice => 1.05,  // Flooded conditions
            CropType::Vegetables => 0.7,
            CropType::Fruits => 0.95,
        }
    }

    /// Create irrigation schedule
    fn create_irrigation_schedule(
        &self,
        field: &Field,
        weather: &WeatherForecast,
        total_requirement: f64,
        objective: OptimizationObjective,
    ) -> IrrigationSchedule {
        let mut applications = Vec::new();

        // Determine irrigation method based on field characteristics
        let method = match field.soil_conditions.texture {
            SoilTexture::Sand | SoilTexture::SandyLoam => IrrigationMethod::DripIrrigation,
            SoilTexture::Clay | SoilTexture::SiltyClay => IrrigationMethod::Sprinkler,
            SoilTexture::Loam | SoilTexture::ClayLoam => IrrigationMethod::CenterPivot,
            _ => IrrigationMethod::Sprinkler,
        };

        // Efficiency varies by method
        let efficiency = match method {
            IrrigationMethod::DripIrrigation => 0.9,
            IrrigationMethod::Sprinkler => 0.75,
            IrrigationMethod::CenterPivot => 0.85,
            IrrigationMethod::FloodIrrigation => 0.6,
        };

        // Schedule applications based on objective
        let application_frequency = match objective {
            OptimizationObjective::MinimizeWaterUse => 14,    // Every 2 weeks
            OptimizationObjective::MaximizeYield => 7,        // Weekly
            _ => 10,                                          // Every 10 days
        };

        let num_applications = (self.config.planning_horizon_days / application_frequency).max(1);
        let amount_per_application = total_requirement / efficiency / num_applications as f64;

        for i in 0..num_applications {
            applications.push(IrrigationApplication {
                day: i * application_frequency,
                amount_mm: amount_per_application,
                method,
            });
        }

        IrrigationSchedule {
            field_id: field.id,
            total_water_mm: total_requirement / efficiency,
            applications,
            efficiency,
        }
    }

    /// Create fertilizer plan
    fn create_fertilizer_plan(
        &self,
        field: &Field,
        objective: OptimizationObjective,
    ) -> FertilizerPlan {
        // Calculate NPK requirements based on soil test and crop needs
        let base_n = self.get_nitrogen_requirement(field.crop);
        let base_p = self.get_phosphorus_requirement(field.crop);
        let base_k = self.get_potassium_requirement(field.crop);

        // Adjust based on soil levels
        let n_needed = (base_n - field.soil_conditions.nitrogen_ppm / 10.0).max(0.0);
        let p_needed = (base_p - field.soil_conditions.phosphorus_ppm / 10.0).max(0.0);
        let k_needed = (base_k - field.soil_conditions.potassium_ppm / 10.0).max(0.0);

        // Adjust based on objective
        let (n_applied, p_applied, k_applied) = match objective {
            OptimizationObjective::MinimizeCost => {
                (n_needed * 0.8, p_needed * 0.8, k_needed * 0.8)
            }
            OptimizationObjective::MaximizeYield => {
                (n_needed * 1.2, p_needed * 1.1, k_needed * 1.1)
            }
            OptimizationObjective::MaximizeSustainability => {
                (n_needed * 0.9, p_needed * 1.0, k_needed * 1.0)
            }
            _ => (n_needed, p_needed, k_needed),
        };

        // Split into multiple applications
        let mut applications = Vec::new();

        // Pre-plant application
        applications.push(FertilizerApplication {
            day: 0,
            npk_ratio: (0.5, 1.0, 1.0),  // All P and K, half N
            amount_kg_per_ha: (n_applied * 0.5 + p_applied + k_applied),
        });

        // Side-dress application (mid-season)
        applications.push(FertilizerApplication {
            day: 40,
            npk_ratio: (1.0, 0.0, 0.0),  // Remaining N
            amount_kg_per_ha: n_applied * 0.5,
        });

        let total_cost = (n_applied + p_applied + k_applied) * self.config.fertilizer_price_per_kg;

        FertilizerPlan {
            field_id: field.id,
            nitrogen_kg_per_ha: n_applied,
            phosphorus_kg_per_ha: p_applied,
            potassium_kg_per_ha: k_applied,
            applications,
            cost_per_ha: total_cost,
        }
    }

    /// Get nitrogen requirement for crop
    fn get_nitrogen_requirement(&self, crop: CropType) -> f64 {
        match crop {
            CropType::Corn => 180.0,         // kg/ha
            CropType::Wheat => 120.0,
            CropType::Soybeans => 40.0,      // Nitrogen-fixing
            CropType::Cotton => 150.0,
            CropType::Rice => 100.0,
            CropType::Vegetables => 160.0,
            CropType::Fruits => 100.0,
        }
    }

    /// Get phosphorus requirement for crop
    fn get_phosphorus_requirement(&self, crop: CropType) -> f64 {
        match crop {
            CropType::Corn => 60.0,
            CropType::Wheat => 40.0,
            CropType::Soybeans => 50.0,
            CropType::Cotton => 50.0,
            CropType::Rice => 30.0,
            CropType::Vegetables => 70.0,
            CropType::Fruits => 50.0,
        }
    }

    /// Get potassium requirement for crop
    fn get_potassium_requirement(&self, crop: CropType) -> f64 {
        match crop {
            CropType::Corn => 70.0,
            CropType::Wheat => 45.0,
            CropType::Soybeans => 60.0,
            CropType::Cotton => 55.0,
            CropType::Rice => 40.0,
            CropType::Vegetables => 80.0,
            CropType::Fruits => 90.0,
        }
    }

    /// Predict crop yield
    fn predict_yield(
        &self,
        field: &Field,
        irrigation: &IrrigationSchedule,
        fertilizer: &FertilizerPlan,
        weather: &WeatherForecast,
    ) -> YieldPrediction {
        // Baseline yields (kg/ha)
        let baseline_yield = match field.crop {
            CropType::Corn => 10000.0,
            CropType::Wheat => 5000.0,
            CropType::Soybeans => 3500.0,
            CropType::Cotton => 1200.0,
            CropType::Rice => 7000.0,
            CropType::Vegetables => 15000.0,
            CropType::Fruits => 20000.0,
        };

        // Soil quality factor (0-1)
        let soil_quality = self.calculate_soil_quality_factor(&field.soil_conditions);

        // Water availability factor (0-1)
        let water_factor = (irrigation.total_water_mm / 500.0).min(1.0) *
                          irrigation.efficiency;

        // Nutrient balance factor (0-1)
        let nutrient_factor = ((fertilizer.nitrogen_kg_per_ha / 200.0).min(1.0) +
                              (fertilizer.phosphorus_kg_per_ha / 80.0).min(1.0) +
                              (fertilizer.potassium_kg_per_ha / 100.0).min(1.0)) / 3.0;

        // Climate suitability (based on temperature)
        let avg_temp: f64 = weather.temperature_celsius.iter().sum::<f64>() /
                           weather.temperature_celsius.len() as f64;
        let climate_factor = if (18.0..=28.0).contains(&avg_temp) {
            1.0
        } else {
            0.8
        };

        // Pest pressure (simplified - assumed low for now)
        let pest_factor = 0.95;

        // Combined yield prediction
        let yield_multiplier = soil_quality * water_factor * nutrient_factor *
                              climate_factor * pest_factor;

        let predicted_yield = baseline_yield * yield_multiplier;

        // Confidence based on data quality
        let confidence = (soil_quality + water_factor + nutrient_factor) / 3.0;

        YieldPrediction {
            field_id: field.id,
            predicted_yield_kg_per_ha: predicted_yield,
            confidence,
            factors: YieldFactors {
                soil_quality,
                water_availability: water_factor,
                nutrient_balance: nutrient_factor,
                climate_suitability: climate_factor,
                pest_pressure: pest_factor,
            },
        }
    }

    /// Calculate soil quality factor
    fn calculate_soil_quality_factor(&self, soil: &SoilConditions) -> f64 {
        // pH factor (ideal: 6.0-7.0)
        let ph_factor = if (6.0..=7.0).contains(&soil.ph) {
            1.0
        } else {
            0.8
        };

        // Organic matter factor (ideal: >2%)
        let om_factor = (soil.organic_matter / 3.0).min(1.0);

        // Overall soil quality
        (ph_factor + om_factor) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agriculture_optimizer_creation() {
        let config = AgricultureConfig::default();
        let optimizer = AgricultureOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_crop_yield_prediction() {
        let field = Field {
            id: 0,
            area_hectares: 10.0,
            crop: CropType::Corn,
            soil_conditions: SoilConditions {
                ph: 6.5,
                nitrogen_ppm: 50.0,
                phosphorus_ppm: 30.0,
                potassium_ppm: 150.0,
                organic_matter: 2.5,
                moisture: 0.3,
                texture: SoilTexture::Loam,
            },
            elevation: 200.0,
            slope: 2.0,
            latitude: 40.0,
            longitude: -95.0,
        };

        let weather = WeatherForecast {
            days_ahead: 120,
            temperature_celsius: vec![22.0; 120],
            precipitation_mm: vec![2.0; 120],
            humidity: vec![0.6; 120],
            solar_radiation: vec![20.0; 120],
            wind_speed: vec![3.0; 120],
        };

        let config = AgricultureConfig::default();
        let mut optimizer = AgricultureOptimizer::new(config).unwrap();

        let result = optimizer.optimize(
            &[field],
            &weather,
            OptimizationObjective::Balanced,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.yield_predictions.len(), 1);
        assert!(result.yield_predictions[0].predicted_yield_kg_per_ha > 0.0);
        assert!(result.total_cost > 0.0);
    }

    #[test]
    fn test_irrigation_scheduling() {
        let field = Field {
            id: 0,
            area_hectares: 5.0,
            crop: CropType::Wheat,
            soil_conditions: SoilConditions {
                ph: 6.8,
                nitrogen_ppm: 40.0,
                phosphorus_ppm: 25.0,
                potassium_ppm: 120.0,
                organic_matter: 2.0,
                moisture: 0.25,
                texture: SoilTexture::ClayLoam,
            },
            elevation: 150.0,
            slope: 1.0,
            latitude: 35.0,
            longitude: -100.0,
        };

        let weather = WeatherForecast {
            days_ahead: 90,
            temperature_celsius: vec![18.0; 90],
            precipitation_mm: vec![1.5; 90],
            humidity: vec![0.5; 90],
            solar_radiation: vec![18.0; 90],
            wind_speed: vec![2.5; 90],
        };

        let config = AgricultureConfig::default();
        let mut optimizer = AgricultureOptimizer::new(config).unwrap();

        let result = optimizer.optimize(
            &[field],
            &weather,
            OptimizationObjective::MinimizeWaterUse,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.irrigation_schedules.len(), 1);
        assert!(result.irrigation_schedules[0].applications.len() > 0);
        assert!(result.water_use_efficiency > 0.0);
    }
}
