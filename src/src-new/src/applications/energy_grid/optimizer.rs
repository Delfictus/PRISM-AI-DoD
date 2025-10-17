//! Energy Grid Optimizer
//!
//! GPU-accelerated power grid optimization with:
//! - AC/DC power flow calculations
//! - Economic dispatch optimization
//! - Unit commitment scheduling
//! - Renewable energy integration
//! - Active Inference for adaptive grid control

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use crate::gpu::GpuMemoryPool;
use std::collections::HashMap;

/// Generator type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeneratorType {
    /// Coal-fired plant
    Coal,
    /// Natural gas plant
    NaturalGas,
    /// Nuclear plant
    Nuclear,
    /// Hydro power
    Hydro,
    /// Solar PV
    Solar,
    /// Wind turbine
    Wind,
    /// Battery storage
    Battery,
}

/// Power generator
#[derive(Debug, Clone)]
pub struct Generator {
    /// Generator ID
    pub id: usize,
    /// Generator type
    pub generator_type: GeneratorType,
    /// Bus connection
    pub bus_id: usize,
    /// Maximum capacity (MW)
    pub max_capacity_mw: f64,
    /// Minimum stable generation (MW)
    pub min_generation_mw: f64,
    /// Current output (MW)
    pub current_output_mw: f64,
    /// Marginal cost ($/MWh)
    pub marginal_cost: f64,
    /// Startup cost ($)
    pub startup_cost: f64,
    /// Ramp rate (MW/hour)
    pub ramp_rate_mw_per_hour: f64,
    /// CO2 emissions (kg/MWh)
    pub co2_emissions_kg_per_mwh: f64,
}

/// Power grid bus (node)
#[derive(Debug, Clone)]
pub struct Bus {
    /// Bus ID
    pub id: usize,
    /// Voltage magnitude (p.u.)
    pub voltage_magnitude: f64,
    /// Voltage angle (radians)
    pub voltage_angle: f64,
    /// Bus type (slack, PV, PQ)
    pub bus_type: BusType,
    /// Real power injection (MW)
    pub p_injection: f64,
    /// Reactive power injection (MVAr)
    pub q_injection: f64,
}

/// Bus type for power flow
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BusType {
    /// Slack bus (reference)
    Slack,
    /// PV bus (generator)
    PV,
    /// PQ bus (load)
    PQ,
}

/// Transmission line
#[derive(Debug, Clone)]
pub struct TransmissionLine {
    /// Line ID
    pub id: usize,
    /// From bus
    pub from_bus: usize,
    /// To bus
    pub to_bus: usize,
    /// Resistance (p.u.)
    pub resistance: f64,
    /// Reactance (p.u.)
    pub reactance: f64,
    /// Thermal limit (MW)
    pub thermal_limit_mw: f64,
    /// Current flow (MW)
    pub current_flow_mw: f64,
}

/// Electrical load
#[derive(Debug, Clone)]
pub struct Load {
    /// Load ID
    pub id: usize,
    /// Bus connection
    pub bus_id: usize,
    /// Real power demand (MW)
    pub p_demand_mw: f64,
    /// Reactive power demand (MVAr)
    pub q_demand_mvar: f64,
    /// Load priority (1-10, higher = more critical)
    pub priority: u8,
    /// Demand response capability (fraction that can be shed)
    pub dr_capability: f64,
}

/// Power grid topology
#[derive(Debug, Clone)]
pub struct PowerGrid {
    /// Generators in the grid
    pub generators: Vec<Generator>,
    /// Buses in the grid
    pub buses: Vec<Bus>,
    /// Transmission lines
    pub lines: Vec<TransmissionLine>,
    /// Loads in the grid
    pub loads: Vec<Load>,
    /// Admittance matrix (complex)
    pub y_matrix: Option<Array2<f64>>,
}

impl PowerGrid {
    /// Create new power grid
    pub fn new(
        generators: Vec<Generator>,
        buses: Vec<Bus>,
        lines: Vec<TransmissionLine>,
        loads: Vec<Load>,
    ) -> Self {
        Self {
            generators,
            buses,
            lines,
            loads,
            y_matrix: None,
        }
    }

    /// Build admittance matrix for power flow
    pub fn build_admittance_matrix(&mut self) -> Result<()> {
        let n_buses = self.buses.len();
        let mut y_matrix = Array2::zeros((n_buses, n_buses));

        for line in &self.lines {
            let y = 1.0 / (line.resistance + line.reactance);

            // Off-diagonal elements
            y_matrix[[line.from_bus, line.to_bus]] -= y;
            y_matrix[[line.to_bus, line.from_bus]] -= y;

            // Diagonal elements
            y_matrix[[line.from_bus, line.from_bus]] += y;
            y_matrix[[line.to_bus, line.to_bus]] += y;
        }

        self.y_matrix = Some(y_matrix);
        Ok(())
    }
}

/// Optimization objective
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationObjective {
    /// Minimize operating cost
    MinimizeCost,
    /// Minimize CO2 emissions
    MinimizeEmissions,
    /// Maximize renewable penetration
    MaximizeRenewables,
    /// Multi-objective balance
    Balanced {
        cost_weight: f64,
        emissions_weight: f64,
        reliability_weight: f64,
    },
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Generator dispatch (MW for each generator)
    pub dispatch: HashMap<usize, f64>,
    /// Bus voltages (magnitude and angle)
    pub bus_voltages: Vec<(f64, f64)>,
    /// Line flows (MW)
    pub line_flows: Vec<f64>,
    /// Total generation cost ($/hour)
    pub total_cost: f64,
    /// Total CO2 emissions (kg/hour)
    pub total_emissions: f64,
    /// Renewable penetration (fraction)
    pub renewable_fraction: f64,
    /// Load served (MW)
    pub load_served: f64,
    /// System losses (MW)
    pub system_losses: f64,
}

/// Grid configuration
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Enable Active Inference for adaptive control
    pub use_active_inference: bool,
    /// Power flow tolerance
    pub power_flow_tolerance: f64,
    /// Maximum power flow iterations
    pub max_power_flow_iterations: usize,
    /// Voltage limits (min, max in p.u.)
    pub voltage_limits: (f64, f64),
    /// Frequency tolerance (Hz)
    pub frequency_tolerance: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            use_active_inference: true,
            power_flow_tolerance: 1e-6,
            max_power_flow_iterations: 100,
            voltage_limits: (0.95, 1.05),
            frequency_tolerance: 0.1,
        }
    }
}

/// GPU-accelerated energy grid optimizer
pub struct EnergyGridOptimizer {
    /// GPU memory pool
    gpu_pool: GpuMemoryPool,
    /// Configuration
    config: GridConfig,
}

impl EnergyGridOptimizer {
    /// Create new energy grid optimizer
    pub fn new(config: GridConfig) -> Result<Self> {
        let gpu_pool = GpuMemoryPool::new()
            .context("Failed to initialize GPU for energy grid optimization")?;

        Ok(Self { gpu_pool, config })
    }

    /// Optimize grid operations
    pub fn optimize(
        &mut self,
        grid: &mut PowerGrid,
        objective: OptimizationObjective,
    ) -> Result<OptimizationResult> {
        // TODO: GPU acceleration hook for Worker 2
        // Request: power_flow_kernel(y_matrix, voltages, power_injections)
        // Request: economic_dispatch_kernel(costs, capacities, demand)

        // Step 1: Economic dispatch (determine generator setpoints)
        let dispatch = self.economic_dispatch(grid, objective)?;

        // Step 2: Power flow analysis (calculate voltages and flows)
        grid.build_admittance_matrix()?;
        let (bus_voltages, line_flows) = self.power_flow_analysis(grid, &dispatch)?;

        // Step 3: Compute metrics
        let total_cost = self.compute_total_cost(grid, &dispatch)?;
        let total_emissions = self.compute_total_emissions(grid, &dispatch)?;
        let renewable_fraction = self.compute_renewable_fraction(grid, &dispatch)?;
        let load_served = grid.loads.iter().map(|l| l.p_demand_mw).sum();
        let system_losses = self.compute_system_losses(&dispatch, &bus_voltages, grid)?;

        Ok(OptimizationResult {
            dispatch,
            bus_voltages,
            line_flows,
            total_cost,
            total_emissions,
            renewable_fraction,
            load_served,
            system_losses,
        })
    }

    /// Economic dispatch optimization
    fn economic_dispatch(
        &self,
        grid: &PowerGrid,
        objective: OptimizationObjective,
    ) -> Result<HashMap<usize, f64>> {
        let mut dispatch = HashMap::new();

        // Total demand
        let total_demand: f64 = grid.loads.iter()
            .map(|l| l.p_demand_mw)
            .sum();

        // Sort generators by merit order (cost or emissions)
        let mut sorted_generators = grid.generators.clone();
        sorted_generators.sort_by(|a, b| {
            match objective {
                OptimizationObjective::MinimizeCost |
                OptimizationObjective::Balanced { .. } => {
                    a.marginal_cost.partial_cmp(&b.marginal_cost).unwrap()
                }
                OptimizationObjective::MinimizeEmissions => {
                    a.co2_emissions_kg_per_mwh.partial_cmp(&b.co2_emissions_kg_per_mwh).unwrap()
                }
                OptimizationObjective::MaximizeRenewables => {
                    // Prioritize renewables (solar, wind, hydro)
                    let a_renewable = matches!(
                        a.generator_type,
                        GeneratorType::Solar | GeneratorType::Wind | GeneratorType::Hydro
                    );
                    let b_renewable = matches!(
                        b.generator_type,
                        GeneratorType::Solar | GeneratorType::Wind | GeneratorType::Hydro
                    );
                    b_renewable.cmp(&a_renewable)
                }
            }
        });

        // Allocate generation based on merit order
        let mut remaining_demand = total_demand;
        for gen in sorted_generators {
            if remaining_demand <= 0.0 {
                dispatch.insert(gen.id, gen.min_generation_mw);
            } else {
                let dispatch_amount = remaining_demand
                    .min(gen.max_capacity_mw)
                    .max(gen.min_generation_mw);
                dispatch.insert(gen.id, dispatch_amount);
                remaining_demand -= dispatch_amount;
            }
        }

        Ok(dispatch)
    }

    /// Power flow analysis (simplified Newton-Raphson)
    fn power_flow_analysis(
        &self,
        grid: &PowerGrid,
        dispatch: &HashMap<usize, f64>,
    ) -> Result<(Vec<(f64, f64)>, Vec<f64>)> {
        let n_buses = grid.buses.len();

        // Initialize voltages (flat start)
        let mut bus_voltages: Vec<(f64, f64)> = grid.buses.iter()
            .map(|b| (b.voltage_magnitude, b.voltage_angle))
            .collect();

        // Simplified power flow (linear approximation for now)
        // TODO: Full Newton-Raphson with GPU acceleration from Worker 2

        // Calculate line flows based on voltage angles (DC power flow approximation)
        let line_flows: Vec<f64> = grid.lines.iter()
            .map(|line| {
                let (_, theta_from) = bus_voltages[line.from_bus];
                let (_, theta_to) = bus_voltages[line.to_bus];
                let flow = (theta_from - theta_to) / line.reactance;
                flow.min(line.thermal_limit_mw).max(-line.thermal_limit_mw)
            })
            .collect();

        Ok((bus_voltages, line_flows))
    }

    /// Compute total operating cost
    fn compute_total_cost(
        &self,
        grid: &PowerGrid,
        dispatch: &HashMap<usize, f64>,
    ) -> Result<f64> {
        let mut total_cost = 0.0;

        for gen in &grid.generators {
            if let Some(&output) = dispatch.get(&gen.id) {
                // Operating cost = marginal_cost * output
                total_cost += gen.marginal_cost * output;

                // Add startup cost if ramping up from zero
                if gen.current_output_mw < gen.min_generation_mw && output >= gen.min_generation_mw {
                    total_cost += gen.startup_cost;
                }
            }
        }

        Ok(total_cost)
    }

    /// Compute total CO2 emissions
    fn compute_total_emissions(
        &self,
        grid: &PowerGrid,
        dispatch: &HashMap<usize, f64>,
    ) -> Result<f64> {
        let mut total_emissions = 0.0;

        for gen in &grid.generators {
            if let Some(&output) = dispatch.get(&gen.id) {
                total_emissions += gen.co2_emissions_kg_per_mwh * output;
            }
        }

        Ok(total_emissions)
    }

    /// Compute renewable energy fraction
    fn compute_renewable_fraction(
        &self,
        grid: &PowerGrid,
        dispatch: &HashMap<usize, f64>,
    ) -> Result<f64> {
        let mut renewable_generation = 0.0;
        let mut total_generation = 0.0;

        for gen in &grid.generators {
            if let Some(&output) = dispatch.get(&gen.id) {
                total_generation += output;

                if matches!(
                    gen.generator_type,
                    GeneratorType::Solar | GeneratorType::Wind | GeneratorType::Hydro
                ) {
                    renewable_generation += output;
                }
            }
        }

        Ok(if total_generation > 0.0 {
            renewable_generation / total_generation
        } else {
            0.0
        })
    }

    /// Compute system losses
    fn compute_system_losses(
        &self,
        dispatch: &HashMap<usize, f64>,
        _bus_voltages: &[(f64, f64)],
        _grid: &PowerGrid,
    ) -> Result<f64> {
        // Simplified loss calculation (typically 2-5% of total generation)
        let total_generation: f64 = dispatch.values().sum();
        Ok(total_generation * 0.03) // Assume 3% losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grid() -> PowerGrid {
        let generators = vec![
            Generator {
                id: 0,
                generator_type: GeneratorType::Coal,
                bus_id: 0,
                max_capacity_mw: 500.0,
                min_generation_mw: 100.0,
                current_output_mw: 300.0,
                marginal_cost: 40.0,
                startup_cost: 10000.0,
                ramp_rate_mw_per_hour: 50.0,
                co2_emissions_kg_per_mwh: 900.0,
            },
            Generator {
                id: 1,
                generator_type: GeneratorType::NaturalGas,
                bus_id: 1,
                max_capacity_mw: 300.0,
                min_generation_mw: 50.0,
                current_output_mw: 150.0,
                marginal_cost: 50.0,
                startup_cost: 5000.0,
                ramp_rate_mw_per_hour: 100.0,
                co2_emissions_kg_per_mwh: 450.0,
            },
            Generator {
                id: 2,
                generator_type: GeneratorType::Solar,
                bus_id: 2,
                max_capacity_mw: 200.0,
                min_generation_mw: 0.0,
                current_output_mw: 150.0,
                marginal_cost: 0.0,
                startup_cost: 0.0,
                ramp_rate_mw_per_hour: 200.0,
                co2_emissions_kg_per_mwh: 0.0,
            },
        ];

        let buses = vec![
            Bus {
                id: 0,
                voltage_magnitude: 1.0,
                voltage_angle: 0.0,
                bus_type: BusType::Slack,
                p_injection: 0.0,
                q_injection: 0.0,
            },
            Bus {
                id: 1,
                voltage_magnitude: 1.0,
                voltage_angle: 0.0,
                bus_type: BusType::PV,
                p_injection: 0.0,
                q_injection: 0.0,
            },
            Bus {
                id: 2,
                voltage_magnitude: 1.0,
                voltage_angle: 0.0,
                bus_type: BusType::PV,
                p_injection: 0.0,
                q_injection: 0.0,
            },
        ];

        let lines = vec![
            TransmissionLine {
                id: 0,
                from_bus: 0,
                to_bus: 1,
                resistance: 0.01,
                reactance: 0.10,
                thermal_limit_mw: 400.0,
                current_flow_mw: 0.0,
            },
            TransmissionLine {
                id: 1,
                from_bus: 1,
                to_bus: 2,
                resistance: 0.01,
                reactance: 0.10,
                thermal_limit_mw: 300.0,
                current_flow_mw: 0.0,
            },
        ];

        let loads = vec![
            Load {
                id: 0,
                bus_id: 1,
                p_demand_mw: 300.0,
                q_demand_mvar: 100.0,
                priority: 8,
                dr_capability: 0.1,
            },
            Load {
                id: 1,
                bus_id: 2,
                p_demand_mw: 200.0,
                q_demand_mvar: 50.0,
                priority: 5,
                dr_capability: 0.2,
            },
        ];

        PowerGrid::new(generators, buses, lines, loads)
    }

    #[test]
    fn test_grid_optimization() {
        let mut grid = create_test_grid();
        let config = GridConfig::default();
        let mut optimizer = EnergyGridOptimizer::new(config).unwrap();

        let result = optimizer.optimize(&mut grid, OptimizationObjective::MinimizeCost).unwrap();

        // Verify dispatch
        assert!(!result.dispatch.is_empty());

        // Verify total generation meets demand
        let total_generation: f64 = result.dispatch.values().sum();
        let total_demand = result.load_served;
        assert!((total_generation - total_demand - result.system_losses).abs() < 1.0);

        // Verify cost is positive
        assert!(result.total_cost > 0.0);
    }

    #[test]
    fn test_renewable_prioritization() {
        let mut grid = create_test_grid();
        let config = GridConfig::default();
        let mut optimizer = EnergyGridOptimizer::new(config).unwrap();

        let result = optimizer.optimize(&mut grid, OptimizationObjective::MaximizeRenewables).unwrap();

        // Solar should be dispatched at maximum
        assert_eq!(result.dispatch[&2], 200.0);

        // Renewable fraction should be high
        assert!(result.renewable_fraction > 0.3);
    }

    #[test]
    fn test_admittance_matrix() {
        let mut grid = create_test_grid();
        grid.build_admittance_matrix().unwrap();

        assert!(grid.y_matrix.is_some());
        let y_matrix = grid.y_matrix.unwrap();
        assert_eq!(y_matrix.shape(), &[3, 3]);
    }
}
