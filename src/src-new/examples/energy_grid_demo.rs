//! Energy Grid Optimization Demo
//!
//! Demonstrates GPU-accelerated power grid management with renewable
//! energy integration and multi-objective optimization.
//!
//! Run with: cargo run --example energy_grid_demo --features cuda

use prism_ai::applications::energy_grid::{
    EnergyGridOptimizer, PowerGrid, Generator, GeneratorType, Bus, BusType,
    TransmissionLine, Load, OptimizationObjective, GridConfig,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Energy Grid Optimization Demo ===\n");

    // Create a realistic regional power grid
    let mut grid = create_regional_grid();

    println!("Grid Configuration:");
    println!("  Generators: {} ({} renewable)",
        grid.generators.len(),
        grid.generators.iter().filter(|g| matches!(
            g.generator_type,
            GeneratorType::Solar | GeneratorType::Wind | GeneratorType::Hydro
        )).count()
    );
    println!("  Buses: {}", grid.buses.len());
    println!("  Transmission Lines: {}", grid.lines.len());
    println!("  Total Load: {:.1} MW\n",
        grid.loads.iter().map(|l| l.p_demand_mw).sum::<f64>()
    );

    // Initialize GPU-accelerated optimizer
    println!("Initializing GPU-accelerated grid optimizer...");
    let config = GridConfig::default();
    let mut optimizer = EnergyGridOptimizer::new(config)?;
    println!("✓ GPU initialization successful\n");

    // Objective 1: Minimize Operating Cost
    println!("--- Objective 1: Minimize Operating Cost ---");
    let result = optimizer.optimize(&mut grid, OptimizationObjective::MinimizeCost)?;
    print_optimization_result(&result, "Cost Minimization", &grid);

    // Objective 2: Minimize CO2 Emissions
    println!("\n--- Objective 2: Minimize CO2 Emissions ---");
    let result = optimizer.optimize(&mut grid, OptimizationObjective::MinimizeEmissions)?;
    print_optimization_result(&result, "Emissions Minimization", &grid);

    // Objective 3: Maximize Renewable Penetration
    println!("\n--- Objective 3: Maximize Renewable Penetration ---");
    let result = optimizer.optimize(&mut grid, OptimizationObjective::MaximizeRenewables)?;
    print_optimization_result(&result, "Renewable Maximization", &grid);

    // Objective 4: Balanced Multi-Objective
    println!("\n--- Objective 4: Balanced Multi-Objective ---");
    let result = optimizer.optimize(
        &mut grid,
        OptimizationObjective::Balanced {
            cost_weight: 0.4,
            emissions_weight: 0.4,
            reliability_weight: 0.2,
        },
    )?;
    print_optimization_result(&result, "Balanced", &grid);

    // Print generator dispatch details
    println!("\n=== Generator Dispatch (Balanced Strategy) ===");
    for (gen_id, output) in result.dispatch.iter() {
        let gen = &grid.generators[*gen_id];
        println!("{:?} Gen #{}: {:.1} MW ({:.1}% capacity)",
            gen.generator_type,
            gen.id,
            output,
            (output / gen.max_capacity_mw) * 100.0
        );
    }

    println!("\n=== Strategy Comparison ===");
    println!("Cost Minimization: Best for budget-constrained operations");
    println!("Emissions Minimization: Best for environmental compliance");
    println!("Renewable Maximization: Best for clean energy goals");
    println!("Balanced: Best for general grid management with multiple objectives");

    println!("\n✓ Energy grid optimization complete!");

    Ok(())
}

fn create_regional_grid() -> PowerGrid {
    // Mix of conventional and renewable generators
    let generators = vec![
        // Coal plant (baseload)
        Generator {
            id: 0,
            generator_type: GeneratorType::Coal,
            bus_id: 0,
            max_capacity_mw: 600.0,
            min_generation_mw: 200.0,
            current_output_mw: 400.0,
            marginal_cost: 35.0,
            startup_cost: 50000.0,
            ramp_rate_mw_per_hour: 60.0,
            co2_emissions_kg_per_mwh: 950.0,
        },
        // Natural gas (load following)
        Generator {
            id: 1,
            generator_type: GeneratorType::NaturalGas,
            bus_id: 1,
            max_capacity_mw: 400.0,
            min_generation_mw: 80.0,
            current_output_mw: 200.0,
            marginal_cost: 45.0,
            startup_cost: 10000.0,
            ramp_rate_mw_per_hour: 150.0,
            co2_emissions_kg_per_mwh: 400.0,
        },
        // Nuclear (baseload)
        Generator {
            id: 2,
            generator_type: GeneratorType::Nuclear,
            bus_id: 2,
            max_capacity_mw: 1000.0,
            min_generation_mw: 800.0,
            current_output_mw: 950.0,
            marginal_cost: 10.0,
            startup_cost: 100000.0,
            ramp_rate_mw_per_hour: 20.0,
            co2_emissions_kg_per_mwh: 0.0,
        },
        // Hydro (flexible)
        Generator {
            id: 3,
            generator_type: GeneratorType::Hydro,
            bus_id: 3,
            max_capacity_mw: 300.0,
            min_generation_mw: 0.0,
            current_output_mw: 150.0,
            marginal_cost: 5.0,
            startup_cost: 0.0,
            ramp_rate_mw_per_hour: 300.0,
            co2_emissions_kg_per_mwh: 0.0,
        },
        // Solar farm
        Generator {
            id: 4,
            generator_type: GeneratorType::Solar,
            bus_id: 4,
            max_capacity_mw: 250.0,
            min_generation_mw: 0.0,
            current_output_mw: 200.0, // Midday peak
            marginal_cost: 0.0,
            startup_cost: 0.0,
            ramp_rate_mw_per_hour: 250.0,
            co2_emissions_kg_per_mwh: 0.0,
        },
        // Wind farm
        Generator {
            id: 5,
            generator_type: GeneratorType::Wind,
            bus_id: 5,
            max_capacity_mw: 350.0,
            min_generation_mw: 0.0,
            current_output_mw: 280.0, // Good wind conditions
            marginal_cost: 0.0,
            startup_cost: 0.0,
            ramp_rate_mw_per_hour: 350.0,
            co2_emissions_kg_per_mwh: 0.0,
        },
        // Peaker plant (expensive, fast ramp)
        Generator {
            id: 6,
            generator_type: GeneratorType::NaturalGas,
            bus_id: 6,
            max_capacity_mw: 200.0,
            min_generation_mw: 0.0,
            current_output_mw: 0.0,
            marginal_cost: 80.0,
            startup_cost: 2000.0,
            ramp_rate_mw_per_hour: 200.0,
            co2_emissions_kg_per_mwh: 500.0,
        },
    ];

    // 7 buses (matching generators for simplicity)
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
        Bus {
            id: 3,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            bus_type: BusType::PV,
            p_injection: 0.0,
            q_injection: 0.0,
        },
        Bus {
            id: 4,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            bus_type: BusType::PV,
            p_injection: 0.0,
            q_injection: 0.0,
        },
        Bus {
            id: 5,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            bus_type: BusType::PV,
            p_injection: 0.0,
            q_injection: 0.0,
        },
        Bus {
            id: 6,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            bus_type: BusType::PV,
            p_injection: 0.0,
            q_injection: 0.0,
        },
    ];

    // Transmission network
    let lines = vec![
        TransmissionLine {
            id: 0,
            from_bus: 0,
            to_bus: 1,
            resistance: 0.01,
            reactance: 0.10,
            thermal_limit_mw: 500.0,
            current_flow_mw: 0.0,
        },
        TransmissionLine {
            id: 1,
            from_bus: 1,
            to_bus: 2,
            resistance: 0.015,
            reactance: 0.12,
            thermal_limit_mw: 600.0,
            current_flow_mw: 0.0,
        },
        TransmissionLine {
            id: 2,
            from_bus: 2,
            to_bus: 3,
            resistance: 0.01,
            reactance: 0.10,
            thermal_limit_mw: 400.0,
            current_flow_mw: 0.0,
        },
        TransmissionLine {
            id: 3,
            from_bus: 3,
            to_bus: 4,
            resistance: 0.02,
            reactance: 0.15,
            thermal_limit_mw: 350.0,
            current_flow_mw: 0.0,
        },
        TransmissionLine {
            id: 4,
            from_bus: 4,
            to_bus: 5,
            resistance: 0.015,
            reactance: 0.12,
            thermal_limit_mw: 400.0,
            current_flow_mw: 0.0,
        },
        TransmissionLine {
            id: 5,
            from_bus: 5,
            to_bus: 6,
            resistance: 0.01,
            reactance: 0.10,
            thermal_limit_mw: 300.0,
            current_flow_mw: 0.0,
        },
        // Additional lines for redundancy
        TransmissionLine {
            id: 6,
            from_bus: 0,
            to_bus: 3,
            resistance: 0.025,
            reactance: 0.20,
            thermal_limit_mw: 300.0,
            current_flow_mw: 0.0,
        },
    ];

    // Load distribution
    let loads = vec![
        Load {
            id: 0,
            bus_id: 1,
            p_demand_mw: 400.0,
            q_demand_mvar: 120.0,
            priority: 10,
            dr_capability: 0.05,
        },
        Load {
            id: 1,
            bus_id: 2,
            p_demand_mw: 600.0,
            q_demand_mvar: 180.0,
            priority: 10,
            dr_capability: 0.05,
        },
        Load {
            id: 2,
            bus_id: 3,
            p_demand_mw: 300.0,
            q_demand_mvar: 90.0,
            priority: 8,
            dr_capability: 0.10,
        },
        Load {
            id: 3,
            bus_id: 4,
            p_demand_mw: 350.0,
            q_demand_mvar: 105.0,
            priority: 7,
            dr_capability: 0.15,
        },
        Load {
            id: 4,
            bus_id: 5,
            p_demand_mw: 450.0,
            q_demand_mvar: 135.0,
            priority: 9,
            dr_capability: 0.08,
        },
        Load {
            id: 5,
            bus_id: 6,
            p_demand_mw: 250.0,
            q_demand_mvar: 75.0,
            priority: 6,
            dr_capability: 0.20,
        },
    ];

    PowerGrid::new(generators, buses, lines, loads)
}

fn print_optimization_result(
    result: &prism_ai::applications::energy_grid::OptimizationResult,
    strategy_name: &str,
    grid: &PowerGrid,
) {
    println!("Strategy: {}", strategy_name);
    println!("Grid Metrics:");
    println!("  Operating Cost: ${:.2}/hour", result.total_cost);
    println!("  CO2 Emissions: {:.1} tons/hour", result.total_emissions / 1000.0);
    println!("  Renewable Penetration: {:.1}%", result.renewable_fraction * 100.0);
    println!("  Load Served: {:.1} MW", result.load_served);
    println!("  System Losses: {:.1} MW ({:.1}%)",
        result.system_losses,
        (result.system_losses / result.load_served) * 100.0
    );

    // Calculate total generation
    let total_gen: f64 = result.dispatch.values().sum();
    println!("  Total Generation: {:.1} MW", total_gen);
}
