//! Manufacturing Process Optimization Demo
//!
//! Demonstrates GPU-accelerated manufacturing optimization with:
//! - Job shop scheduling
//! - Predictive maintenance
//! - Multi-objective optimization
//!
//! Run with: cargo run --example manufacturing_demo --features cuda

use prism_ai::applications::manufacturing::*;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Manufacturing Process Optimization Demo ===\n");

    // Create production line with multiple machines
    let production_line = create_production_line();

    println!("Production Line Configuration:");
    println!("  Machines: {}", production_line.machines.len());
    for machine in production_line.machines.iter().take(3) {
        println!("    {:?} - {} (${:.0}/hr)",
            machine.machine_type,
            machine.name,
            machine.operating_cost_per_hour);
    }
    println!("  Jobs: {} orders", production_line.jobs.len());
    let total_units: u32 = production_line.jobs.iter().map(|j| j.quantity).sum();
    println!("  Total Units: {}\n", total_units);

    // Initialize optimizer
    println!("Initializing GPU-accelerated manufacturing optimizer...");
    let config = ManufacturingConfig::default();
    let mut optimizer = ManufacturingOptimizer::new(config)?;
    println!("✓ Optimizer initialized\n");

    // Strategy 1: Minimize Makespan
    println!("--- Strategy 1: Minimize Makespan ---");
    let result = optimizer.optimize(&production_line, SchedulingStrategy::MinimizeMakespan)?;
    print_optimization_result(&result, "Makespan Minimization");

    // Strategy 2: Maximize Throughput
    println!("\n--- Strategy 2: Maximize Throughput ---");
    let result = optimizer.optimize(&production_line, SchedulingStrategy::MaximizeThroughput)?;
    print_optimization_result(&result, "Throughput Maximization");

    // Strategy 3: Minimize Cost
    println!("\n--- Strategy 3: Minimize Cost ---");
    let result = optimizer.optimize(&production_line, SchedulingStrategy::MinimizeCost)?;
    print_optimization_result(&result, "Cost Minimization");

    // Strategy 4: Priority-Based
    println!("\n--- Strategy 4: Priority-Based Scheduling ---");
    let result = optimizer.optimize(&production_line, SchedulingStrategy::PriorityBased)?;
    print_optimization_result(&result, "Priority-Based");

    // Strategy 5: Balanced Multi-Objective
    println!("\n--- Strategy 5: Balanced Multi-Objective ---");
    let result = optimizer.optimize(&production_line, SchedulingStrategy::Balanced)?;
    print_optimization_result(&result, "Balanced");

    // Predictive Maintenance Analysis
    println!("\n=== Predictive Maintenance Analysis ===");
    for machine in &production_line.machines {
        if let Some(maintenance) = optimizer.predict_maintenance(machine, 100.0)? {
            println!("{} - Maintenance Required:", machine.name);
            println!("  Type: {:?}", maintenance.maintenance_type);
            println!("  Scheduled: {:.1} hours", maintenance.scheduled_time_hours);
            println!("  Duration: {:.1} hours", maintenance.duration_hours);
            println!("  Priority: {}/10", maintenance.priority);
        }
    }

    println!("\n=== Strategy Comparison ===");
    println!("Makespan Minimization: Best for on-time delivery");
    println!("Throughput Maximization: Best for production volume");
    println!("Cost Minimization: Best for budget constraints");
    println!("Priority-Based: Best for urgent orders");
    println!("Balanced: Best for general manufacturing with multiple objectives");

    println!("\n✓ Manufacturing optimization complete!");

    Ok(())
}

fn create_production_line() -> ProductionLine {
    // Create diverse machine fleet
    let machines = vec![
        Machine {
            id: 0,
            machine_type: MachineType::CNC,
            name: "CNC-Mill-1".to_string(),
            capacity_units_per_hour: 12.0,
            setup_time_minutes: 20.0,
            operating_cost_per_hour: 75.0,
            failure_rate: 0.002,
            current_utilization: 0.0,
            maintenance_due_hours: 500.0,
        },
        Machine {
            id: 1,
            machine_type: MachineType::CNC,
            name: "CNC-Lathe-1".to_string(),
            capacity_units_per_hour: 15.0,
            setup_time_minutes: 15.0,
            operating_cost_per_hour: 65.0,
            failure_rate: 0.0015,
            current_utilization: 0.0,
            maintenance_due_hours: 600.0,
        },
        Machine {
            id: 2,
            machine_type: MachineType::Assembly,
            name: "Assembly-Line-1".to_string(),
            capacity_units_per_hour: 25.0,
            setup_time_minutes: 10.0,
            operating_cost_per_hour: 45.0,
            failure_rate: 0.001,
            current_utilization: 0.0,
            maintenance_due_hours: 800.0,
        },
        Machine {
            id: 3,
            machine_type: MachineType::Welding,
            name: "Welding-Station-1".to_string(),
            capacity_units_per_hour: 10.0,
            setup_time_minutes: 25.0,
            operating_cost_per_hour: 80.0,
            failure_rate: 0.003,
            current_utilization: 0.0,
            maintenance_due_hours: 400.0,
        },
        Machine {
            id: 4,
            machine_type: MachineType::Painting,
            name: "Paint-Booth-1".to_string(),
            capacity_units_per_hour: 20.0,
            setup_time_minutes: 30.0,
            operating_cost_per_hour: 55.0,
            failure_rate: 0.0012,
            current_utilization: 0.0,
            maintenance_due_hours: 700.0,
        },
        Machine {
            id: 5,
            machine_type: MachineType::Inspection,
            name: "QC-Station-1".to_string(),
            capacity_units_per_hour: 30.0,
            setup_time_minutes: 5.0,
            operating_cost_per_hour: 35.0,
            failure_rate: 0.0005,
            current_utilization: 0.0,
            maintenance_due_hours: 1000.0,
        },
        Machine {
            id: 6,
            machine_type: MachineType::Packaging,
            name: "Packaging-Line-1".to_string(),
            capacity_units_per_hour: 35.0,
            setup_time_minutes: 8.0,
            operating_cost_per_hour: 30.0,
            failure_rate: 0.0008,
            current_utilization: 0.0,
            maintenance_due_hours: 900.0,
        },
    ];

    // Create diverse job mix
    let jobs = vec![
        Job {
            id: 0,
            product_type: "Widget-A".to_string(),
            quantity: 150,
            priority: 9,
            due_date_hours: 48.0,
            processing_sequence: vec![
                MachineType::CNC,
                MachineType::Assembly,
                MachineType::Inspection,
                MachineType::Packaging,
            ],
            processing_times: vec![4.0, 3.0, 1.0, 0.5],  // minutes per unit
        },
        Job {
            id: 1,
            product_type: "Widget-B".to_string(),
            quantity: 200,
            priority: 7,
            due_date_hours: 72.0,
            processing_sequence: vec![
                MachineType::CNC,
                MachineType::Welding,
                MachineType::Painting,
                MachineType::Inspection,
                MachineType::Packaging,
            ],
            processing_times: vec![5.0, 6.0, 4.0, 1.0, 0.5],
        },
        Job {
            id: 2,
            product_type: "Widget-C".to_string(),
            quantity: 100,
            priority: 10,
            due_date_hours: 24.0,
            processing_sequence: vec![
                MachineType::CNC,
                MachineType::Assembly,
                MachineType::Inspection,
                MachineType::Packaging,
            ],
            processing_times: vec![3.0, 2.0, 1.0, 0.5],
        },
        Job {
            id: 3,
            product_type: "Widget-D".to_string(),
            quantity: 250,
            priority: 6,
            due_date_hours: 96.0,
            processing_sequence: vec![
                MachineType::CNC,
                MachineType::Painting,
                MachineType::Inspection,
                MachineType::Packaging,
            ],
            processing_times: vec![4.0, 5.0, 1.0, 0.5],
        },
    ];

    ProductionLine {
        machines,
        jobs,
        maintenance_schedules: Vec::new(),
    }
}

fn print_optimization_result(result: &OptimizationResult, strategy_name: &str) {
    println!("Strategy: {}", strategy_name);
    println!("Production Metrics:");
    println!("  Makespan: {:.1} hours ({:.1} days)", result.makespan_hours, result.makespan_hours / 24.0);
    println!("  Total Throughput: {} units", result.total_throughput);
    println!("  Total Cost: ${:.2}", result.total_cost);
    println!("  Average Utilization: {:.1}%", result.average_utilization * 100.0);
    println!("  Late Jobs: {}", result.late_jobs.len());

    println!("Quality Metrics:");
    println!("  First Pass Yield: {:.1}%", result.quality_metrics.first_pass_yield);
    println!("  Defect Rate: {:.2} per 1000 units", result.quality_metrics.defect_rate);
    println!("  Scrap Rate: {:.1}%", result.quality_metrics.scrap_rate);
    println!("  Rework Rate: {:.1}%", result.quality_metrics.rework_rate);
}
