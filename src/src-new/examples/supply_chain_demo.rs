//! Supply Chain Optimization Demo
//!
//! Demonstrates GPU-accelerated supply chain management with inventory
//! optimization and logistics routing.
//!
//! Run with: cargo run --example supply_chain_demo --features cuda

use prism_ai::applications::supply_chain::{
    SupplyChainOptimizer, LogisticsNetwork, Warehouse, Customer,
    Vehicle, OptimizationStrategy, SupplyChainConfig,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Supply Chain Optimization Demo ===\n");

    // Create a realistic US distribution network
    let network = create_us_distribution_network();

    println!("Network Configuration:");
    println!("  Warehouses: {} (major distribution centers)", network.warehouses.len());
    println!("  Customers: {} (retail locations)", network.customers.len());
    println!("  Vehicles: {} (delivery fleet)\n", network.vehicles.len());

    // Initialize GPU-accelerated optimizer
    println!("Initializing GPU-accelerated supply chain optimizer...");
    let config = SupplyChainConfig {
        use_active_inference: true,
        order_cost: 500.0,
        holding_cost_rate: 0.20,
        target_service_level: 0.95,
        max_route_duration: 10.0,
    };
    let mut optimizer = SupplyChainOptimizer::new(config)?;
    println!("✓ GPU initialization successful\n");

    // Strategy 1: Minimize Cost
    println!("--- Strategy 1: Minimize Total Cost ---");
    let result = optimizer.optimize(&network, OptimizationStrategy::MinimizeCost)?;
    print_optimization_result(&result, "Cost Minimization");

    // Strategy 2: Minimize Time
    println!("\n--- Strategy 2: Minimize Delivery Time ---");
    let result = optimizer.optimize(&network, OptimizationStrategy::MinimizeTime)?;
    print_optimization_result(&result, "Time Minimization");

    // Strategy 3: Maximize Service Level
    println!("\n--- Strategy 3: Maximize Service Level ---");
    let result = optimizer.optimize(&network, OptimizationStrategy::MaximizeServiceLevel)?;
    print_optimization_result(&result, "Service Maximization");

    // Strategy 4: Balanced Multi-Objective
    println!("\n--- Strategy 4: Balanced Multi-Objective ---");
    let result = optimizer.optimize(
        &network,
        OptimizationStrategy::Balanced {
            cost_weight: 0.5,
            time_weight: 0.3,
            service_weight: 0.2,
        },
    )?;
    print_optimization_result(&result, "Balanced");

    // Print inventory policies
    println!("\n=== Inventory Policies ===");
    for (i, policy) in result.inventory_policies.iter().enumerate() {
        println!("Warehouse {}:", get_warehouse_name(i));
        println!("  Economic Order Quantity: {:.0} units", policy.order_quantity);
        println!("  Reorder Point: {:.0} units", policy.reorder_point);
        println!("  Safety Stock: {:.0} units", policy.safety_stock);
        println!("  Target Service Level: {:.1}%", policy.service_level * 100.0);
    }

    // Print sample routes
    println!("\n=== Sample Delivery Routes (first 3) ===");
    for (i, route) in result.routes.iter().take(3).enumerate() {
        println!("Route {}:", i + 1);
        println!("  Origin: {}", get_warehouse_name(route.warehouse_id));
        println!("  Vehicle: {}", route.vehicle_id);
        println!("  Customers: {}", route.customer_sequence.len());
        println!("  Distance: {:.1} km", route.total_distance_km);
        println!("  Duration: {:.2} hours", route.total_time_hours);
        println!("  Cost: ${:.2}", route.total_cost);
        println!("  Load: {:.0} units ({:.1}% capacity)",
            route.total_demand,
            (route.total_demand / 1000.0) * 100.0 // Assuming 1000 capacity
        );
    }

    println!("\n=== Strategy Comparison ===");
    println!("Cost Minimization: Best for budget-constrained operations");
    println!("Time Minimization: Best for urgent/perishable goods");
    println!("Service Maximization: Best for high-value customer relationships");
    println!("Balanced: Best for general-purpose supply chain management");

    println!("\n✓ Supply chain optimization complete!");

    Ok(())
}

fn create_us_distribution_network() -> LogisticsNetwork {
    // Major distribution centers
    let warehouses = vec![
        Warehouse {
            id: 0,
            location: (40.7128, -74.0060), // New York, NY
            capacity: 50000.0,
            current_inventory: 25000.0,
            fixed_cost: 100000.0,
            holding_cost_per_unit: 5.0,
        },
        Warehouse {
            id: 1,
            location: (41.8781, -87.6298), // Chicago, IL
            capacity: 60000.0,
            current_inventory: 30000.0,
            fixed_cost: 90000.0,
            holding_cost_per_unit: 4.5,
        },
        Warehouse {
            id: 2,
            location: (33.7490, -84.3880), // Atlanta, GA
            capacity: 45000.0,
            current_inventory: 22000.0,
            fixed_cost: 85000.0,
            holding_cost_per_unit: 4.0,
        },
        Warehouse {
            id: 3,
            location: (34.0522, -118.2437), // Los Angeles, CA
            capacity: 55000.0,
            current_inventory: 28000.0,
            fixed_cost: 110000.0,
            holding_cost_per_unit: 6.0,
        },
    ];

    // Customer locations (retail stores)
    let customers = vec![
        Customer {
            id: 0,
            location: (42.3601, -71.0589), // Boston, MA
            demand: 500.0,
            service_level_target: 0.98,
            stockout_penalty: 100.0,
        },
        Customer {
            id: 1,
            location: (39.9526, -75.1652), // Philadelphia, PA
            demand: 450.0,
            service_level_target: 0.95,
            stockout_penalty: 80.0,
        },
        Customer {
            id: 2,
            location: (38.9072, -77.0369), // Washington, DC
            demand: 400.0,
            service_level_target: 0.95,
            stockout_penalty: 75.0,
        },
        Customer {
            id: 3,
            location: (35.2271, -80.8431), // Charlotte, NC
            demand: 350.0,
            service_level_target: 0.92,
            stockout_penalty: 70.0,
        },
        Customer {
            id: 4,
            location: (25.7617, -80.1918), // Miami, FL
            demand: 550.0,
            service_level_target: 0.95,
            stockout_penalty: 90.0,
        },
        Customer {
            id: 5,
            location: (39.7392, -104.9903), // Denver, CO
            demand: 380.0,
            service_level_target: 0.90,
            stockout_penalty: 65.0,
        },
        Customer {
            id: 6,
            location: (29.7604, -95.3698), // Houston, TX
            demand: 480.0,
            service_level_target: 0.93,
            stockout_penalty: 75.0,
        },
        Customer {
            id: 7,
            location: (32.7767, -96.7970), // Dallas, TX
            demand: 420.0,
            service_level_target: 0.92,
            stockout_penalty: 70.0,
        },
        Customer {
            id: 8,
            location: (33.4484, -112.0740), // Phoenix, AZ
            demand: 400.0,
            service_level_target: 0.90,
            stockout_penalty: 65.0,
        },
        Customer {
            id: 9,
            location: (37.7749, -122.4194), // San Francisco, CA
            demand: 600.0,
            service_level_target: 0.98,
            stockout_penalty: 120.0,
        },
        Customer {
            id: 10,
            location: (47.6062, -122.3321), // Seattle, WA
            demand: 520.0,
            service_level_target: 0.95,
            stockout_penalty: 85.0,
        },
        Customer {
            id: 11,
            location: (43.0731, -89.4012), // Madison, WI
            demand: 280.0,
            service_level_target: 0.88,
            stockout_penalty: 55.0,
        },
    ];

    // Delivery fleet
    let vehicles = vec![
        Vehicle {
            id: 0,
            capacity: 1000.0,
            cost_per_km: 2.5,
            speed_kmh: 90.0,
            max_range_km: 1000.0,
        },
        Vehicle {
            id: 1,
            capacity: 1000.0,
            cost_per_km: 2.5,
            speed_kmh: 90.0,
            max_range_km: 1000.0,
        },
        Vehicle {
            id: 2,
            capacity: 800.0,
            cost_per_km: 2.0,
            speed_kmh: 85.0,
            max_range_km: 900.0,
        },
        Vehicle {
            id: 3,
            capacity: 800.0,
            cost_per_km: 2.0,
            speed_kmh: 85.0,
            max_range_km: 900.0,
        },
        Vehicle {
            id: 4,
            capacity: 600.0,
            cost_per_km: 1.8,
            speed_kmh: 80.0,
            max_range_km: 800.0,
        },
        Vehicle {
            id: 5,
            capacity: 600.0,
            cost_per_km: 1.8,
            speed_kmh: 80.0,
            max_range_km: 800.0,
        },
    ];

    LogisticsNetwork::new(warehouses, customers, vehicles)
}

fn get_warehouse_name(id: usize) -> &'static str {
    match id {
        0 => "New York",
        1 => "Chicago",
        2 => "Atlanta",
        3 => "Los Angeles",
        _ => "Unknown",
    }
}

fn print_optimization_result(
    result: &prism_ai::applications::supply_chain::OptimizationResult,
    strategy_name: &str,
) {
    println!("Strategy: {}", strategy_name);
    println!("System Metrics:");
    println!("  Total Cost: ${:.2}", result.total_cost);
    println!("  Avg Delivery Time: {:.2} hours", result.avg_delivery_time);
    println!("  Service Level: {:.1}%", result.service_level * 100.0);
    println!("  Vehicle Utilization: {:.1}%", result.vehicle_utilization * 100.0);
    println!("  Routes Created: {}", result.routes.len());
}
