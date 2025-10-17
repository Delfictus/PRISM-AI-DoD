//! Telecom Network Optimization Demo
//!
//! Demonstrates GPU-accelerated network routing and traffic engineering.
//!
//! Run with: cargo run --example telecom_network_demo --features cuda

use prism_ai::applications::telecom::{
    NetworkOptimizer, NetworkTopology, NetworkNode, NetworkLink,
    RoutingStrategy, TrafficDemand, NetworkConfig, NodeType,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Telecom Network Optimization Demo ===\n");

    // Create a realistic metro network topology
    // 6 cities: NYC, CHI, DEN, LA, SF, SEA
    let topology = create_metro_network();

    println!("Network Topology:");
    println!("  Nodes: {} (Core routers in major US cities)", topology.nodes.len());
    println!("  Links: {} (High-capacity fiber links)\n", topology.links.len());

    // Define traffic demands
    let demands = vec![
        TrafficDemand {
            source: 0,  // NYC
            destination: 3,  // LA
            bandwidth_gbps: 25.0,
            max_latency_ms: 100.0,
            priority: 7,  // High priority (video conferencing)
        },
        TrafficDemand {
            source: 1,  // CHI
            destination: 4,  // SF
            bandwidth_gbps: 40.0,
            max_latency_ms: 80.0,
            priority: 5,  // Normal priority (data transfer)
        },
        TrafficDemand {
            source: 0,  // NYC
            destination: 5,  // SEA
            bandwidth_gbps: 15.0,
            max_latency_ms: 150.0,
            priority: 3,  // Best effort (backup traffic)
        },
    ];

    println!("Traffic Demands:");
    for (i, demand) in demands.iter().enumerate() {
        println!("  Demand {}: {} → {} ({:.1} Gbps, max {:.0}ms latency, priority {})",
            i + 1,
            get_city_name(demand.source),
            get_city_name(demand.destination),
            demand.bandwidth_gbps,
            demand.max_latency_ms,
            demand.priority
        );
    }
    println!();

    // Initialize GPU-accelerated optimizer
    println!("Initializing GPU-accelerated network optimizer...");
    let config = NetworkConfig::default();
    let mut optimizer = NetworkOptimizer::new(config)?;
    println!("✓ GPU initialization successful\n");

    // Strategy 1: Minimize Latency
    println!("--- Strategy 1: Minimize Latency ---");
    let result = optimizer.optimize(&topology, &demands, RoutingStrategy::MinLatency)?;
    print_optimization_result(&result, "Min Latency", &demands);

    // Strategy 2: Maximize Bandwidth
    println!("\n--- Strategy 2: Maximize Bandwidth ---");
    let result = optimizer.optimize(&topology, &demands, RoutingStrategy::MaxBandwidth)?;
    print_optimization_result(&result, "Max Bandwidth", &demands);

    // Strategy 3: Load Balancing
    println!("\n--- Strategy 3: Load Balancing ---");
    let result = optimizer.optimize(&topology, &demands, RoutingStrategy::LoadBalance)?;
    print_optimization_result(&result, "Load Balance", &demands);

    // Strategy 4: QoS-Optimized (multi-objective)
    println!("\n--- Strategy 4: QoS-Optimized (Multi-Objective) ---");
    let result = optimizer.optimize(
        &topology,
        &demands,
        RoutingStrategy::QoSOptimized {
            latency_weight: 1.0,
            bandwidth_weight: 0.8,
            cost_weight: 0.5,
        },
    )?;
    print_optimization_result(&result, "QoS-Optimized", &demands);

    // Compare strategies
    println!("\n=== Strategy Comparison ===");
    println!("Min Latency: Best for real-time applications (video, gaming)");
    println!("Max Bandwidth: Best for high-throughput data transfers");
    println!("Load Balance: Best for distributing traffic evenly");
    println!("QoS-Optimized: Best for mixed traffic with different requirements");

    println!("\n✓ Network optimization complete!");

    Ok(())
}

fn create_metro_network() -> NetworkTopology {
    let nodes = vec![
        NetworkNode {
            id: 0,
            name: "NYC".to_string(),
            location: (40.7128, -74.0060),
            capacity_pps: 2_000_000.0,
            current_load_pps: 800_000.0,
            node_type: NodeType::Core,
        },
        NetworkNode {
            id: 1,
            name: "CHI".to_string(),
            location: (41.8781, -87.6298),
            capacity_pps: 1_800_000.0,
            current_load_pps: 700_000.0,
            node_type: NodeType::Core,
        },
        NetworkNode {
            id: 2,
            name: "DEN".to_string(),
            location: (39.7392, -104.9903),
            capacity_pps: 1_500_000.0,
            current_load_pps: 500_000.0,
            node_type: NodeType::Core,
        },
        NetworkNode {
            id: 3,
            name: "LA".to_string(),
            location: (34.0522, -118.2437),
            capacity_pps: 1_800_000.0,
            current_load_pps: 900_000.0,
            node_type: NodeType::Core,
        },
        NetworkNode {
            id: 4,
            name: "SF".to_string(),
            location: (37.7749, -122.4194),
            capacity_pps: 1_600_000.0,
            current_load_pps: 650_000.0,
            node_type: NodeType::Core,
        },
        NetworkNode {
            id: 5,
            name: "SEA".to_string(),
            location: (47.6062, -122.3321),
            capacity_pps: 1_400_000.0,
            current_load_pps: 550_000.0,
            node_type: NodeType::Core,
        },
    ];

    let links = vec![
        // NYC <-> CHI
        NetworkLink {
            source: 0,
            destination: 1,
            bandwidth_gbps: 100.0,
            utilization: 0.4,
            latency_ms: 10.0,
            cost_per_gbps: 1200.0,
            reliability: 0.999,
        },
        NetworkLink {
            source: 1,
            destination: 0,
            bandwidth_gbps: 100.0,
            utilization: 0.35,
            latency_ms: 10.0,
            cost_per_gbps: 1200.0,
            reliability: 0.999,
        },
        // CHI <-> DEN
        NetworkLink {
            source: 1,
            destination: 2,
            bandwidth_gbps: 100.0,
            utilization: 0.3,
            latency_ms: 12.0,
            cost_per_gbps: 1000.0,
            reliability: 0.998,
        },
        NetworkLink {
            source: 2,
            destination: 1,
            bandwidth_gbps: 100.0,
            utilization: 0.28,
            latency_ms: 12.0,
            cost_per_gbps: 1000.0,
            reliability: 0.998,
        },
        // DEN <-> SF
        NetworkLink {
            source: 2,
            destination: 4,
            bandwidth_gbps: 100.0,
            utilization: 0.25,
            latency_ms: 18.0,
            cost_per_gbps: 1100.0,
            reliability: 0.997,
        },
        NetworkLink {
            source: 4,
            destination: 2,
            bandwidth_gbps: 100.0,
            utilization: 0.22,
            latency_ms: 18.0,
            cost_per_gbps: 1100.0,
            reliability: 0.997,
        },
        // DEN <-> LA
        NetworkLink {
            source: 2,
            destination: 3,
            bandwidth_gbps: 100.0,
            utilization: 0.5,
            latency_ms: 15.0,
            cost_per_gbps: 950.0,
            reliability: 0.998,
        },
        NetworkLink {
            source: 3,
            destination: 2,
            bandwidth_gbps: 100.0,
            utilization: 0.48,
            latency_ms: 15.0,
            cost_per_gbps: 950.0,
            reliability: 0.998,
        },
        // LA <-> SF
        NetworkLink {
            source: 3,
            destination: 4,
            bandwidth_gbps: 100.0,
            utilization: 0.55,
            latency_ms: 8.0,
            cost_per_gbps: 800.0,
            reliability: 0.999,
        },
        NetworkLink {
            source: 4,
            destination: 3,
            bandwidth_gbps: 100.0,
            utilization: 0.52,
            latency_ms: 8.0,
            cost_per_gbps: 800.0,
            reliability: 0.999,
        },
        // SF <-> SEA
        NetworkLink {
            source: 4,
            destination: 5,
            bandwidth_gbps: 100.0,
            utilization: 0.4,
            latency_ms: 14.0,
            cost_per_gbps: 1050.0,
            reliability: 0.998,
        },
        NetworkLink {
            source: 5,
            destination: 4,
            bandwidth_gbps: 100.0,
            utilization: 0.38,
            latency_ms: 14.0,
            cost_per_gbps: 1050.0,
            reliability: 0.998,
        },
        // NYC <-> DEN (direct, higher latency)
        NetworkLink {
            source: 0,
            destination: 2,
            bandwidth_gbps: 50.0,
            utilization: 0.6,
            latency_ms: 35.0,
            cost_per_gbps: 1300.0,
            reliability: 0.995,
        },
        NetworkLink {
            source: 2,
            destination: 0,
            bandwidth_gbps: 50.0,
            utilization: 0.58,
            latency_ms: 35.0,
            cost_per_gbps: 1300.0,
            reliability: 0.995,
        },
    ];

    NetworkTopology::new(nodes, links)
}

fn get_city_name(id: usize) -> &'static str {
    match id {
        0 => "NYC",
        1 => "CHI",
        2 => "DEN",
        3 => "LA",
        4 => "SF",
        5 => "SEA",
        _ => "Unknown",
    }
}

fn print_optimization_result(
    result: &prism_ai::applications::telecom::OptimizationResult,
    strategy_name: &str,
    demands: &[TrafficDemand],
) {
    println!("Strategy: {}", strategy_name);
    println!("Network Metrics:");
    println!("  Max Link Utilization: {:.1}%", result.max_link_utilization * 100.0);
    println!("  Average Latency: {:.1} ms", result.avg_latency_ms);
    println!("  Congested Links: {}", result.congested_links_count);
    println!("  Objective Value: {:.2}", result.objective_value);

    println!("\nRouting Paths:");
    for demand in demands {
        let key = (demand.source, demand.destination);
        if let Some(path) = result.routing_paths.get(&key) {
            print!("  {} → {}: ",
                get_city_name(demand.source),
                get_city_name(demand.destination)
            );

            for (i, &node_id) in path.nodes.iter().enumerate() {
                if i > 0 {
                    print!(" → ");
                }
                print!("{}", get_city_name(node_id));
            }

            println!();
            println!("    Latency: {:.1} ms, Bandwidth: {:.1} Gbps, Cost: ${:.0}/mo, Reliability: {:.3}",
                path.total_latency_ms,
                path.bottleneck_bandwidth_gbps,
                path.total_cost,
                path.reliability
            );
        }
    }
}
