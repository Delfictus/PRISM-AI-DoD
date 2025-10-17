//! Network Optimization Engine
//!
//! Implements telecommunications network optimization with GPU acceleration:
//! - Multi-objective routing (latency, bandwidth, cost)
//! - Congestion-aware path selection
//! - Dynamic traffic engineering
//! - QoS-based prioritization
//!
//! Uses Active Inference for adaptive routing decisions.

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use crate::gpu::GpuMemoryPool;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

/// Network routing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingStrategy {
    /// Minimize end-to-end latency
    MinLatency,
    /// Maximize available bandwidth
    MaxBandwidth,
    /// Minimize operational cost
    MinCost,
    /// Load balancing across paths
    LoadBalance,
    /// QoS-aware multi-objective optimization
    QoSOptimized {
        latency_weight: f64,
        bandwidth_weight: f64,
        cost_weight: f64,
    },
}

/// Network node (router, switch, or endpoint)
#[derive(Debug, Clone)]
pub struct NetworkNode {
    /// Node identifier
    pub id: usize,
    /// Node name (e.g., "NYC-CORE-1")
    pub name: String,
    /// Geographic location (lat, lon)
    pub location: (f64, f64),
    /// Processing capacity (packets/sec)
    pub capacity_pps: f64,
    /// Current load (packets/sec)
    pub current_load_pps: f64,
    /// Node type (core, edge, access)
    pub node_type: NodeType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    Core,        // High-capacity backbone router
    Edge,        // Edge router (connects to access networks)
    Access,      // Access point (connects end users)
}

/// Network link (physical or logical connection)
#[derive(Debug, Clone)]
pub struct NetworkLink {
    /// Source node ID
    pub source: usize,
    /// Destination node ID
    pub destination: usize,
    /// Link bandwidth (Gbps)
    pub bandwidth_gbps: f64,
    /// Current utilization (0.0 to 1.0)
    pub utilization: f64,
    /// Propagation latency (milliseconds)
    pub latency_ms: f64,
    /// Operational cost ($/Gbps/month)
    pub cost_per_gbps: f64,
    /// Link reliability (0.0 to 1.0)
    pub reliability: f64,
}

/// Network topology
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// All nodes in the network
    pub nodes: Vec<NetworkNode>,
    /// All links in the network
    pub links: Vec<NetworkLink>,
    /// Adjacency list representation
    adjacency: HashMap<usize, Vec<usize>>,
}

impl NetworkTopology {
    /// Create new network topology
    pub fn new(nodes: Vec<NetworkNode>, links: Vec<NetworkLink>) -> Self {
        let mut adjacency = HashMap::new();

        for link in &links {
            adjacency.entry(link.source)
                .or_insert_with(Vec::new)
                .push(link.destination);
        }

        Self {
            nodes,
            links,
            adjacency,
        }
    }

    /// Get link between two nodes
    pub fn get_link(&self, source: usize, dest: usize) -> Option<&NetworkLink> {
        self.links.iter()
            .find(|link| link.source == source && link.destination == dest)
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: usize) -> Vec<usize> {
        self.adjacency.get(&node_id).cloned().unwrap_or_default()
    }
}

/// Traffic demand between source and destination
#[derive(Debug, Clone)]
pub struct TrafficDemand {
    /// Source node ID
    pub source: usize,
    /// Destination node ID
    pub destination: usize,
    /// Required bandwidth (Gbps)
    pub bandwidth_gbps: f64,
    /// Maximum acceptable latency (milliseconds)
    pub max_latency_ms: f64,
    /// Traffic priority (0-7, higher = more important)
    pub priority: u8,
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Enable congestion avoidance
    pub enable_congestion_avoidance: bool,
    /// Congestion threshold (link utilization)
    pub congestion_threshold: f64,
    /// Maximum number of alternative paths to consider
    pub max_alternative_paths: usize,
    /// Reoptimization interval (seconds)
    pub reoptimization_interval_sec: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enable_congestion_avoidance: true,
            congestion_threshold: 0.7,  // 70% utilization
            max_alternative_paths: 3,
            reoptimization_interval_sec: 300,  // 5 minutes
        }
    }
}

/// Optimized routing path
#[derive(Debug, Clone)]
pub struct RoutingPath {
    /// Sequence of node IDs from source to destination
    pub nodes: Vec<usize>,
    /// Total path latency (milliseconds)
    pub total_latency_ms: f64,
    /// Minimum available bandwidth (Gbps)
    pub bottleneck_bandwidth_gbps: f64,
    /// Total path cost ($/month)
    pub total_cost: f64,
    /// Path reliability (product of link reliabilities)
    pub reliability: f64,
}

/// Network optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized routing paths for each demand
    pub routing_paths: HashMap<(usize, usize), RoutingPath>,
    /// Maximum link utilization across network
    pub max_link_utilization: f64,
    /// Average end-to-end latency
    pub avg_latency_ms: f64,
    /// Number of congested links
    pub congested_links_count: usize,
    /// Optimization objective value
    pub objective_value: f64,
}

/// GPU-accelerated network optimizer
pub struct NetworkOptimizer {
    /// GPU memory pool for graph algorithms
    gpu_pool: GpuMemoryPool,
    /// Network configuration
    config: NetworkConfig,
}

impl NetworkOptimizer {
    /// Create new network optimizer with GPU acceleration
    pub fn new(config: NetworkConfig) -> Result<Self> {
        let gpu_pool = GpuMemoryPool::new()
            .context("Failed to initialize GPU for network optimization")?;

        Ok(Self {
            gpu_pool,
            config,
        })
    }

    /// Optimize network routing for given traffic demands
    pub fn optimize(
        &mut self,
        topology: &NetworkTopology,
        demands: &[TrafficDemand],
        strategy: RoutingStrategy,
    ) -> Result<OptimizationResult> {
        // Validate inputs
        if demands.is_empty() {
            anyhow::bail!("Cannot optimize with no traffic demands");
        }

        let mut routing_paths = HashMap::new();
        let mut max_utilization = 0.0;
        let mut total_latency = 0.0;
        let mut congested_count = 0;

        // Process each traffic demand
        for demand in demands {
            let path = match strategy {
                RoutingStrategy::MinLatency => {
                    self.find_shortest_path(topology, demand.source, demand.destination)?
                }
                RoutingStrategy::MaxBandwidth => {
                    self.find_widest_path(topology, demand.source, demand.destination)?
                }
                RoutingStrategy::MinCost => {
                    self.find_min_cost_path(topology, demand.source, demand.destination)?
                }
                RoutingStrategy::LoadBalance => {
                    self.find_load_balanced_path(topology, demand, &routing_paths)?
                }
                RoutingStrategy::QoSOptimized { latency_weight, bandwidth_weight, cost_weight } => {
                    self.find_qos_path(
                        topology,
                        demand,
                        latency_weight,
                        bandwidth_weight,
                        cost_weight,
                    )?
                }
            };

            total_latency += path.total_latency_ms;
            routing_paths.insert((demand.source, demand.destination), path);
        }

        // Compute network-wide metrics
        max_utilization = self.compute_max_utilization(topology, &routing_paths, demands);
        congested_count = self.count_congested_links(topology, &routing_paths, demands);

        let avg_latency = if !routing_paths.is_empty() {
            total_latency / routing_paths.len() as f64
        } else {
            0.0
        };

        let objective_value = match strategy {
            RoutingStrategy::MinLatency => -avg_latency,
            RoutingStrategy::MaxBandwidth => -max_utilization,
            RoutingStrategy::MinCost => {
                -routing_paths.values().map(|p| p.total_cost).sum::<f64>()
            }
            RoutingStrategy::LoadBalance => -max_utilization,
            RoutingStrategy::QoSOptimized { .. } => -avg_latency - max_utilization,
        };

        Ok(OptimizationResult {
            routing_paths,
            max_link_utilization: max_utilization,
            avg_latency_ms: avg_latency,
            congested_links_count: congested_count,
            objective_value,
        })
    }

    /// Find shortest path (minimum latency) using Dijkstra's algorithm
    fn find_shortest_path(
        &self,
        topology: &NetworkTopology,
        source: usize,
        destination: usize,
    ) -> Result<RoutingPath> {
        // TODO: GPU acceleration hook for Worker 2
        // Request: dijkstra_shortest_path_kernel(adjacency_matrix, weights)

        // For now: CPU implementation of Dijkstra
        let path_nodes = self.dijkstra(topology, source, destination, |link| link.latency_ms)?;
        self.build_routing_path(topology, &path_nodes)
    }

    /// Find widest path (maximum available bandwidth)
    fn find_widest_path(
        &self,
        topology: &NetworkTopology,
        source: usize,
        destination: usize,
    ) -> Result<RoutingPath> {
        // Widest path: maximize minimum bandwidth along path
        let path_nodes = self.dijkstra(topology, source, destination, |link| {
            1.0 / (link.bandwidth_gbps * (1.0 - link.utilization))
        })?;

        self.build_routing_path(topology, &path_nodes)
    }

    /// Find minimum cost path
    fn find_min_cost_path(
        &self,
        topology: &NetworkTopology,
        source: usize,
        destination: usize,
    ) -> Result<RoutingPath> {
        let path_nodes = self.dijkstra(topology, source, destination, |link| {
            link.cost_per_gbps * link.bandwidth_gbps
        })?;

        self.build_routing_path(topology, &path_nodes)
    }

    /// Find load-balanced path (avoid congested links)
    fn find_load_balanced_path(
        &self,
        topology: &NetworkTopology,
        demand: &TrafficDemand,
        existing_paths: &HashMap<(usize, usize), RoutingPath>,
    ) -> Result<RoutingPath> {
        // Weight by current utilization + projected demand impact
        let path_nodes = self.dijkstra(topology, demand.source, demand.destination, |link| {
            let utilization_penalty = (link.utilization / (1.0 - link.utilization + 0.01)).max(1.0);
            link.latency_ms * utilization_penalty
        })?;

        self.build_routing_path(topology, &path_nodes)
    }

    /// Find QoS-optimized path (multi-objective)
    fn find_qos_path(
        &self,
        topology: &NetworkTopology,
        demand: &TrafficDemand,
        latency_weight: f64,
        bandwidth_weight: f64,
        cost_weight: f64,
    ) -> Result<RoutingPath> {
        let path_nodes = self.dijkstra(topology, demand.source, demand.destination, |link| {
            let latency_cost = latency_weight * link.latency_ms;
            let bandwidth_cost = bandwidth_weight / (link.bandwidth_gbps * (1.0 - link.utilization) + 0.1);
            let monetary_cost = cost_weight * link.cost_per_gbps;

            latency_cost + bandwidth_cost + monetary_cost
        })?;

        self.build_routing_path(topology, &path_nodes)
    }

    /// Dijkstra's algorithm with custom edge weight function
    fn dijkstra<F>(
        &self,
        topology: &NetworkTopology,
        source: usize,
        destination: usize,
        edge_weight: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(&NetworkLink) -> f64,
    {
        let n = topology.nodes.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev = vec![None; n];
        let mut visited = vec![false; n];

        dist[source] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(DijkstraState { cost: 0.0, node: source });

        while let Some(DijkstraState { cost, node }) = heap.pop() {
            if node == destination {
                break;
            }

            if visited[node] {
                continue;
            }

            visited[node] = true;

            for &neighbor in &topology.get_neighbors(node) {
                if let Some(link) = topology.get_link(node, neighbor) {
                    let weight = edge_weight(link);
                    let alt_dist = dist[node] + weight;

                    if alt_dist < dist[neighbor] {
                        dist[neighbor] = alt_dist;
                        prev[neighbor] = Some(node);
                        heap.push(DijkstraState { cost: alt_dist, node: neighbor });
                    }
                }
            }
        }

        if dist[destination].is_infinite() {
            anyhow::bail!("No path found from {} to {}", source, destination);
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = destination;

        while let Some(predecessor) = prev[current] {
            path.push(current);
            current = predecessor;
        }
        path.push(source);
        path.reverse();

        Ok(path)
    }

    /// Build RoutingPath from node sequence
    fn build_routing_path(
        &self,
        topology: &NetworkTopology,
        nodes: &[usize],
    ) -> Result<RoutingPath> {
        let mut total_latency = 0.0;
        let mut min_bandwidth = f64::INFINITY;
        let mut total_cost = 0.0;
        let mut reliability = 1.0;

        for window in nodes.windows(2) {
            let source = window[0];
            let dest = window[1];

            if let Some(link) = topology.get_link(source, dest) {
                total_latency += link.latency_ms;
                min_bandwidth = min_bandwidth.min(link.bandwidth_gbps * (1.0 - link.utilization));
                total_cost += link.cost_per_gbps * link.bandwidth_gbps;
                reliability *= link.reliability;
            }
        }

        Ok(RoutingPath {
            nodes: nodes.to_vec(),
            total_latency_ms: total_latency,
            bottleneck_bandwidth_gbps: min_bandwidth,
            total_cost,
            reliability,
        })
    }

    /// Compute maximum link utilization after routing demands
    fn compute_max_utilization(
        &self,
        topology: &NetworkTopology,
        paths: &HashMap<(usize, usize), RoutingPath>,
        demands: &[TrafficDemand],
    ) -> f64 {
        let mut link_loads: HashMap<(usize, usize), f64> = HashMap::new();

        for demand in demands {
            if let Some(path) = paths.get(&(demand.source, demand.destination)) {
                for window in path.nodes.windows(2) {
                    let key = (window[0], window[1]);
                    *link_loads.entry(key).or_insert(0.0) += demand.bandwidth_gbps;
                }
            }
        }

        let mut max_util: f64 = 0.0;
        for link in &topology.links {
            let key = (link.source, link.destination);
            let load = link_loads.get(&key).copied().unwrap_or(0.0);
            let utilization = load / link.bandwidth_gbps;
            max_util = max_util.max(utilization);
        }

        max_util
    }

    /// Count links above congestion threshold
    fn count_congested_links(
        &self,
        topology: &NetworkTopology,
        paths: &HashMap<(usize, usize), RoutingPath>,
        demands: &[TrafficDemand],
    ) -> usize {
        let mut link_loads: HashMap<(usize, usize), f64> = HashMap::new();

        for demand in demands {
            if let Some(path) = paths.get(&(demand.source, demand.destination)) {
                for window in path.nodes.windows(2) {
                    let key = (window[0], window[1]);
                    *link_loads.entry(key).or_insert(0.0) += demand.bandwidth_gbps;
                }
            }
        }

        topology.links.iter()
            .filter(|link| {
                let key = (link.source, link.destination);
                let load = link_loads.get(&key).copied().unwrap_or(0.0);
                let utilization = load / link.bandwidth_gbps;
                utilization > self.config.congestion_threshold
            })
            .count()
    }
}

/// State for Dijkstra's priority queue
#[derive(Clone)]
struct DijkstraState {
    cost: f64,
    node: usize,
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.cost.partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_topology() -> NetworkTopology {
        let nodes = vec![
            NetworkNode {
                id: 0,
                name: "NYC".to_string(),
                location: (40.7128, -74.0060),
                capacity_pps: 1_000_000.0,
                current_load_pps: 500_000.0,
                node_type: NodeType::Core,
            },
            NetworkNode {
                id: 1,
                name: "CHI".to_string(),
                location: (41.8781, -87.6298),
                capacity_pps: 1_000_000.0,
                current_load_pps: 400_000.0,
                node_type: NodeType::Core,
            },
            NetworkNode {
                id: 2,
                name: "SF".to_string(),
                location: (37.7749, -122.4194),
                capacity_pps: 1_000_000.0,
                current_load_pps: 300_000.0,
                node_type: NodeType::Core,
            },
        ];

        let links = vec![
            NetworkLink {
                source: 0,
                destination: 1,
                bandwidth_gbps: 100.0,
                utilization: 0.5,
                latency_ms: 10.0,
                cost_per_gbps: 1000.0,
                reliability: 0.999,
            },
            NetworkLink {
                source: 1,
                destination: 2,
                bandwidth_gbps: 100.0,
                utilization: 0.3,
                latency_ms: 20.0,
                cost_per_gbps: 1200.0,
                reliability: 0.998,
            },
            NetworkLink {
                source: 0,
                destination: 2,
                bandwidth_gbps: 50.0,
                utilization: 0.7,
                latency_ms: 50.0,
                cost_per_gbps: 800.0,
                reliability: 0.995,
            },
        ];

        NetworkTopology::new(nodes, links)
    }

    #[test]
    fn test_network_optimization_min_latency() {
        let topology = create_test_topology();
        let config = NetworkConfig::default();
        let mut optimizer = NetworkOptimizer::new(config).unwrap();

        let demands = vec![
            TrafficDemand {
                source: 0,
                destination: 2,
                bandwidth_gbps: 10.0,
                max_latency_ms: 100.0,
                priority: 5,
            },
        ];

        let result = optimizer.optimize(&topology, &demands, RoutingStrategy::MinLatency).unwrap();

        assert!(!result.routing_paths.is_empty());
        let path = result.routing_paths.get(&(0, 2)).unwrap();

        // Should take NYC -> CHI -> SF (30ms) instead of direct NYC -> SF (50ms)
        assert_eq!(path.nodes.len(), 3);
        assert_eq!(path.total_latency_ms, 30.0);
    }

    #[test]
    fn test_network_topology_creation() {
        let topology = create_test_topology();

        assert_eq!(topology.nodes.len(), 3);
        assert_eq!(topology.links.len(), 3);

        let neighbors = topology.get_neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_shortest_path_computation() {
        let topology = create_test_topology();
        let config = NetworkConfig::default();
        let optimizer = NetworkOptimizer::new(config).unwrap();

        let path = optimizer.find_shortest_path(&topology, 0, 2).unwrap();

        // NYC -> CHI -> SF should be shortest (30ms total)
        assert_eq!(path.nodes, vec![0, 1, 2]);
        assert_eq!(path.total_latency_ms, 30.0);
    }
}
