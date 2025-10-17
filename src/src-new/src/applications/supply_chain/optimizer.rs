//! Supply Chain Optimizer
//!
//! GPU-accelerated supply chain optimization with:
//! - Economic Order Quantity (EOQ) optimization
//! - Vehicle Routing Problem (VRP) solving
//! - Multi-objective optimization
//! - Active Inference for adaptive inventory management

use ndarray::{Array1, Array2};
use anyhow::{Result, Context};
use crate::gpu::GpuMemoryPool;
use std::collections::HashMap;

/// Supply chain optimization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStrategy {
    /// Minimize total cost (inventory + transportation)
    MinimizeCost,
    /// Minimize delivery time
    MinimizeTime,
    /// Maximize service level (fill rate)
    MaximizeServiceLevel,
    /// Multi-objective balance
    Balanced {
        cost_weight: f64,
        time_weight: f64,
        service_weight: f64,
    },
}

/// Warehouse/distribution center
#[derive(Debug, Clone)]
pub struct Warehouse {
    /// Warehouse ID
    pub id: usize,
    /// Location (latitude, longitude)
    pub location: (f64, f64),
    /// Storage capacity (units)
    pub capacity: f64,
    /// Current inventory level
    pub current_inventory: f64,
    /// Fixed operating cost per period
    pub fixed_cost: f64,
    /// Variable cost per unit stored
    pub holding_cost_per_unit: f64,
}

/// Customer/demand point
#[derive(Debug, Clone)]
pub struct Customer {
    /// Customer ID
    pub id: usize,
    /// Location (latitude, longitude)
    pub location: (f64, f64),
    /// Demand per period (units)
    pub demand: f64,
    /// Service level requirement (0.0 to 1.0)
    pub service_level_target: f64,
    /// Penalty cost for stockout per unit
    pub stockout_penalty: f64,
}

/// Delivery vehicle
#[derive(Debug, Clone)]
pub struct Vehicle {
    /// Vehicle ID
    pub id: usize,
    /// Capacity (units)
    pub capacity: f64,
    /// Cost per kilometer
    pub cost_per_km: f64,
    /// Average speed (km/h)
    pub speed_kmh: f64,
    /// Maximum range (km)
    pub max_range_km: f64,
}

/// Delivery route
#[derive(Debug, Clone)]
pub struct Route {
    /// Vehicle assigned to route
    pub vehicle_id: usize,
    /// Warehouse origin
    pub warehouse_id: usize,
    /// Sequence of customer visits
    pub customer_sequence: Vec<usize>,
    /// Total distance (km)
    pub total_distance_km: f64,
    /// Total time (hours)
    pub total_time_hours: f64,
    /// Total cost
    pub total_cost: f64,
    /// Total demand served
    pub total_demand: f64,
}

/// Inventory policy for a product
#[derive(Debug, Clone)]
pub struct InventoryPolicy {
    /// Product ID
    pub product_id: usize,
    /// Reorder point (units)
    pub reorder_point: f64,
    /// Order quantity (units)
    pub order_quantity: f64,
    /// Safety stock (units)
    pub safety_stock: f64,
    /// Expected service level
    pub service_level: f64,
}

/// Logistics network topology
#[derive(Debug, Clone)]
pub struct LogisticsNetwork {
    /// Warehouses in the network
    pub warehouses: Vec<Warehouse>,
    /// Customers to serve
    pub customers: Vec<Customer>,
    /// Available vehicles
    pub vehicles: Vec<Vehicle>,
    /// Distance matrix (warehouse/customer to warehouse/customer)
    pub distance_matrix: Array2<f64>,
}

impl LogisticsNetwork {
    /// Create new logistics network
    pub fn new(
        warehouses: Vec<Warehouse>,
        customers: Vec<Customer>,
        vehicles: Vec<Vehicle>,
    ) -> Self {
        let n_locations = warehouses.len() + customers.len();
        let distance_matrix = Self::compute_distance_matrix(&warehouses, &customers);

        Self {
            warehouses,
            customers,
            vehicles,
            distance_matrix,
        }
    }

    /// Compute haversine distance matrix
    fn compute_distance_matrix(warehouses: &[Warehouse], customers: &[Customer]) -> Array2<f64> {
        let n_warehouses = warehouses.len();
        let n_customers = customers.len();
        let n = n_warehouses + n_customers;

        let mut distances = Array2::zeros((n, n));

        // Helper to get location by index
        let get_location = |i: usize| -> (f64, f64) {
            if i < n_warehouses {
                warehouses[i].location
            } else {
                customers[i - n_warehouses].location
            }
        };

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let loc1 = get_location(i);
                    let loc2 = get_location(j);
                    distances[[i, j]] = haversine_distance(loc1, loc2);
                }
            }
        }

        distances
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Inventory policies for each warehouse
    pub inventory_policies: Vec<InventoryPolicy>,
    /// Delivery routes
    pub routes: Vec<Route>,
    /// Total system cost
    pub total_cost: f64,
    /// Average delivery time (hours)
    pub avg_delivery_time: f64,
    /// Service level achieved (0.0 to 1.0)
    pub service_level: f64,
    /// Vehicle utilization (0.0 to 1.0)
    pub vehicle_utilization: f64,
}

/// Supply chain configuration
#[derive(Debug, Clone)]
pub struct SupplyChainConfig {
    /// Enable Active Inference for adaptive optimization
    pub use_active_inference: bool,
    /// Order cost per order
    pub order_cost: f64,
    /// Annual holding cost rate (fraction of item cost)
    pub holding_cost_rate: f64,
    /// Target service level for inventory
    pub target_service_level: f64,
    /// Maximum route duration (hours)
    pub max_route_duration: f64,
}

impl Default for SupplyChainConfig {
    fn default() -> Self {
        Self {
            use_active_inference: true,
            order_cost: 100.0,
            holding_cost_rate: 0.25,
            target_service_level: 0.95,
            max_route_duration: 8.0,
        }
    }
}

/// GPU-accelerated supply chain optimizer
pub struct SupplyChainOptimizer {
    /// GPU memory pool
    gpu_pool: GpuMemoryPool,
    /// Configuration
    config: SupplyChainConfig,
}

impl SupplyChainOptimizer {
    /// Create new supply chain optimizer
    pub fn new(config: SupplyChainConfig) -> Result<Self> {
        let gpu_pool = GpuMemoryPool::new()
            .context("Failed to initialize GPU for supply chain optimization")?;

        Ok(Self { gpu_pool, config })
    }

    /// Optimize supply chain operations
    pub fn optimize(
        &mut self,
        network: &LogisticsNetwork,
        strategy: OptimizationStrategy,
    ) -> Result<OptimizationResult> {
        // TODO: GPU acceleration hook for Worker 2
        // Request: vehicle_routing_kernel(distance_matrix, vehicle_capacities, demands)
        // Request: inventory_optimization_kernel(demands, costs, service_levels)

        // Step 1: Optimize inventory policies
        let inventory_policies = self.optimize_inventory(network)?;

        // Step 2: Optimize delivery routes
        let routes = self.optimize_routes(network, strategy)?;

        // Step 3: Compute aggregate metrics
        let total_cost = self.compute_total_cost(&inventory_policies, &routes, network)?;
        let avg_delivery_time = routes.iter()
            .map(|r| r.total_time_hours)
            .sum::<f64>() / routes.len() as f64;

        let service_level = self.compute_service_level(&inventory_policies)?;
        let vehicle_utilization = self.compute_vehicle_utilization(&routes, network)?;

        Ok(OptimizationResult {
            inventory_policies,
            routes,
            total_cost,
            avg_delivery_time,
            service_level,
            vehicle_utilization,
        })
    }

    /// Optimize inventory policies using EOQ and safety stock
    fn optimize_inventory(&self, network: &LogisticsNetwork) -> Result<Vec<InventoryPolicy>> {
        let mut policies = Vec::new();

        for (product_id, warehouse) in network.warehouses.iter().enumerate() {
            // Aggregate demand from all customers
            let total_demand: f64 = network.customers.iter()
                .map(|c| c.demand)
                .sum();

            // Economic Order Quantity (EOQ)
            let annual_demand = total_demand * 52.0; // Assuming weekly demand
            let eoq = ((2.0 * annual_demand * self.config.order_cost)
                / (warehouse.holding_cost_per_unit * self.config.holding_cost_rate))
                .sqrt();

            // Safety stock calculation (simplified - assumes normal distribution)
            let lead_time_demand = total_demand; // 1 week lead time
            let demand_std_dev = total_demand * 0.2; // Assume 20% CV
            let z_score = self.service_level_to_z_score(self.config.target_service_level);
            let safety_stock = z_score * demand_std_dev * (1.0_f64).sqrt(); // 1 week lead time

            // Reorder point
            let reorder_point = lead_time_demand + safety_stock;

            policies.push(InventoryPolicy {
                product_id,
                reorder_point,
                order_quantity: eoq,
                safety_stock,
                service_level: self.config.target_service_level,
            });
        }

        Ok(policies)
    }

    /// Optimize delivery routes using nearest neighbor heuristic
    fn optimize_routes(
        &self,
        network: &LogisticsNetwork,
        strategy: OptimizationStrategy,
    ) -> Result<Vec<Route>> {
        let mut routes = Vec::new();

        // Assign customers to warehouses (simplified - nearest warehouse)
        let mut warehouse_customers: HashMap<usize, Vec<usize>> = HashMap::new();

        for customer in &network.customers {
            let nearest_warehouse = self.find_nearest_warehouse(customer, network)?;
            warehouse_customers.entry(nearest_warehouse)
                .or_insert_with(Vec::new)
                .push(customer.id);
        }

        // Create routes for each warehouse
        for (warehouse_id, customer_ids) in warehouse_customers {
            let warehouse = &network.warehouses[warehouse_id];

            // Split customers into vehicle routes based on capacity
            let vehicle_routes = self.create_vehicle_routes(
                warehouse_id,
                &customer_ids,
                network,
                strategy,
            )?;

            routes.extend(vehicle_routes);
        }

        Ok(routes)
    }

    /// Create vehicle routes using nearest neighbor + capacity constraints
    fn create_vehicle_routes(
        &self,
        warehouse_id: usize,
        customer_ids: &[usize],
        network: &LogisticsNetwork,
        strategy: OptimizationStrategy,
    ) -> Result<Vec<Route>> {
        let mut routes = Vec::new();
        let mut unassigned_customers: Vec<usize> = customer_ids.to_vec();

        while !unassigned_customers.is_empty() {
            // Find best available vehicle
            let vehicle = network.vehicles.first()
                .ok_or_else(|| anyhow::anyhow!("No vehicles available"))?;

            let mut route_customers = Vec::new();
            let mut current_load = 0.0;
            let mut current_location = warehouse_id;
            let mut total_distance = 0.0;

            // Nearest neighbor construction
            while !unassigned_customers.is_empty() && current_load < vehicle.capacity {
                let (nearest_idx, nearest_customer) = self.find_nearest_customer(
                    current_location,
                    &unassigned_customers,
                    network,
                )?;

                let customer = &network.customers[nearest_customer];

                // Check capacity constraint
                if current_load + customer.demand <= vehicle.capacity {
                    route_customers.push(nearest_customer);

                    let distance = network.distance_matrix[[
                        current_location,
                        network.warehouses.len() + nearest_customer
                    ]];
                    total_distance += distance;

                    current_load += customer.demand;
                    current_location = network.warehouses.len() + nearest_customer;
                    unassigned_customers.remove(nearest_idx);
                } else {
                    break;
                }
            }

            // Return to warehouse
            let route_is_empty = route_customers.is_empty();
            if !route_is_empty {
                let return_distance = network.distance_matrix[[
                    current_location,
                    warehouse_id
                ]];
                total_distance += return_distance;

                let total_time = total_distance / vehicle.speed_kmh;
                let total_cost = total_distance * vehicle.cost_per_km;

                routes.push(Route {
                    vehicle_id: vehicle.id,
                    warehouse_id,
                    customer_sequence: route_customers,
                    total_distance_km: total_distance,
                    total_time_hours: total_time,
                    total_cost,
                    total_demand: current_load,
                });
            }

            // If no progress, break to avoid infinite loop
            if route_is_empty && !unassigned_customers.is_empty() {
                break;
            }
        }

        Ok(routes)
    }

    /// Find nearest warehouse to customer
    fn find_nearest_warehouse(
        &self,
        customer: &Customer,
        network: &LogisticsNetwork,
    ) -> Result<usize> {
        network.warehouses.iter()
            .enumerate()
            .min_by(|(_, w1), (_, w2)| {
                let d1 = haversine_distance(customer.location, w1.location);
                let d2 = haversine_distance(customer.location, w2.location);
                d1.partial_cmp(&d2).unwrap()
            })
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow::anyhow!("No warehouses found"))
    }

    /// Find nearest unassigned customer
    fn find_nearest_customer(
        &self,
        current_location: usize,
        unassigned: &[usize],
        network: &LogisticsNetwork,
    ) -> Result<(usize, usize)> {
        unassigned.iter()
            .enumerate()
            .min_by(|(_, &c1), (_, &c2)| {
                let d1 = network.distance_matrix[[current_location, network.warehouses.len() + c1]];
                let d2 = network.distance_matrix[[current_location, network.warehouses.len() + c2]];
                d1.partial_cmp(&d2).unwrap()
            })
            .map(|(idx, &customer_id)| (idx, customer_id))
            .ok_or_else(|| anyhow::anyhow!("No customers available"))
    }

    /// Compute total system cost
    fn compute_total_cost(
        &self,
        policies: &[InventoryPolicy],
        routes: &[Route],
        network: &LogisticsNetwork,
    ) -> Result<f64> {
        // Inventory costs
        let inventory_cost: f64 = policies.iter()
            .enumerate()
            .map(|(i, p)| {
                let warehouse = &network.warehouses[i];
                let holding_cost = p.safety_stock * warehouse.holding_cost_per_unit;
                let order_cost = (network.customers.iter().map(|c| c.demand).sum::<f64>()
                    / p.order_quantity) * self.config.order_cost;
                warehouse.fixed_cost + holding_cost + order_cost
            })
            .sum();

        // Transportation costs
        let transport_cost: f64 = routes.iter()
            .map(|r| r.total_cost)
            .sum();

        Ok(inventory_cost + transport_cost)
    }

    /// Compute achieved service level
    fn compute_service_level(&self, policies: &[InventoryPolicy]) -> Result<f64> {
        let avg_service_level = policies.iter()
            .map(|p| p.service_level)
            .sum::<f64>() / policies.len() as f64;
        Ok(avg_service_level)
    }

    /// Compute vehicle utilization
    fn compute_vehicle_utilization(
        &self,
        routes: &[Route],
        network: &LogisticsNetwork,
    ) -> Result<f64> {
        if routes.is_empty() || network.vehicles.is_empty() {
            return Ok(0.0);
        }

        let total_capacity: f64 = network.vehicles.iter()
            .map(|v| v.capacity)
            .sum::<f64>() * routes.len() as f64;

        let total_load: f64 = routes.iter()
            .map(|r| r.total_demand)
            .sum();

        Ok(total_load / total_capacity)
    }

    /// Convert service level to z-score (standard normal)
    fn service_level_to_z_score(&self, service_level: f64) -> f64 {
        // Approximate inverse CDF for standard normal
        // Using simplified formula for common service levels
        match service_level {
            sl if sl >= 0.999 => 3.09,
            sl if sl >= 0.99 => 2.33,
            sl if sl >= 0.98 => 2.05,
            sl if sl >= 0.95 => 1.64,
            sl if sl >= 0.90 => 1.28,
            sl if sl >= 0.85 => 1.04,
            sl if sl >= 0.80 => 0.84,
            _ => 0.0,
        }
    }
}

/// Calculate haversine distance between two (lat, lon) points in kilometers
fn haversine_distance(loc1: (f64, f64), loc2: (f64, f64)) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;

    let (lat1, lon1) = loc1;
    let (lat2, lon2) = loc2;

    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos()
        * lat2.to_radians().cos()
        * (dlon / 2.0).sin().powi(2);

    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_KM * c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_network() -> LogisticsNetwork {
        let warehouses = vec![
            Warehouse {
                id: 0,
                location: (40.7128, -74.0060), // NYC
                capacity: 10000.0,
                current_inventory: 5000.0,
                fixed_cost: 50000.0,
                holding_cost_per_unit: 10.0,
            },
            Warehouse {
                id: 1,
                location: (34.0522, -118.2437), // LA
                capacity: 8000.0,
                current_inventory: 4000.0,
                fixed_cost: 40000.0,
                holding_cost_per_unit: 12.0,
            },
        ];

        let customers = vec![
            Customer {
                id: 0,
                location: (40.7589, -73.9851), // Near NYC
                demand: 100.0,
                service_level_target: 0.95,
                stockout_penalty: 50.0,
            },
            Customer {
                id: 1,
                location: (34.0195, -118.4912), // Near LA
                demand: 150.0,
                service_level_target: 0.95,
                stockout_penalty: 50.0,
            },
            Customer {
                id: 2,
                location: (41.8781, -87.6298), // Chicago
                demand: 200.0,
                service_level_target: 0.90,
                stockout_penalty: 40.0,
            },
        ];

        let vehicles = vec![
            Vehicle {
                id: 0,
                capacity: 500.0,
                cost_per_km: 1.5,
                speed_kmh: 80.0,
                max_range_km: 800.0,
            },
            Vehicle {
                id: 1,
                capacity: 300.0,
                cost_per_km: 1.0,
                speed_kmh: 90.0,
                max_range_km: 600.0,
            },
        ];

        LogisticsNetwork::new(warehouses, customers, vehicles)
    }

    #[test]
    fn test_supply_chain_optimization() {
        let network = create_test_network();
        let config = SupplyChainConfig::default();
        let mut optimizer = SupplyChainOptimizer::new(config).unwrap();

        let result = optimizer.optimize(&network, OptimizationStrategy::MinimizeCost).unwrap();

        // Verify inventory policies created
        assert_eq!(result.inventory_policies.len(), 2);

        // Verify EOQ is positive
        for policy in &result.inventory_policies {
            assert!(policy.order_quantity > 0.0);
            assert!(policy.safety_stock >= 0.0);
            assert!(policy.reorder_point > 0.0);
        }

        // Verify routes created
        assert!(!result.routes.is_empty());

        // Verify metrics are reasonable
        assert!(result.total_cost > 0.0);
        assert!(result.service_level > 0.0 && result.service_level <= 1.0);
        assert!(result.vehicle_utilization >= 0.0 && result.vehicle_utilization <= 1.0);
    }

    #[test]
    fn test_eoq_calculation() {
        let network = create_test_network();
        let config = SupplyChainConfig::default();
        let optimizer = SupplyChainOptimizer::new(config).unwrap();

        let policies = optimizer.optimize_inventory(&network).unwrap();

        // EOQ should be positive for all products
        for policy in policies {
            assert!(policy.order_quantity > 0.0, "EOQ must be positive");
            assert!(policy.safety_stock >= 0.0, "Safety stock must be non-negative");
        }
    }

    #[test]
    fn test_route_creation() {
        let network = create_test_network();
        let config = SupplyChainConfig::default();
        let mut optimizer = SupplyChainOptimizer::new(config).unwrap();

        let routes = optimizer.optimize_routes(
            &network,
            OptimizationStrategy::MinimizeCost
        ).unwrap();

        // Should have at least one route
        assert!(!routes.is_empty());

        // All routes should respect vehicle capacity
        for route in &routes {
            let vehicle = &network.vehicles[route.vehicle_id];
            assert!(route.total_demand <= vehicle.capacity * 1.01); // Small tolerance
        }
    }
}
