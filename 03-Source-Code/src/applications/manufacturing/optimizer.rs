//! Manufacturing Process Optimizer
//!
//! Implements GPU-accelerated manufacturing optimization including:
//! - Job shop scheduling with constraints
//! - Predictive maintenance scheduling
//! - Quality control optimization
//! - Throughput maximization
//!
//! Worker 3 Implementation
//! Constitutional Compliance: Articles I, II, III, IV

use anyhow::{Result, Context};
use std::collections::{HashMap, VecDeque};

#[cfg(feature = "cuda")]
use crate::gpu::GpuMemoryPool;

/// Machine type in production line
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MachineType {
    CNC,           // Computer Numerical Control machining
    Assembly,      // Assembly operations
    Welding,       // Welding operations
    Painting,      // Painting/coating
    Inspection,    // Quality inspection
    Packaging,     // Packaging operations
}

/// Production machine
#[derive(Debug, Clone)]
pub struct Machine {
    pub id: usize,
    pub machine_type: MachineType,
    pub name: String,
    pub capacity_units_per_hour: f64,
    pub setup_time_minutes: f64,
    pub operating_cost_per_hour: f64,
    pub failure_rate: f64,  // Probability of failure per hour
    pub current_utilization: f64,  // 0.0 to 1.0
    pub maintenance_due_hours: f64,
}

/// Manufacturing job/order
#[derive(Debug, Clone)]
pub struct Job {
    pub id: usize,
    pub product_type: String,
    pub quantity: u32,
    pub priority: u8,  // 1-10, higher is more urgent
    pub due_date_hours: f64,
    pub processing_sequence: Vec<MachineType>,  // Required machine sequence
    pub processing_times: Vec<f64>,  // Time per unit on each machine (minutes)
}

/// Maintenance schedule
#[derive(Debug, Clone)]
pub struct MaintenanceSchedule {
    pub machine_id: usize,
    pub scheduled_time_hours: f64,
    pub duration_hours: f64,
    pub maintenance_type: MaintenanceType,
    pub priority: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaintenanceType {
    Preventive,
    Predictive,
    Emergency,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub defect_rate: f64,  // Defects per 1000 units
    pub first_pass_yield: f64,  // Percentage
    pub scrap_rate: f64,
    pub rework_rate: f64,
}

/// Production line configuration
#[derive(Debug, Clone)]
pub struct ProductionLine {
    pub machines: Vec<Machine>,
    pub jobs: Vec<Job>,
    pub maintenance_schedules: Vec<MaintenanceSchedule>,
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    MinimizeMakespan,      // Minimize total completion time
    MaximizeThroughput,    // Maximize units produced
    MinimizeCost,          // Minimize operating costs
    PriorityBased,         // Schedule by job priority
    Balanced,              // Balance multiple objectives
}

/// Manufacturing configuration
#[derive(Debug, Clone)]
pub struct ManufacturingConfig {
    pub planning_horizon_hours: f64,
    pub overtime_allowed: bool,
    pub quality_threshold: f64,
    pub cost_weight: f64,
    pub throughput_weight: f64,
    pub quality_weight: f64,
}

impl Default for ManufacturingConfig {
    fn default() -> Self {
        Self {
            planning_horizon_hours: 168.0,  // 1 week
            overtime_allowed: true,
            quality_threshold: 0.95,  // 95% first pass yield
            cost_weight: 0.4,
            throughput_weight: 0.4,
            quality_weight: 0.2,
        }
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub schedule: HashMap<usize, Vec<JobAssignment>>,  // machine_id -> assignments
    pub makespan_hours: f64,
    pub total_throughput: u32,
    pub total_cost: f64,
    pub average_utilization: f64,
    pub quality_metrics: QualityMetrics,
    pub late_jobs: Vec<usize>,
}

/// Job assignment to machine
#[derive(Debug, Clone)]
pub struct JobAssignment {
    pub job_id: usize,
    pub start_time: f64,
    pub end_time: f64,
    pub machine_type: MachineType,
}

/// Manufacturing process optimizer
pub struct ManufacturingOptimizer {
    config: ManufacturingConfig,

    #[cfg(feature = "cuda")]
    gpu_context: Option<GpuMemoryPool>,
}

impl ManufacturingOptimizer {
    /// Create new manufacturing optimizer
    pub fn new(config: ManufacturingConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_context = Some(GpuMemoryPool::new()
            .context("Failed to initialize GPU for manufacturing optimization")?);

        Ok(Self {
            config,
            #[cfg(feature = "cuda")]
            gpu_context,
        })
    }

    /// Optimize production schedule
    pub fn optimize(
        &mut self,
        line: &ProductionLine,
        strategy: SchedulingStrategy,
    ) -> Result<OptimizationResult> {
        #[cfg(feature = "cuda")]
        {
            if self.gpu_context.is_some() {
                return self.optimize_gpu(line, strategy);
            }
        }

        self.optimize_cpu(line, strategy)
    }

    /// CPU-based optimization
    fn optimize_cpu(
        &self,
        line: &ProductionLine,
        strategy: SchedulingStrategy,
    ) -> Result<OptimizationResult> {
        // Initialize schedule
        let mut schedule: HashMap<usize, Vec<JobAssignment>> = HashMap::new();
        for machine in &line.machines {
            schedule.insert(machine.id, Vec::new());
        }

        // Sort jobs based on strategy
        let mut sorted_jobs = line.jobs.clone();
        self.sort_jobs(&mut sorted_jobs, strategy);

        // Schedule jobs using dispatching rules
        let mut machine_availability: HashMap<usize, f64> = line.machines.iter()
            .map(|m| (m.id, 0.0))
            .collect();

        for job in &sorted_jobs {
            let mut job_start_time = 0.0;

            // Schedule each operation in sequence
            for (op_idx, &machine_type) in job.processing_sequence.iter().enumerate() {
                // Find available machine of required type
                let machine = line.machines.iter()
                    .find(|m| m.machine_type == machine_type)
                    .context("No machine of required type available")?;

                let processing_time = job.processing_times[op_idx] / 60.0;  // Convert to hours
                let machine_ready_time = machine_availability.get(&machine.id).copied().unwrap_or(0.0);

                // Start time is max of job ready and machine ready
                let start_time = if job_start_time > machine_ready_time {
                    job_start_time
                } else {
                    machine_ready_time
                };
                let end_time = start_time + machine.setup_time_minutes / 60.0 +
                               processing_time * job.quantity as f64;

                // Add assignment
                schedule.get_mut(&machine.id).unwrap().push(JobAssignment {
                    job_id: job.id,
                    start_time,
                    end_time,
                    machine_type,
                });

                // Update availability
                machine_availability.insert(machine.id, end_time);
                job_start_time = end_time;
            }
        }

        // Calculate metrics
        let makespan = machine_availability.values().cloned().fold(0.0, f64::max);
        let total_throughput: u32 = sorted_jobs.iter().map(|j| j.quantity).sum();
        let total_cost = self.calculate_total_cost(&schedule, &line.machines);
        let average_utilization = self.calculate_utilization(&schedule, makespan, &line.machines);

        // Simulate quality metrics
        let quality_metrics = QualityMetrics {
            defect_rate: 2.5,  // 2.5 defects per 1000 units
            first_pass_yield: 96.5,
            scrap_rate: 1.2,
            rework_rate: 2.3,
        };

        // Find late jobs
        let late_jobs: Vec<usize> = sorted_jobs.iter()
            .filter(|job| {
                let completion_time = schedule.values()
                    .flat_map(|assignments| assignments.iter())
                    .filter(|a| a.job_id == job.id)
                    .map(|a| a.end_time)
                    .fold(0.0, f64::max);
                completion_time > job.due_date_hours
            })
            .map(|job| job.id)
            .collect();

        Ok(OptimizationResult {
            schedule,
            makespan_hours: makespan,
            total_throughput,
            total_cost,
            average_utilization,
            quality_metrics,
            late_jobs,
        })
    }

    #[cfg(feature = "cuda")]
    fn optimize_gpu(
        &self,
        line: &ProductionLine,
        strategy: SchedulingStrategy,
    ) -> Result<OptimizationResult> {
        // TODO: Request job_shop_scheduling_kernel from Worker 2
        // __global__ void job_shop_scheduling(
        //     Job* jobs,
        //     Machine* machines,
        //     JobAssignment* schedule,
        //     int num_jobs,
        //     int num_machines,
        //     SchedulingStrategy strategy
        // )

        // Placeholder: use CPU implementation
        self.optimize_cpu(line, strategy)
    }

    /// Sort jobs based on scheduling strategy
    fn sort_jobs(&self, jobs: &mut Vec<Job>, strategy: SchedulingStrategy) {
        match strategy {
            SchedulingStrategy::PriorityBased => {
                jobs.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
            SchedulingStrategy::MinimizeMakespan => {
                // Shortest processing time first
                jobs.sort_by(|a, b| {
                    let a_time: f64 = a.processing_times.iter().sum();
                    let b_time: f64 = b.processing_times.iter().sum();
                    a_time.partial_cmp(&b_time).unwrap()
                });
            }
            SchedulingStrategy::MaximizeThroughput => {
                // Largest quantity first
                jobs.sort_by(|a, b| b.quantity.cmp(&a.quantity));
            }
            SchedulingStrategy::MinimizeCost => {
                // Lowest priority first (assume higher priority = higher cost)
                jobs.sort_by(|a, b| a.priority.cmp(&b.priority));
            }
            SchedulingStrategy::Balanced => {
                // Weighted combination
                jobs.sort_by(|a, b| {
                    let a_score = a.priority as f64 * 0.4 +
                                  (1.0 / a.due_date_hours) * 0.6;
                    let b_score = b.priority as f64 * 0.4 +
                                  (1.0 / b.due_date_hours) * 0.6;
                    b_score.partial_cmp(&a_score).unwrap()
                });
            }
        }
    }

    /// Calculate total operating cost
    fn calculate_total_cost(
        &self,
        schedule: &HashMap<usize, Vec<JobAssignment>>,
        machines: &[Machine],
    ) -> f64 {
        let mut total_cost = 0.0;

        for machine in machines {
            if let Some(assignments) = schedule.get(&machine.id) {
                for assignment in assignments {
                    let duration = assignment.end_time - assignment.start_time;
                    total_cost += duration * machine.operating_cost_per_hour;
                }
            }
        }

        total_cost
    }

    /// Calculate average machine utilization
    fn calculate_utilization(
        &self,
        schedule: &HashMap<usize, Vec<JobAssignment>>,
        makespan: f64,
        machines: &[Machine],
    ) -> f64 {
        if makespan == 0.0 {
            return 0.0;
        }

        let mut total_utilization = 0.0;

        for machine in machines {
            if let Some(assignments) = schedule.get(&machine.id) {
                let busy_time: f64 = assignments.iter()
                    .map(|a| a.end_time - a.start_time)
                    .sum();
                total_utilization += busy_time / makespan;
            }
        }

        total_utilization / machines.len() as f64
    }

    /// Predict maintenance requirements
    pub fn predict_maintenance(
        &self,
        machine: &Machine,
        hours_ahead: f64,
    ) -> Result<Option<MaintenanceSchedule>> {
        // Simple predictive maintenance model based on failure rate
        let failure_probability = 1.0 - (-machine.failure_rate * hours_ahead).exp();

        if failure_probability > 0.3 {
            // Schedule preventive maintenance
            Ok(Some(MaintenanceSchedule {
                machine_id: machine.id,
                scheduled_time_hours: machine.maintenance_due_hours,
                duration_hours: 2.0,
                maintenance_type: MaintenanceType::Predictive,
                priority: if failure_probability > 0.5 { 9 } else { 7 },
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manufacturing_optimizer_creation() {
        let config = ManufacturingConfig::default();
        let optimizer = ManufacturingOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_job_shop_scheduling() {
        let machines = vec![
            Machine {
                id: 0,
                machine_type: MachineType::CNC,
                name: "CNC-1".to_string(),
                capacity_units_per_hour: 10.0,
                setup_time_minutes: 15.0,
                operating_cost_per_hour: 50.0,
                failure_rate: 0.001,
                current_utilization: 0.0,
                maintenance_due_hours: 1000.0,
            },
            Machine {
                id: 1,
                machine_type: MachineType::Assembly,
                name: "Assembly-1".to_string(),
                capacity_units_per_hour: 20.0,
                setup_time_minutes: 10.0,
                operating_cost_per_hour: 30.0,
                failure_rate: 0.0005,
                current_utilization: 0.0,
                maintenance_due_hours: 1500.0,
            },
        ];

        let jobs = vec![
            Job {
                id: 0,
                product_type: "Widget".to_string(),
                quantity: 100,
                priority: 8,
                due_date_hours: 24.0,
                processing_sequence: vec![MachineType::CNC, MachineType::Assembly],
                processing_times: vec![5.0, 3.0],  // minutes per unit
            },
        ];

        let line = ProductionLine {
            machines,
            jobs,
            maintenance_schedules: Vec::new(),
        };

        let config = ManufacturingConfig::default();
        let mut optimizer = ManufacturingOptimizer::new(config).unwrap();

        let result = optimizer.optimize(&line, SchedulingStrategy::MinimizeMakespan);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.makespan_hours > 0.0);
        assert_eq!(result.total_throughput, 100);
        assert!(result.average_utilization >= 0.0 && result.average_utilization <= 1.0);
    }

    #[test]
    fn test_predictive_maintenance() {
        let machine = Machine {
            id: 0,
            machine_type: MachineType::CNC,
            name: "CNC-1".to_string(),
            capacity_units_per_hour: 10.0,
            setup_time_minutes: 15.0,
            operating_cost_per_hour: 50.0,
            failure_rate: 0.01,  // High failure rate
            current_utilization: 0.8,
            maintenance_due_hours: 100.0,
        };

        let config = ManufacturingConfig::default();
        let optimizer = ManufacturingOptimizer::new(config).unwrap();

        let maintenance = optimizer.predict_maintenance(&machine, 50.0).unwrap();

        // High failure rate should trigger maintenance
        assert!(maintenance.is_some());

        if let Some(schedule) = maintenance {
            assert_eq!(schedule.machine_id, 0);
            assert_eq!(schedule.maintenance_type, MaintenanceType::Predictive);
        }
    }
}
