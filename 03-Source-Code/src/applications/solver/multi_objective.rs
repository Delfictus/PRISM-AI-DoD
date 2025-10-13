//! Multi-Objective Optimization - Worker 4
//!
//! Pareto-optimal solution generation using NSGA-II (Non-dominated Sorting Genetic Algorithm II).
//! Handles problems with multiple conflicting objectives.
//!
//! # Algorithm: NSGA-II
//!
//! 1. **Non-dominated Sorting**: Rank solutions by Pareto dominance
//! 2. **Crowding Distance**: Maintain diversity in objective space
//! 3. **Selection**: Tournament selection with rank and crowding
//! 4. **Variation**: Crossover and mutation operators
//! 5. **Elitism**: Preserve best solutions across generations
//!
//! # Mathematical Foundation
//!
//! **Pareto Dominance**: Solution x dominates y if:
//! - ∀i: f_i(x) ≤ f_i(y) (minimization)
//! - ∃j: f_j(x) < f_j(y)
//!
//! **Crowding Distance**: Measure of solution density in objective space
//! CD_i = Σ_m (f_m^(i+1) - f_m^(i-1)) / (f_m^max - f_m^min)

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{Problem, Solution, ProblemType};

/// Multi-objective problem specification
pub struct MultiObjectiveProblem {
    /// Number of objectives
    pub num_objectives: usize,

    /// Objective names
    pub objective_names: Vec<String>,

    /// Whether each objective should be minimized (true) or maximized (false)
    pub minimize: Vec<bool>,

    /// Underlying problem data
    pub problem: Problem,

    /// Objective evaluation function
    /// Takes a solution vector and returns objective values
    pub evaluate_objectives: Box<dyn Fn(&Array1<f64>) -> Vec<f64> + Send + Sync>,
}

/// Individual solution in multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveSolution {
    /// Solution vector
    pub solution: Vec<f64>,

    /// Objective values
    pub objectives: Vec<f64>,

    /// Pareto rank (1 = non-dominated front)
    pub rank: usize,

    /// Crowding distance (larger = less crowded)
    pub crowding_distance: f64,

    /// Whether this solution is on the Pareto front
    pub is_pareto_optimal: bool,
}

impl MultiObjectiveSolution {
    /// Check if this solution dominates another
    pub fn dominates(&self, other: &Self, minimize: &[bool]) -> bool {
        let mut at_least_one_better = false;
        let mut all_as_good_or_better = true;

        for (i, (&self_obj, &other_obj)) in self.objectives.iter().zip(other.objectives.iter()).enumerate() {
            let comparison = if minimize[i] {
                // Minimization: lower is better
                if self_obj < other_obj {
                    at_least_one_better = true;
                } else if self_obj > other_obj {
                    all_as_good_or_better = false;
                }
            } else {
                // Maximization: higher is better
                if self_obj > other_obj {
                    at_least_one_better = true;
                } else if self_obj < other_obj {
                    all_as_good_or_better = false;
                }
            };
        }

        at_least_one_better && all_as_good_or_better
    }
}

/// Pareto front result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFront {
    /// Solutions on the Pareto front
    pub solutions: Vec<MultiObjectiveSolution>,

    /// Number of objectives
    pub num_objectives: usize,

    /// Objective names
    pub objective_names: Vec<String>,

    /// Hypervolume indicator (quality metric)
    pub hypervolume: Option<f64>,
}

impl ParetoFront {
    /// Get solution with best value for a specific objective
    pub fn best_for_objective(&self, objective_idx: usize, minimize: bool) -> Option<&MultiObjectiveSolution> {
        self.solutions.iter().min_by(|a, b| {
            let cmp = a.objectives[objective_idx].partial_cmp(&b.objectives[objective_idx]).unwrap();
            if minimize {
                cmp
            } else {
                cmp.reverse()
            }
        })
    }

    /// Get knee point (balanced solution)
    /// Uses distance from ideal point heuristic
    pub fn knee_point(&self, minimize: &[bool]) -> Option<&MultiObjectiveSolution> {
        if self.solutions.is_empty() {
            return None;
        }

        // Find ideal point (best values for each objective)
        let mut ideal_point = vec![0.0; self.num_objectives];
        for i in 0..self.num_objectives {
            ideal_point[i] = if minimize[i] {
                self.solutions
                    .iter()
                    .map(|s| s.objectives[i])
                    .fold(f64::INFINITY, f64::min)
            } else {
                self.solutions
                    .iter()
                    .map(|s| s.objectives[i])
                    .fold(f64::NEG_INFINITY, f64::max)
            };
        }

        // Normalize objectives
        let mut ranges = vec![1.0; self.num_objectives];
        for i in 0..self.num_objectives {
            let min_val = self.solutions.iter().map(|s| s.objectives[i]).fold(f64::INFINITY, f64::min);
            let max_val = self.solutions.iter().map(|s| s.objectives[i]).fold(f64::NEG_INFINITY, f64::max);
            ranges[i] = (max_val - min_val).max(1e-10);
        }

        // Find solution closest to ideal point
        self.solutions.iter().min_by(|a, b| {
            let dist_a: f64 = a.objectives
                .iter()
                .zip(ideal_point.iter())
                .zip(ranges.iter())
                .enumerate()
                .map(|(i, ((&obj, &ideal), &range))| {
                    let normalized = (obj - ideal) / range;
                    let signed = if minimize[i] { normalized } else { -normalized };
                    signed * signed
                })
                .sum();

            let dist_b: f64 = b.objectives
                .iter()
                .zip(ideal_point.iter())
                .zip(ranges.iter())
                .enumerate()
                .map(|(i, ((&obj, &ideal), &range))| {
                    let normalized = (obj - ideal) / range;
                    let signed = if minimize[i] { normalized } else { -normalized };
                    signed * signed
                })
                .sum();

            dist_a.partial_cmp(&dist_b).unwrap()
        })
    }
}

/// NSGA-II optimizer configuration
#[derive(Debug, Clone)]
pub struct NsgaIIConfig {
    /// Population size
    pub population_size: usize,

    /// Number of generations
    pub num_generations: usize,

    /// Crossover probability
    pub crossover_prob: f64,

    /// Mutation probability
    pub mutation_prob: f64,

    /// Tournament size for selection
    pub tournament_size: usize,

    /// Random seed
    pub seed: u64,
}

impl Default for NsgaIIConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            num_generations: 50,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            tournament_size: 2,
            seed: 42,
        }
    }
}

/// NSGA-II multi-objective optimizer
pub struct NsgaIIOptimizer {
    config: NsgaIIConfig,
    generation: usize,
}

impl NsgaIIOptimizer {
    /// Create a new NSGA-II optimizer
    pub fn new(config: NsgaIIConfig) -> Self {
        Self {
            config,
            generation: 0,
        }
    }

    /// Optimize a multi-objective problem
    pub fn optimize(&mut self, problem: &MultiObjectiveProblem) -> Result<ParetoFront> {
        // Initialize population
        let mut population = self.initialize_population(problem)?;

        // Evaluate initial population
        self.evaluate_population(&mut population, problem);

        // Evolution loop
        for gen in 0..self.config.num_generations {
            self.generation = gen;

            // Non-dominated sorting
            self.fast_non_dominated_sort(&mut population, &problem.minimize);

            // Calculate crowding distances
            self.calculate_crowding_distance(&mut population);

            // Create offspring
            let offspring = self.create_offspring(&population, problem)?;

            // Combine parent and offspring
            population.extend(offspring);

            // Evaluate new solutions
            self.evaluate_population(&mut population, problem);

            // Non-dominated sorting (combined population)
            self.fast_non_dominated_sort(&mut population, &problem.minimize);

            // Calculate crowding distances
            self.calculate_crowding_distance(&mut population);

            // Environmental selection (keep best N)
            population = self.environmental_selection(population);
        }

        // Extract Pareto front
        self.extract_pareto_front(population, problem)
    }

    /// Initialize random population
    fn initialize_population(&self, problem: &MultiObjectiveProblem) -> Result<Vec<MultiObjectiveSolution>> {
        let mut population = Vec::with_capacity(self.config.population_size);

        // Simple random initialization (uniform distribution)
        // In production, would use problem-specific initialization
        let dimension = 10; // Default dimension

        for i in 0..self.config.population_size {
            let seed = self.config.seed + i as u64;
            let mut solution = vec![0.0; dimension];

            for j in 0..dimension {
                let random_seed = seed + j as u64;
                let random_value = (random_seed as f64 / u64::MAX as f64).clamp(0.0, 1.0);
                solution[j] = random_value;
            }

            population.push(MultiObjectiveSolution {
                solution,
                objectives: vec![],
                rank: 0,
                crowding_distance: 0.0,
                is_pareto_optimal: false,
            });
        }

        Ok(population)
    }

    /// Evaluate objectives for all solutions
    fn evaluate_population(&self, population: &mut [MultiObjectiveSolution], problem: &MultiObjectiveProblem) {
        for solution in population.iter_mut() {
            let solution_array = Array1::from_vec(solution.solution.clone());
            solution.objectives = (problem.evaluate_objectives)(&solution_array);
        }
    }

    /// Fast non-dominated sorting (NSGA-II)
    fn fast_non_dominated_sort(&self, population: &mut [MultiObjectiveSolution], minimize: &[bool]) {
        let n = population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if population[i].dominates(&population[j], minimize) {
                    dominated_solutions[i].push(j);
                } else if population[j].dominates(&population[i], minimize) {
                    domination_count[i] += 1;
                }
            }

            if domination_count[i] == 0 {
                population[i].rank = 1;
                population[i].is_pareto_optimal = true;
                fronts[0].push(i);
            }
        }

        // Build subsequent fronts
        let mut current_front = 0;
        while !fronts[current_front].is_empty() {
            let mut next_front = Vec::new();

            for &i in &fronts[current_front] {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        population[j].rank = current_front + 2;
                        next_front.push(j);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
                current_front += 1;
            } else {
                break;
            }
        }
    }

    /// Calculate crowding distance for each solution
    fn calculate_crowding_distance(&self, population: &mut [MultiObjectiveSolution]) {
        if population.is_empty() {
            return;
        }

        let num_objectives = population[0].objectives.len();

        // Initialize crowding distances
        for solution in population.iter_mut() {
            solution.crowding_distance = 0.0;
        }

        // For each objective
        for obj_idx in 0..num_objectives {
            // Sort by objective value
            let mut indices: Vec<usize> = (0..population.len()).collect();
            indices.sort_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap()
            });

            // Boundary solutions get infinite distance
            population[indices[0]].crowding_distance = f64::INFINITY;
            population[indices[indices.len() - 1]].crowding_distance = f64::INFINITY;

            // Calculate range
            let obj_range = population[indices[indices.len() - 1]].objectives[obj_idx]
                - population[indices[0]].objectives[obj_idx];

            if obj_range == 0.0 {
                continue;
            }

            // Calculate crowding distance for intermediate solutions
            for i in 1..indices.len() - 1 {
                let distance = (population[indices[i + 1]].objectives[obj_idx]
                    - population[indices[i - 1]].objectives[obj_idx])
                    / obj_range;
                population[indices[i]].crowding_distance += distance;
            }
        }
    }

    /// Create offspring through selection, crossover, and mutation
    fn create_offspring(
        &self,
        population: &[MultiObjectiveSolution],
        problem: &MultiObjectiveProblem,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        let mut offspring = Vec::with_capacity(self.config.population_size);

        for i in 0..self.config.population_size / 2 {
            // Tournament selection
            let parent1 = self.tournament_selection(population, i * 2);
            let parent2 = self.tournament_selection(population, i * 2 + 1);

            // Crossover
            let (mut child1, mut child2) = if ((self.config.seed + i as u64) as f64 / u64::MAX as f64) < self.config.crossover_prob {
                self.simulated_binary_crossover(&parent1.solution, &parent2.solution, i)
            } else {
                (parent1.solution.clone(), parent2.solution.clone())
            };

            // Mutation
            self.polynomial_mutation(&mut child1, i * 2);
            self.polynomial_mutation(&mut child2, i * 2 + 1);

            offspring.push(MultiObjectiveSolution {
                solution: child1,
                objectives: vec![],
                rank: 0,
                crowding_distance: 0.0,
                is_pareto_optimal: false,
            });

            offspring.push(MultiObjectiveSolution {
                solution: child2,
                objectives: vec![],
                rank: 0,
                crowding_distance: 0.0,
                is_pareto_optimal: false,
            });
        }

        Ok(offspring)
    }

    /// Tournament selection based on rank and crowding distance
    fn tournament_selection<'a>(&self, population: &'a [MultiObjectiveSolution], seed_offset: usize) -> &'a MultiObjectiveSolution {
        let seed = self.config.seed + seed_offset as u64;
        let idx1 = (seed % population.len() as u64) as usize;
        let idx2 = ((seed + 1000) % population.len() as u64) as usize;

        let sol1 = &population[idx1];
        let sol2 = &population[idx2];

        // Compare by rank first, then by crowding distance
        if sol1.rank < sol2.rank {
            sol1
        } else if sol1.rank > sol2.rank {
            sol2
        } else if sol1.crowding_distance > sol2.crowding_distance {
            sol1
        } else {
            sol2
        }
    }

    /// Simulated binary crossover (SBX)
    fn simulated_binary_crossover(&self, parent1: &[f64], parent2: &[f64], seed_offset: usize) -> (Vec<f64>, Vec<f64>) {
        let n = parent1.len();
        let mut child1 = parent1.to_vec();
        let mut child2 = parent2.to_vec();
        let eta = 20.0; // Distribution index

        for i in 0..n {
            let seed = self.config.seed + seed_offset as u64 + i as u64;
            let u = (seed as f64 / u64::MAX as f64).clamp(1e-10, 1.0 - 1e-10);

            let beta = if u <= 0.5 {
                (2.0 * u).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
            };

            child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i]);
            child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i]);

            child1[i] = child1[i].clamp(0.0, 1.0);
            child2[i] = child2[i].clamp(0.0, 1.0);
        }

        (child1, child2)
    }

    /// Polynomial mutation
    fn polynomial_mutation(&self, solution: &mut [f64], seed_offset: usize) {
        let eta = 20.0; // Distribution index

        for i in 0..solution.len() {
            let seed = self.config.seed + seed_offset as u64 + i as u64;
            let u = (seed as f64 / u64::MAX as f64).clamp(0.0, 1.0);

            if u < self.config.mutation_prob {
                let r = ((seed + 1) as f64 / u64::MAX as f64).clamp(1e-10, 1.0 - 1e-10);

                let delta = if r < 0.5 {
                    (2.0 * r).powf(1.0 / (eta + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - r)).powf(1.0 / (eta + 1.0))
                };

                solution[i] = (solution[i] + delta).clamp(0.0, 1.0);
            }
        }
    }

    /// Environmental selection: keep best N solutions
    fn environmental_selection(&self, mut population: Vec<MultiObjectiveSolution>) -> Vec<MultiObjectiveSolution> {
        // Sort by rank, then by crowding distance
        population.sort_by(|a, b| {
            if a.rank != b.rank {
                a.rank.cmp(&b.rank)
            } else {
                b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap()
            }
        });

        population.truncate(self.config.population_size);
        population
    }

    /// Extract Pareto front from final population
    fn extract_pareto_front(
        &self,
        population: Vec<MultiObjectiveSolution>,
        problem: &MultiObjectiveProblem,
    ) -> Result<ParetoFront> {
        let pareto_solutions: Vec<MultiObjectiveSolution> = population
            .into_iter()
            .filter(|s| s.is_pareto_optimal)
            .collect();

        Ok(ParetoFront {
            solutions: pareto_solutions,
            num_objectives: problem.num_objectives,
            objective_names: problem.objective_names.clone(),
            hypervolume: None, // TODO: Calculate hypervolume
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    // Test objective function: ZDT1 benchmark
    fn zdt1_objectives(x: &Array1<f64>) -> Vec<f64> {
        let n = x.len();
        let f1 = x[0];

        let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (n as f64 - 1.0);
        let f2 = g * (1.0 - (f1 / g).sqrt());

        vec![f1, f2]
    }

    #[test]
    fn test_pareto_dominance() {
        let sol1 = MultiObjectiveSolution {
            solution: vec![0.1, 0.2],
            objectives: vec![1.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
            is_pareto_optimal: false,
        };

        let sol2 = MultiObjectiveSolution {
            solution: vec![0.3, 0.4],
            objectives: vec![1.5, 2.5],
            rank: 0,
            crowding_distance: 0.0,
            is_pareto_optimal: false,
        };

        let minimize = vec![true, true];
        assert!(sol1.dominates(&sol2, &minimize));
        assert!(!sol2.dominates(&sol1, &minimize));
    }

    #[test]
    fn test_nsga_ii_initialization() {
        let config = NsgaIIConfig {
            population_size: 10,
            ..Default::default()
        };

        let optimizer = NsgaIIOptimizer::new(config);

        let problem = MultiObjectiveProblem {
            num_objectives: 2,
            objective_names: vec!["f1".to_string(), "f2".to_string()],
            minimize: vec![true, true],
            problem: Problem {
                problem_type: ProblemType::ContinuousOptimization,
                description: "Test problem".to_string(),
                data: super::super::problem::ProblemData::Continuous {
                    variables: (0..10).map(|i| format!("x{}", i)).collect(),
                    bounds: vec![(0.0, 1.0); 10],
                    objective: super::super::problem::ObjectiveFunction::Minimize("test".to_string()),
                },
                constraints: Vec::new(),
                metadata: std::collections::HashMap::new(),
            },
            evaluate_objectives: Box::new(zdt1_objectives),
        };

        let population = optimizer.initialize_population(&problem);
        assert!(population.is_ok());
        assert_eq!(population.unwrap().len(), 10);
    }

    #[test]
    fn test_nsga_ii_optimization() {
        let config = NsgaIIConfig {
            population_size: 20,
            num_generations: 10,
            ..Default::default()
        };

        let mut optimizer = NsgaIIOptimizer::new(config);

        let problem = MultiObjectiveProblem {
            num_objectives: 2,
            objective_names: vec!["f1".to_string(), "f2".to_string()],
            minimize: vec![true, true],
            problem: Problem {
                problem_type: ProblemType::ContinuousOptimization,
                description: "Test problem".to_string(),
                data: super::super::problem::ProblemData::Continuous {
                    variables: (0..5).map(|i| format!("x{}", i)).collect(),
                    bounds: vec![(0.0, 1.0); 5],
                    objective: super::super::problem::ObjectiveFunction::Minimize("test".to_string()),
                },
                constraints: Vec::new(),
                metadata: std::collections::HashMap::new(),
            },
            evaluate_objectives: Box::new(zdt1_objectives),
        };

        let result = optimizer.optimize(&problem);
        assert!(result.is_ok());

        let front = result.unwrap();
        assert!(!front.solutions.is_empty());
        assert!(front.solutions.len() <= 20);
    }

    #[test]
    fn test_pareto_front_knee_point() {
        let front = ParetoFront {
            solutions: vec![
                MultiObjectiveSolution {
                    solution: vec![0.0],
                    objectives: vec![0.0, 1.0],
                    rank: 1,
                    crowding_distance: 0.0,
                    is_pareto_optimal: true,
                },
                MultiObjectiveSolution {
                    solution: vec![0.5],
                    objectives: vec![0.5, 0.5],
                    rank: 1,
                    crowding_distance: 0.0,
                    is_pareto_optimal: true,
                },
                MultiObjectiveSolution {
                    solution: vec![1.0],
                    objectives: vec![1.0, 0.0],
                    rank: 1,
                    crowding_distance: 0.0,
                    is_pareto_optimal: true,
                },
            ],
            num_objectives: 2,
            objective_names: vec!["f1".to_string(), "f2".to_string()],
            hypervolume: None,
        };

        let minimize = vec![true, true];
        let knee = front.knee_point(&minimize);
        assert!(knee.is_some());

        // Knee point should be the middle solution
        let knee_sol = knee.unwrap();
        assert!((knee_sol.objectives[0] - 0.5).abs() < 0.1);
    }
}
