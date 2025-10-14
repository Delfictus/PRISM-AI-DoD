//! Problem Specification Types for Universal Solver

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ProblemType;

/// Problem specification for universal solver
#[derive(Debug, Clone)]
pub struct Problem {
    /// Problem type (auto-detected if Unknown)
    pub problem_type: ProblemType,

    /// Problem description
    pub description: String,

    /// Problem data (flexible key-value store)
    pub data: ProblemData,

    /// Constraints (optional)
    pub constraints: Vec<Constraint>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Problem data container
#[derive(Debug, Clone)]
pub enum ProblemData {
    /// Continuous vector data
    Continuous {
        variables: Vec<String>,
        bounds: Vec<(f64, f64)>,
        objective: ObjectiveFunction,
    },

    /// Discrete/integer data
    Discrete {
        variables: Vec<String>,
        domains: Vec<Vec<i64>>,
        objective: ObjectiveFunction,
    },

    /// Graph data
    Graph {
        adjacency_matrix: Array2<bool>,
        node_labels: Option<Vec<String>>,
        edge_weights: Option<Array2<f64>>,
    },

    /// Time series data
    TimeSeries {
        series: Array1<f64>,
        timestamps: Option<Vec<f64>>,
        horizon: usize,
    },

    /// Portfolio data
    Portfolio {
        assets: Vec<AssetSpec>,
        target_return: Option<f64>,
        max_risk: Option<f64>,
    },

    /// Generic tabular data
    Tabular {
        features: Array2<f64>,
        targets: Option<Array1<f64>>,
        feature_names: Vec<String>,
    },
}

/// Asset specification for portfolio problems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSpec {
    pub symbol: String,
    pub name: String,
    pub current_price: f64,
    pub historical_returns: Vec<f64>,
}

/// Objective function specification
#[derive(Debug, Clone)]
pub enum ObjectiveFunction {
    /// Minimize expression
    Minimize(String),

    /// Maximize expression
    Maximize(String),

    /// Custom evaluation function (not serializable)
    Custom(fn(&[f64]) -> f64),
}

/// Constraint specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Linear equality: Ax = b
    LinearEquality {
        coefficients: Vec<f64>,
        rhs: f64,
    },

    /// Linear inequality: Ax <= b
    LinearInequality {
        coefficients: Vec<f64>,
        rhs: f64,
    },

    /// Bounds: l <= x <= u
    Bounds {
        variable_index: usize,
        lower: f64,
        upper: f64,
    },

    /// Non-linear constraint
    NonLinear {
        expression: String,
        constraint_type: ConstraintType,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Equality,
    Inequality,
}

impl Problem {
    /// Create a new problem
    pub fn new(problem_type: ProblemType, description: String, data: ProblemData) -> Self {
        Self {
            problem_type,
            description,
            data,
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Get problem dimension
    pub fn dimension(&self) -> Result<usize> {
        match &self.data {
            ProblemData::Continuous { variables, .. } => Ok(variables.len()),
            ProblemData::Discrete { variables, .. } => Ok(variables.len()),
            ProblemData::Graph { adjacency_matrix, .. } => Ok(adjacency_matrix.nrows()),
            ProblemData::TimeSeries { series, .. } => Ok(series.len()),
            ProblemData::Portfolio { assets, .. } => Ok(assets.len()),
            ProblemData::Tabular { features, .. } => Ok(features.ncols()),
        }
    }

    /// Check if problem has constraints
    pub fn has_constraints(&self) -> bool {
        !self.constraints.is_empty()
    }
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Problem: {} ({})\nDimension: {:?}\nConstraints: {}",
            self.description,
            self.problem_type.description(),
            self.dimension(),
            self.constraints.len()
        )
    }
}
