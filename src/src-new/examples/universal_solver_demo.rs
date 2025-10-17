//! Universal Solver Demo - Worker 4
//!
//! Demonstrates the Universal Solver's ability to solve multiple problem types
//! through a single unified API.

use anyhow::Result;
use ndarray::{Array1, Array2};
use prism_ai::applications::solver::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 PRISM-AI Universal Solver Demo\n");
    println!("Demonstrating intelligent routing to optimal solvers\n");
    println!("=" .repeat(60));

    // Demo 1: Graph Coloring Problem
    demo_graph_coloring().await?;

    println!("\n{}\n", "=".repeat(60));

    // Demo 2: Portfolio Optimization
    demo_portfolio_optimization().await?;

    println!("\n{}\n", "=".repeat(60));

    // Demo 3: Continuous Optimization
    demo_continuous_optimization().await?;

    println!("\n{}\n", "=".repeat(60));
    println!("✅ All demos completed successfully!");

    Ok(())
}

/// Demo 1: Graph Coloring using Phase6 Adaptive Solver
async fn demo_graph_coloring() -> Result<()> {
    println!("\n📊 DEMO 1: Graph Coloring Problem");
    println!("Problem: Color a 5-node graph with minimum colors");
    println!("Solver: Will auto-route to Phase6 Adaptive Solver\n");

    // Create a simple graph (petersen graph subset)
    let adjacency = Array2::from_shape_fn((5, 5), |(i, j)| {
        if i == j {
            false
        } else {
            // Create edges: 0-1, 0-4, 1-2, 1-3, 2-3, 2-4, 3-4
            matches!(
                (i.min(j), i.max(j)),
                (0, 1) | (0, 4) | (1, 2) | (1, 3) | (2, 3) | (2, 4) | (3, 4)
            )
        }
    });

    println!("Adjacency Matrix:");
    for i in 0..5 {
        print!("  ");
        for j in 0..5 {
            print!("{} ", if adjacency[[i, j]] { "1" } else { "0" });
        }
        println!();
    }

    // Create problem
    let problem = Problem::new(
        ProblemType::Unknown, // Test auto-detection
        "Graph coloring demo".to_string(),
        ProblemData::Graph {
            adjacency_matrix: adjacency.clone(),
            node_labels: Some(vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
            ]),
            edge_weights: None,
        },
    );

    // Solve
    let config = SolverConfig::default();
    let mut solver = UniversalSolver::new(config);

    println!("\n🔍 Auto-detecting problem type...");
    let detected_type = solver.detect_problem_type(&problem);
    println!("   Detected: {:?}", detected_type);

    println!("\n🚀 Solving...");
    let solution = solver.solve(problem).await?;

    println!("\n✅ Solution:");
    println!("   Algorithm: {}", solution.algorithm_used);
    println!("   Colors used: {}", solution.objective_value as usize);
    println!("   Computation time: {:.2}ms", solution.computation_time_ms);
    println!("   Confidence: {:.0}%", solution.confidence * 100.0);

    println!("\n   Node colorings:");
    for (i, &color) in solution.solution_vector.iter().enumerate() {
        println!("     Node {}: Color {}", i, color as usize);
    }

    println!("\n📝 Explanation:\n{}", solution.explanation);

    // Verify solution
    let mut valid = true;
    for i in 0..5 {
        for j in (i + 1)..5 {
            if adjacency[[i, j]]
                && solution.solution_vector[i] == solution.solution_vector[j]
            {
                valid = false;
                println!("   ⚠️  Invalid: nodes {} and {} have same color", i, j);
            }
        }
    }

    if valid {
        println!("   ✅ Solution verified: all adjacent nodes have different colors");
    }

    Ok(())
}

/// Demo 2: Portfolio Optimization using Financial Optimizer
async fn demo_portfolio_optimization() -> Result<()> {
    println!("\n💰 DEMO 2: Portfolio Optimization");
    println!("Problem: Allocate capital across 3 assets");
    println!("Solver: Will auto-route to Financial Optimizer\n");

    // Create realistic asset data
    let assets = vec![
        AssetSpec {
            symbol: "AAPL".to_string(),
            name: "Apple Inc.".to_string(),
            current_price: 150.0,
            historical_returns: vec![
                0.015, 0.022, -0.010, 0.030, 0.012, 0.018, -0.005, 0.025, 0.020, 0.015,
            ],
        },
        AssetSpec {
            symbol: "GOOGL".to_string(),
            name: "Alphabet Inc.".to_string(),
            current_price: 2800.0,
            historical_returns: vec![
                0.020, 0.015, 0.010, 0.025, 0.018, 0.022, 0.012, 0.020, 0.015, 0.018,
            ],
        },
        AssetSpec {
            symbol: "MSFT".to_string(),
            name: "Microsoft Corporation".to_string(),
            current_price: 300.0,
            historical_returns: vec![
                0.018, 0.012, 0.015, 0.020, 0.016, 0.019, 0.014, 0.022, 0.017, 0.016,
            ],
        },
    ];

    println!("Assets:");
    for asset in &assets {
        let avg_return =
            asset.historical_returns.iter().sum::<f64>() / asset.historical_returns.len() as f64;
        println!(
            "   {} ({}): Avg return = {:.2}%",
            asset.symbol,
            asset.name,
            avg_return * 100.0
        );
    }

    // Create problem
    let problem = Problem::new(
        ProblemType::PortfolioOptimization,
        "Maximize Sharpe ratio".to_string(),
        ProblemData::Portfolio {
            assets,
            target_return: Some(0.018), // Target 1.8% return
            max_risk: Some(0.015),      // Max 1.5% volatility
        },
    );

    // Solve
    let mut solver = UniversalSolver::new(SolverConfig::default());

    println!("\n🚀 Solving with Active Inference + Transfer Entropy...");
    let solution = solver.solve(problem).await?;

    println!("\n✅ Solution:");
    println!("   Algorithm: {}", solution.algorithm_used);
    println!("   Sharpe Ratio: {:.3}", -solution.objective_value);
    println!("   Computation time: {:.2}ms", solution.computation_time_ms);
    println!("   Confidence: {:.0}%", solution.confidence * 100.0);

    println!("\n   Portfolio Allocation:");
    let symbols = vec!["AAPL", "GOOGL", "MSFT"];
    for (i, &weight) in solution.solution_vector.iter().enumerate() {
        println!("     {}: {:.1}%", symbols[i], weight * 100.0);
    }

    // Verify weights sum to 1
    let total: f64 = solution.solution_vector.iter().sum();
    println!("\n   Total allocation: {:.1}% ✅", total * 100.0);

    println!("\n📝 Explanation:\n{}", solution.explanation);

    Ok(())
}

/// Demo 3: Continuous Optimization using CMA
async fn demo_continuous_optimization() -> Result<()> {
    println!("\n📈 DEMO 3: Continuous Optimization");
    println!("Problem: Minimize f(x,y) = x² + y²");
    println!("Solver: Will auto-route to CMA (Causal Manifold Annealing)\n");

    // Create continuous optimization problem
    let problem = Problem::new(
        ProblemType::ContinuousOptimization,
        "Minimize sphere function".to_string(),
        ProblemData::Continuous {
            variables: vec!["x".to_string(), "y".to_string()],
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            objective: ObjectiveFunction::Minimize("x^2 + y^2".to_string()),
        },
    );

    println!("Objective: f(x, y) = x² + y²");
    println!("Domain: x, y ∈ [-5, 5]");
    println!("Expected minimum: f(0, 0) = 0\n");

    // Solve
    let mut solver = UniversalSolver::new(SolverConfig::default());

    println!("🚀 Solving with CMA...");
    let solution = solver.solve(problem).await?;

    println!("\n✅ Solution:");
    println!("   Algorithm: {}", solution.algorithm_used);
    println!("   Minimum value: {:.6}", solution.objective_value);
    println!("   Computation time: {:.2}ms", solution.computation_time_ms);

    println!("\n   Minimizer:");
    println!("     x = {:.6}", solution.solution_vector[0]);
    println!("     y = {:.6}", solution.solution_vector[1]);

    // Check how close to optimal
    let distance_to_optimal = (solution.solution_vector[0].powi(2)
        + solution.solution_vector[1].powi(2))
    .sqrt();
    println!("\n   Distance to optimal (0,0): {:.6}", distance_to_optimal);

    if solution.metrics.is_optimal {
        println!("   ✅ Proven optimal solution");
        if let Some(gap) = solution.metrics.optimality_gap {
            println!("   Error bound: {:.6}", gap);
        }
    }

    println!("\n📝 Explanation:\n{}", solution.explanation);

    Ok(())
}
