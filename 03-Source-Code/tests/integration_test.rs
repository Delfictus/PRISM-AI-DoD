//! Integration Tests - Worker 4 Applications Domain
//!
//! End-to-end tests validating:
//! - Financial Portfolio Optimization
//! - Universal Solver routing and problem solving
//! - Integration with Phase6, CMA, and Transfer Entropy

use anyhow::Result;
use ndarray::{Array1, Array2};
use prism_ai::applications::solver::{*, problem::ObjectiveFunction};
use prism_ai::applications::financial::*;

/// Test 1: Graph coloring end-to-end
#[tokio::test]
async fn test_graph_coloring_integration() -> Result<()> {
    // Create a simple 4-coloring problem (K4 - complete graph on 4 vertices)
    let adjacency = Array2::from_shape_fn((4, 4), |(i, j)| i != j);

    let problem = Problem::new(
        ProblemType::Unknown, // Test auto-detection
        "K4 graph coloring".to_string(),
        ProblemData::Graph {
            adjacency_matrix: adjacency.clone(),
            node_labels: Some(vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]),
            edge_weights: None,
        },
    );

    let mut solver = UniversalSolver::new(SolverConfig::default());

    // Test auto-detection
    let detected_type = solver.detect_problem_type(&problem);
    assert_eq!(detected_type, ProblemType::GraphProblem);

    // Solve
    let solution = solver.solve(problem).await?;

    // Validate solution
    assert_eq!(solution.problem_type, ProblemType::GraphProblem);
    assert!(solution.algorithm_used.contains("Phase6"));
    assert_eq!(solution.solution_vector.len(), 4);
    assert!(solution.confidence > 0.0);
    assert!(!solution.explanation.is_empty());

    // Verify coloring is valid (no adjacent nodes have same color)
    for i in 0..4 {
        for j in (i + 1)..4 {
            if adjacency[[i, j]] {
                assert_ne!(
                    solution.solution_vector[i] as usize,
                    solution.solution_vector[j] as usize,
                    "Adjacent nodes {} and {} have same color",
                    i,
                    j
                );
            }
        }
    }

    // K4 requires exactly 4 colors (chromatic number = 4)
    let num_colors = solution.objective_value as usize;
    assert_eq!(num_colors, 4, "K4 should use exactly 4 colors");

    println!("✅ Graph coloring integration test passed");
    Ok(())
}

/// Test 2: Portfolio optimization end-to-end
#[tokio::test]
async fn test_portfolio_optimization_integration() -> Result<()> {
    // Create realistic assets
    let assets = vec![
        AssetSpec {
            symbol: "TECH".to_string(),
            name: "Tech Fund".to_string(),
            current_price: 100.0,
            historical_returns: vec![
                0.02, 0.03, -0.01, 0.04, 0.02, 0.03, 0.01, 0.02, 0.03, 0.02,
            ],
        },
        AssetSpec {
            symbol: "BOND".to_string(),
            name: "Bond Fund".to_string(),
            current_price: 50.0,
            historical_returns: vec![
                0.005, 0.006, 0.005, 0.007, 0.005, 0.006, 0.005, 0.006, 0.005, 0.006,
            ],
        },
        AssetSpec {
            symbol: "GOLD".to_string(),
            name: "Gold ETF".to_string(),
            current_price: 75.0,
            historical_returns: vec![
                0.01, -0.005, 0.015, 0.01, -0.01, 0.02, 0.01, 0.005, 0.01, 0.015,
            ],
        },
    ];

    let problem = Problem::new(
        ProblemType::PortfolioOptimization,
        "Diversified portfolio".to_string(),
        ProblemData::Portfolio {
            assets,
            target_return: Some(0.015), // 1.5% target
            max_risk: Some(0.02),       // 2% max risk
        },
    );

    let mut solver = UniversalSolver::new(SolverConfig::default());
    let solution = solver.solve(problem).await?;

    // Validate solution
    assert_eq!(solution.problem_type, ProblemType::PortfolioOptimization);
    assert!(solution.algorithm_used.contains("ActiveInference") || solution.algorithm_used.contains("MPT"));
    assert_eq!(solution.solution_vector.len(), 3); // 3 assets

    // Check weights sum to 1.0 (fully invested)
    let total_weight: f64 = solution.solution_vector.iter().sum();
    assert!((total_weight - 1.0).abs() < 1e-6, "Weights should sum to 1.0, got {}", total_weight);

    // Check all weights are non-negative (no short selling by default)
    for &weight in &solution.solution_vector {
        assert!(weight >= -1e-6, "Weight should be non-negative, got {}", weight);
        assert!(weight <= 1.0 + 1e-6, "Weight should not exceed 1.0, got {}", weight);
    }

    // Check Sharpe ratio is reasonable (objective is negative Sharpe)
    let sharpe_ratio = -solution.objective_value;
    assert!(sharpe_ratio > 0.0, "Sharpe ratio should be positive");

    println!("✅ Portfolio optimization integration test passed");
    println!("   Allocation: TECH={:.1}%, BOND={:.1}%, GOLD={:.1}%",
        solution.solution_vector[0] * 100.0,
        solution.solution_vector[1] * 100.0,
        solution.solution_vector[2] * 100.0
    );
    println!("   Sharpe Ratio: {:.3}", sharpe_ratio);

    Ok(())
}

/// Test 3: Financial optimizer direct usage
#[test]
fn test_financial_optimizer_direct() -> Result<()> {
    let assets = vec![
        Asset {
            symbol: "ASSET1".to_string(),
            name: "Asset One".to_string(),
            current_price: 100.0,
            historical_returns: vec![0.01, 0.02, 0.015, 0.02, 0.018],
        },
        Asset {
            symbol: "ASSET2".to_string(),
            name: "Asset Two".to_string(),
            current_price: 50.0,
            historical_returns: vec![0.005, 0.008, 0.006, 0.007, 0.006],
        },
    ];

    let mut optimizer = PortfolioOptimizer::new(OptimizationConfig::default());
    let portfolio = optimizer.optimize(assets)?;

    // Validate portfolio
    assert_eq!(portfolio.weights.len(), 2);

    // Check weights sum to 1
    let total: f64 = portfolio.weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "Weights should sum to 1.0");

    // Check non-negative weights
    for &w in portfolio.weights.iter() {
        assert!(w >= -1e-6, "Weights should be non-negative");
    }

    // Check Sharpe ratio is computed
    assert!(portfolio.sharpe_ratio.is_finite());

    println!("✅ Financial optimizer direct test passed");
    println!("   Weights: {:?}", portfolio.weights.to_vec());
    println!("   Expected Return: {:.2}%", portfolio.expected_return * 100.0);
    println!("   Risk: {:.2}%", portfolio.risk * 100.0);
    println!("   Sharpe: {:.3}", portfolio.sharpe_ratio);

    Ok(())
}

/// Test 4: Problem type auto-detection
#[test]
fn test_problem_type_detection() {
    let solver = UniversalSolver::new(SolverConfig::default());

    // Test Graph detection
    let graph_problem = Problem::new(
        ProblemType::Unknown,
        "test".to_string(),
        ProblemData::Graph {
            adjacency_matrix: Array2::from_elem((3, 3), false),
            node_labels: None,
            edge_weights: None,
        },
    );
    assert_eq!(solver.detect_problem_type(&graph_problem), ProblemType::GraphProblem);

    // Test Portfolio detection
    let portfolio_problem = Problem::new(
        ProblemType::Unknown,
        "test".to_string(),
        ProblemData::Portfolio {
            assets: vec![],
            target_return: None,
            max_risk: None,
        },
    );
    assert_eq!(solver.detect_problem_type(&portfolio_problem), ProblemType::PortfolioOptimization);

    // Test Continuous detection
    let continuous_problem = Problem::new(
        ProblemType::Unknown,
        "test".to_string(),
        ProblemData::Continuous {
            variables: vec!["x".to_string()],
            bounds: vec![(-1.0, 1.0)],
            objective: ObjectiveFunction::Minimize("x^2".to_string()),
        },
    );
    assert_eq!(solver.detect_problem_type(&continuous_problem), ProblemType::ContinuousOptimization);

    // Test TimeSeries detection
    let timeseries_problem = Problem::new(
        ProblemType::Unknown,
        "test".to_string(),
        ProblemData::TimeSeries {
            series: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            timestamps: None,
            horizon: 5,
        },
    );
    assert_eq!(solver.detect_problem_type(&timeseries_problem), ProblemType::TimeSeriesForecast);

    println!("✅ Problem type detection test passed");
}

/// Test 5: Solution metrics validation
#[tokio::test]
async fn test_solution_metrics() -> Result<()> {
    // Create simple graph problem
    let adjacency = Array2::from_shape_fn((3, 3), |(i, j)| i != j && (i + j) % 2 == 0);

    let problem = Problem::new(
        ProblemType::GraphProblem,
        "Small graph".to_string(),
        ProblemData::Graph {
            adjacency_matrix: adjacency,
            node_labels: None,
            edge_weights: None,
        },
    );

    let mut solver = UniversalSolver::new(SolverConfig::default());
    let solution = solver.solve(problem).await?;

    // Validate metrics
    assert!(solution.metrics.iterations > 0, "Should have non-zero iterations");
    assert!(solution.metrics.convergence_rate >= 0.0, "Convergence rate should be non-negative");
    assert!(solution.metrics.quality_score >= 0.0 && solution.metrics.quality_score <= 1.0,
        "Quality score should be in [0,1]");
    assert!(solution.computation_time_ms > 0.0, "Should have non-zero computation time");

    println!("✅ Solution metrics validation test passed");
    Ok(())
}

/// Test 6: Multi-problem solving (stress test)
#[tokio::test]
async fn test_multi_problem_solving() -> Result<()> {
    let mut solver = UniversalSolver::new(SolverConfig::default());
    let mut successes = 0;

    // Solve multiple problems of different types
    for i in 0..5 {
        // Graph problem
        let adjacency = Array2::from_shape_fn((i + 3, i + 3), |(a, b)| a != b && (a + b) % 2 == 0);
        let graph_problem = Problem::new(
            ProblemType::GraphProblem,
            format!("Graph {}", i),
            ProblemData::Graph {
                adjacency_matrix: adjacency,
                node_labels: None,
                edge_weights: None,
            },
        );

        let result = solver.solve(graph_problem).await;
        if result.is_ok() {
            successes += 1;
        }
    }

    assert!(successes >= 4, "Should successfully solve at least 4 out of 5 problems");

    println!("✅ Multi-problem solving test passed ({}/5 successful)", successes);
    Ok(())
}

/// Test 7: Explanation generation
#[tokio::test]
async fn test_explanation_generation() -> Result<()> {
    let adjacency = Array2::from_elem((2, 2), false);
    let problem = Problem::new(
        ProblemType::GraphProblem,
        "Explanation test".to_string(),
        ProblemData::Graph {
            adjacency_matrix: adjacency,
            node_labels: None,
            edge_weights: None,
        },
    );

    let mut solver = UniversalSolver::new(SolverConfig::default());
    let solution = solver.solve(problem).await?;

    // Check explanation is present and informative
    assert!(!solution.explanation.is_empty(), "Explanation should not be empty");
    assert!(solution.explanation.len() > 50, "Explanation should be detailed");
    assert!(solution.explanation.contains("Phase 6") || solution.explanation.contains("solver"),
        "Explanation should mention algorithm");

    println!("✅ Explanation generation test passed");
    println!("   Explanation: {}", solution.explanation);

    Ok(())
}

/// Test 8: Solver configuration options
#[test]
fn test_solver_configuration() {
    // Test with auto-detection disabled
    let mut config = SolverConfig::default();
    config.auto_detect_type = false;
    config.gpu_accelerated = true;

    let _solver = UniversalSolver::new(config);
    // Config is private, so we can't directly check it
    // Just verify it compiles and runs

    // Test with time limit
    let mut config2 = SolverConfig::default();
    config2.max_time_ms = Some(1000);
    let _solver2 = UniversalSolver::new(config2);

    println!("✅ Solver configuration test passed");
}
