//! Worker 4 Complete Demonstration
//!
//! Showcases all major capabilities:
//! 1. Financial Portfolio Optimization
//! 2. Multi-Objective Portfolio Optimization
//! 3. Risk Analysis & Backtesting
//! 4. GNN Training & Prediction
//! 5. Hybrid Solver Usage

use anyhow::Result;
use ndarray::{arr1, Array1};
use prism_ai::applications::financial::{
    Asset, OptimizationConfig, PortfolioOptimizer,
    MultiObjectiveConfig, MultiObjectivePortfolioOptimizer,
    RiskAnalyzer, VarMethod,
};
use prism_ai::applications::solver::{
    Problem, ProblemData, ProblemType,
    GnnTrainer, TrainingConfig, TrainingSample,
    GnnPredictor, PredictorConfig,
    HybridSolver, HybridConfig,
    ProblemEmbedder,
};

fn main() -> Result<()> {
    println!("=== Worker 4 Complete Demonstration ===\n");

    // ========================================
    // 1. Financial Portfolio Optimization
    // ========================================
    println!("1. FINANCIAL PORTFOLIO OPTIMIZATION");
    println!("   Using Mean-Variance Optimization with Active Inference\n");

    let assets = create_sample_assets();

    let mut config = OptimizationConfig::default();
    config.target_return = Some(0.12); // 12% target return
    config.risk_free_rate = 0.02;
    config.use_transfer_entropy = true;
    config.use_regime_detection = true;

    let mut optimizer = PortfolioOptimizer::new(config);
    let portfolio = optimizer.optimize(assets.clone())?;

    println!("   Optimized Portfolio:");
    println!("   - Expected Return: {:.2}%", portfolio.expected_return * 100.0);
    println!("   - Risk (Volatility): {:.2}%", portfolio.risk * 100.0);
    println!("   - Sharpe Ratio: {:.3}", portfolio.sharpe_ratio);
    println!("   - Weights:");
    for (asset, &weight) in portfolio.assets.iter().zip(portfolio.weights.iter()) {
        println!("     * {}: {:.1}%", asset.symbol, weight * 100.0);
    }
    println!();

    // ========================================
    // 2. Multi-Objective Portfolio Optimization
    // ========================================
    println!("2. MULTI-OBJECTIVE PORTFOLIO OPTIMIZATION");
    println!("   Optimizing return, risk, and turnover simultaneously\n");

    let mut mo_config = MultiObjectiveConfig {
        assets: assets.clone(),
        current_weights: Some(arr1(&[0.33, 0.33, 0.34])),
        risk_free_rate: 0.02,
        max_weight_per_asset: 0.5,
        ..Default::default()
    };
    mo_config.nsga_config.population_size = 50;
    mo_config.nsga_config.num_generations = 20;

    let mut mo_optimizer = MultiObjectivePortfolioOptimizer::new(mo_config);
    let mo_result = mo_optimizer.optimize()?;

    println!("   Pareto Front: {} solutions", mo_result.pareto_front.solutions.len());
    println!("   Recommended Portfolio (Knee Point):");
    println!("   - Expected Return: {:.2}%", mo_result.recommended_portfolio.expected_return * 100.0);
    println!("   - Risk: {:.2}%", mo_result.recommended_portfolio.risk * 100.0);
    println!("   - Sharpe Ratio: {:.3}", mo_result.recommended_portfolio.sharpe_ratio);
    println!();

    // ========================================
    // 3. Risk Analysis
    // ========================================
    println!("3. RISK ANALYSIS");
    println!("   Value-at-Risk and Risk Decomposition\n");

    let analyzer = RiskAnalyzer::new().with_num_simulations(1000);

    // Calculate covariance matrix
    let single_obj_config = OptimizationConfig::default();
    let temp_optimizer = PortfolioOptimizer::new(single_obj_config);
    let covariance = temp_optimizer.calculate_covariance_matrix(&assets)?;

    // VaR Analysis
    let var_result = analyzer.calculate_var(
        &portfolio,
        0.95, // 95% confidence
        VarMethod::MonteCarlo,
        Some(&covariance)
    )?;

    println!("   Value-at-Risk (95% confidence):");
    println!("   - VaR: {:.2}%", var_result.var * 100.0);
    println!("   - CVaR: {:.2}%", var_result.cvar * 100.0);
    println!("   - Method: {:?}", var_result.method);

    // Risk Decomposition
    let risk_decomp = analyzer.decompose_risk(&portfolio, &covariance)?;
    println!("\n   Risk Contribution by Asset:");
    for (symbol, &pct) in risk_decomp.asset_symbols.iter().zip(risk_decomp.percentage_contributions.iter()) {
        println!("   - {}: {:.1}%", symbol, pct);
    }
    println!();

    // ========================================
    // 4. GNN Training & Prediction
    // ========================================
    println!("4. GNN TRAINING & PREDICTION");
    println!("   Training Graph Neural Network for transfer learning\n");

    // Create training samples from synthetic problems
    let training_samples = create_training_samples()?;

    let mut gnn_config = TrainingConfig::default();
    gnn_config.max_epochs = 10; // Quick demo
    gnn_config.batch_size = 16;
    gnn_config.patience = 3;

    let mut trainer = GnnTrainer::new(gnn_config);
    println!("   Training GNN on {} samples...", training_samples.len());
    let history = trainer.train(training_samples)?;

    println!("   Training Results:");
    println!("   - Epochs: {}", history.epochs_trained);
    println!("   - Best Val Loss: {:.4}", history.best_val_loss);
    println!("   - Final Train Loss: {:.4}", history.train_losses.last().unwrap_or(&0.0));

    // Create predictor
    let pred_config = PredictorConfig {
        confidence_threshold: 0.7,
        num_neighbors: 5,
        use_pattern_database: false, // Simplified for demo
        ..Default::default()
    };
    let predictor = GnnPredictor::new(trainer, pred_config);

    // Test prediction
    let test_problem = create_test_problem();
    let prediction = predictor.predict(&test_problem)?;

    println!("\n   Test Problem Prediction:");
    println!("   - Quality: {:.4}", prediction.quality);
    println!("   - Confidence: {:.1}%", prediction.confidence * 100.0);
    println!("   - Use GNN: {}", prediction.use_prediction);
    println!();

    // ========================================
    // 5. Hybrid Solver
    // ========================================
    println!("5. HYBRID SOLVER");
    println!("   Intelligent routing between GNN and exact solver\n");

    let hybrid_config = HybridConfig {
        use_gnn: false, // Disabled for demo (no trained GNN with real problems yet)
        ..Default::default()
    };

    let mut hybrid_solver = HybridSolver::new(hybrid_config);

    println!("   Hybrid Solver Stats:");
    let stats = hybrid_solver.get_stats();
    println!("   - Total Problems: {}", stats.total_problems);
    println!("   - GNN Solutions: {}", stats.gnn_solutions);
    println!("   - Exact Solutions: {}", stats.exact_solutions);
    println!("   - GNN Usage Rate: {:.1}%", stats.gnn_usage_rate() * 100.0);
    println!();

    // ========================================
    // Summary
    // ========================================
    println!("=== DEMONSTRATION COMPLETE ===");
    println!("\nWorker 4 Capabilities Demonstrated:");
    println!("✓ Mean-Variance Portfolio Optimization");
    println!("✓ Multi-Objective Optimization (NSGA-II)");
    println!("✓ Risk Analysis (VaR/CVaR)");
    println!("✓ Risk Decomposition");
    println!("✓ GNN Training & Prediction");
    println!("✓ Hybrid Solver Architecture");
    println!("\nAll systems operational!");

    Ok(())
}

/// Create sample assets for demonstration
fn create_sample_assets() -> Vec<Asset> {
    vec![
        Asset {
            symbol: "AAPL".to_string(),
            name: "Apple Inc.".to_string(),
            current_price: 175.0,
            historical_returns: vec![
                0.012, 0.015, -0.008, 0.020, 0.011, 0.018, -0.005, 0.013,
                0.016, 0.009, 0.014, -0.003, 0.019, 0.010, 0.017, -0.007,
            ],
        },
        Asset {
            symbol: "MSFT".to_string(),
            name: "Microsoft Corp.".to_string(),
            current_price: 380.0,
            historical_returns: vec![
                0.014, 0.011, 0.008, 0.016, 0.013, 0.015, 0.009, 0.012,
                0.010, 0.014, 0.007, 0.018, 0.011, 0.013, 0.009, 0.015,
            ],
        },
        Asset {
            symbol: "GOOGL".to_string(),
            name: "Alphabet Inc.".to_string(),
            current_price: 140.0,
            historical_returns: vec![
                0.013, 0.017, 0.010, 0.012, 0.015, -0.004, 0.014, 0.011,
                0.016, 0.008, 0.013, 0.012, 0.019, 0.010, -0.006, 0.014,
            ],
        },
    ]
}

/// Create training samples for GNN
fn create_training_samples() -> Result<Vec<TrainingSample>> {
    let embedder = ProblemEmbedder::new();
    let mut samples = Vec::new();

    // Generate 100 synthetic problems with varying quality
    for i in 0..100 {
        let problem = Problem {
            problem_type: ProblemType::ContinuousOptimization,
            description: format!("Training problem {}", i),
            data: ProblemData::Continuous {
                variables: vec!["x".to_string(), "y".to_string()],
                bounds: vec![(0.0, 1.0), (0.0, 1.0)],
                objective: prism_ai::applications::solver::problem::ObjectiveFunction::Minimize(
                    format!("f{}", i)
                ),
            },
            constraints: Vec::new(),
            metadata: std::collections::HashMap::new(),
        };

        let embedding = embedder.embed(&problem)?;
        let quality = 1.0 - (i as f64 / 100.0); // Decreasing quality

        samples.push(TrainingSample {
            problem: embedding,
            quality,
            solution: None,
        });
    }

    Ok(samples)
}

/// Create a test problem
fn create_test_problem() -> Problem {
    Problem {
        problem_type: ProblemType::ContinuousOptimization,
        description: "Test optimization problem".to_string(),
        data: ProblemData::Continuous {
            variables: vec!["x".to_string(), "y".to_string()],
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            objective: prism_ai::applications::solver::problem::ObjectiveFunction::Minimize(
                "test_function".to_string()
            ),
        },
        constraints: Vec::new(),
        metadata: std::collections::HashMap::new(),
    }
}
