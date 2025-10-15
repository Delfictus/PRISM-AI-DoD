//! Portfolio Optimization Demo
//!
//! Demonstrates GPU-accelerated portfolio optimization using Modern Portfolio Theory.
//!
//! Run with: cargo run --example portfolio_optimization_demo --features cuda

use prism_ai::finance::{
    PortfolioOptimizer, OptimizationStrategy, Asset, PortfolioConfig,
};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM Portfolio Optimization Demo ===\n");

    // Create sample portfolio assets
    let assets = vec![
        Asset {
            ticker: "AAPL".to_string(),
            expected_return: 0.12,  // 12% annual return
            prices: vec![
                100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 110.0, 109.0,
                112.0, 115.0, 117.0, 120.0, 118.0, 121.0, 124.0,
            ],
            min_weight: 0.0,
            max_weight: 0.5,  // Max 50% allocation
        },
        Asset {
            ticker: "GOOGL".to_string(),
            expected_return: 0.15,  // 15% annual return
            prices: vec![
                200.0, 205.0, 203.0, 210.0, 215.0, 220.0, 218.0, 225.0,
                230.0, 228.0, 235.0, 240.0, 238.0, 245.0, 250.0,
            ],
            min_weight: 0.0,
            max_weight: 0.5,
        },
        Asset {
            ticker: "MSFT".to_string(),
            expected_return: 0.10,  // 10% annual return
            prices: vec![
                150.0, 151.0, 153.0, 152.0, 155.0, 157.0, 156.0, 159.0,
                162.0, 161.0, 164.0, 167.0, 166.0, 169.0, 172.0,
            ],
            min_weight: 0.0,
            max_weight: 0.5,
        },
        Asset {
            ticker: "BND".to_string(),  // Bond ETF (lower risk, lower return)
            expected_return: 0.04,  // 4% annual return
            prices: vec![
                80.0, 80.2, 80.1, 80.3, 80.5, 80.4, 80.6, 80.8,
                80.7, 81.0, 81.2, 81.1, 81.4, 81.6, 81.8,
            ],
            min_weight: 0.0,
            max_weight: 0.5,
        },
    ];

    // Configure portfolio optimization
    let config = PortfolioConfig {
        risk_free_rate: 0.02,           // 2% risk-free rate
        target_return: None,
        max_position_size: 0.5,         // Max 50% in any single asset
        allow_short: false,             // Long-only portfolio
        rebalance_freq: 30,             // Monthly rebalancing
    };

    println!("Assets in Portfolio:");
    for asset in &assets {
        println!("  {} - Expected Return: {:.1}%", asset.ticker, asset.expected_return * 100.0);
    }
    println!();

    // Initialize GPU-accelerated optimizer
    println!("Initializing GPU-accelerated portfolio optimizer...");
    let mut optimizer = PortfolioOptimizer::new(config)?;
    println!("✓ GPU initialization successful\n");

    // Strategy 1: Maximum Sharpe Ratio
    println!("--- Strategy 1: Maximum Sharpe Ratio ---");
    let result = optimizer.optimize(&assets, OptimizationStrategy::MaxSharpe)?;
    print_optimization_result(&result, "Max Sharpe");

    // Strategy 2: Minimum Variance
    println!("\n--- Strategy 2: Minimum Variance ---");
    let result = optimizer.optimize(&assets, OptimizationStrategy::MinVariance)?;
    print_optimization_result(&result, "Min Variance");

    // Strategy 3: Risk Parity
    println!("\n--- Strategy 3: Risk Parity ---");
    let result = optimizer.optimize(&assets, OptimizationStrategy::RiskParity)?;
    print_optimization_result(&result, "Risk Parity");

    // Compare strategies
    println!("\n=== Strategy Comparison ===");
    println!("Max Sharpe: Best risk-adjusted returns (highest Sharpe ratio)");
    println!("Min Variance: Lowest portfolio volatility (conservative)");
    println!("Risk Parity: Equal risk contribution from each asset (balanced)");

    println!("\n✓ Portfolio optimization complete!");

    Ok(())
}

fn print_optimization_result(
    result: &prism_ai::finance::OptimizationResult,
    strategy_name: &str,
) {
    println!("Strategy: {}", strategy_name);
    println!("Optimization:");
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.converged);
    println!("  Objective Value: {:.6}", result.objective_value);

    println!("\nPortfolio Metrics:");
    println!("  Expected Return: {:.2}%", result.portfolio.expected_return * 100.0);
    println!("  Volatility (Risk): {:.2}%", result.portfolio.volatility * 100.0);
    println!("  Sharpe Ratio: {:.4}", result.portfolio.sharpe_ratio);

    println!("\nAsset Allocations:");
    for (i, ticker) in result.portfolio.assets.iter().enumerate() {
        let weight = result.portfolio.weights[i];
        println!("  {}: {:.1}%", ticker, weight * 100.0);
    }

    // Verify weights sum to 100%
    let total: f64 = result.portfolio.weights.iter().sum();
    println!("\nTotal Allocation: {:.1}%", total * 100.0);
}
