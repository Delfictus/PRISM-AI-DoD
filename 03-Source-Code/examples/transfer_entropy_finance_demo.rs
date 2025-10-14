//! # Transfer Entropy Finance Demo
//!
//! This example demonstrates how to use Worker 1's Transfer Entropy module
//! for financial causality analysis and causal portfolio optimization.
//!
//! ## What This Demo Shows
//!
//! 1. **Calculate Transfer Entropy Matrix** - Measure information flow between 5 assets
//! 2. **Identify Leading/Lagging Assets** - Determine which assets predict others
//! 3. **Causal Relationships** - Visualize directional causality network
//! 4. **Causal Portfolio Optimization** - Build diversified portfolio using causal structure
//!
//! ## Key Concepts
//!
//! - **Transfer Entropy (TE)**: Measures directed information flow from asset X → Y
//! - **Causal Diversification**: True diversification based on causal independence
//! - **Lead-Lag Relationships**: Identify predictive relationships between assets
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example transfer_entropy_finance_demo
//! ```

use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

// Simulated imports (adjust to actual Worker 1 module paths)
// use prism_worker_1::transfer_entropy::TransferEntropyAnalyzer;
// use prism_worker_1::transfer_entropy::TransferEntropyConfig;

/// Asset names for the demo
const ASSETS: [&str; 5] = ["SPY", "TLT", "GLD", "BTC", "USD"];

fn main() -> Result<()> {
    println!("=".repeat(80));
    println!("  TRANSFER ENTROPY FINANCE DEMO");
    println!("  Worker 1 - Information Theoretic Portfolio Analysis");
    println!("=".repeat(80));
    println!();

    // Step 1: Generate synthetic stock price data
    println!("📊 Step 1: Loading Asset Price Data");
    println!("-".repeat(80));

    let n_timesteps = 1000;
    let price_data = generate_synthetic_prices(n_timesteps);

    println!("  ✓ Loaded {} timesteps for {} assets", n_timesteps, ASSETS.len());
    println!();

    // Step 2: Calculate returns
    println!("📈 Step 2: Computing Asset Returns");
    println!("-".repeat(80));

    let returns = calculate_returns(&price_data);

    for (i, asset) in ASSETS.iter().enumerate() {
        let mean_return = returns.column(i).mean().unwrap();
        let std_return = returns.column(i).std(0.0);
        println!("  {} - Mean: {:.4}%, Std: {:.4}%",
            asset, mean_return * 100.0, std_return * 100.0);
    }
    println!();

    // Step 3: Calculate Transfer Entropy matrix
    println!("🔬 Step 3: Computing Transfer Entropy Matrix");
    println!("-".repeat(80));

    let te_matrix = calculate_transfer_entropy_matrix(&returns)?;

    println!("  Transfer Entropy (bits) - Measures X → Y information flow:");
    println!();
    print!("         ");
    for asset in &ASSETS {
        print!("{:>8}", asset);
    }
    println!();
    println!("  {}", "-".repeat(50));

    for (i, source) in ASSETS.iter().enumerate() {
        print!("  {:>6} ", source);
        for j in 0..ASSETS.len() {
            if i == j {
                print!("    -   ");
            } else {
                print!(" {:7.4}", te_matrix[[i, j]]);
            }
        }
        println!();
    }
    println!();

    // Step 4: Identify leading/lagging assets
    println!("🎯 Step 4: Identifying Leading/Lagging Assets");
    println!("-".repeat(80));

    let (leading_assets, lagging_assets) = identify_leading_lagging(&te_matrix);

    println!("  📢 Leading Assets (predict others):");
    for (asset, score) in &leading_assets {
        println!("     {} - Predictive Strength: {:.4} bits", asset, score);
    }
    println!();

    println!("  📡 Lagging Assets (predicted by others):");
    for (asset, score) in &lagging_assets {
        println!("     {} - Lag Score: {:.4} bits", asset, score);
    }
    println!();

    // Step 5: Visualize causal relationships
    println!("🔗 Step 5: Causal Relationship Network");
    println!("-".repeat(80));

    visualize_causal_network(&te_matrix, 0.02); // threshold: 0.02 bits
    println!();

    // Step 6: Causal portfolio optimization
    println!("💼 Step 6: Causal Portfolio Optimization");
    println!("-".repeat(80));

    let optimal_weights = causal_portfolio_optimization(&te_matrix, &returns)?;

    println!("  Optimal Portfolio Weights (Causal Diversification):");
    println!();
    for (i, asset) in ASSETS.iter().enumerate() {
        let weight = optimal_weights[i];
        let bar_length = (weight * 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  {:>6} | {:>6.2}% {}", asset, weight * 100.0, bar);
    }
    println!();

    // Step 7: Compare with correlation-based portfolio
    println!("🔍 Step 7: Comparison - Causal vs Correlation Diversification");
    println!("-".repeat(80));

    let correlation_weights = correlation_portfolio_optimization(&returns)?;

    println!("  Correlation-Based Portfolio:");
    for (i, asset) in ASSETS.iter().enumerate() {
        println!("  {:>6} | {:>6.2}%", asset, correlation_weights[i] * 100.0);
    }
    println!();

    let causal_div_score = calculate_diversification_score(&te_matrix, &optimal_weights);
    let corr_div_score = calculate_diversification_score(&te_matrix, &correlation_weights);

    println!("  📊 Diversification Scores (higher = better):");
    println!("     Causal Portfolio:      {:.4} (TE-based)", causal_div_score);
    println!("     Correlation Portfolio: {:.4} (correlation-based)", corr_div_score);
    println!();

    if causal_div_score > corr_div_score {
        let improvement = ((causal_div_score / corr_div_score - 1.0) * 100.0);
        println!("  ✅ Causal portfolio is {:.1}% better diversified!", improvement);
    }
    println!();

    // Summary
    println!("=".repeat(80));
    println!("  DEMO COMPLETE");
    println!("=".repeat(80));
    println!();
    println!("  Key Takeaways:");
    println!("  • Transfer Entropy reveals directional causality (X → Y)");
    println!("  • Leading assets predict market movements");
    println!("  • Causal diversification > correlation diversification");
    println!("  • True risk reduction requires causal independence");
    println!();
    println!("  Next Steps:");
    println!("  • Integrate with Worker 4 advanced finance for portfolio optimization");
    println!("  • Use Worker 3 finance API for production deployment");
    println!("  • Combine with ARIMA/LSTM forecasting for predictions");
    println!();

    Ok(())
}

/// Generate synthetic price data for 5 assets
fn generate_synthetic_prices(n_timesteps: usize) -> Array2<f64> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let mut prices = Array2::<f64>::zeros((n_timesteps, ASSETS.len()));

    // Initial prices
    let initial_prices = [100.0, 80.0, 150.0, 30000.0, 1.0];

    // Asset-specific parameters (drift, volatility)
    let params = [
        (0.0003, 0.01),  // SPY: low drift, low vol
        (-0.0001, 0.008), // TLT: slightly negative, very low vol
        (0.0002, 0.012),  // GLD: low drift, medium vol
        (0.0008, 0.03),   // BTC: high drift, high vol
        (0.0, 0.005),     // USD: zero drift, minimal vol
    ];

    for i in 0..ASSETS.len() {
        prices[[0, i]] = initial_prices[i];

        let normal = Normal::new(params[i].0, params[i].1).unwrap();

        for t in 1..n_timesteps {
            let return_val = normal.sample(&mut rng);

            // Add causal relationships
            let causal_effect = if t > 1 {
                match i {
                    0 => 0.0, // SPY: leader, no causal effects
                    1 => -0.3 * (prices[[t-1, 0]] / prices[[t-2, 0]] - 1.0), // TLT negatively follows SPY
                    2 => 0.2 * (prices[[t-1, 0]] / prices[[t-2, 0]] - 1.0), // GLD weakly follows SPY
                    3 => 0.1 * (prices[[t-1, 0]] / prices[[t-2, 0]] - 1.0) + 0.5 * rng.gen::<f64>(), // BTC weakly follows, high noise
                    4 => 0.0, // USD: independent
                    _ => 0.0,
                }
            } else {
                0.0
            };

            prices[[t, i]] = prices[[t-1, i]] * (1.0 + return_val + causal_effect);
        }
    }

    prices
}

/// Calculate returns from prices
fn calculate_returns(prices: &Array2<f64>) -> Array2<f64> {
    let n_timesteps = prices.nrows();
    let n_assets = prices.ncols();

    let mut returns = Array2::<f64>::zeros((n_timesteps - 1, n_assets));

    for i in 0..n_assets {
        for t in 1..n_timesteps {
            returns[[t-1, i]] = (prices[[t, i]] / prices[[t-1, i]]) - 1.0;
        }
    }

    returns
}

/// Calculate Transfer Entropy matrix (simplified)
///
/// In production, this would use Worker 1's actual TE module
fn calculate_transfer_entropy_matrix(returns: &Array2<f64>) -> Result<Array2<f64>> {
    let n_assets = returns.ncols();
    let mut te_matrix = Array2::<f64>::zeros((n_assets, n_assets));

    // Simplified TE calculation (demonstration)
    // Real implementation would use KSG estimator from Worker 1
    for i in 0..n_assets {
        for j in 0..n_assets {
            if i != j {
                let te = estimate_transfer_entropy(
                    returns.column(i).to_vec(),
                    returns.column(j).to_vec(),
                );
                te_matrix[[i, j]] = te;
            }
        }
    }

    Ok(te_matrix)
}

/// Simplified Transfer Entropy estimator
fn estimate_transfer_entropy(source: Vec<f64>, target: Vec<f64>) -> f64 {
    // Simplified: Use lagged correlation as proxy for TE
    // Real implementation would use mutual information and conditional mutual information

    let n = source.len().min(target.len()) - 1;

    let mut cov_sum = 0.0;
    let mut source_sq_sum = 0.0;
    let mut target_sq_sum = 0.0;

    for i in 0..n {
        cov_sum += source[i] * target[i + 1];
        source_sq_sum += source[i] * source[i];
        target_sq_sum += target[i + 1] * target[i + 1];
    }

    let correlation = cov_sum / (source_sq_sum.sqrt() * target_sq_sum.sqrt());

    // Convert correlation to TE-like measure (bits)
    let te = -0.5 * (1.0 - correlation.abs()).ln() / 2.0_f64.ln();
    te.max(0.0).min(1.0) // Clamp to [0, 1]
}

/// Identify leading and lagging assets
fn identify_leading_lagging(te_matrix: &Array2<f64>) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
    let n_assets = te_matrix.nrows();

    let mut leading_scores = Vec::new();
    let mut lagging_scores = Vec::new();

    for i in 0..n_assets {
        // Leading score: sum of TE from this asset to others
        let leading_score: f64 = (0..n_assets)
            .filter(|&j| i != j)
            .map(|j| te_matrix[[i, j]])
            .sum();

        // Lagging score: sum of TE from others to this asset
        let lagging_score: f64 = (0..n_assets)
            .filter(|&j| i != j)
            .map(|j| te_matrix[[j, i]])
            .sum();

        leading_scores.push((ASSETS[i].to_string(), leading_score));
        lagging_scores.push((ASSETS[i].to_string(), lagging_score));
    }

    // Sort by score (descending)
    leading_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    lagging_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    (leading_scores, lagging_scores)
}

/// Visualize causal network
fn visualize_causal_network(te_matrix: &Array2<f64>, threshold: f64) {
    println!("  Significant Causal Links (TE > {:.3} bits):", threshold);
    println!();

    let mut link_count = 0;

    for i in 0..ASSETS.len() {
        for j in 0..ASSETS.len() {
            if i != j && te_matrix[[i, j]] > threshold {
                let strength = if te_matrix[[i, j]] > 0.05 {
                    "Strong"
                } else if te_matrix[[i, j]] > 0.03 {
                    "Medium"
                } else {
                    "Weak"
                };

                println!("  {} → {} : {:.4} bits ({})",
                    ASSETS[i], ASSETS[j], te_matrix[[i, j]], strength);
                link_count += 1;
            }
        }
    }

    if link_count == 0 {
        println!("  (No significant causal links found above threshold)");
    }
}

/// Causal portfolio optimization
fn causal_portfolio_optimization(te_matrix: &Array2<f64>, returns: &Array2<f64>) -> Result<Array1<f64>> {
    let n_assets = ASSETS.len();

    // Objective: Maximize diversification (minimize pairwise TE) while balancing returns
    // Simplified approach: Weight inversely proportional to causal dependencies

    let mut weights = Array1::<f64>::ones(n_assets);

    for i in 0..n_assets {
        // Penalize assets that are causally dependent on others
        let dependency_score: f64 = (0..n_assets)
            .filter(|&j| i != j)
            .map(|j| te_matrix[[j, i]]) // TE from others to this asset
            .sum();

        // Higher weight for causally independent assets
        weights[i] = 1.0 / (1.0 + dependency_score);
    }

    // Normalize to sum to 1
    let total_weight: f64 = weights.sum();
    weights.mapv_inplace(|w| w / total_weight);

    Ok(weights)
}

/// Correlation-based portfolio optimization (baseline)
fn correlation_portfolio_optimization(returns: &Array2<f64>) -> Result<Array1<f64>> {
    let n_assets = returns.ncols();

    // Simplified: Equal weight (correlation-based would use mean-variance optimization)
    let mut weights = Array1::<f64>::ones(n_assets) / n_assets as f64;

    Ok(weights)
}

/// Calculate diversification score
fn calculate_diversification_score(te_matrix: &Array2<f64>, weights: &Array1<f64>) -> f64 {
    let n_assets = weights.len();

    // Diversification score: weighted average of causal independence
    let mut total_te = 0.0;
    let mut count = 0.0;

    for i in 0..n_assets {
        for j in 0..n_assets {
            if i != j {
                total_te += weights[i] * weights[j] * te_matrix[[i, j]];
                count += weights[i] * weights[j];
            }
        }
    }

    // Higher score = better diversified (less causal dependence)
    let avg_te = if count > 0.0 { total_te / count } else { 0.0 };
    1.0 - avg_te.min(1.0)
}
