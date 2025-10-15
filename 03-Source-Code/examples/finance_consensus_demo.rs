//! Finance Consensus Mechanisms Demo
//!
//! Demonstrates quantum voting and thermodynamic consensus for achieving
//! portfolio optimization consensus across multiple strategies.

use prism_ai::finance::{
    Asset, PortfolioConfig, PortfolioOptimizer, OptimizationStrategy,
    QuantumVotingEngine, QuantumVotingConfig,
    ThermodynamicConsensusEngine, ThermodynamicConfig,
};
use ndarray::Array1;
use anyhow::Result;

fn create_demo_assets() -> Vec<Asset> {
    // Create 5 tech stocks with different risk/return profiles
    vec![
        Asset {
            ticker: "AAPL".to_string(),
            expected_return: 0.12,
            prices: vec![150.0, 152.0, 155.0, 153.0, 157.0, 160.0],
            min_weight: 0.0,
            max_weight: 0.4,
        },
        Asset {
            ticker: "GOOGL".to_string(),
            expected_return: 0.15,
            prices: vec![2800.0, 2850.0, 2900.0, 2880.0, 2920.0, 2950.0],
            min_weight: 0.0,
            max_weight: 0.4,
        },
        Asset {
            ticker: "MSFT".to_string(),
            expected_return: 0.10,
            prices: vec![300.0, 305.0, 310.0, 308.0, 312.0, 315.0],
            min_weight: 0.0,
            max_weight: 0.4,
        },
        Asset {
            ticker: "TSLA".to_string(),
            expected_return: 0.20,
            prices: vec![700.0, 720.0, 750.0, 730.0, 760.0, 780.0],
            min_weight: 0.0,
            max_weight: 0.3,
        },
        Asset {
            ticker: "NVDA".to_string(),
            expected_return: 0.18,
            prices: vec![500.0, 520.0, 540.0, 530.0, 550.0, 570.0],
            min_weight: 0.0,
            max_weight: 0.3,
        },
    ]
}

fn main() -> Result<()> {
    println!("=================================================");
    println!("  Finance Consensus Mechanisms Demonstration");
    println!("=================================================\n");

    let assets = create_demo_assets();
    let asset_names: Vec<String> = assets.iter().map(|a| a.ticker.clone()).collect();

    // Step 1: Generate multiple portfolio strategies
    println!("Step 1: Generating Portfolio Strategies\n");
    println!("---------------------------------------");

    let mut portfolios = Vec::new();
    let mut strategy_names = Vec::new();

    let config = PortfolioConfig::default();
    let mut optimizer = PortfolioOptimizer::new(config)?;

    // Strategy 1: Maximum Sharpe Ratio
    println!("1. Maximum Sharpe Ratio Strategy:");
    let result1 = optimizer.optimize(&assets, OptimizationStrategy::MaxSharpe)?;
    println!("   Sharpe Ratio: {:.3}", result1.portfolio.sharpe_ratio);
    println!("   Return: {:.2}%, Risk: {:.2}%",
             result1.portfolio.expected_return * 100.0,
             result1.portfolio.volatility * 100.0);
    portfolios.push(result1.portfolio.weights.clone());
    strategy_names.push("MaxSharpe".to_string());

    // Strategy 2: Minimum Variance
    println!("\n2. Minimum Variance Strategy:");
    let result2 = optimizer.optimize(&assets, OptimizationStrategy::MinVariance)?;
    println!("   Volatility: {:.2}%", result2.portfolio.volatility * 100.0);
    println!("   Return: {:.2}%, Sharpe: {:.3}",
             result2.portfolio.expected_return * 100.0,
             result2.portfolio.sharpe_ratio);
    portfolios.push(result2.portfolio.weights.clone());
    strategy_names.push("MinVariance".to_string());

    // Strategy 3: Risk Parity
    println!("\n3. Risk Parity Strategy:");
    let result3 = optimizer.optimize(&assets, OptimizationStrategy::RiskParity)?;
    println!("   Balanced Risk Contribution");
    println!("   Return: {:.2}%, Risk: {:.2}%",
             result3.portfolio.expected_return * 100.0,
             result3.portfolio.volatility * 100.0);
    portfolios.push(result3.portfolio.weights.clone());
    strategy_names.push("RiskParity".to_string());

    // Strategy 4: Target Return (12%)
    println!("\n4. Target Return Strategy (12%):");
    let result4 = optimizer.optimize(&assets, OptimizationStrategy::TargetReturn(0.12))?;
    println!("   Target: 12.0%, Actual: {:.2}%", result4.portfolio.expected_return * 100.0);
    println!("   Risk: {:.2}%, Sharpe: {:.3}",
             result4.portfolio.volatility * 100.0,
             result4.portfolio.sharpe_ratio);
    portfolios.push(result4.portfolio.weights.clone());
    strategy_names.push("Target12%".to_string());

    // Display individual strategy allocations
    println!("\n\nIndividual Strategy Allocations:");
    println!("---------------------------------------");
    println!("{:<12} {:>8} {:>8} {:>8} {:>8}", "Strategy", "MaxSharpe", "MinVar", "RiskPar", "Target12%");
    println!("{}", "-".repeat(52));
    for (i, name) in asset_names.iter().enumerate() {
        print!("{:<12}", name);
        for portfolio in &portfolios {
            print!(" {:>7.1}%", portfolio[i] * 100.0);
        }
        println!();
    }

    // Step 2: Quantum Voting Consensus
    println!("\n\n=================================================");
    println!("Step 2: Quantum Voting Consensus");
    println!("=================================================\n");

    let quantum_config = QuantumVotingConfig {
        decoherence_rate: 0.1,
        entanglement_threshold: 0.6,
        measurement_iterations: 50,
        temperature: 0.01,
    };

    let quantum_engine = QuantumVotingEngine::new(quantum_config);

    println!("Computing quantum-inspired consensus via superposition and entanglement...\n");
    let quantum_result = quantum_engine.vote(&portfolios, &strategy_names, &asset_names)?;

    quantum_result.print_summary(&asset_names, &strategy_names);

    // Compute quantum discord
    let quantum_state = quantum_engine.create_quantum_state(&portfolios, &strategy_names, &asset_names)?;
    let discord = quantum_engine.compute_discord(&quantum_state)?;
    println!("\nQuantum Discord: {:.4} (non-classical correlations)", discord);

    // Step 3: Thermodynamic Consensus
    println!("\n\n=================================================");
    println!("Step 3: Thermodynamic Consensus");
    println!("=================================================\n");

    let thermo_config = ThermodynamicConfig {
        temperature: 1.0,
        cooling_rate: 0.95,
        min_temperature: 0.01,
        equilibration_steps: 100,
        risk_penalty: 1.0,
    };

    let thermo_engine = ThermodynamicConsensusEngine::new(thermo_config);

    // Extract expected returns and risks
    let expected_returns = Array1::from_vec(
        assets.iter().map(|a| a.expected_return).collect()
    );

    let risks = Array1::from_vec(vec![
        result1.portfolio.volatility, // Use portfolio volatilities as proxy
        result2.portfolio.volatility,
        result3.portfolio.volatility,
        result4.portfolio.volatility,
        0.15, // NVDA estimated risk
    ]);

    println!("Performing simulated annealing to minimize free energy...\n");
    let thermo_result = thermo_engine.anneal(&portfolios, &expected_returns, &risks)?;

    thermo_result.print_summary(&asset_names);

    let diversification = thermo_result.diversification_score();
    println!("\nDiversification Score: {:.2}% (normalized entropy)", diversification * 100.0);

    // Compute specific heat
    let specific_heat = thermo_engine.specific_heat(&thermo_result.energy_landscape, 0.5)?;
    println!("Specific Heat: {:.6} (thermal fluctuation sensitivity)", specific_heat);

    // Step 4: Comparison
    println!("\n\n=================================================");
    println!("Step 4: Consensus Comparison");
    println!("=================================================\n");

    println!("{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}", "Method", "AAPL", "GOOGL", "MSFT", "TSLA", "NVDA");
    println!("{}", "-".repeat(70));

    // Original strategies
    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "MaxSharpe",
             portfolios[0][0] * 100.0, portfolios[0][1] * 100.0, portfolios[0][2] * 100.0,
             portfolios[0][3] * 100.0, portfolios[0][4] * 100.0);

    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "MinVariance",
             portfolios[1][0] * 100.0, portfolios[1][1] * 100.0, portfolios[1][2] * 100.0,
             portfolios[1][3] * 100.0, portfolios[1][4] * 100.0);

    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "RiskParity",
             portfolios[2][0] * 100.0, portfolios[2][1] * 100.0, portfolios[2][2] * 100.0,
             portfolios[2][3] * 100.0, portfolios[2][4] * 100.0);

    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "Target12%",
             portfolios[3][0] * 100.0, portfolios[3][1] * 100.0, portfolios[3][2] * 100.0,
             portfolios[3][3] * 100.0, portfolios[3][4] * 100.0);

    println!("{}", "-".repeat(70));

    // Consensus methods
    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "Quantum",
             quantum_result.consensus_weights[0] * 100.0,
             quantum_result.consensus_weights[1] * 100.0,
             quantum_result.consensus_weights[2] * 100.0,
             quantum_result.consensus_weights[3] * 100.0,
             quantum_result.consensus_weights[4] * 100.0);

    println!("{:<15} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "Thermodynamic",
             thermo_result.consensus_weights[0] * 100.0,
             thermo_result.consensus_weights[1] * 100.0,
             thermo_result.consensus_weights[2] * 100.0,
             thermo_result.consensus_weights[3] * 100.0,
             thermo_result.consensus_weights[4] * 100.0);

    // Metrics comparison
    println!("\n\nConsensus Metrics:");
    println!("---------------------------------------");
    println!("Quantum Voting:");
    println!("  Coherence: {:.4} (strategy agreement)", quantum_result.consensus_coherence);
    println!("  Discord: {:.4} (quantum correlations)", discord);

    println!("\nThermodynamic Consensus:");
    println!("  Free Energy: {:.6} (system stability)", thermo_result.free_energy);
    println!("  Entropy: {:.6} (diversification)", thermo_result.entropy);
    println!("  Diversification: {:.1}%", diversification * 100.0);

    // Summary
    println!("\n\n=================================================");
    println!("Summary");
    println!("=================================================\n");

    println!("✓ Generated 4 portfolio optimization strategies");
    println!("✓ Quantum voting achieved consensus via superposition");
    println!("✓ Thermodynamic consensus minimized free energy");
    println!("\nBoth methods successfully aggregate multiple strategies into");
    println!("a unified consensus portfolio, each with different properties:");
    println!("\n• Quantum: Emphasizes strategy correlations (entanglement)");
    println!("• Thermodynamic: Balances return vs diversification (energy vs entropy)");

    println!("\n=================================================\n");

    Ok(())
}
