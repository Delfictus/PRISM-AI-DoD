//! Finance Portfolio Forecasting Demo
//!
//! Demonstrates time series forecasting integration with portfolio optimization

use prism_ai::finance::{Asset, PortfolioConfig, PortfolioForecaster, ForecastConfig, OptimizationStrategy};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Finance Portfolio Forecasting Demo ===\n");

    // Create test portfolio with 60 days of price history
    let assets = create_portfolio_with_history();

    println!("Portfolio: {} assets with 60 days of history", assets.len());
    for asset in &assets {
        println!("  {} - Current price: ${:.2}, Expected return: {:.1}%",
                 asset.ticker,
                 asset.prices.last().unwrap(),
                 asset.expected_return * 100.0);
    }

    // Configure portfolio optimizer
    let portfolio_config = PortfolioConfig {
        risk_free_rate: 0.02,
        max_position_size: 0.5,
        allow_short: false,
        ..Default::default()
    };

    // Configure forecasting
    let forecast_config = ForecastConfig {
        horizon: 20,  // 20 trading days (~1 month)
        use_arima: true,
        ..Default::default()
    };

    // Create forecaster
    let mut forecaster = PortfolioForecaster::new(portfolio_config, forecast_config)?;

    println!("\n--- Optimizing Portfolio with Forecasting ---");

    // Forecast and optimize
    let forecasted = forecaster.forecast_and_optimize(&assets, OptimizationStrategy::MaxSharpe)?;

    // Print results
    forecasted.print_summary();

    // Generate 3-period rebalancing schedule
    println!("\n--- Generating 3-Period Rebalancing Schedule ---");
    let schedule = forecaster.generate_rebalancing_schedule(&assets, OptimizationStrategy::MaxSharpe, 3)?;

    for (i, portfolio) in schedule.iter().enumerate() {
        println!("\nPeriod {}:", i + 1);
        println!("  Return: {:.2}%, Vol: {:.2}%, Sharpe: {:.2}",
                 portfolio.expected_return * 100.0,
                 portfolio.volatility * 100.0,
                 portfolio.sharpe_ratio);
        println!("  Weights: ");
        for (j, ticker) in portfolio.assets.iter().enumerate() {
            println!("    {}: {:.1}%", ticker, portfolio.weights[j] * 100.0);
        }
    }

    println!("\n✅ Finance forecasting demo complete!");

    Ok(())
}

fn create_portfolio_with_history() -> Vec<Asset> {
    // Generate realistic price histories for 3 assets
    let mut aapl_prices = vec![100.0];
    let mut googl_prices = vec![200.0];
    let mut msft_prices = vec![150.0];

    // 60 trading days with trend + noise
    for i in 1..60 {
        let t = i as f64 * 0.1;
        let noise_aapl = (i as f64 * 0.7).sin() * 2.0;
        let noise_googl = (i as f64 * 1.1).sin() * 3.0;
        let noise_msft = (i as f64 * 0.9).sin() * 1.5;

        aapl_prices.push(100.0 + t * 0.8 + noise_aapl);
        googl_prices.push(200.0 + t * 1.2 + noise_googl);
        msft_prices.push(150.0 + t * 0.6 + noise_msft);
    }

    vec![
        Asset {
            ticker: "AAPL".to_string(),
            expected_return: 0.12,  // 12% annual
            prices: aapl_prices,
            min_weight: 0.0,
            max_weight: 1.0,
        },
        Asset {
            ticker: "GOOGL".to_string(),
            expected_return: 0.15,  // 15% annual
            prices: googl_prices,
            min_weight: 0.0,
            max_weight: 1.0,
        },
        Asset {
            ticker: "MSFT".to_string(),
            expected_return: 0.10,  // 10% annual
            prices: msft_prices,
            min_weight: 0.0,
            max_weight: 1.0,
        },
    ]
}
