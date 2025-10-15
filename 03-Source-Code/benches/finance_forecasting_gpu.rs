//! Finance Portfolio Forecasting GPU Benchmarks
//!
//! Phase 3 Task 2: Validate GPU acceleration performance
//!
//! Benchmarks:
//! 1. CPU vs GPU ARIMA forecasting (3-asset portfolio)
//! 2. CPU vs GPU LSTM forecasting (3-asset portfolio)
//! 3. End-to-end portfolio optimization with GPU acceleration
//!
//! Expected Results:
//! - ARIMA: 15-25x speedup (GPU vs CPU)
//! - LSTM: 50-100x speedup (GPU vs CPU)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_ai::finance::portfolio_forecaster::{PortfolioForecaster, ForecastConfig};
use prism_ai::finance::portfolio_optimizer::{Asset, PortfolioConfig, OptimizationStrategy};
use prism_ai::time_series::{ArimaConfig, LstmConfig};

fn create_benchmark_assets() -> Vec<Asset> {
    // Create 3 assets with 60 days of price history
    let mut aapl_prices = vec![100.0];
    let mut googl_prices = vec![200.0];
    let mut msft_prices = vec![150.0];

    for i in 1..60 {
        let t = i as f64 * 0.1;
        aapl_prices.push(100.0 + 5.0 * t.sin() + t * 0.5);
        googl_prices.push(200.0 + 10.0 * (t * 1.2).sin() + t * 0.8);
        msft_prices.push(150.0 + 3.0 * (t * 0.8).sin() + t * 0.3);
    }

    vec![
        Asset {
            ticker: "AAPL".to_string(),
            expected_return: 0.12,
            prices: aapl_prices,
            min_weight: 0.0,
            max_weight: 1.0,
        },
        Asset {
            ticker: "GOOGL".to_string(),
            expected_return: 0.15,
            prices: googl_prices,
            min_weight: 0.0,
            max_weight: 1.0,
        },
        Asset {
            ticker: "MSFT".to_string(),
            expected_return: 0.10,
            prices: msft_prices,
            min_weight: 0.0,
            max_weight: 1.0,
        },
    ]
}

fn bench_arima_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("finance_arima_forecasting");
    group.sample_size(20); // Reduce sample size for GPU benchmarks

    let assets = create_benchmark_assets();
    let portfolio_config = PortfolioConfig::default();

    // ARIMA configuration
    let mut forecast_config = ForecastConfig::default();
    forecast_config.use_arima = true;
    forecast_config.horizon = 20; // 20 trading days
    forecast_config.arima_config = ArimaConfig {
        p: 2,
        d: 1,
        q: 1,
        include_constant: true,
    };

    group.bench_function("3_asset_portfolio_arima", |b| {
        b.iter(|| {
            let mut forecaster = PortfolioForecaster::new(
                portfolio_config.clone(),
                forecast_config.clone()
            ).unwrap();

            let result = forecaster.forecast_and_optimize(
                black_box(&assets),
                black_box(OptimizationStrategy::MaxSharpe)
            );

            result.unwrap()
        })
    });

    group.finish();
}

fn bench_lstm_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("finance_lstm_forecasting");
    group.sample_size(10); // Smaller sample size for LSTM (slower)

    let assets = create_benchmark_assets();
    let portfolio_config = PortfolioConfig::default();

    // LSTM configuration
    let mut forecast_config = ForecastConfig::default();
    forecast_config.use_arima = false; // Use LSTM
    forecast_config.horizon = 20;
    forecast_config.lstm_config = LstmConfig {
        hidden_size: 20,
        sequence_length: 10,
        epochs: 50,
        ..Default::default()
    };

    group.bench_function("3_asset_portfolio_lstm", |b| {
        b.iter(|| {
            let mut forecaster = PortfolioForecaster::new(
                portfolio_config.clone(),
                forecast_config.clone()
            ).unwrap();

            let result = forecaster.forecast_and_optimize(
                black_box(&assets),
                black_box(OptimizationStrategy::MaxSharpe)
            );

            result.unwrap()
        })
    });

    group.finish();
}

fn bench_portfolio_horizons(c: &mut Criterion) {
    let mut group = c.benchmark_group("finance_forecast_horizons");
    group.sample_size(15);

    let assets = create_benchmark_assets();
    let portfolio_config = PortfolioConfig::default();

    for horizon in [5, 10, 20, 40].iter() {
        let mut forecast_config = ForecastConfig::default();
        forecast_config.use_arima = true;
        forecast_config.horizon = *horizon;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}day_horizon", horizon)),
            horizon,
            |b, _| {
                b.iter(|| {
                    let mut forecaster = PortfolioForecaster::new(
                        portfolio_config.clone(),
                        forecast_config.clone()
                    ).unwrap();

                    forecaster.forecast_and_optimize(
                        black_box(&assets),
                        black_box(OptimizationStrategy::MaxSharpe)
                    ).unwrap()
                })
            }
        );
    }

    group.finish();
}

fn bench_multi_asset_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("finance_multi_asset_scaling");
    group.sample_size(15);

    let portfolio_config = PortfolioConfig::default();
    let mut forecast_config = ForecastConfig::default();
    forecast_config.use_arima = true;
    forecast_config.horizon = 10; // Shorter horizon for scaling test

    // Create assets with different counts
    for n_assets in [3, 5, 10].iter() {
        let mut assets = Vec::new();
        for i in 0..*n_assets {
            let mut prices = vec![100.0 + i as f64 * 10.0];
            for j in 1..60 {
                let t = j as f64 * 0.1;
                prices.push(prices[0] + 5.0 * (t * (1.0 + i as f64 * 0.1)).sin() + t * 0.5);
            }

            assets.push(Asset {
                ticker: format!("ASSET{}", i),
                expected_return: 0.10 + i as f64 * 0.01,
                prices,
                min_weight: 0.0,
                max_weight: 1.0,
            });
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_assets", n_assets)),
            n_assets,
            |b, _| {
                b.iter(|| {
                    let mut forecaster = PortfolioForecaster::new(
                        portfolio_config.clone(),
                        forecast_config.clone()
                    ).unwrap();

                    forecaster.forecast_and_optimize(
                        black_box(&assets),
                        black_box(OptimizationStrategy::MaxSharpe)
                    ).unwrap()
                })
            }
        );
    }

    group.finish();
}

fn bench_rebalancing_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("finance_rebalancing_schedule");
    group.sample_size(10);

    let assets = create_benchmark_assets();
    let portfolio_config = PortfolioConfig::default();
    let mut forecast_config = ForecastConfig::default();
    forecast_config.use_arima = true;
    forecast_config.horizon = 5; // Short horizon for multi-period

    for periods in [2, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_periods", periods)),
            periods,
            |b, &periods| {
                b.iter(|| {
                    let mut forecaster = PortfolioForecaster::new(
                        portfolio_config.clone(),
                        forecast_config.clone()
                    ).unwrap();

                    forecaster.generate_rebalancing_schedule(
                        black_box(&assets),
                        black_box(OptimizationStrategy::MaxSharpe),
                        black_box(periods)
                    ).unwrap()
                })
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_arima_forecasting,
    bench_lstm_forecasting,
    bench_portfolio_horizons,
    bench_multi_asset_scaling,
    bench_rebalancing_schedule
);

criterion_main!(benches);
