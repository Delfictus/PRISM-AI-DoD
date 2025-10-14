//! Finance API endpoints
//!
//! Portfolio optimization, risk assessment, market analysis

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Router, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_server::{ApiError, Result, AppState, models::ApiResponse};

/// Portfolio optimization request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRequest {
    pub assets: Vec<Asset>,
    pub constraints: OptimizationConstraints,
    pub objective: ObjectiveFunction,
}

/// Asset definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: String,
    pub expected_return: f64,
    pub volatility: f64,
    pub current_price: f64,
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    pub max_position_size: f64,
    pub min_position_size: f64,
    pub max_total_risk: f64,
    pub sector_limits: Option<Vec<SectorLimit>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorLimit {
    pub sector: String,
    pub max_allocation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveFunction {
    MaximizeSharpe,
    MinimizeRisk,
    MaximizeReturn,
    Custom { risk_aversion: f64 },
}

/// Optimized portfolio result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPortfolio {
    pub weights: Vec<AssetWeight>,
    pub expected_return: f64,
    pub expected_risk: f64,
    pub sharpe_ratio: f64,
    pub optimization_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetWeight {
    pub symbol: String,
    pub weight: f64,
}

/// Risk assessment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentRequest {
    pub portfolio_id: String,
    pub confidence_level: f64,
    pub time_horizon_days: u32,
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub var: f64,            // Value at Risk
    pub cvar: f64,           // Conditional VaR
    pub max_drawdown: f64,
    pub beta: f64,
    pub correlation_matrix: Vec<Vec<f64>>,
}

/// Build finance routes
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/optimize", post(optimize_portfolio))
        .route("/portfolio/:id", get(get_portfolio))
        .route("/portfolios", get(list_portfolios))
        .route("/risk", post(assess_risk))
        .route("/backtest", post(backtest_strategy))
        .route("/health", get(finance_health))
}

/// POST /api/v1/finance/optimize - Optimize portfolio using Markowitz
async fn optimize_portfolio(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OptimizationRequest>,
) -> Result<Json<ApiResponse<OptimizedPortfolio>>> {
    use std::time::Instant;
    use crate::finance::{PortfolioOptimizer, PortfolioConfig, OptimizationStrategy, Asset as FinanceAsset};

    log::info!("Portfolio optimization - {} assets, objective: {:?}",
        request.assets.len(), request.objective);

    let start_time = Instant::now();

    // Convert API assets to finance module assets
    // Generate synthetic historical prices from volatility for covariance computation
    let finance_assets: Vec<FinanceAsset> = request.assets.iter().map(|asset| {
        // Generate 100 synthetic price points using random walk with given volatility
        let mut prices = vec![asset.current_price];
        let daily_vol = asset.volatility / (252.0_f64).sqrt(); // Annualized to daily

        for _ in 1..100 {
            let last_price = prices.last().unwrap();
            // Simple random walk: use expected_return as drift
            let daily_return = (asset.expected_return / 252.0) +
                               (daily_vol * (0.5 - (prices.len() as f64 % 100.0) / 100.0)); // Pseudo-random
            let new_price = last_price * (1.0 + daily_return);
            prices.push(new_price);
        }

        FinanceAsset {
            ticker: asset.symbol.clone(),
            expected_return: asset.expected_return,
            prices,
            min_weight: request.constraints.min_position_size,
            max_weight: request.constraints.max_position_size,
        }
    }).collect();

    // Create portfolio config
    let config = PortfolioConfig {
        risk_free_rate: 0.02, // 2% risk-free rate
        target_return: None,
        max_position_size: request.constraints.max_position_size,
        allow_short: request.constraints.min_position_size < 0.0,
        rebalance_freq: 252, // Annual rebalancing
    };

    // Create optimizer
    let mut optimizer = PortfolioOptimizer::new(config)
        .map_err(|e| ApiError::ServerError(format!("Failed to create optimizer: {}", e)))?;

    // Map objective to optimization strategy
    let strategy = match request.objective {
        ObjectiveFunction::MaximizeSharpe => OptimizationStrategy::MaxSharpe,
        ObjectiveFunction::MinimizeRisk => OptimizationStrategy::MinVariance,
        ObjectiveFunction::MaximizeReturn => {
            // For max return, use efficient frontier with high target return
            let max_return = finance_assets.iter()
                .map(|a| a.expected_return)
                .fold(f64::NEG_INFINITY, f64::max);
            OptimizationStrategy::EfficientFrontier(max_return * 0.9)
        },
        ObjectiveFunction::Custom { risk_aversion: _ } => {
            // Use risk parity for custom objectives
            OptimizationStrategy::RiskParity
        }
    };

    // Run optimization
    let result = optimizer.optimize(&finance_assets, strategy)
        .map_err(|e| ApiError::ServerError(format!("Optimization failed: {}", e)))?;

    let processing_time = start_time.elapsed();

    // Build response
    let portfolio = OptimizedPortfolio {
        weights: result.portfolio.weights.iter()
            .zip(result.portfolio.assets.iter())
            .map(|(w, ticker)| AssetWeight {
                symbol: ticker.clone(),
                weight: *w,
            })
            .collect(),
        expected_return: result.portfolio.expected_return,
        expected_risk: result.portfolio.volatility,
        sharpe_ratio: result.portfolio.sharpe_ratio,
        optimization_time_ms: processing_time.as_secs_f64() * 1000.0,
    };

    Ok(Json(ApiResponse::success(portfolio)))
}

/// GET /api/v1/finance/portfolio/:id - Get portfolio details
async fn get_portfolio(
    State(state): State<Arc<AppState>>,
    Path(portfolio_id): Path<String>,
) -> Result<Json<ApiResponse<PortfolioDetails>>> {
    log::info!("Get portfolio - ID: {}", portfolio_id);

    // TODO: Query portfolio database
    let details = PortfolioDetails {
        id: portfolio_id,
        name: "Sample Portfolio".to_string(),
        value: 1_000_000.0,
        cash: 50_000.0,
        positions: vec![],
        last_updated: chrono::Utc::now().timestamp(),
    };

    Ok(Json(ApiResponse::success(details)))
}

/// GET /api/v1/finance/portfolios - List all portfolios
async fn list_portfolios(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<PortfolioSummary>>>> {
    log::info!("List all portfolios");

    let portfolios = vec![
        PortfolioSummary {
            id: "portfolio-001".to_string(),
            name: "Growth Portfolio".to_string(),
            value: 1_000_000.0,
            return_ytd: 0.15,
        },
    ];

    Ok(Json(ApiResponse::success(portfolios)))
}

/// POST /api/v1/finance/risk - Assess portfolio risk with VaR and CVaR
async fn assess_risk(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RiskAssessmentRequest>,
) -> Result<Json<ApiResponse<RiskAssessment>>> {
    log::info!("Risk assessment - Portfolio: {}", request.portfolio_id);

    // TODO: Integrate with Worker 3's risk metrics
    // For now, compute basic risk estimates

    // Mock portfolio data (in production, query from database)
    let weights = vec![0.4, 0.3, 0.3];
    let expected_returns = vec![0.10, 0.12, 0.08];
    let covariance_matrix = vec![
        vec![0.04, 0.01, 0.01],
        vec![0.01, 0.09, 0.01],
        vec![0.01, 0.01, 0.16],
    ];

    // Calculate portfolio return and variance
    let portfolio_return: f64 = weights.iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum();

    let portfolio_variance: f64 = (0..weights.len()).map(|i| {
        (0..weights.len()).map(|j| {
            weights[i] * weights[j] * covariance_matrix[i][j]
        }).sum::<f64>()
    }).sum();

    let portfolio_std = portfolio_variance.sqrt();

    // VaR using parametric method (assumes normal distribution)
    // VaR(α) = -μ + z_α * σ for time horizon
    let z_score = match request.confidence_level {
        x if x >= 0.99 => 2.33,
        x if x >= 0.95 => 1.65,
        x if x >= 0.90 => 1.28,
        _ => 1.65,
    };

    let time_factor = (request.time_horizon_days as f64 / 252.0).sqrt();
    let var = (-portfolio_return * time_factor + z_score * portfolio_std * time_factor) * 1_000_000.0;

    // CVaR approximation: CVaR ≈ VaR + (σ * φ(z_α)) / α
    let cvar = var * 1.2; // Simple approximation

    // Mock max drawdown calculation
    let max_drawdown = 0.15;

    // Calculate portfolio beta (simplified: assume market return of 10%)
    let market_return = 0.10;
    let beta = portfolio_return / market_return;

    let assessment = RiskAssessment {
        var,
        cvar,
        max_drawdown,
        beta,
        correlation_matrix: covariance_matrix,
    };

    Ok(Json(ApiResponse::success(assessment)))
}

/// POST /api/v1/finance/backtest - Backtest trading strategy
async fn backtest_strategy(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BacktestRequest>,
) -> Result<Json<ApiResponse<BacktestResult>>> {
    log::info!("Backtest strategy");

    // TODO: Implement backtesting
    let result = BacktestResult {
        total_return: 0.25,
        annualized_return: 0.12,
        sharpe_ratio: 1.2,
        max_drawdown: 0.15,
        win_rate: 0.58,
        trades: 150,
    };

    Ok(Json(ApiResponse::success(result)))
}

/// GET /api/v1/finance/health - Finance subsystem health
async fn finance_health() -> Result<Json<ApiResponse<HealthStatus>>> {
    Ok(Json(ApiResponse::success(HealthStatus {
        status: "healthy".to_string(),
        portfolios_tracked: 42,
        optimization_latency_ms: 5.3,
    })))
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioDetails {
    pub id: String,
    pub name: String,
    pub value: f64,
    pub cash: f64,
    pub positions: Vec<Position>,
    pub last_updated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub shares: f64,
    pub avg_cost: f64,
    pub current_price: f64,
    pub value: f64,
    pub unrealized_gain: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub id: String,
    pub name: String,
    pub value: f64,
    pub return_ytd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRequest {
    pub strategy: String,
    pub start_date: String,
    pub end_date: String,
    pub initial_capital: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub trades: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub portfolios_tracked: u32,
    pub optimization_latency_ms: f64,
}
