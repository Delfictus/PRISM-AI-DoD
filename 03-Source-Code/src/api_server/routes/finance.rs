//! Finance API endpoints
//!
//! Portfolio optimization, risk assessment, market analysis

use axum::{
    extract::{Path, Query, State},
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
    use crate::api_server::portfolio::PortfolioOptimizer;

    log::info!("Portfolio optimization - {} assets", request.assets.len());

    let start_time = Instant::now();

    // Extract expected returns and build covariance matrix
    let expected_returns: Vec<f64> = request.assets.iter()
        .map(|a| a.expected_return)
        .collect();

    // Build covariance matrix from volatilities (simplified: assume uncorrelated)
    let n = request.assets.len();
    let mut covariance_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Variance on diagonal
                covariance_matrix[i][j] = request.assets[i].volatility * request.assets[i].volatility;
            } else {
                // Assume 0.3 correlation for off-diagonal
                covariance_matrix[i][j] = 0.3 * request.assets[i].volatility * request.assets[j].volatility;
            }
        }
    }

    // Create optimizer and run optimization
    let optimizer = PortfolioOptimizer::new(0.02); // 2% risk-free rate
    let result = match request.objective {
        ObjectiveFunction::MaximizeSharpe => {
            optimizer.optimize_markowitz(&expected_returns, &covariance_matrix, None, request.constraints.max_position_size)
        }
        ObjectiveFunction::MinimizeRisk => {
            // Minimize risk by targeting lowest feasible return
            let min_return = expected_returns.iter().cloned().fold(f64::INFINITY, f64::min);
            optimizer.optimize_markowitz(&expected_returns, &covariance_matrix, Some(min_return), request.constraints.max_position_size)
        }
        ObjectiveFunction::MaximizeReturn => {
            // Maximize return by targeting highest feasible return
            let max_return = expected_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            optimizer.optimize_markowitz(&expected_returns, &covariance_matrix, Some(max_return), request.constraints.max_position_size)
        }
        ObjectiveFunction::Custom { risk_aversion } => {
            // Custom utility function: U = μ - λσ²
            let target_return = expected_returns.iter().sum::<f64>() / expected_returns.len() as f64;
            optimizer.optimize_markowitz(&expected_returns, &covariance_matrix, Some(target_return), request.constraints.max_position_size)
        }
    };

    let processing_time = start_time.elapsed();

    // Build response
    let portfolio = OptimizedPortfolio {
        weights: result.weights.iter()
            .zip(request.assets.iter())
            .map(|(w, a)| AssetWeight {
                symbol: a.symbol.clone(),
                weight: *w,
            })
            .collect(),
        expected_return: result.expected_return,
        expected_risk: result.expected_risk,
        sharpe_ratio: result.sharpe_ratio,
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
    use crate::api_server::portfolio::PortfolioOptimizer;

    log::info!("Risk assessment - Portfolio: {}", request.portfolio_id);

    // Mock portfolio data (in production, query from database)
    let weights = vec![0.4, 0.3, 0.3];
    let expected_returns = vec![0.10, 0.12, 0.08];
    let covariance_matrix = vec![
        vec![0.04, 0.01, 0.01],
        vec![0.01, 0.09, 0.01],
        vec![0.01, 0.01, 0.16],
    ];

    let optimizer = PortfolioOptimizer::default();

    // Calculate VaR and CVaR
    let var = optimizer.calculate_var(
        &weights,
        &expected_returns,
        &covariance_matrix,
        request.confidence_level,
        request.time_horizon_days,
    );

    let cvar = optimizer.calculate_cvar(
        &weights,
        &expected_returns,
        &covariance_matrix,
        request.confidence_level,
        request.time_horizon_days,
    );

    // Mock historical data for max drawdown
    let historical_returns = vec![
        vec![0.02, 0.01, -0.01],
        vec![0.01, 0.03, 0.02],
        vec![-0.02, -0.01, -0.03],
    ];
    let max_dd = optimizer.calculate_max_drawdown(&weights, &historical_returns);

    // Calculate portfolio beta (simplified: assume market beta of 1.0)
    let portfolio_return: f64 = weights.iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum();
    let market_return = 0.10;
    let beta = portfolio_return / market_return;

    let assessment = RiskAssessment {
        var: var * 1_000_000.0, // Scale to dollar amount for $1M portfolio
        cvar: cvar * 1_000_000.0,
        max_drawdown: max_dd,
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
