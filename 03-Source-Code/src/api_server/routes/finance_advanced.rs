//! Advanced Finance API Routes (Worker 4)
//!
//! Provides REST endpoints for Worker 4's advanced/quantitative finance capabilities:
//! - Portfolio Optimization: Max Sharpe, Min volatility, Risk parity, Multi-objective
//! - GNN-based Portfolio Prediction: Graph neural network asset relationships
//! - Transfer Entropy Causality: Causal relationships between assets
//! - Hybrid Solver: Confidence-based routing between traditional and GNN methods
//! - Advanced Risk Analysis: VaR, CVaR, risk decomposition
//! - Portfolio Rebalancing: Dynamic rebalancing strategies

use axum::{
    Router,
    routing::{get, post},
    extract::{State, Json},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_server::{AppState, Result, ApiError};

// ============================================================================
// Request/Response Types
// ============================================================================

/// Portfolio optimization strategy
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationStrategy {
    MaximizeSharpe,
    MinimizeVolatility,
    MaximizeReturn,
    RiskParity,
    MultiObjective,
}

/// Asset input for portfolio optimization
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AdvancedAssetInput {
    /// Asset symbol/ticker
    pub symbol: String,
    /// Expected return (annualized)
    pub expected_return: f64,
    /// Volatility (annualized std dev)
    pub volatility: f64,
    /// Historical price series (for GNN/TE)
    pub price_history: Option<Vec<f64>>,
}

/// Advanced portfolio optimization request
#[derive(Debug, Deserialize, Serialize)]
pub struct AdvancedPortfolioOptimizationRequest {
    /// Assets to optimize
    pub assets: Vec<AdvancedAssetInput>,
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Use GNN predictor
    pub use_gnn: bool,
    /// Risk-free rate (for Sharpe ratio)
    pub risk_free_rate: Option<f64>,
    /// Constraints
    pub constraints: Option<PortfolioConstraints>,
}

/// Portfolio constraints
#[derive(Debug, Deserialize, Serialize)]
pub struct PortfolioConstraints {
    /// Minimum weight per asset
    pub min_weight: Option<f64>,
    /// Maximum weight per asset
    pub max_weight: Option<f64>,
    /// Target return (for min volatility)
    pub target_return: Option<f64>,
}

/// Advanced portfolio optimization response
#[derive(Debug, Deserialize, Serialize)]
pub struct AdvancedPortfolioOptimizationResponse {
    /// Optimized asset weights
    pub weights: Vec<AssetWeight>,
    /// Expected portfolio return
    pub expected_return: f64,
    /// Portfolio risk (volatility)
    pub portfolio_risk: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Optimization method used
    pub method: String,
    /// GNN confidence (if GNN used)
    pub gnn_confidence: Option<f64>,
}

/// Asset weight result
#[derive(Debug, Deserialize, Serialize)]
pub struct AssetWeight {
    pub symbol: String,
    pub weight: f64,
}

/// GNN portfolio prediction request
#[derive(Debug, Deserialize, Serialize)]
pub struct GnnPortfolioPredictionRequest {
    /// Assets with price histories
    pub assets: Vec<AdvancedAssetInput>,
    /// Prediction horizon (days)
    pub horizon: usize,
}

/// GNN portfolio prediction response
#[derive(Debug, Deserialize, Serialize)]
pub struct GnnPortfolioPredictionResponse {
    /// Predicted returns for each asset
    pub predicted_returns: Vec<AssetReturn>,
    /// Asset relationship graph
    pub asset_relationships: Vec<AssetRelationship>,
    /// Confidence score
    pub confidence: f64,
    /// Recommended weights
    pub recommended_weights: Vec<AssetWeight>,
}

/// Asset return prediction
#[derive(Debug, Deserialize, Serialize)]
pub struct AssetReturn {
    pub symbol: String,
    pub predicted_return: f64,
    pub confidence: f64,
}

/// Asset relationship from GNN
#[derive(Debug, Deserialize, Serialize)]
pub struct AssetRelationship {
    pub from_asset: String,
    pub to_asset: String,
    pub correlation: f64,
    pub causal_strength: f64,
}

/// Transfer Entropy causality analysis request
#[derive(Debug, Deserialize, Serialize)]
pub struct TransferEntropyCausalityRequest {
    /// Time series data for assets
    pub time_series: Vec<TimeSeriesData>,
    /// Analysis window (days)
    pub window: Option<usize>,
}

/// Time series data for one asset
#[derive(Debug, Deserialize, Serialize)]
pub struct TimeSeriesData {
    pub symbol: String,
    pub values: Vec<f64>,
}

/// Transfer Entropy causality analysis response
#[derive(Debug, Deserialize, Serialize)]
pub struct TransferEntropyCausalityResponse {
    /// Pairwise causal relationships
    pub causal_relationships: Vec<CausalRelationship>,
    /// Causal network summary
    pub network_summary: String,
    /// Most influential assets
    pub influential_assets: Vec<String>,
}

/// Causal relationship between two assets
#[derive(Debug, Deserialize, Serialize)]
pub struct CausalRelationship {
    pub source: String,
    pub target: String,
    pub transfer_entropy: f64,
    pub significance: f64,
    pub causal_strength: String, // "STRONG", "MODERATE", "WEAK"
}

/// Portfolio rebalancing request
#[derive(Debug, Deserialize, Serialize)]
pub struct PortfolioRebalancingRequest {
    /// Current portfolio weights
    pub current_weights: Vec<AssetWeight>,
    /// Target weights (from optimization)
    pub target_weights: Vec<AssetWeight>,
    /// Rebalancing frequency (days)
    pub frequency: usize,
    /// Transaction cost (%)
    pub transaction_cost: f64,
}

/// Portfolio rebalancing response
#[derive(Debug, Deserialize, Serialize)]
pub struct PortfolioRebalancingResponse {
    /// Rebalancing schedule
    pub schedule: Vec<RebalancingAction>,
    /// Total transaction cost
    pub total_cost: f64,
    /// Projected return after rebalancing
    pub projected_return: f64,
}

/// Single rebalancing action
#[derive(Debug, Deserialize, Serialize)]
pub struct RebalancingAction {
    pub day: usize,
    pub trades: Vec<Trade>,
    pub cost: f64,
}

/// Trade instruction
#[derive(Debug, Deserialize, Serialize)]
pub struct Trade {
    pub symbol: String,
    pub action: String, // "BUY" or "SELL"
    pub weight_change: f64,
}

// ============================================================================
// Route Handlers
// ============================================================================

/// Advanced portfolio optimization endpoint
async fn optimize_portfolio_advanced(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<AdvancedPortfolioOptimizationRequest>,
) -> Result<Json<AdvancedPortfolioOptimizationResponse>> {
    // TODO: Integrate with actual Worker 4 advanced finance module
    // For now, return mock response based on strategy

    let method = if req.use_gnn {
        "GNN-Enhanced Optimization"
    } else {
        match req.strategy {
            OptimizationStrategy::MaximizeSharpe => "Interior Point QP (Max Sharpe)",
            OptimizationStrategy::MinimizeVolatility => "Interior Point QP (Min Volatility)",
            OptimizationStrategy::MaximizeReturn => "Linear Programming (Max Return)",
            OptimizationStrategy::RiskParity => "Risk Parity Algorithm",
            OptimizationStrategy::MultiObjective => "Multi-Objective Optimization (Pareto)",
        }
    };

    let weights = vec![
        AssetWeight { symbol: "AAPL".to_string(), weight: 0.30 },
        AssetWeight { symbol: "GOOGL".to_string(), weight: 0.40 },
        AssetWeight { symbol: "MSFT".to_string(), weight: 0.30 },
    ];

    Ok(Json(AdvancedPortfolioOptimizationResponse {
        weights,
        expected_return: 0.15,
        portfolio_risk: 0.18,
        sharpe_ratio: 0.72,
        method: method.to_string(),
        gnn_confidence: if req.use_gnn { Some(0.85) } else { None },
    }))
}

/// GNN portfolio prediction endpoint
async fn predict_portfolio_gnn(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<GnnPortfolioPredictionRequest>,
) -> Result<Json<GnnPortfolioPredictionResponse>> {
    // TODO: Integrate with actual Worker 4 GNN module
    Ok(Json(GnnPortfolioPredictionResponse {
        predicted_returns: vec![
            AssetReturn {
                symbol: "AAPL".to_string(),
                predicted_return: 0.12,
                confidence: 0.88,
            },
            AssetReturn {
                symbol: "GOOGL".to_string(),
                predicted_return: 0.15,
                confidence: 0.82,
            },
        ],
        asset_relationships: vec![
            AssetRelationship {
                from_asset: "AAPL".to_string(),
                to_asset: "GOOGL".to_string(),
                correlation: 0.75,
                causal_strength: 0.42,
            },
        ],
        confidence: 0.85,
        recommended_weights: vec![
            AssetWeight { symbol: "AAPL".to_string(), weight: 0.45 },
            AssetWeight { symbol: "GOOGL".to_string(), weight: 0.55 },
        ],
    }))
}

/// Transfer Entropy causality analysis endpoint
async fn analyze_causality_transfer_entropy(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<TransferEntropyCausalityRequest>,
) -> Result<Json<TransferEntropyCausalityResponse>> {
    // TODO: Integrate with actual Worker 4 causal analysis module
    Ok(Json(TransferEntropyCausalityResponse {
        causal_relationships: vec![
            CausalRelationship {
                source: "AAPL".to_string(),
                target: "MSFT".to_string(),
                transfer_entropy: 0.42,
                significance: 0.95,
                causal_strength: "STRONG".to_string(),
            },
            CausalRelationship {
                source: "GOOGL".to_string(),
                target: "AAPL".to_string(),
                transfer_entropy: 0.28,
                significance: 0.82,
                causal_strength: "MODERATE".to_string(),
            },
        ],
        network_summary: "2 significant causal relationships identified. AAPL is the most influential asset.".to_string(),
        influential_assets: vec!["AAPL".to_string(), "GOOGL".to_string()],
    }))
}

/// Portfolio rebalancing endpoint
async fn rebalance_portfolio(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<PortfolioRebalancingRequest>,
) -> Result<Json<PortfolioRebalancingResponse>> {
    // TODO: Integrate with actual Worker 4 rebalancing module
    Ok(Json(PortfolioRebalancingResponse {
        schedule: vec![
            RebalancingAction {
                day: 0,
                trades: vec![
                    Trade {
                        symbol: "AAPL".to_string(),
                        action: "BUY".to_string(),
                        weight_change: 0.05,
                    },
                    Trade {
                        symbol: "MSFT".to_string(),
                        action: "SELL".to_string(),
                        weight_change: -0.05,
                    },
                ],
                cost: 0.002,
            },
        ],
        total_cost: 0.002,
        projected_return: 0.14,
    }))
}

// ============================================================================
// Router Setup
// ============================================================================

/// Create advanced finance routes
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // Portfolio optimization
        .route("/optimize_advanced", post(optimize_portfolio_advanced))

        // GNN prediction
        .route("/gnn/predict", post(predict_portfolio_gnn))

        // Transfer Entropy causality
        .route("/causality/transfer_entropy", post(analyze_causality_transfer_entropy))

        // Rebalancing
        .route("/rebalance", post(rebalance_portfolio))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::ApiConfig;

    #[tokio::test]
    async fn test_advanced_portfolio_optimization() {
        let state = Arc::new(AppState::new(ApiConfig::default()));
        let req = AdvancedPortfolioOptimizationRequest {
            assets: vec![
                AdvancedAssetInput {
                    symbol: "AAPL".to_string(),
                    expected_return: 0.12,
                    volatility: 0.20,
                    price_history: None,
                },
            ],
            strategy: OptimizationStrategy::MaximizeSharpe,
            use_gnn: false,
            risk_free_rate: Some(0.03),
            constraints: None,
        };

        let response = optimize_portfolio_advanced(State(state), Json(req)).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_gnn_portfolio_prediction() {
        let state = Arc::new(AppState::new(ApiConfig::default()));
        let req = GnnPortfolioPredictionRequest {
            assets: vec![
                AdvancedAssetInput {
                    symbol: "AAPL".to_string(),
                    expected_return: 0.12,
                    volatility: 0.20,
                    price_history: Some(vec![100.0, 102.0, 104.0]),
                },
            ],
            horizon: 30,
        };

        let response = predict_portfolio_gnn(State(state), Json(req)).await;
        assert!(response.is_ok());
    }
}
