// Integration tests for Finance endpoints

use serde_json::{json, Value};

use super::common::*;

#[tokio::test]
async fn test_finance_portfolio_optimization() {
    let payload = json!({
        "assets": [
            {
                "symbol": "AAPL",
                "expected_return": 0.12,
                "volatility": 0.25,
                "current_price": 150.0
            },
            {
                "symbol": "GOOGL",
                "expected_return": 0.15,
                "volatility": 0.30,
                "current_price": 2800.0
            },
            {
                "symbol": "MSFT",
                "expected_return": 0.13,
                "volatility": 0.22,
                "current_price": 380.0
            }
        ],
        "constraints": {
            "max_position_size": 0.5,
            "min_position_size": 0.1,
            "max_total_risk": 0.20
        },
        "objective": "maximize_sharpe"
    });

    let response = post_authenticated("/api/v1/finance/optimize", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    verify_api_response(&body);

    let data = &body["data"];
    assert!(data["weights"].is_array(), "Should have weights array");
    assert!(data["expected_return"].is_number(), "Should have expected_return");
    assert!(data["expected_risk"].is_number(), "Should have expected_risk");
    assert!(data["sharpe_ratio"].is_number(), "Should have sharpe_ratio");
}

#[tokio::test]
async fn test_finance_risk_assessment() {
    let payload = json!({
        "portfolio_id": "test_portfolio_001",
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "entry_price": 145.0,
                "current_price": 150.0
            },
            {
                "symbol": "TSLA",
                "quantity": 50,
                "entry_price": 200.0,
                "current_price": 190.0
            }
        ],
        "risk_metrics": ["var", "cvar", "max_drawdown", "beta"]
    });

    let response = post_authenticated("/api/v1/finance/risk", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["var"].is_number(), "Should have VaR");
    assert!(data["cvar"].is_number(), "Should have CVaR");
    assert!(data["max_drawdown"].is_number(), "Should have max drawdown");
}

#[tokio::test]
async fn test_finance_backtest() {
    let payload = json!({
        "strategy_id": "momentum_strategy",
        "parameters": {
            "lookback_period": 20,
            "rebalance_frequency": "monthly"
        },
        "historical_data": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "symbols": ["AAPL", "GOOGL", "MSFT"]
        },
        "initial_capital": 100000.0
    });

    let response = post_authenticated("/api/v1/finance/backtest", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["total_return"].is_number());
    assert!(data["sharpe_ratio"].is_number());
    assert!(data["trades"].is_array());
}

#[tokio::test]
async fn test_finance_invalid_constraints() {
    let payload = json!({
        "assets": [
            {
                "symbol": "AAPL",
                "expected_return": 0.12,
                "volatility": 0.25,
                "current_price": 150.0
            }
        ],
        "constraints": {
            "max_position_size": 0.5,
            "min_position_size": 0.8,  // Invalid: min > max
            "max_total_risk": 0.20
        },
        "objective": "maximize_sharpe"
    });

    let response = post_authenticated("/api/v1/finance/optimize", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 400, "Should return 400 for invalid constraints");
}

#[tokio::test]
async fn test_finance_portfolio_rebalance() {
    let payload = json!({
        "portfolio_id": "test_portfolio_001",
        "current_positions": [
            {"symbol": "AAPL", "quantity": 100, "current_price": 150.0},
            {"symbol": "GOOGL", "quantity": 50, "current_price": 2800.0}
        ],
        "target_allocation": {
            "AAPL": 0.6,
            "GOOGL": 0.4
        },
        "rebalance_threshold": 0.05
    });

    let response = post_authenticated("/api/v1/finance/rebalance", DEFAULT_API_KEY, &payload)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: Value = response.json().await.unwrap();
    let data = &body["data"];

    assert!(data["trades"].is_array(), "Should have trades array");
    assert!(data["rebalance_needed"].is_boolean());
}
