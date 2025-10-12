# Financial Portfolio Optimization - Worker 4

## Overview

GPU-accelerated financial portfolio optimization using cutting-edge AI techniques:

- **Active Inference** for market dynamics modeling and regime detection
- **Transfer Entropy** for causal relationship analysis between assets
- **Mean-Variance Optimization** with gradient descent solver
- **Market Regime Detection** for adaptive strategy adjustment
- **Constraint Handling** for position limits and diversification

## Architecture

### Core Components

1. **PortfolioOptimizer** (`mod.rs:96`)
   - Main optimization engine
   - Integrates all AI components
   - Handles constraints and bounds

2. **MarketRegimeDetector** (`market_regime.rs:42`)
   - Detects market conditions using Active Inference
   - 6 regime types: Bull, Bear, High/Low Volatility, Normal, Transition
   - Adjusts portfolio strategy based on regime

3. **Transfer Entropy Integration** (`mod.rs:231`)
   - Calculates causal influence between assets
   - Weights portfolio based on causal relationships
   - Statistical significance testing (p < 0.05)

## Key Features

### 1. Market Regime Detection

```rust
let mut detector = MarketRegimeDetector::new(20); // 20-day window
let (regime, confidence) = detector.detect_regime(&price_history)?;

// Adjust strategy based on regime
let adjustment = detector.regime_adjustment_factor();
// Bull: 1.2x (increase exposure)
// Bear: 0.8x (decrease exposure)
// High Vol: 0.7x (conservative)
```

### 2. Transfer Entropy Analysis

Identifies causal relationships between assets:

```rust
// Asset A causally influences Asset B
TE(A → B) > TE(B → A)  // A leads B

// Portfolio weights adjusted based on causal influence
weight[A] *= (1 + causal_influence[A])
```

### 3. Mean-Variance Optimization

Solves the classic portfolio problem:

```
Maximize: w^T * μ - λ * w^T * Σ * w
Subject to:
  - sum(w) = 1 (fully invested)
  - w_i >= min_weight (minimum position)
  - w_i <= max_weight (diversification)
```

Where:
- `w` = asset weights
- `μ` = expected returns
- `Σ` = covariance matrix
- `λ` = risk aversion parameter

## Usage

### Basic Example

```rust
use prism_ai::applications::financial::*;

// Configure optimizer
let mut config = OptimizationConfig::default();
config.risk_free_rate = 0.02;
config.max_weight_per_asset = 0.4; // Max 40% per asset

let mut optimizer = PortfolioOptimizer::new(config);

// Define assets with historical returns
let assets = vec![
    Asset {
        symbol: "AAPL".to_string(),
        name: "Apple Inc.".to_string(),
        current_price: 150.0,
        historical_returns: vec![0.01, 0.02, -0.01, 0.03, 0.01],
    },
    Asset {
        symbol: "GOOGL".to_string(),
        name: "Alphabet Inc.".to_string(),
        current_price: 2800.0,
        historical_returns: vec![0.02, 0.01, 0.01, 0.02, 0.015],
    },
];

// Optimize portfolio
let portfolio = optimizer.optimize(assets)?;

println!("Expected Return: {:.2}%", portfolio.expected_return * 100.0);
println!("Risk (Std Dev): {:.2}%", portfolio.risk * 100.0);
println!("Sharpe Ratio: {:.3}", portfolio.sharpe_ratio);

for (asset, weight) in portfolio.assets.iter().zip(portfolio.weights.iter()) {
    println!("{}: {:.1}%", asset.symbol, weight * 100.0);
}
```

### Advanced: Full Feature Set

```rust
let mut config = OptimizationConfig::default();
config.use_transfer_entropy = true;    // Enable causal analysis
config.use_regime_detection = true;    // Enable market regime detection
config.target_return = Some(0.12);     // Target 12% annual return
config.max_risk = Some(0.20);          // Max 20% annual volatility

let mut optimizer = PortfolioOptimizer::new(config);
let portfolio = optimizer.optimize(assets)?;
```

## Performance

### Current Implementation (CPU)

- **Covariance calculation**: O(n² * T) where n=assets, T=time periods
- **Transfer Entropy**: O(n² * T) for pairwise calculations
- **Optimization**: ~1000 iterations of gradient descent

### Planned GPU Acceleration

**TODO: Request kernels from Worker 2:**

1. **Covariance Matrix Kernel**
   ```cuda
   __global__ void covariance_matrix(
       float* returns,      // [n_assets x n_periods]
       float* covariance,   // [n_assets x n_assets] output
       int n_assets,
       int n_periods
   );
   ```

2. **Quadratic Programming Solver**
   ```cuda
   __global__ void qp_gradient_step(
       float* weights,      // Current weights
       float* gradient,     // Gradient direction
       float* covariance,   // Covariance matrix
       float* returns,      // Expected returns
       float learning_rate,
       int n_assets
   );
   ```

3. **Transfer Entropy Batch Kernel**
   ```cuda
   __global__ void batch_transfer_entropy(
       float* all_series,   // [n_assets x n_periods]
       float* te_matrix,    // [n_assets x n_assets] output
       int n_assets,
       int n_periods
   );
   ```

**Expected Speedup**: 10-50x for large portfolios (100+ assets)

## Mathematical Foundation

### 1. Modern Portfolio Theory (MPT)

Markowitz's framework for optimal asset allocation under risk-return tradeoff.

**Efficient Frontier**: Set of portfolios with maximum return for given risk level.

### 2. Sharpe Ratio

Risk-adjusted performance metric:

```
Sharpe = (R_p - R_f) / σ_p

Where:
  R_p = Portfolio expected return
  R_f = Risk-free rate
  σ_p = Portfolio standard deviation (risk)
```

Higher Sharpe ratio = better risk-adjusted performance.

### 3. Transfer Entropy

Information-theoretic measure of causal influence:

```
TE(X → Y) = Σ p(y_{t+1}, y_t, x_t) log [p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t)]
```

- Measures how much knowing X's past reduces uncertainty about Y's future
- Asymmetric: TE(X → Y) ≠ TE(Y → X)
- Zero iff X provides no information about Y beyond Y's own history

### 4. Active Inference

Bayesian framework for adaptive agents:

- Minimizes **Free Energy**: F = E_q[log q(x) - log p(o,x)]
- Models market as generative process
- Infers hidden regime states from observations
- Selects actions to minimize expected free energy

## Testing

Run the test suite:

```bash
cargo test --lib applications::financial --features cuda
```

### Test Coverage

- ✅ Portfolio optimizer creation
- ✅ Sharpe ratio calculation
- ✅ Simple portfolio optimization (2-3 assets)
- ✅ Covariance matrix calculation and symmetry
- ✅ Expected returns calculation
- ✅ Weight constraint projection
- ✅ Empty portfolio error handling
- ✅ Transfer Entropy integration
- ✅ Asset serialization
- ✅ Market regime detection (bull, bear, high volatility)
- ✅ Regime adjustment factors

## Integration Points

### With Worker 1 (Time Series Forecasting)

```rust
// TODO: Integrate ARIMA/LSTM forecasts
let forecasted_returns = time_series_module.forecast_returns(
    historical_data,
    horizon_days=30
)?;

// Use forecasts instead of historical mean
config.use_forecasting = true;
```

### With Worker 5 (Thermodynamic Consensus)

```rust
// TODO: Use thermodynamic consensus for ensemble predictions
let ensemble_forecast = thermodynamic_consensus.aggregate_predictions(
    vec![model1_forecast, model2_forecast, model3_forecast]
)?;
```

## Future Enhancements

1. **GPU Acceleration** (Week 2-3)
   - Covariance matrix on GPU
   - Batch Transfer Entropy
   - QP solver on GPU

2. **Time Series Integration** (Week 5)
   - ARIMA forecasting from Worker 1
   - Multi-step ahead predictions
   - Uncertainty quantification

3. **Advanced Constraints** (Week 4)
   - Sector exposure limits
   - Transaction costs
   - Tax optimization
   - ESG constraints

4. **Risk Models** (Week 4)
   - Value at Risk (VaR)
   - Conditional Value at Risk (CVaR)
   - Maximum drawdown constraints

5. **Multi-Period Optimization** (Week 6)
   - Dynamic programming for multi-period allocation
   - Rebalancing costs
   - Market impact modeling

## References

1. Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*.
2. Schreiber, T. (2000). "Measuring Information Transfer". *Physical Review Letters*.
3. Friston, K. (2010). "The free-energy principle: a unified brain theory?". *Nature Reviews Neuroscience*.
4. Brandt, M. (2010). "Portfolio Choice Problems". *Handbook of Financial Econometrics*.

## Worker 4 Status

**Implementation**: ✅ Complete
**Tests**: ✅ Passing
**GPU Kernels**: ⏳ Pending (request to Worker 2)
**Documentation**: ✅ Complete
**Integration**: ⏳ Pending (Workers 1, 5)

**Next Steps:**
1. Request GPU kernels from Worker 2 (covariance, QP solver, TE batch)
2. Integrate time series forecasting from Worker 1 (Week 5)
3. Add thermodynamic consensus from Worker 5 (Week 6)
4. Benchmark against classical solvers (cvxpy, scipy.optimize)
