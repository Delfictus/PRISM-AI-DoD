# PRISM-AI Go Client Library

Official Go SDK for the PRISM-AI REST API.

## Installation

```bash
go get github.com/your-org/prism-ai/client-go
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "time"

    prismclient "github.com/your-org/prism-ai/client-go"
)

func main() {
    // Initialize client
    client := prismclient.NewClient(prismclient.ClientConfig{
        APIKey:  "your-api-key-here",
        BaseURL: "http://localhost:8080",
        Timeout: 30 * time.Second,
        VerifySSL: true,
    })

    // Check API health
    health, err := client.Health()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(health)

    // Detect threat
    threat, err := client.DetectThreat(
        42,          // sv_id
        1234567890,  // timestamp
        map[string]interface{}{
            "width":         640,
            "height":        480,
            "centroid_x":    320.0,
            "centroid_y":    240.0,
            "hotspot_count": 5,
        },
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Threat: %s (%.1f%% confidence)\n", threat.ThreatType, threat.Confidence*100)

    // Optimize portfolio
    portfolio, err := client.OptimizePortfolio(
        []map[string]interface{}{
            {"symbol": "AAPL", "expected_return": 0.12, "volatility": 0.25, "current_price": 150.0},
            {"symbol": "GOOGL", "expected_return": 0.15, "volatility": 0.30, "current_price": 2800.0},
        },
        map[string]interface{}{
            "max_position_size": 0.5,
            "min_position_size": 0.1,
            "max_total_risk":    0.20,
        },
        "maximize_sharpe",
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Sharpe ratio: %.2f\n", portfolio.SharpeRatio)

    // Query LLM
    response, err := client.QueryLLM(
        "Explain quantum computing in one sentence.",
        nil,  // use default model
        0.7,  // temperature
        100,  // max_tokens
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("LLM: %s\n", response.Text)
}
```

## Usage Examples

### PWSA - Threat Detection

```go
// Detect threat from IR frame
threat, err := client.DetectThreat(
    42,          // sv_id
    1234567890,  // timestamp
    map[string]interface{}{
        "width":         640,
        "height":        480,
        "centroid_x":    320.0,
        "centroid_y":    240.0,
        "hotspot_count": 5,
    },
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Threat ID: %s\n", threat.ThreatID)
fmt.Printf("Type: %s\n", threat.ThreatType)
fmt.Printf("Confidence: %.1f%%\n", threat.Confidence*100)
fmt.Printf("Position: %v\n", threat.Position)
```

### PWSA - Sensor Fusion

```go
// Fuse multi-sensor data
fusion, err := client.FuseSensors(
    42,          // sv_id
    1234567890,  // timestamp
    map[string]interface{}{
        "ir": map[string]interface{}{
            "frame_id": 1234,
            "targets": []map[string]interface{}{
                {
                    "id":                   "target_1",
                    "azimuth":              10.0,
                    "elevation":            5.0,
                    "range":                25000.0,
                    "velocity":             500.0,
                    "ir_signature":         0.85,
                    "radar_cross_section":  1.5,
                },
            },
        },
        "radar": map[string]interface{}{
            "scan_id": 5678,
            "targets": []map[string]interface{}{},
        },
    },
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Tracks: %d\n", fusion.NumTracks)
fmt.Printf("Fusion quality: %.1f%%\n", fusion.FusionQuality*100)
```

### PWSA - Trajectory Prediction

```go
// Predict threat trajectory
prediction, err := client.PredictTrajectory(
    "threat_001",  // track_id
    []map[string]interface{}{
        {
            "timestamp": 1234567890,
            "position":  []float64{10000.0, 20000.0, 5000.0},
            "velocity":  []float64{800.0, -200.0, 0.0},
        },
        {
            "timestamp": 1234567900,
            "position":  []float64{18000.0, 18000.0, 5000.0},
            "velocity":  []float64{800.0, -200.0, 0.0},
        },
    },
    30,               // prediction_horizon
    "kalman_filter",  // model
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Confidence: %.1f%%\n", prediction.Confidence*100)
if prediction.TimeToImpact != nil {
    fmt.Printf("Time to impact: %.0fs\n", *prediction.TimeToImpact)
}
```

### Finance - Portfolio Optimization

```go
// Optimize portfolio
portfolio, err := client.OptimizePortfolio(
    []map[string]interface{}{
        {
            "symbol":          "AAPL",
            "expected_return": 0.12,
            "volatility":      0.25,
            "current_price":   150.0,
        },
        {
            "symbol":          "GOOGL",
            "expected_return": 0.15,
            "volatility":      0.30,
            "current_price":   2800.0,
        },
        {
            "symbol":          "MSFT",
            "expected_return": 0.13,
            "volatility":      0.22,
            "current_price":   380.0,
        },
    },
    map[string]interface{}{
        "max_position_size": 0.5,
        "min_position_size": 0.1,
        "max_total_risk":    0.20,
    },
    "maximize_sharpe",  // objective
)
if err != nil {
    log.Fatal(err)
}

fmt.Println("Optimal weights:")
for _, weight := range portfolio.Weights {
    fmt.Printf("  %s: %.1f%%\n", weight.Symbol, weight.Weight*100)
}
fmt.Printf("Expected return: %.1f%%\n", portfolio.ExpectedReturn*100)
fmt.Printf("Expected risk: %.1f%%\n", portfolio.ExpectedRisk*100)
fmt.Printf("Sharpe ratio: %.2f\n", portfolio.SharpeRatio)
```

### Finance - Risk Assessment

```go
// Assess portfolio risk
risk, err := client.AssessRisk(
    "portfolio_001",  // portfolio_id
    []map[string]interface{}{
        {"symbol": "AAPL", "quantity": 100, "entry_price": 145.0, "current_price": 150.0},
        {"symbol": "TSLA", "quantity": 50, "entry_price": 200.0, "current_price": 190.0},
    },
    []string{"var", "cvar", "max_drawdown", "beta"},  // risk_metrics
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("VaR: %.2f%%\n", risk["var"].(float64)*100)
fmt.Printf("CVaR: %.2f%%\n", risk["cvar"].(float64)*100)
fmt.Printf("Max Drawdown: %.2f%%\n", risk["max_drawdown"].(float64)*100)
```

### LLM - Single Query

```go
// Query language model
response, err := client.QueryLLM(
    "What is the capital of France?",
    nil,  // use default model
    0.1,  // temperature
    50,   // max_tokens
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Response: %s\n", response.Text)
fmt.Printf("Model: %s\n", response.ModelUsed)
fmt.Printf("Tokens: %d\n", response.TokensUsed)
fmt.Printf("Cost: $%.4f\n", response.CostUsd)
```

### LLM - Multi-Model Consensus

```go
// Get consensus from multiple models
consensus, err := client.LLMConsensus(
    "What is the capital of Australia?",
    []map[string]interface{}{
        {"name": "gpt-4", "weight": 1.0},
        {"name": "gpt-3.5-turbo", "weight": 0.8},
        {"name": "claude-3", "weight": 1.0},
    },
    "majority_vote",  // strategy
    0.1,              // temperature
    50,               // max_tokens
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Consensus: %s\n", consensus.ConsensusText)
fmt.Printf("Confidence: %.1f%%\n", consensus.Confidence*100)
fmt.Printf("Agreement: %.1f%%\n", consensus.AgreementRate*100)
fmt.Printf("Total cost: $%.4f\n", consensus.TotalCostUsd)
```

### Time Series - Forecasting

```go
// Forecast time series
forecast, err := client.ForecastTimeSeries(
    "sales_data",                                     // series_id
    []float64{100, 105, 110, 108, 115, 120, 125, 130}, // historical_data
    nil,                                              // timestamps (optional)
    5,                                                // horizon
    "arima",                                          // method
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Predictions: %v\n", forecast.Predictions)
fmt.Printf("Confidence intervals: %v\n", forecast.ConfidenceIntervals)
fmt.Printf("Metrics: %v\n", forecast.Metrics)
```

### Pixel Processing

```go
// Generate sample IR frame
width := 640
height := 480
pixels := make([]int, width*height)
for i := range pixels {
    pixels[i] = rand.Intn(256)
}

// Process pixels
result, err := client.ProcessPixels(
    12345,  // frame_id
    width,
    height,
    pixels,
    map[string]interface{}{
        "detect_hotspots": true,
        "compute_entropy": true,
        "apply_tda":       true,
    },
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Hotspots detected: %d\n", len(result.Hotspots))
if result.Entropy != nil {
    fmt.Printf("Entropy: %.3f\n", *result.Entropy)
}
if result.TDAFeatures != nil {
    fmt.Printf("TDA features: %d\n", len(result.TDAFeatures))
}
```

## Error Handling

```go
import (
    prismclient "github.com/your-org/prism-ai/client-go"
)

threat, err := client.DetectThreat(42, 1234567890, map[string]interface{}{})
if err != nil {
    switch e := err.(type) {
    case *prismclient.AuthenticationError:
        fmt.Println("Invalid API key")
    case *prismclient.RateLimitError:
        fmt.Printf("Rate limited. Retry after %d seconds\n", e.RetryAfter)
    case *prismclient.ValidationError:
        fmt.Printf("Validation error: %s\n", e.Message)
    case *prismclient.ServerError:
        fmt.Println("Server error occurred")
    case *prismclient.NetworkError:
        fmt.Println("Network connection failed")
    case *prismclient.TimeoutError:
        fmt.Println("Request timed out")
    default:
        fmt.Printf("Unknown error: %v\n", err)
    }
    return
}
```

## Configuration

### Custom Base URL

```go
client := prismclient.NewClient(prismclient.ClientConfig{
    APIKey:  "your-key",
    BaseURL: "https://api.prism-ai.example.com",
})
```

### Custom Timeout

```go
client := prismclient.NewClient(prismclient.ClientConfig{
    APIKey:  "your-key",
    Timeout: 60 * time.Second, // 60 seconds
})
```

### Disable SSL Verification (development only)

```go
client := prismclient.NewClient(prismclient.ClientConfig{
    APIKey:    "your-key",
    VerifySSL: false, // NOT recommended for production
})
```

## API Reference

### Client Methods

**Health & Info:**
- `Health()` - Check API health
- `Info()` - Get API information

**PWSA:**
- `DetectThreat()` - Detect threats from IR data
- `FuseSensors()` - Fuse multi-sensor data
- `PredictTrajectory()` - Predict threat trajectory
- `PrioritizeThreats()` - Prioritize multiple threats

**Finance:**
- `OptimizePortfolio()` - Optimize portfolio allocation
- `AssessRisk()` - Assess portfolio risk
- `BacktestStrategy()` - Backtest trading strategy

**LLM:**
- `QueryLLM()` - Query language model
- `LLMConsensus()` - Multi-model consensus
- `ListLLMModels()` - List available models

**Time Series:**
- `ForecastTimeSeries()` - Forecast time series

**Pixel Processing:**
- `ProcessPixels()` - Process pixel data

## Testing

```bash
go test -v ./...
```

## License

MIT License

## Support

- Documentation: https://docs.prism-ai.example.com
- Issues: https://github.com/your-org/prism-ai/issues
- Email: support@prism-ai.example.com
