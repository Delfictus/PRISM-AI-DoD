# PRISM-AI JavaScript Client Library

Official JavaScript/Node.js SDK for the PRISM-AI REST API.

## Installation

### With npm

```bash
npm install @prism-ai/client
```

### With yarn

```bash
yarn add @prism-ai/client
```

### From source

```bash
cd examples/javascript
npm install
npm link
```

## Quick Start

```javascript
const { PrismClient } = require('@prism-ai/client');

// Initialize client
const client = new PrismClient({ apiKey: 'your-api-key-here' });

// Check API health
const health = await client.health();
console.log(health);

// Detect threat
const threat = await client.detectThreat(
  42, // sv_id
  1234567890, // timestamp
  {
    width: 640,
    height: 480,
    centroid_x: 320.0,
    centroid_y: 240.0,
    hotspot_count: 5
  }
);
console.log(`Threat: ${threat.threatType} (${(threat.confidence * 100).toFixed(1)}% confidence)`);

// Optimize portfolio
const portfolio = await client.optimizePortfolio(
  [
    { symbol: 'AAPL', expected_return: 0.12, volatility: 0.25, current_price: 150.0 },
    { symbol: 'GOOGL', expected_return: 0.15, volatility: 0.30, current_price: 2800.0 },
  ],
  {
    max_position_size: 0.5,
    min_position_size: 0.1,
    max_total_risk: 0.20
  }
);
console.log(`Sharpe ratio: ${portfolio.sharpeRatio.toFixed(2)}`);

// Query LLM
const response = await client.queryLLM(
  'Explain quantum computing in one sentence.',
  null, // use default model
  0.7, // temperature
  100 // max_tokens
);
console.log(`LLM: ${response.text}`);
```

## Usage Examples

### ES6 Imports

```javascript
import { PrismClient, AuthenticationError, RateLimitError } from '@prism-ai/client';

const client = new PrismClient({ apiKey: 'your-key' });
```

### CommonJS

```javascript
const { PrismClient } = require('@prism-ai/client');

const client = new PrismClient({ apiKey: 'your-key' });
```

### PWSA - Threat Detection

```javascript
// Detect threat from IR frame
const threat = await client.detectThreat(
  42, // sv_id
  1234567890, // timestamp
  {
    width: 640,
    height: 480,
    centroid_x: 320.0,
    centroid_y: 240.0,
    hotspot_count: 5
  }
);

console.log(`Threat ID: ${threat.threatId}`);
console.log(`Type: ${threat.threatType}`);
console.log(`Confidence: ${(threat.confidence * 100).toFixed(1)}%`);
console.log(`Position: ${threat.position}`);
```

### PWSA - Sensor Fusion

```javascript
// Fuse multi-sensor data
const fusion = await client.fuseSensors(
  42, // sv_id
  1234567890, // timestamp
  {
    ir: {
      frame_id: 1234,
      targets: [
        {
          id: 'target_1',
          azimuth: 10.0,
          elevation: 5.0,
          range: 25000.0,
          velocity: 500.0,
          ir_signature: 0.85,
          radar_cross_section: 1.5
        }
      ]
    },
    radar: {
      scan_id: 5678,
      targets: []
    }
  }
);

console.log(`Tracks: ${fusion.numTracks}`);
console.log(`Fusion quality: ${(fusion.fusionQuality * 100).toFixed(1)}%`);
```

### PWSA - Trajectory Prediction

```javascript
// Predict threat trajectory
const prediction = await client.predictTrajectory(
  'threat_001', // track_id
  [
    {
      timestamp: 1234567890,
      position: [10000.0, 20000.0, 5000.0],
      velocity: [800.0, -200.0, 0.0]
    },
    {
      timestamp: 1234567900,
      position: [18000.0, 18000.0, 5000.0],
      velocity: [800.0, -200.0, 0.0]
    }
  ],
  30, // prediction_horizon
  'kalman_filter' // model
);

console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
console.log(`Time to impact: ${prediction.timeToImpact}s`);
```

### Finance - Portfolio Optimization

```javascript
// Optimize portfolio
const portfolio = await client.optimizePortfolio(
  [
    {
      symbol: 'AAPL',
      expected_return: 0.12,
      volatility: 0.25,
      current_price: 150.0
    },
    {
      symbol: 'GOOGL',
      expected_return: 0.15,
      volatility: 0.30,
      current_price: 2800.0
    },
    {
      symbol: 'MSFT',
      expected_return: 0.13,
      volatility: 0.22,
      current_price: 380.0
    }
  ],
  {
    max_position_size: 0.5,
    min_position_size: 0.1,
    max_total_risk: 0.20
  },
  'maximize_sharpe' // objective
);

console.log('Optimal weights:');
portfolio.weights.forEach(w => {
  console.log(`  ${w.symbol}: ${(w.weight * 100).toFixed(1)}%`);
});
console.log(`Expected return: ${(portfolio.expectedReturn * 100).toFixed(1)}%`);
console.log(`Expected risk: ${(portfolio.expectedRisk * 100).toFixed(1)}%`);
console.log(`Sharpe ratio: ${portfolio.sharpeRatio.toFixed(2)}`);
```

### Finance - Risk Assessment

```javascript
// Assess portfolio risk
const risk = await client.assessRisk(
  'portfolio_001', // portfolio_id
  [
    { symbol: 'AAPL', quantity: 100, entry_price: 145.0, current_price: 150.0 },
    { symbol: 'TSLA', quantity: 50, entry_price: 200.0, current_price: 190.0 }
  ],
  ['var', 'cvar', 'max_drawdown', 'beta'] // risk_metrics
);

console.log(`VaR: ${(risk.var * 100).toFixed(2)}%`);
console.log(`CVaR: ${(risk.cvar * 100).toFixed(2)}%`);
console.log(`Max Drawdown: ${(risk.max_drawdown * 100).toFixed(2)}%`);
```

### LLM - Single Query

```javascript
// Query language model
const response = await client.queryLLM(
  'What is the capital of France?',
  null, // use default model
  0.1, // temperature
  50 // max_tokens
);

console.log(`Response: ${response.text}`);
console.log(`Model: ${response.modelUsed}`);
console.log(`Tokens: ${response.tokensUsed}`);
console.log(`Cost: $${response.costUsd.toFixed(4)}`);
```

### LLM - Multi-Model Consensus

```javascript
// Get consensus from multiple models
const consensus = await client.llmConsensus(
  'What is the capital of Australia?',
  [
    { name: 'gpt-4', weight: 1.0 },
    { name: 'gpt-3.5-turbo', weight: 0.8 },
    { name: 'claude-3', weight: 1.0 }
  ],
  'majority_vote', // strategy
  0.1, // temperature
  50 // max_tokens
);

console.log(`Consensus: ${consensus.consensusText}`);
console.log(`Confidence: ${(consensus.confidence * 100).toFixed(1)}%`);
console.log(`Agreement: ${(consensus.agreementRate * 100).toFixed(1)}%`);
console.log(`Total cost: $${consensus.totalCostUsd.toFixed(4)}`);
```

### Time Series - Forecasting

```javascript
// Forecast time series
const forecast = await client.forecastTimeSeries(
  'sales_data', // series_id
  [100, 105, 110, 108, 115, 120, 125, 130], // historical_data
  null, // timestamps (optional)
  5, // horizon
  'arima' // method
);

console.log(`Predictions: ${forecast.predictions}`);
console.log(`Confidence intervals: ${forecast.confidenceIntervals}`);
console.log(`Metrics: ${JSON.stringify(forecast.metrics)}`);
```

### Pixel Processing

```javascript
// Generate sample IR frame
const width = 640;
const height = 480;
const pixels = Array.from({ length: width * height }, () => Math.floor(Math.random() * 256));

// Process pixels
const result = await client.processPixels(
  12345, // frame_id
  width,
  height,
  pixels,
  {
    detect_hotspots: true,
    compute_entropy: true,
    apply_tda: true
  }
);

console.log(`Hotspots detected: ${result.hotspots.length}`);
console.log(`Entropy: ${result.entropy.toFixed(3)}`);
console.log(`TDA features: ${result.tdaFeatures.length}`);
```

## Error Handling

```javascript
const {
  PrismClient,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  ServerError,
} = require('@prism-ai/client');

const client = new PrismClient({ apiKey: 'your-key' });

try {
  const threat = await client.detectThreat(42, 1234567890, {});
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key');
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limited. Retry after ${error.retryAfter}s`);
  } else if (error instanceof ValidationError) {
    console.error(`Validation error: ${error.message}`);
  } else if (error instanceof ServerError) {
    console.error('Server error occurred');
  } else {
    console.error(`Unknown error: ${error.message}`);
  }
}
```

## Async/Await Pattern

```javascript
async function detectThreats() {
  const client = new PrismClient({ apiKey: 'your-key' });

  try {
    const threat = await client.detectThreat(42, Date.now(), {
      width: 640,
      height: 480,
      centroid_x: 320.0,
      centroid_y: 240.0,
      hotspot_count: 5
    });

    console.log(`Detected: ${threat.threatType}`);
  } catch (error) {
    console.error(`Error: ${error.message}`);
  }
}

detectThreats();
```

## Promise Pattern

```javascript
const client = new PrismClient({ apiKey: 'your-key' });

client.detectThreat(42, Date.now(), {
  width: 640,
  height: 480,
  centroid_x: 320.0,
  centroid_y: 240.0,
  hotspot_count: 5
})
.then(threat => {
  console.log(`Detected: ${threat.threatType}`);
})
.catch(error => {
  console.error(`Error: ${error.message}`);
});
```

## Configuration

### Custom Base URL

```javascript
const client = new PrismClient({
  apiKey: 'your-key',
  baseUrl: 'https://api.prism-ai.example.com'
});
```

### Custom Timeout

```javascript
const client = new PrismClient({
  apiKey: 'your-key',
  timeout: 60000 // 60 seconds
});
```

### Disable SSL Verification (development only)

```javascript
const client = new PrismClient({
  apiKey: 'your-key',
  verifySsl: false // NOT recommended for production
});
```

## API Reference

### Client Methods

**Health & Info:**
- `health()` - Check API health
- `info()` - Get API information

**PWSA:**
- `detectThreat()` - Detect threats from IR data
- `fuseSensors()` - Fuse multi-sensor data
- `predictTrajectory()` - Predict threat trajectory
- `prioritizeThreats()` - Prioritize multiple threats

**Finance:**
- `optimizePortfolio()` - Optimize portfolio allocation
- `assessRisk()` - Assess portfolio risk
- `backtestStrategy()` - Backtest trading strategy

**LLM:**
- `queryLLM()` - Query language model
- `llmConsensus()` - Multi-model consensus
- `listLLMModels()` - List available models

**Time Series:**
- `forecastTimeSeries()` - Forecast time series

**Pixel Processing:**
- `processPixels()` - Process pixel data

## TypeScript Support

This library includes TypeScript definitions. Use it with full type safety:

```typescript
import { PrismClient, ThreatDetection } from '@prism-ai/client';

const client = new PrismClient({ apiKey: 'your-key' });

const threat: ThreatDetection = await client.detectThreat(
  42,
  Date.now(),
  {
    width: 640,
    height: 480,
    centroid_x: 320.0,
    centroid_y: 240.0,
    hotspot_count: 5
  }
);
```

## Development

### Setup Development Environment

```bash
cd examples/javascript
npm install
```

### Run Tests

```bash
npm test
```

### Code Formatting

```bash
npm run format
```

### Linting

```bash
npm run lint
```

## License

MIT License

## Support

- Documentation: https://docs.prism-ai.example.com
- Issues: https://github.com/your-org/prism-ai/issues
- Email: support@prism-ai.example.com
