#!/bin/bash
# GraphQL API Test Script - Worker 8 Dual API Validation

set -e

API_URL="http://localhost:8080/graphql"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "PRISM-AI GraphQL API Test Suite"
echo "Worker 8 - Dual API Integration"
echo "======================================"
echo ""

# Check if server is running
echo "Checking if API server is running..."
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo -e "${RED}❌ API server not running!${NC}"
    echo "Start server with: cargo run --bin api_server"
    exit 1
fi
echo -e "${GREEN}✅ API server is running${NC}"
echo ""

# Test 1: Health Check Query
echo "Test 1: Health Check Query"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{"query":"query { health { status version uptimeSeconds } }"}')

if echo "$RESPONSE" | grep -q '"status"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 2: GPU Status Query
echo "Test 2: GPU Status Query"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{"query":"query { gpuStatus { available deviceCount utilizationPercent } }"}')

if echo "$RESPONSE" | grep -q '"available"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 3: Time Series Forecast Query
echo "Test 3: Time Series Forecast (ARIMA)"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "query ForecastTimeSeries($input: TimeSeriesForecastInput!) { forecastTimeSeries(input: $input) { predictions method horizon } }",
    "variables": {
      "input": {
        "historicalData": [100.0, 102.0, 101.0, 105.0, 108.0],
        "horizon": 5,
        "method": "ARIMA"
      }
    }
  }')

if echo "$RESPONSE" | grep -q '"predictions"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
    echo "Predictions: $(echo $RESPONSE | grep -o '"predictions":\[[^]]*\]')"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 4: Portfolio Optimization Query
echo "Test 4: Portfolio Optimization (Max Sharpe)"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "query OptimizePortfolio($input: PortfolioOptimizationInput!) { optimizePortfolio(input: $input) { weights { symbol weight } expectedReturn sharpeRatio } }",
    "variables": {
      "input": {
        "assets": [
          {"symbol": "AAPL", "expectedReturn": 0.12, "volatility": 0.20},
          {"symbol": "GOOGL", "expectedReturn": 0.15, "volatility": 0.25}
        ],
        "objective": "MaximizeSharpe"
      }
    }
  }')

if echo "$RESPONSE" | grep -q '"weights"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
    echo "Sharpe Ratio: $(echo $RESPONSE | grep -o '"sharpeRatio":[0-9.]*')"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 5: Robot Motion Planning Query
echo "Test 5: Robot Motion Planning"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "query PlanRobotMotion($input: MotionPlanInput!) { planRobotMotion(input: $input) { waypoints { time } totalTime isCollisionFree } }",
    "variables": {
      "input": {
        "robotId": "robot-1",
        "start": {"x": 0.0, "y": 0.0, "z": 0.0},
        "goal": {"x": 5.0, "y": 3.0, "z": 0.0}
      }
    }
  }')

if echo "$RESPONSE" | grep -q '"waypoints"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
    echo "Total Time: $(echo $RESPONSE | grep -o '"totalTime":[0-9.]*')"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 6: Performance Metrics Query
echo "Test 6: Performance Metrics"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{"query":"query { performanceMetrics { endpoint avgResponseTimeMs requestsPerSecond } }"}')

if echo "$RESPONSE" | grep -q '"endpoint"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 7: Dashboard Combined Query
echo "Test 7: Dashboard Combined Query (Multiple Resources)"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{"query":"query Dashboard { health { status } gpuStatus { available } performanceMetrics { endpoint } }"}')

if echo "$RESPONSE" | grep -q '"health"' && echo "$RESPONSE" | grep -q '"gpuStatus"' && echo "$RESPONSE" | grep -q '"performanceMetrics"'; then
    echo -e "${GREEN}✅ PASSED - Single query returned 3 resources!${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 8: Submit Forecast Mutation
echo "Test 8: Submit Forecast Mutation"
RESPONSE=$(curl -s -X POST $API_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "mutation SubmitForecast($input: TimeSeriesForecastInput!) { submitForecast(input: $input) { predictions method } }",
    "variables": {
      "input": {
        "historicalData": [100.0, 102.0, 104.0],
        "horizon": 3,
        "method": "LSTM"
      }
    }
  }')

if echo "$RESPONSE" | grep -q '"predictions"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 9: GraphQL Schema Introspection
echo "Test 9: GraphQL Schema Introspection"
RESPONSE=$(curl -s http://localhost:8080/graphql/schema)

if echo "$RESPONSE" | grep -q '"sdl"' && echo "$RESPONSE" | grep -q '"endpoints"'; then
    echo -e "${GREEN}✅ PASSED${NC}"
    echo "Available endpoints:"
    echo "$RESPONSE" | grep -o '"playground":"[^"]*"'
    echo "$RESPONSE" | grep -o '"endpoint":"[^"]*"'
    echo "$RESPONSE" | grep -o '"schema":"[^"]*"'
else
    echo -e "${RED}❌ FAILED${NC}"
    echo "Response: $RESPONSE"
fi
echo ""

# Test 10: GraphQL Playground HTML
echo "Test 10: GraphQL Playground UI"
RESPONSE=$(curl -s -H "Accept: text/html" http://localhost:8080/graphql)

if echo "$RESPONSE" | grep -q "GraphQL Playground"; then
    echo -e "${GREEN}✅ PASSED${NC}"
    echo "GraphQL Playground available at: http://localhost:8080/graphql"
else
    echo -e "${RED}❌ FAILED${NC}"
fi
echo ""

echo "======================================"
echo "Test Suite Complete"
echo "======================================"
echo ""
echo "Next Steps:"
echo "1. Visit http://localhost:8080/graphql for interactive playground"
echo "2. Review docs/DUAL_API_GUIDE.md for usage examples"
echo "3. Compare with REST API: ./run_integration_tests.sh"
echo ""
echo -e "${YELLOW}Note: Some tests may fail if Worker 1/2 GPU kernel dependencies are not resolved${NC}"
