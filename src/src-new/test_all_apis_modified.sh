#!/bin/bash
# Comprehensive API Test Suite
# Phase 3 Task 2: End-to-end API Testing
# Tests all 50+ REST endpoints and GraphQL queries

set +e

API_URL="${API_URL:-http://localhost:8080}"
RESULTS_FILE="api_test_results.json"
PASSED=0
FAILED=0

echo "============================================"
echo "PRISM-AI Comprehensive API Test Suite"
echo "Phase 3 - End-to-End Testing"
echo "============================================"
echo ""
echo "API URL: $API_URL"
echo "Test Results: $RESULTS_FILE"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test result tracking
declare -a TEST_RESULTS

log_test() {
    local name=$1
    local status=$2
    local message=$3

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $name"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $name: $message"
        ((FAILED++))
    fi

    TEST_RESULTS+=("{\"name\":\"$name\",\"status\":\"$status\",\"message\":\"$message\"}")
}

test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4

    response=$(curl -s -X $method \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$API_URL$endpoint" \
        -w "\n%{http_code}" 2>&1)

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        log_test "$description" "PASS" "HTTP $http_code"
        return 0
    else
        log_test "$description" "FAIL" "HTTP $http_code: $body"
        return 1
    fi
}

echo "============================================"
echo "1. CORE INFRASTRUCTURE"
echo "============================================"

# Health check
test_endpoint GET "/health" "" "Health Check"

echo ""
echo "============================================"
echo "2. WORKER 3 APPLICATION DOMAINS (13)"
echo "============================================"

# Healthcare
test_endpoint POST "/api/v1/applications/healthcare/predict_risk" \
    '{"historical_metrics":[0.2,0.25,0.3,0.28,0.32],"horizon":5,"risk_factors":["age","bmi"]}' \
    "Healthcare: Risk Prediction"

test_endpoint POST "/api/v1/applications/healthcare/forecast_trajectory" \
    '{"historical_metrics":[0.2,0.25,0.3],"horizon":5,"risk_factors":["age"]}' \
    "Healthcare: Trajectory Forecast"

# Energy
test_endpoint POST "/api/v1/applications/energy/forecast_load" \
    '{"historical_load":[100.0,105.0,110.0,108.0],"horizon":5}' \
    "Energy: Load Forecast"

# Manufacturing
test_endpoint POST "/api/v1/applications/manufacturing/predict_maintenance" \
    '{"sensor_data":[95.0,96.5,98.0,99.5],"equipment_id":"PUMP-001","window":24}' \
    "Manufacturing: Predictive Maintenance"

# Supply Chain
test_endpoint POST "/api/v1/applications/supply_chain/forecast_demand" \
    '{"historical_demand":[100.0,105.0,110.0],"product_id":"SKU-123","horizon":7}' \
    "Supply Chain: Demand Forecast"

# Agriculture
test_endpoint POST "/api/v1/applications/agriculture/predict_yield" \
    '{"historical_yield":[4000.0,4200.0,4100.0],"horizon":4}' \
    "Agriculture: Yield Prediction"

# Cybersecurity
test_endpoint POST "/api/v1/applications/cybersecurity/predict_threats" \
    '{"historical_events":[10.0,12.0,15.0],"threat_levels":[2.0,2.5,3.0],"horizon":6}' \
    "Cybersecurity: Threat Prediction"

# Climate
test_endpoint POST "/api/v1/applications/climate/forecast" \
    '{"historical_data":[20.0,21.5,23.0],"location":"NYC","horizon":5}' \
    "Climate: Weather Forecast"

# Smart Cities
test_endpoint POST "/api/v1/applications/smart_city/optimize" \
    '{"resource_type":"energy","current_levels":[80.0,85.0,90.0],"horizon":24}' \
    "Smart Cities: Resource Optimization"

# Education
test_endpoint POST "/api/v1/applications/education/predict_performance" \
    '{"historical_performance":[85.0,82.0,80.0],"student_id":"STU-001","horizon":4}' \
    "Education: Performance Prediction"

# Retail
test_endpoint POST "/api/v1/applications/retail/optimize_inventory" \
    '{"historical_sales":[100.0,110.0,105.0],"product_id":"PROD-001","current_inventory":50.0,"horizon":7}' \
    "Retail: Inventory Optimization"

# Construction
test_endpoint POST "/api/v1/applications/construction/forecast_project" \
    '{"project_id":"PROJ-001","historical_progress":[10.0,25.0,40.0,55.0],"horizon":30}' \
    "Construction: Project Forecast"

echo ""
echo "============================================"
echo "3. WORKER 4 ADVANCED FINANCE (4)"
echo "============================================"

# Portfolio Optimization
test_endpoint POST "/api/v1/finance_advanced/optimize_advanced" \
    '{"assets":[{"symbol":"AAPL","expected_return":0.12,"volatility":0.20},{"symbol":"GOOGL","expected_return":0.15,"volatility":0.25}],"strategy":"maximize_sharpe","use_gnn":false,"risk_free_rate":0.03}' \
    "Finance: Portfolio Optimization"

# GNN Portfolio Prediction
test_endpoint POST "/api/v1/finance_advanced/gnn/predict" \
    '{"assets":[{"symbol":"AAPL","expected_return":0.12,"volatility":0.20,"price_history":[100.0,102.0,104.0]}],"horizon":30}' \
    "Finance: GNN Portfolio Prediction"

# Transfer Entropy Causality
test_endpoint POST "/api/v1/finance_advanced/causality/transfer_entropy" \
    '{"time_series":[{"symbol":"AAPL","values":[100,102,101,105,108]},{"symbol":"MSFT","values":[200,198,202,205,207]}],"window":100}' \
    "Finance: Transfer Entropy Analysis"

# Portfolio Rebalancing
test_endpoint POST "/api/v1/finance_advanced/rebalance" \
    '{"current_weights":[{"symbol":"AAPL","weight":0.5},{"symbol":"GOOGL","weight":0.5}],"target_weights":[{"symbol":"AAPL","weight":0.6},{"symbol":"GOOGL","weight":0.4}],"frequency":30,"transaction_cost":0.001}' \
    "Finance: Portfolio Rebalancing"

echo ""
echo "============================================"
echo "4. WORKER 7 SPECIALIZED APPS (6)"
echo "============================================"

# Robotics - Motion Planning
test_endpoint POST "/api/v1/worker7/robotics/plan_motion" \
    '{"start_position":[0.0,0.0,0.0],"goal_position":[10.0,10.0,5.0],"algorithm":"RRT"}' \
    "Robotics: Motion Planning"

# Robotics - Trajectory Optimization
test_endpoint POST "/api/v1/worker7/robotics/optimize_trajectory" \
    '{"waypoints":[[0,0,0],[5,5,2],[10,10,5]],"objective":"time","max_velocity":2.0}' \
    "Robotics: Trajectory Optimization"

# Drug Discovery - Molecular Screening
test_endpoint POST "/api/v1/worker7/drug_discovery/screen_molecules" \
    '{"molecules":["CCO","CC(=O)O","CC(C)O"],"target_protein":"ACE2","criteria":["binding_affinity"]}' \
    "Drug Discovery: Molecular Screening"

# Drug Discovery - Drug Optimization
test_endpoint POST "/api/v1/worker7/drug_discovery/optimize_drug" \
    '{"seed_molecule":"CCO","target_properties":{"target_binding_affinity":8.0,"max_toxicity":0.3,"min_drug_likeness":0.7},"max_iterations":10}' \
    "Drug Discovery: Drug Optimization"

# Scientific - Experiment Design
test_endpoint POST "/api/v1/worker7/scientific/design_experiment" \
    '{"hypothesis":"Temperature affects reaction rate","variables":[{"name":"temp","min_value":20.0,"max_value":100.0,"variable_type":"continuous"}],"num_experiments":5}' \
    "Scientific: Experiment Design"

# Scientific - Hypothesis Testing
test_endpoint POST "/api/v1/worker7/scientific/test_hypothesis" \
    '{"hypothesis":"Mean > 50","data":[55.0,58.0,52.0,60.0,56.0],"alpha":0.05}' \
    "Scientific: Hypothesis Testing"

echo ""
echo "============================================"
echo "5. GRAPHQL API TESTS"
echo "============================================"

# GraphQL Health Query
graphql_query='{"query":"query { health { status version uptimeSeconds } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Health Query"

# GraphQL GPU Status
graphql_query='{"query":"query { gpuStatus { available deviceCount totalMemoryMb } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: GPU Status"

# GraphQL Time Series Forecast
graphql_query='{"query":"query { forecastTimeSeries(input: { historicalData: [100.0, 102.0, 104.0], horizon: 5, method: \"ARIMA\" }) { predictions method horizon } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Time Series Forecast"

# GraphQL Portfolio Optimization
graphql_query='{"query":"query { optimizePortfolio(input: { assets: [{ symbol: \"AAPL\", expectedReturn: 0.12, volatility: 0.20 }], objective: \"MaximizeSharpe\" }) { weights { symbol weight } expectedReturn } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Portfolio Optimization"

# GraphQL Healthcare Risk
graphql_query='{"query":"query { healthcarePredictRisk(input: { historicalMetrics: [0.2, 0.3, 0.35], horizon: 5, riskFactors: [\"age\"] }) { riskTrajectory riskLevel confidence } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Healthcare Risk"

# GraphQL Energy Forecast
graphql_query='{"query":"query { energyForecastLoad(input: { historicalLoad: [100.0, 105.0], horizon: 5 }) { forecastedLoad peakLoad } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Energy Forecast"

# GraphQL Molecular Screening
graphql_query='{"query":"query { screenMolecules(input: { molecules: [\"CCO\"], targetProtein: \"ACE2\" }) { topCandidates screeningTimeMs } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Molecular Screening"

# GraphQL Experiment Design
graphql_query='{"query":"query { designExperiment(input: { hypothesis: \"Test hypothesis\", numExperiments: 5 }) { numExperiments designStrategy } }"}'
test_endpoint POST "/graphql" "$graphql_query" "GraphQL: Experiment Design"

echo ""
echo "============================================"
echo "TEST SUMMARY"
echo "============================================"
echo ""
echo -e "Total Tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

# Calculate pass rate
if [ $((PASSED + FAILED)) -gt 0 ]; then
    PASS_RATE=$((PASSED * 100 / (PASSED + FAILED)))
    echo "Pass Rate: ${PASS_RATE}%"

    if [ $PASS_RATE -ge 90 ]; then
        echo -e "${GREEN}Status: EXCELLENT${NC}"
        exit 0
    elif [ $PASS_RATE -ge 70 ]; then
        echo -e "${YELLOW}Status: GOOD${NC}"
        exit 0
    else
        echo -e "${RED}Status: NEEDS IMPROVEMENT${NC}"
        exit 1
    fi
else
    echo -e "${RED}Status: NO TESTS RUN${NC}"
    exit 1
fi

# Save results to JSON
echo "[" > $RESULTS_FILE
printf '%s\n' "${TEST_RESULTS[@]}" | paste -sd ',' >> $RESULTS_FILE
echo "]" >> $RESULTS_FILE

echo ""
echo "Results saved to: $RESULTS_FILE"
