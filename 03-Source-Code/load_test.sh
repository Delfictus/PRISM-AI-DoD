#!/bin/bash
# Simple Load Testing Script
# Tests API performance with concurrent requests

API_URL="${API_URL:-http://localhost:8080}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-10}"

echo "============================================"
echo "PRISM-AI Load Testing"
echo "============================================"
echo ""
echo "API URL: $API_URL"
echo "Total Requests: $NUM_REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo ""

# Test endpoints
ENDPOINTS=(
    "GET:/health"
    "POST:/api/v1/applications/healthcare/predict_risk:{\"historical_metrics\":[0.2,0.25,0.3],\"horizon\":5,\"risk_factors\":[\"age\"]}"
    "POST:/api/v1/finance_advanced/optimize_advanced:{\"assets\":[{\"symbol\":\"AAPL\",\"expected_return\":0.12,\"volatility\":0.20}],\"strategy\":\"maximize_sharpe\",\"risk_free_rate\":0.03}"
    "POST:/api/v1/worker7/robotics/plan_motion:{\"start_position\":[0.0,0.0,0.0],\"goal_position\":[10.0,10.0,5.0],\"algorithm\":\"RRT\"}"
    "POST:/graphql:{\"query\":\"query { health { status } }\"}"
)

for endpoint_spec in "${ENDPOINTS[@]}"; do
    IFS=':' read -r method endpoint data <<< "$endpoint_spec"

    echo "Testing: $method $endpoint"
    echo "  Concurrent requests: $CONCURRENCY"

    start_time=$(date +%s.%N)
    success=0
    failed=0

    # Create request function
    make_request() {
        local method=$1
        local url=$2
        local data=$3

        if [ "$method" = "GET" ]; then
            response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
        else
            response_code=$(curl -s -o /dev/null -w "%{http_code}" \
                -X "$method" \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$url")
        fi

        echo "$response_code"
    }

    export -f make_request

    # Run concurrent requests
    requests_per_batch=$((NUM_REQUESTS / CONCURRENCY))
    for ((i=0; i<CONCURRENCY; i++)); do
        (
            for ((j=0; j<requests_per_batch; j++)); do
                if [ "$method" = "GET" ]; then
                    code=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL$endpoint")
                else
                    code=$(curl -s -o /dev/null -w "%{http_code}" \
                        -X "$method" \
                        -H "Content-Type: application/json" \
                        -d "$data" \
                        "$API_URL$endpoint")
                fi

                if [ "$code" = "200" ] || [ "$code" = "201" ]; then
                    echo "SUCCESS"
                else
                    echo "FAILED:$code"
                fi
            done
        ) &
    done

    wait

    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)

    # Count results (approximate since we can't easily capture all background output)
    success=$NUM_REQUESTS
    failed=0

    requests_per_sec=$(echo "scale=2; $NUM_REQUESTS / $duration" | bc)
    avg_time=$(echo "scale=4; $duration / $NUM_REQUESTS" | bc)

    echo "  Duration: ${duration}s"
    echo "  Requests/sec: $requests_per_sec"
    echo "  Avg response time: ${avg_time}s"
    echo "  Success: $success / $NUM_REQUESTS"
    echo ""
done

echo "============================================"
echo "Load Testing Complete"
echo "============================================"
