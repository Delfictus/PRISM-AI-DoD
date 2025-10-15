#!/bin/bash
#
# Integration Test Runner
# Starts API server, runs tests, and cleans up
#

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  PRISM-AI API Integration Test Runner                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Configuration
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-0.0.0.0}
API_KEY=${API_KEY:-test-key}
ADMIN_API_KEY=${ADMIN_API_KEY:-admin-key}
READ_ONLY_API_KEY=${READ_ONLY_API_KEY:-readonly-key}

export API_PORT
export API_HOST
export API_KEY
export ADMIN_API_KEY
export READ_ONLY_API_KEY

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  API URL: http://${API_HOST}:${API_PORT}"
echo "  API Key: ${API_KEY}"
echo ""

# Check if server is already running
if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
    echo "⚠️  API server already running on port ${API_PORT}"
    echo "   Using existing server for tests"
    echo ""
    SERVER_STARTED=false
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Starting API Server"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Start API server in background
    cd 03-Source-Code
    cargo run --bin api_server --features api_server > ../api_server.log 2>&1 &
    SERVER_PID=$!
    cd ..

    echo "  Server PID: ${SERVER_PID}"
    echo "  Log file: api_server.log"
    echo ""

    # Wait for server to be ready
    echo "⏳ Waiting for server to start..."
    MAX_WAIT=30
    WAITED=0

    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
            echo "✅ Server is ready!"
            echo ""
            break
        fi
        sleep 1
        WAITED=$((WAITED + 1))
        echo -n "."
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""
        echo "❌ Server failed to start within ${MAX_WAIT} seconds"
        echo "   Check api_server.log for errors"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    SERVER_STARTED=true
fi

# Function to cleanup
cleanup() {
    if [ "$SERVER_STARTED" = true ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🧹 Cleaning up"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Stopping API server (PID: ${SERVER_PID})"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "✅ Server stopped"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Run tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Running Integration Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd tests

# Parse command line arguments
TEST_FILTER="${1:-}"
NOCAPTURE="${2:-}"

if [ -n "$TEST_FILTER" ]; then
    echo "Running tests matching: $TEST_FILTER"
    if [ "$NOCAPTURE" = "--nocapture" ]; then
        cargo test --test integration "$TEST_FILTER" --features api_server -- --nocapture
    else
        cargo test --test integration "$TEST_FILTER" --features api_server
    fi
else
    echo "Running all integration tests"
    if [ "$NOCAPTURE" = "--nocapture" ]; then
        cargo test --test integration --features api_server -- --nocapture
    else
        cargo test --test integration --features api_server
    fi
fi

TEST_EXIT_CODE=$?

cd ..

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  ✅ All Tests Passed                                      ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
else
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  ❌ Some Tests Failed                                     ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Check test output above for details"
    echo "Server logs: api_server.log"
fi

exit $TEST_EXIT_CODE
