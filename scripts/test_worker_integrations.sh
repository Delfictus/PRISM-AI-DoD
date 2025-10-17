#!/bin/bash
#
# Quick Worker Integration Test Runner
# Tests Workers 1, 3, 7 integrations via library calls (no server needed)
#

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  PRISM-AI Worker Integration Tests (1, 3, 7)             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

cd 03-Source-Code

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing Worker Integrations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run unit/integration tests for worker modules
echo "📦 Worker 1: Time Series Forecasting..."
cargo test --lib time_series -- --nocapture 2>&1 | grep -E "(test |passed|FAILED)" || true
echo ""

echo "📦 Worker 3: Finance Portfolio Optimization..."
cargo test --lib finance -- --nocapture 2>&1 | grep -E "(test |passed|FAILED)" || true
echo ""

echo "📦 Worker 7: Robotics Motion Planning..."
cargo test --lib applications::robotics -- --nocapture 2>&1 | grep -E "(test |passed|FAILED)" || true
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Integration Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run comprehensive integration tests
cargo test --lib integration_tests --no-fail-fast

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ✅ Worker Integration Tests Complete                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
