#!/bin/bash
#
# Dependency Checker for Workers
# Usage: ./check_dependencies.sh <worker-number>
#
# Checks if required deliverables are available for a given worker
#

WORKER_ID=$1
DELIVERABLES_LOG=".worker-deliverables.log"
MAIN_REPO="/home/diddy/Desktop/PRISM-AI-DoD"

if [ -z "$WORKER_ID" ]; then
    echo "Usage: ./check_dependencies.sh <worker-number>"
    echo "Example: ./check_dependencies.sh 3"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Dependency Check: Worker $WORKER_ID                            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Fetch latest deliverables
cd "$MAIN_REPO"
git fetch origin deliverables 2>/dev/null || true

# Check deliverables log
if [ ! -f "$DELIVERABLES_LOG" ]; then
    echo "⚠️  Deliverables log not found"
    echo "   Creating from remote..."
    git show origin/deliverables:.worker-deliverables.log > "$DELIVERABLES_LOG" 2>/dev/null || {
        echo "❌ Could not fetch deliverables log"
        exit 1
    }
fi

echo "📋 Checking dependencies for Worker $WORKER_ID..."
echo ""

case $WORKER_ID in
    1)
        echo "🔍 Worker 1 Dependencies:"
        echo ""
        echo "Required from Worker 2:"
        echo "  • Base GPU kernels (Week 1)"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 2: Base GPU kernels.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
        else
            echo "    ❌ NOT READY"
        fi

        echo ""
        echo "  • Time series kernels (Week 2)"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 2: Time series.*kernels.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
            echo "    → Can proceed with time series implementation"
        else
            echo "    ❌ NOT READY"
            echo "    → Wait for Worker 2 Week 2 completion"
        fi
        ;;

    2)
        echo "🔍 Worker 2 Dependencies:"
        echo ""
        echo "  ✅ Worker 2 has no dependencies (GPU infrastructure is foundation)"
        echo "  → Can proceed with all GPU kernel development"
        ;;

    3)
        echo "🔍 Worker 3 Dependencies:"
        echo ""
        echo "Required from Worker 2:"
        echo "  • Pixel processing kernels (Week 3)"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 2: Pixel.*kernels.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
            echo ""
            echo "📥 To integrate:"
            echo "   cd /home/diddy/Desktop/PRISM-Worker-3"
            echo "   git fetch origin deliverables"
            echo "   git merge origin/deliverables"
            echo "   cargo check --features cuda"
        else
            echo "    ❌ NOT READY"
            echo "    → Worker 2 must complete pixel kernels first"
            echo ""
            echo "⏳ While waiting, Worker 3 can:"
            echo "   • Work on PWSA frame-level processing"
            echo "   • Implement Finance portfolio optimization"
            echo "   • Prepare pixel integration code"
        fi
        ;;

    4)
        echo "🔍 Worker 4 Dependencies:"
        echo ""
        echo "Required from Worker 1:"
        echo "  • Core AI infrastructure"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 1: Core AI.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
        else
            echo "    ❌ NOT READY"
        fi

        echo ""
        echo "Required from Worker 2:"
        echo "  • Base GPU kernels"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 2: Base GPU.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
        else
            echo "    ❌ NOT READY"
        fi
        ;;

    5)
        echo "🔍 Worker 5 Dependencies:"
        echo ""
        echo "Required from Worker 1:"
        echo "  • Time series forecasting module (Week 3)"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 1: Time series.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
            echo ""
            echo "📥 To integrate:"
            echo "   cd /home/diddy/Desktop/PRISM-Worker-5"
            echo "   git fetch origin deliverables"
            echo "   git merge origin/deliverables"
            echo "   cargo test --lib time_series"
            echo ""
            echo "🚀 Can now implement:"
            echo "   • LLM cost forecasting (use time series)"
            echo "   • Proactive model selection"
        else
            echo "    ❌ NOT READY - BLOCKING"
            echo "    → Worker 1 must complete time series module (Week 3)"
            echo ""
            echo "⏳ While waiting, Worker 5 can:"
            echo "   • Implement replica exchange (no dependency)"
            echo "   • Advanced energy functions (no dependency)"
            echo "   • Bayesian learning (no dependency)"
            echo ""
            echo "📊 Status check:"
            git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -i "worker 1.*time series" || echo "   No time series updates yet"
        fi
        ;;

    6)
        echo "🔍 Worker 6 Dependencies:"
        echo ""
        echo "Required from Worker 2:"
        echo "  • Base GPU kernels"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 2: Base GPU.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
            echo "    → Can proceed with advanced LLM features"
        else
            echo "    ❌ NOT READY"
        fi
        ;;

    7)
        echo "🔍 Worker 7 Dependencies:"
        echo ""
        echo "Required from Worker 1:"
        echo "  • Time series forecasting module (Week 3)"

        if git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 1: Time series.*AVAILABLE"; then
            echo "    ✅ AVAILABLE"
            echo ""
            echo "📥 To integrate:"
            echo "   cd /home/diddy/Desktop/PRISM-Worker-7"
            echo "   git fetch origin deliverables"
            echo "   git merge origin/deliverables"
            echo ""
            echo "🚀 Can now implement:"
            echo "   • Robotics trajectory forecasting"
            echo "   • Environment dynamics prediction"
            echo "   • Multi-agent motion planning"
        else
            echo "    ❌ NOT READY - BLOCKING"
            echo "    → Worker 1 must complete time series module (Week 3)"
            echo ""
            echo "⏳ While waiting, Worker 7 can:"
            echo "   • Implement drug discovery features (no dependency)"
            echo "   • Basic robotics motion planning (no forecasting)"
            echo "   • Prepare trajectory integration code"
        fi
        ;;

    8)
        echo "🔍 Worker 8 Dependencies:"
        echo ""
        echo "Required: Core features from Workers 1-7"
        echo ""

        # Check critical deliverables
        W1_READY=false
        W2_READY=false
        W3_READY=false
        W4_READY=false

        git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 1.*AVAILABLE" && W1_READY=true
        git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 2.*AVAILABLE" && W2_READY=true
        git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 3.*AVAILABLE" && W3_READY=true
        git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep -q "Worker 4.*AVAILABLE" && W4_READY=true

        [ "$W1_READY" = true ] && echo "  ✅ Worker 1 (AI core)" || echo "  ⏳ Worker 1 (AI core)"
        [ "$W2_READY" = true ] && echo "  ✅ Worker 2 (GPU)" || echo "  ⏳ Worker 2 (GPU)"
        [ "$W3_READY" = true ] && echo "  ✅ Worker 3 (PWSA/Finance)" || echo "  ⏳ Worker 3 (PWSA/Finance)"
        [ "$W4_READY" = true ] && echo "  ✅ Worker 4 (Telecom/Robotics)" || echo "  ⏳ Worker 4 (Telecom/Robotics)"

        echo ""
        if [ "$W1_READY" = true ] && [ "$W2_READY" = true ]; then
            echo "✅ Core infrastructure ready - can begin API development"
        else
            echo "⏳ Waiting for core infrastructure (Week 5-6)"
        fi
        ;;

    *)
        echo "❌ Unknown worker: $WORKER_ID"
        echo "   Valid workers: 1-8"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Overall Integration Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show recent deliverables
echo "Recent deliverables (last 5):"
git show origin/deliverables:.worker-deliverables.log 2>/dev/null | grep "^- ✅" | tail -5 | sed 's/^/  /'

echo ""
echo "🔄 To pull latest deliverables:"
echo "   git fetch origin deliverables"
echo "   git merge origin/deliverables"
echo ""
echo "📝 Full status: cat .worker-deliverables.log"
echo "📋 Manifest: cat DELIVERABLES.md"
echo ""
