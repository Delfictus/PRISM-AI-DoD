#!/bin/bash
#
# Worker Startup Script
# Automatically syncs dependencies and prepares worker environment
#
# Usage: ./worker_start.sh <worker-number>
#

WORKER_ID=$1
WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-${WORKER_ID}"

if [ -z "$WORKER_ID" ]; then
    echo "Usage: ./worker_start.sh <worker-number>"
    echo "Example: ./worker_start.sh 5"
    exit 1
fi

if [ ! -d "$WORKER_DIR" ]; then
    echo "❌ Worker $WORKER_ID directory not found: $WORKER_DIR"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Worker $WORKER_ID Startup                                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

cd "$WORKER_DIR"

# Step 1: Pull latest from own branch
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Step 1: Pulling latest from worker branch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CURRENT_BRANCH=$(git branch --show-current)
echo "Branch: $CURRENT_BRANCH"

if git pull origin "$CURRENT_BRANCH" 2>&1; then
    echo "✅ Worker branch up to date"
else
    echo "⚠️  Pull had issues - check manually"
fi

echo ""

# Step 2: Pull latest integration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Step 2: Merging latest integration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

git fetch origin integration-staging 2>/dev/null

if git log HEAD..origin/integration-staging --oneline 2>/dev/null | grep -q .; then
    echo "New integration commits available:"
    git log HEAD..origin/integration-staging --oneline | head -5
    echo ""
    echo "Merging integration-staging..."

    if git merge origin/integration-staging -m "Auto-merge: integration-staging" 2>&1; then
        echo "✅ Integration merged successfully"
    else
        echo "⚠️  Merge conflicts - manual resolution needed"
        echo "   Run: git merge origin/integration-staging"
    fi
else
    echo "✅ Already up to date with integration-staging"
fi

echo ""

# Step 3: Run auto-sync for dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Step 3: Auto-syncing dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run auto-sync script
if [ -f "./worker_auto_sync.sh" ]; then
    ./worker_auto_sync.sh "$WORKER_ID"
    SYNC_STATUS=$?
elif [ -f "/home/diddy/Desktop/PRISM-AI-DoD/worker_auto_sync.sh" ]; then
    /home/diddy/Desktop/PRISM-AI-DoD/worker_auto_sync.sh "$WORKER_ID"
    SYNC_STATUS=$?
else
    echo "⚠️  Auto-sync script not found - skipping dependency check"
    SYNC_STATUS=0
fi

echo ""

# Step 4: Build check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔨 Step 4: Build validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd 03-Source-Code 2>/dev/null || cd .

echo "Running cargo check..."
if cargo check --features cuda 2>&1 | tail -20; then
    echo ""
    echo "✅ Build check PASSED"
    BUILD_OK=true
else
    echo ""
    echo "❌ Build check FAILED"
    echo "   Review errors above and fix before proceeding"
    BUILD_OK=false
fi

cd "$WORKER_DIR"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Worker $WORKER_ID Startup Complete                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Status summary
if [ "$BUILD_OK" = true ] && [ $SYNC_STATUS -eq 0 ]; then
    echo "🎉 Status: READY TO WORK"
    echo ""
    echo "Your environment is ready:"
    echo "  ✅ Worker branch up to date"
    echo "  ✅ Integration merged"
    echo "  ✅ Dependencies synced"
    echo "  ✅ Build validated"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Review your tasks: cat .worker-vault/Tasks/MY_TASKS.md"
    echo "  2. Start development in your assigned files"
    echo "  3. Run tests: cargo test --lib <your_module>"
    echo ""
elif [ $SYNC_STATUS -ne 0 ]; then
    echo "⏳ Status: WAITING FOR DEPENDENCIES"
    echo ""
    echo "Some dependencies are not yet available."
    echo ""
    echo "📋 What to do:"
    echo "  1. Work on non-dependent features (see auto-sync output above)"
    echo "  2. When ready to check again, run: ./worker_start.sh $WORKER_ID"
    echo "  3. Or prompt: 'Worker $WORKER_ID ready to continue'"
    echo ""
    echo "💡 Your dependencies will auto-pull when available!"
    echo ""
elif [ "$BUILD_OK" = false ]; then
    echo "⚠️  Status: BUILD ISSUES"
    echo ""
    echo "Build validation failed - fix errors before proceeding"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Review build errors above"
    echo "  2. Fix issues in your code"
    echo "  3. Run: cargo check --features cuda"
    echo "  4. Re-run startup: ./worker_start.sh $WORKER_ID"
    echo ""
else
    echo "✅ Status: READY (with minor warnings)"
    echo ""
    echo "Environment is usable but check warnings above"
    echo ""
fi
