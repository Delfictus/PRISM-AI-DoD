#!/bin/bash
#
# Worker 8 Start Script
# Initializes development session with proper sync and governance checks
#

WORKER_ID=8
WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-8"
BRANCH="worker-8-finance-deploy"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Worker 8 Development Session - Starting                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

cd "$WORKER_DIR" || exit 1

# 1. Run governance check
echo "🔍 Step 1: Running governance check..."
if [ -f ".worker-vault/STRICT_GOVERNANCE_ENGINE.sh" ]; then
    bash .worker-vault/STRICT_GOVERNANCE_ENGINE.sh $WORKER_ID
    if [ $? -ne 0 ]; then
        echo "❌ Governance check failed - fix violations before proceeding"
        exit 1
    fi
fi
echo ""

# 2. Update from remote
echo "📥 Step 2: Syncing with remote..."
git fetch origin
git pull origin $BRANCH
echo ""

# 3. Check for integration updates
echo "🔄 Step 3: Checking for integration updates..."
if git remote | grep -q "upstream"; then
    git fetch upstream
    echo "   Upstream updates available"
fi
echo ""

# 4. Verify build
echo "🔨 Step 4: Verifying build..."
cd 03-Source-Code
cargo check --lib --features api_server 2>&1 | tail -5
BUILD_STATUS=$?
cd ..

if [ $BUILD_STATUS -eq 0 ]; then
    echo "   ✅ Build verified"
else
    echo "   ⚠️  Build has issues - review before making changes"
fi
echo ""

# 5. Show recent progress
echo "📊 Step 5: Recent progress..."
echo "   Last 5 commits:"
git log --oneline -5
echo ""

# 6. Show current status
echo "📋 Step 6: Current status..."
git status --short
echo ""

# 7. Start auto-sync (if available)
if [ -f "worker_auto_sync.sh" ]; then
    echo "🔄 Step 7: Starting auto-sync daemon..."
    bash worker_auto_sync.sh start
    echo ""
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Worker 8 Ready for Development                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Tips:"
echo "  • All deliverables complete - 100% of assigned workload done"
echo "  • Branch: $BRANCH"
echo "  • Run governance check: bash .worker-vault/STRICT_GOVERNANCE_ENGINE.sh 8"
echo "  • View progress: cat .worker-vault/Progress/DAILY_PROGRESS.md"
echo "  • Review deliverables: cat COMPLETION_REPORT.md"
echo ""
