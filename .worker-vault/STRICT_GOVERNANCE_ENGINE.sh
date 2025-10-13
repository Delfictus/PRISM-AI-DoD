#!/bin/bash
#
# STRICT GOVERNANCE ENGINE
# Enforces development rules and integration protocol for all workers
#
# This engine runs BEFORE any worker can proceed with work
# Violations result in immediate blocking until resolved
#

WORKER_ID=$1
WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-${WORKER_ID}"
GOVERNANCE_LOG="/home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/Enforcement/governance.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [ -z "$WORKER_ID" ]; then
    echo "❌ GOVERNANCE VIOLATION: No worker ID specified"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  STRICT GOVERNANCE ENGINE - Worker $WORKER_ID                  ║"
echo "║  Enforcement Level: MAXIMUM                               ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

cd "$WORKER_DIR" 2>/dev/null || {
    echo "❌ CRITICAL: Worker directory not found"
    exit 1
}

VIOLATIONS=0
WARNINGS=0

# ============================================================================
# RULE 1: FILE OWNERSHIP - Workers can ONLY edit their assigned files
# ============================================================================
echo "🔍 Rule 1: Checking File Ownership Compliance..."

# Get list of modified files
MODIFIED_FILES=$(git diff --name-only HEAD 2>/dev/null)

if [ ! -z "$MODIFIED_FILES" ]; then
    while IFS= read -r file; do
        # Determine if worker owns this file
        OWNS_FILE=false

        # Skip vault progress files (workers maintain their own progress)
        if [[ "$file" =~ ^\.worker-vault/Progress/ ]]; then
            OWNS_FILE=true
        # Skip vault constitution references (read-only, but workers can view/reference)
        elif [[ "$file" =~ ^\.worker-vault/Constitution/WORKER_${WORKER_ID}_CONSTITUTION\.md$ ]]; then
            OWNS_FILE=true
        else
            case $WORKER_ID in
                1)
                    if [[ "$file" =~ ^03-Source-Code/src/(active_inference|orchestration/routing|time_series|information_theory)/ ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                2)
                    if [[ "$file" =~ ^03-Source-Code/src/(gpu|orchestration/local_llm.*gpu|production)/|\.cu$ ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                3)
                    if [[ "$file" =~ ^03-Source-Code/src/(pwsa|finance)/.*portfolio ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                4)
                    if [[ "$file" =~ ^03-Source-Code/src/(telecom|robotics)/.*motion ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                5)
                    if [[ "$file" =~ ^03-Source-Code/src/orchestration/(thermodynamic|routing/.*advanced) ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                6)
                    if [[ "$file" =~ ^03-Source-Code/src/orchestration/local_llm/.*transformer ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                7)
                    if [[ "$file" =~ ^03-Source-Code/src/(drug_discovery|robotics)/.*advanced ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
                8)
                    if [[ "$file" =~ ^03-Source-Code/(src/api_server|deployment|docs)/ ]]; then
                        OWNS_FILE=true
                    fi
                    ;;
            esac
        fi

        if [ "$OWNS_FILE" = false ]; then
            # Check if it's a shared file (requires coordination)
            if [[ "$file" =~ ^03-Source-Code/src/(lib\.rs|orchestration/thermodynamic/mod\.rs|orchestration/mod\.rs|gpu/kernel_executor\.rs)$|^03-Source-Code/Cargo\.toml$ ]]; then
                echo "  ⚠️  WARNING: Editing shared file: $file"
                echo "     → Shared module files (mod.rs) are allowed with coordination"
                echo "     → Ensure exports are for your owned modules only"
                ((WARNINGS++))
            else
                echo "  ❌ VIOLATION: Editing file outside ownership: $file"
                echo "     → Worker $WORKER_ID does NOT own this file"
                ((VIOLATIONS++))
            fi
        fi
    done <<< "$MODIFIED_FILES"
fi

if [ $VIOLATIONS -eq 0 ]; then
    echo "  ✅ File ownership compliance: PASSED"
else
    echo "  ❌ File ownership violations: $VIOLATIONS"
fi

echo ""

# ============================================================================
# RULE 2: DEPENDENCIES - Must have required dependencies before proceeding
# ============================================================================
echo "🔍 Rule 2: Checking Dependency Requirements..."

DEPENDENCIES_MET=true

case $WORKER_ID in
    5|7)
        # Workers 5 and 7 REQUIRE Worker 1's time series for critical features
        if ! git log --all --oneline | grep -q "time series"; then
            if git diff --name-only HEAD | grep -q "cost_forecasting\|trajectory"; then
                echo "  ❌ VIOLATION: Attempting work that requires time series module"
                echo "     → Worker 1's time series NOT integrated yet"
                echo "     → Cannot proceed with forecasting features"
                DEPENDENCIES_MET=false
                ((VIOLATIONS++))
            fi
        fi
        ;;
    3)
        # Worker 3 requires pixel kernels for pixel processing
        if ! git log --all --oneline | grep -q "pixel.*kernel"; then
            if git diff --name-only HEAD | grep -q "pixel_processor\|pixel_tda"; then
                echo "  ❌ VIOLATION: Attempting pixel processing without pixel kernels"
                echo "     → Worker 2's pixel kernels NOT integrated yet"
                echo "     → Cannot proceed with pixel features"
                DEPENDENCIES_MET=false
                ((VIOLATIONS++))
            fi
        fi
        ;;
esac

if [ "$DEPENDENCIES_MET" = true ]; then
    echo "  ✅ Dependency requirements: MET"
else
    echo "  ❌ Dependency requirements: VIOLATED"
fi

echo ""

# ============================================================================
# RULE 3: INTEGRATION PROTOCOL - Must follow deliverable publishing process
# ============================================================================
echo "🔍 Rule 3: Checking Integration Protocol Compliance..."

# Check if worker has unpublished completed features
UNPUBLISHED_FEATURES=$(git log origin/$(git branch --show-current)..HEAD --oneline 2>/dev/null | grep -c "feat:")

if [ $UNPUBLISHED_FEATURES -gt 5 ]; then
    echo "  ⚠️  WARNING: $UNPUBLISHED_FEATURES completed features not published to deliverables"
    echo "     → Should publish completed features to deliverables branch"
    echo "     → Run: git checkout deliverables && git cherry-pick <commits>"
    ((WARNINGS++))
else
    echo "  ✅ Integration protocol: COMPLIANT"
fi

echo ""

# ============================================================================
# RULE 4: BUILD HYGIENE - Code must build before committing
# ============================================================================
echo "🔍 Rule 4: Checking Build Hygiene..."

if [ -d "03-Source-Code" ]; then
    cd 03-Source-Code

    echo "  Running cargo check (library)..."
    # Use --lib to only check library code (workers don't own bins)
    if cargo check --lib --features cuda 2>&1 | tail -10 | grep -q "error:"; then
        echo "  ❌ VIOLATION: Code has build errors"
        echo "     → Cannot commit code that doesn't build"
        echo "     → Fix errors before proceeding"
        ((VIOLATIONS++))
    else
        echo "  ✅ Build hygiene: PASSED (library compiles)"
    fi

    cd ..
else
    echo "  ⚠️  WARNING: 03-Source-Code directory not found"
    ((WARNINGS++))
fi

echo ""

# ============================================================================
# RULE 5: COMMIT DISCIPLINE - Proper commit messages and frequency
# ============================================================================
echo "🔍 Rule 5: Checking Commit Discipline..."

RECENT_COMMITS=$(git log --since="24 hours ago" --oneline | wc -l)

if [ $RECENT_COMMITS -eq 0 ]; then
    echo "  ⚠️  WARNING: No commits in last 24 hours"
    echo "     → Should commit progress at least daily"
    ((WARNINGS++))
else
    # Check commit message quality
    BAD_MESSAGES=$(git log --since="24 hours ago" --oneline | grep -c "^[a-f0-9]\+ \(WIP\|temp\|test\|foo\|bar\)" || true)

    if [ $BAD_MESSAGES -gt 0 ]; then
        echo "  ⚠️  WARNING: $BAD_MESSAGES commits with poor messages"
        echo "     → Use descriptive commit messages (feat:, fix:, refactor:)"
        ((WARNINGS++))
    else
        echo "  ✅ Commit discipline: GOOD"
    fi
fi

echo ""

# ============================================================================
# RULE 6: AUTO-SYNC COMPLIANCE - Must use auto-sync system
# ============================================================================
echo "🔍 Rule 6: Checking Auto-Sync System Usage..."

if [ ! -f "worker_start.sh" ] || [ ! -f "worker_auto_sync.sh" ]; then
    echo "  ❌ VIOLATION: Auto-sync scripts missing"
    echo "     → Scripts must be present in worker directory"
    ((VIOLATIONS++))
else
    echo "  ✅ Auto-sync system: PRESENT"
fi

echo ""

# ============================================================================
# RULE 7: GPU MANDATE - All compute code MUST use GPU
# ============================================================================
echo "🔍 Rule 7: Checking GPU Utilization Mandate..."

if [ -d "03-Source-Code/src" ]; then
    # Check for CPU loops in new code
    MODIFIED_RUST=$(git diff --name-only HEAD | grep "\.rs$" || true)

    if [ ! -z "$MODIFIED_RUST" ]; then
        CPU_LOOPS=0
        while IFS= read -r file; do
            if [ -f "$file" ]; then
                # Check for CPU loops without GPU calls
                if grep -q "for.*in.*{" "$file" && ! grep -q "gpu\|cuda\|kernel" "$file"; then
                    echo "  ⚠️  WARNING: Potential CPU loop in $file"
                    echo "     → Computational code should use GPU kernels"
                    ((CPU_LOOPS++))
                fi
            fi
        done <<< "$MODIFIED_RUST"

        if [ $CPU_LOOPS -gt 0 ]; then
            echo "  ⚠️  Found $CPU_LOOPS files with potential CPU loops"
            ((WARNINGS++))
        else
            echo "  ✅ GPU utilization: COMPLIANT"
        fi
    else
        echo "  ✅ No new Rust files to check"
    fi
fi

echo ""

# ============================================================================
# GOVERNANCE DECISION
# ============================================================================
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  GOVERNANCE VERDICT                                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

echo "📊 Summary:"
echo "   Violations: $VIOLATIONS"
echo "   Warnings: $WARNINGS"
echo ""

# Log to governance log
echo "[$TIMESTAMP] Worker $WORKER_ID | Violations: $VIOLATIONS | Warnings: $WARNINGS" >> "$GOVERNANCE_LOG"

if [ $VIOLATIONS -gt 0 ]; then
    echo "❌ GOVERNANCE STATUS: BLOCKED"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⛔ Worker $WORKER_ID is BLOCKED from proceeding"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "VIOLATIONS DETECTED:"
    echo "  • You have $VIOLATIONS rule violations"
    echo "  • These MUST be fixed before proceeding"
    echo ""
    echo "REQUIRED ACTIONS:"
    echo "  1. Review violations above"
    echo "  2. Fix all issues"
    echo "  3. Run governance check again"
    echo "  4. Only proceed when governance passes"
    echo ""
    echo "CONTACT: Worker 0-Alpha if you need guidance"
    echo ""

    exit 1

elif [ $WARNINGS -gt 0 ]; then
    echo "⚠️  GOVERNANCE STATUS: CAUTION"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  Worker $WORKER_ID has warnings"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "WARNINGS DETECTED:"
    echo "  • You have $WARNINGS warnings"
    echo "  • These should be addressed but don't block work"
    echo ""
    echo "RECOMMENDED ACTIONS:"
    echo "  1. Review warnings above"
    echo "  2. Address when convenient"
    echo "  3. Can proceed with caution"
    echo ""
    echo "✅ PROCEEDING WITH CAUTION"
    echo ""

    exit 0

else
    echo "✅ GOVERNANCE STATUS: APPROVED"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Worker $WORKER_ID is CLEARED to proceed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "ALL RULES COMPLIANT:"
    echo "  ✅ File ownership respected"
    echo "  ✅ Dependencies met"
    echo "  ✅ Integration protocol followed"
    echo "  ✅ Build hygiene maintained"
    echo "  ✅ Commit discipline good"
    echo "  ✅ Auto-sync system present"
    echo "  ✅ GPU utilization compliant"
    echo ""
    echo "🚀 APPROVED TO PROCEED WITH DEVELOPMENT"
    echo ""

    exit 0
fi
