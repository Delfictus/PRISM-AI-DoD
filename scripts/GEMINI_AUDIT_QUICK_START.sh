#!/bin/bash
# PRISM-AI Google Gemini Audit - Quick Start Script
# Created: October 14, 2025
# Purpose: Prepare environment for Gemini code audit

set -e

echo "=========================================="
echo "PRISM-AI Google Gemini Audit Setup"
echo "=========================================="
echo ""

# Configuration
AUDIT_WORKSPACE="/home/diddy/Desktop/PRISM-AI-CODE-AUDIT-2025"
CODEBASE="/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code"
DIRECTIVE_DIR="/home/diddy/Desktop/PRISM-AI-DoD"

# Verify audit package files exist
echo "✓ Verifying audit package files..."
REQUIRED_FILES=(
    "$DIRECTIVE_DIR/GEMINI_AUDIT_PACKAGE_README.md"
    "$DIRECTIVE_DIR/GEMINI_CODE_AUDIT_DIRECTIVE.md"
    "$DIRECTIVE_DIR/GEMINI_AUDIT_FILE_LIST.md"
    "$DIRECTIVE_DIR/GEMINI_AUDIT_EVALUATION_RUBRIC.md"
    "$DIRECTIVE_DIR/GEMINI_AUDIT_EXECUTION_GUIDE.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: Missing required file: $file"
        exit 1
    fi
done
echo "✓ All audit package files present"
echo ""

# Verify codebase exists
echo "✓ Verifying codebase location..."
if [ ! -d "$CODEBASE" ]; then
    echo "❌ ERROR: Codebase not found at $CODEBASE"
    exit 1
fi
echo "✓ Codebase found at $CODEBASE"
echo ""

# Create workspace directory
echo "✓ Creating audit workspace..."
mkdir -p "$AUDIT_WORKSPACE"
cd "$AUDIT_WORKSPACE"

# Create phase findings templates
echo "✓ Creating phase findings templates..."
for i in {1..10}; do
    if [ ! -f "phase${i}_findings.md" ]; then
        cat > "phase${i}_findings.md" << EOF
# Phase ${i} Findings

## Score: [X/10]

## Evidence Reviewed
- [ ] File 1: path/to/file.rs
- [ ] File 2: path/to/file.rs

## Strengths
1. [Finding with evidence: file:line]

## Weaknesses
1. [Finding with evidence: file:line]

## Critical Issues
- None / [Issue description with evidence]

## Notes
- [Additional observations]
EOF
    fi
done

# Create additional templates
if [ ! -f "critical_issues.md" ]; then
    cat > "critical_issues.md" << EOF
# Critical Issues (Show-Stoppers)

## Issue 1: [Title]
**Severity**: CRITICAL
**Evidence**: file:line
**Description**: [Detailed description]
**Impact**: [Impact on production readiness]
**Recommendation**: [How to fix with effort estimate]

---
EOF
fi

if [ ! -f "evidence_log.md" ]; then
    cat > "evidence_log.md" << EOF
# Evidence Log

## Date: $(date +"%Y-%m-%d")

### GPU Infrastructure Evidence
- [ ] Real CUDA API calls found in: [file:line]
- [ ] GPU memory allocation in: [file:line]
- [ ] Kernel loading in: [file:line]

### CUDA Kernels Evidence
- [ ] Kernel count: [N kernels]
- [ ] Kernel files: [list files]
- [ ] Kernel launches: [file:line examples]

### Space Force SBIR Evidence
- [ ] Graph coloring implementation: [file:line]
- [ ] TSP implementation: [file:line]
- [ ] Performance benchmarks: [file:line]

### Other Evidence
- [Add as needed]
EOF
fi

echo "✓ Templates created"
echo ""

# Print summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Audit Workspace: $AUDIT_WORKSPACE"
echo "Codebase Location: $CODEBASE"
echo ""
echo "Next Steps:"
echo "1. Review audit package README:"
echo "   cat $DIRECTIVE_DIR/GEMINI_AUDIT_PACKAGE_README.md"
echo ""
echo "2. Read main directive:"
echo "   cat $DIRECTIVE_DIR/GEMINI_CODE_AUDIT_DIRECTIVE.md"
echo ""
echo "3. Start with Phase 1 (GPU Infrastructure):"
echo "   cd $CODEBASE"
echo "   # Review: src/gpu/context.rs, src/gpu/memory.rs, src/gpu/module.rs"
echo ""
echo "4. Document findings in:"
echo "   $AUDIT_WORKSPACE/phase1_findings.md"
echo ""
echo "5. Continue through all 10 phases following:"
echo "   $DIRECTIVE_DIR/GEMINI_AUDIT_EXECUTION_GUIDE.md"
echo ""
echo "6. Generate final report at:"
echo "   $AUDIT_WORKSPACE/GEMINI_PRODUCTION_AUDIT_REPORT.md"
echo ""
echo "=========================================="
echo "Key Reminders:"
echo "=========================================="
echo "- WALK THROUGH ACTUAL CODE (not documentation)"
echo "- Verify GPU reality (check for CUDA API calls)"
echo "- Count actual kernels (PTX/CUDA files)"
echo "- Assess Space Force SBIR readiness (GO/NO-GO)"
echo "- Provide honest assessment with evidence"
echo "=========================================="
echo ""
echo "Estimated Time: 21.5 hours (2-3 days)"
echo ""
