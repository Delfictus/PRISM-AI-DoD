#!/bin/bash
# PRISM-AI Build Integrity Verification
# Purpose: Ensure Worker 0-Alpha doesn't break the build
# Run this BEFORE and AFTER any Cargo.toml changes

set -e

echo "=========================================="
echo "PRISM-AI Build Integrity Verification"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track status
ALL_PASSED=true

# Function to check status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ALL_PASSED=false
        return 1
    fi
}

cd /home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code

echo "1. Checking workspace structure..."
if grep -q "^\[workspace\]" Cargo.toml && grep -q "members = \[" Cargo.toml; then
    echo -e "${GREEN}✅ Workspace structure intact${NC}"
else
    echo -e "${RED}❌ CRITICAL: Workspace structure broken!${NC}"
    echo "   The [workspace] section is missing or commented out."
    ALL_PASSED=false
fi
echo ""

echo "2. Checking workspace members..."
EXPECTED_MEMBERS=(
    "src/neuromorphic"
    "src/quantum"
    "src/foundation"
    "src/shared-types"
    "src/prct-core"
    "src/mathematics"
    "validation"
)

for member in "${EXPECTED_MEMBERS[@]}"; do
    if grep -q "\"$member\"" Cargo.toml; then
        echo -e "   ${GREEN}✅${NC} $member"
    else
        echo -e "   ${RED}❌${NC} $member - MISSING!"
        ALL_PASSED=false
    fi
done
echo ""

echo "3. Checking path dependencies..."
PATH_DEPS=(
    "neuromorphic-engine.*path.*src/neuromorphic"
    "quantum-engine.*path.*src/quantum"
    "platform-foundation.*path.*src/foundation"
    "shared-types.*path.*src/shared-types"
    "prct-core.*path.*src/prct-core"
    "mathematics.*path.*src/mathematics"
)

for dep in "${PATH_DEPS[@]}"; do
    if grep -qP "$dep" Cargo.toml; then
        echo -e "   ${GREEN}✅${NC} Path dependency intact"
    else
        echo -e "   ${RED}❌${NC} Path dependency broken: $dep"
        ALL_PASSED=false
    fi
done
echo ""

echo "4. Verifying sub-crate Cargo.toml files exist..."
SUB_CRATES=(
    "src/neuromorphic/Cargo.toml"
    "src/quantum/Cargo.toml"
    "src/foundation/Cargo.toml"
    "src/shared-types/Cargo.toml"
    "src/prct-core/Cargo.toml"
    "src/mathematics/Cargo.toml"
    "validation/Cargo.toml"
)

for crate in "${SUB_CRATES[@]}"; do
    if [ -f "$crate" ]; then
        echo -e "   ${GREEN}✅${NC} $crate"
    else
        echo -e "   ${RED}❌${NC} $crate - MISSING!"
        ALL_PASSED=false
    fi
done
echo ""

echo "5. Testing workspace detection..."
if cargo tree --workspace >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Cargo recognizes workspace${NC}"
else
    echo -e "${RED}❌ Cargo cannot recognize workspace!${NC}"
    ALL_PASSED=false
fi
echo ""

echo "6. Checking for build dependencies..."
if grep -q "^\[build-dependencies\]" Cargo.toml; then
    echo -e "${GREEN}✅ Build dependencies present${NC}"
else
    echo -e "${RED}❌ Build dependencies missing!${NC}"
    ALL_PASSED=false
fi
echo ""

echo "7. Verifying CUDA configuration..."
if grep -q "cudarc.*git.*cudarc.git" Cargo.toml; then
    echo -e "${GREEN}✅ CUDA 13 configuration intact${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA configuration may have changed${NC}"
fi
echo ""

echo "=========================================="
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
    echo "Build integrity is intact."
    echo "Safe to proceed with release tasks."
    exit 0
else
    echo -e "${RED}❌ INTEGRITY CHECKS FAILED${NC}"
    echo ""
    echo "CRITICAL: Build structure has been compromised!"
    echo ""
    echo "Actions to take:"
    echo "1. DO NOT commit these changes"
    echo "2. Revert Cargo.toml: git restore Cargo.toml"
    echo "3. Read: WORKER_0_ALPHA_BUILD_SAFETY_DIRECTIVE.md"
    echo "4. Avoid workspace structure changes"
    echo ""
    exit 1
fi
