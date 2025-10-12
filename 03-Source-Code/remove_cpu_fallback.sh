#!/bin/bash
#
# AUTOMATED CPU FALLBACK REMOVAL
# Systematically removes all CPU fallback code
#

set -e

REPO_ROOT="/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code"

echo "====================================="
echo "  AUTOMATED CPU FALLBACK REMOVAL"
echo "====================================="
echo

cd "$REPO_ROOT"

# Phase 1: Delete old/unused files with CPU fallback
echo "[Phase 1] Removing old/unused GPU files..."
rm -f src/gpu/gpu_enabled_old.rs 2>/dev/null || true
rm -f src/gpu/simple_gpu_v2.rs 2>/dev/null || true
rm -f src/gpu/gpu_real.rs 2>/dev/null || true
echo "✅ Old files removed"

# Phase 2: Delete memory_simple.rs and memory_manager.rs (superseded by kernel_executor)
echo "[Phase 2] Removing superseded memory managers..."
rm -f src/gpu/memory_simple.rs 2>/dev/null || true
rm -f src/gpu/memory_manager.rs 2>/dev/null || true
echo "✅ Superseded memory files removed"

# Phase 3: Delete kernel_launcher.rs and tensor_ops.rs (superseded by kernel_executor)
echo "[Phase 3] Removing superseded launchers..."
rm -f src/gpu/kernel_launcher.rs 2>/dev/null || true
rm -f src/gpu/tensor_ops.rs 2>/dev/null || true
echo "✅ Superseded launcher files removed"

# Phase 4: Delete gpu_executor.rs (has CPU fallback and superseded)
echo "[Phase 4] Removing old gpu_executor..."
rm -f src/gpu/gpu_executor.rs 2>/dev/null || true
echo "✅ Old executor removed"

# Phase 5: Delete gpu_launcher.rs (top-level, has CPU fallback)
echo "[Phase 5] Removing top-level gpu_launcher..."
rm -f src/gpu_launcher.rs 2>/dev/null || true
echo "✅ Top-level launcher removed"

echo
echo "====================================="
echo "✅ CLEANUP COMPLETE"
echo "====================================="
echo
echo "Remaining files to fix manually:"
echo "  - src/pwsa/gpu_kernels.rs"
echo "  - src/pwsa/active_inference_classifier.rs"
echo "  - src/active_inference/gpu_policy_eval.rs"
echo "  - src/active_inference/gpu_inference.rs"
echo "  - src/cma/transfer_entropy_gpu.rs"
echo "  - src/cma/quantum/pimc_gpu.rs"
echo "  - src/cma/gpu_integration.rs"
echo "  - src/quantum_mlir/runtime.rs"
echo "  - src/statistical_mechanics/gpu_bindings.rs"
echo "  - src/information_theory/gpu_transfer_entropy.rs"
echo