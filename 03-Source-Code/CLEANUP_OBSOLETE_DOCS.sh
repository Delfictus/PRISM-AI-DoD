#!/bin/bash
# Remove obsolete documentation from worker worktrees
# Keep only relevant docs for each worker

OBSOLETE_DOCS=(
    "CUDA13_FINAL_STATUS.md"
    "CUDA13_STATUS.md"
    "GPU_VALIDATION_COMPLETE.md"
    "GPU_API_FIX_PLAN.md"
    "GPU_ENABLED_SUCCESS.md"
    "GPU_KERNEL_SUCCESS.md"
    "GPU_TEST_SUCCESS.md"
    "CPU_FALLBACK_AUDIT.md"
    "CPU_FALLBACK_ELIMINATION_PLAN.md"
    "PTX_LOADING_REQUIREMENTS.md"
    "GPU_MODULE_PRIORITY.md"
    "GPU_ACTION_SUMMARY.md"
    "GPU_IMPLEMENTATION_PLAN.md"
    "GPU_DRIVER_STATUS.md"
    "GPU_MIGRATION_PLAN.md"
    "GPU_DETAILED_IMPLEMENTATION.md"
    "GPU_PERSISTENCE_TEST_RESULTS.md"
    "GPU_INTEGRATION_REPORT.md"
    "SECURE_BOOT_FIX.md"
    "GPU_FINAL_COMPLETE_STATUS.md"
    "GPU_FINAL_GPU_STATUS_SUMMARY.md"
    "GPU_TRUTH_STATUS.md"
    "PWSA_GPU_TASKS.md"
)

echo "Archiving obsolete docs..."
mkdir -p archive/old-status-files

for doc in "${OBSOLETE_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" archive/old-status-files/
        echo "Archived: $doc"
    fi
done

echo "✅ Cleanup complete"
