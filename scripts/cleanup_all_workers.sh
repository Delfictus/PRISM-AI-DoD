#!/bin/bash
# Clean up obsolete docs from all worker worktrees

# Docs to remove (obsolete planning/status files)
REMOVE_DOCS=(
    "ACTIVE_DEVELOPMENT_STATUS.md"
    "CONSTITUTIONAL_PHASE_6_PROPOSAL.md"
    "GPU_ACCELERATION_STATUS.md"
    "MISSION_CHARLIE_COMPLETE.md"
    "VAULT_STRUCTURE.md"
    "PHASE_6_EXPANDED_CAPABILITIES.md"
)

for worker in {1..8}; do
    WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-$worker"
    
    if [ -d "$WORKER_DIR" ]; then
        echo "Cleaning Worker $worker..."
        cd "$WORKER_DIR"
        
        for doc in "${REMOVE_DOCS[@]}"; do
            if [ -f "$doc" ]; then
                rm -f "$doc"
                echo "  Removed: $doc"
            fi
        done
        
        # Keep only: README.md, WORKER_X_README.md, PARALLEL_DEV_SETUP_SUMMARY.md
        echo "  ✅ Worker $worker cleaned"
    fi
done

echo ""
echo "✅ All workers cleaned - only essential docs remain"
