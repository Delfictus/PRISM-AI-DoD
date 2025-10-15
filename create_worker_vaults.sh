#!/bin/bash

for i in {1..8}; do
    WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-$i"
    VAULT_DIR="$WORKER_DIR/.worker-vault"
    
    echo "Creating specialized vault for Worker $i..."
    
    mkdir -p "$VAULT_DIR"/{Constitution,Tasks,Progress,Reference}
    
    # Copy GPU Constitution (all workers need this)
    cp /home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/Constitution/GPU_CONSTITUTION.md \
       "$VAULT_DIR/Constitution/"
    
    # Copy only their specific task file
    case $i in
        1|2)
            # Workers 1-2 use original worker guides
            cp /home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/WORKER_A_TASKS.md \
               "$VAULT_DIR/Tasks/MY_TASKS.md" 2>/dev/null || \
            echo "# Worker $i Tasks - See 8_WORKER_ENHANCED_PLAN.md" > "$VAULT_DIR/Tasks/MY_TASKS.md"
            ;;
        *)
            echo "# Worker $i Tasks - See 8_WORKER_ENHANCED_PLAN.md in main vault" > "$VAULT_DIR/Tasks/MY_TASKS.md"
            ;;
    esac
    
    # Copy 8-worker enhanced plan (all need this)
    cp /home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/8_WORKER_ENHANCED_PLAN.md \
       "$VAULT_DIR/Reference/" 2>/dev/null || echo "Plan not found"
    
    # Copy worktree setup guide (all need this)
    cp /home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/GIT_WORKTREE_SETUP.md \
       "$VAULT_DIR/Reference/" 2>/dev/null || echo "Worktree guide not found"
    
    # Copy production upgrade plan (reference)
    cp /home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/PRODUCTION_UPGRADE_PLAN.md \
       "$VAULT_DIR/Reference/" 2>/dev/null || echo "Production plan not found"
    
    # Create worker-specific constitution
    cat > "$VAULT_DIR/Constitution/WORKER_${i}_CONSTITUTION.md" << CONST
# Worker $i Development Constitution

**Immutable Laws for Worker $i**

## Article I: File Ownership

YOU SHALL:
- Only edit files assigned to Worker $i
- Request kernels from Worker 2 via GitHub issues
- Coordinate shared file edits
- Follow daily sync protocol

YOU SHALL NOT:
- Edit files owned by other workers
- Modify kernel_executor.rs (Worker 2 only)
- Skip daily merges from parallel-development

## Article II: GPU Acceleration

ALL computational code SHALL:
- Use GPU kernels (not CPU loops)
- Request new kernels if needed (don't implement yourself unless you're Worker 2)
- Verify GPU execution with tests
- Maintain 95%+ GPU utilization for compute

## Article III: Testing

YOU SHALL:
- Test after every significant change
- Maintain 90%+ coverage for your modules
- Run full build before pushing
- Fix broken tests before EOD

## Article IV: Daily Protocol

MORNING (9 AM):
\`\`\`bash
cd $WORKER_DIR
git pull origin worker-$i-[branch]
git merge parallel-development
cargo build --features cuda
\`\`\`

EVENING (5 PM):
\`\`\`bash
git add -A
git commit -m "feat: [your work]"
git push origin worker-$i-[branch]
\`\`\`

CONST

    # Create progress tracker
    cat > "$VAULT_DIR/Progress/DAILY_PROGRESS.md" << PROGRESS
# Worker $i - Daily Progress Tracker

## Week 1
- [ ] Day 1:
- [ ] Day 2:
- [ ] Day 3:
- [ ] Day 4:
- [ ] Day 5:

## Week 2
- [ ] Day 1:
- [ ] Day 2:
- [ ] Day 3:
- [ ] Day 4:
- [ ] Day 5:

(Continue for 7 weeks)

Update this daily with what you accomplished.
PROGRESS

    # Create quick reference
    cat > "$VAULT_DIR/QUICK_REFERENCE.md" << QUICKREF
# Worker $i Quick Reference

## Your Worktree
\`/home/diddy/Desktop/PRISM-Worker-$i\`

## Your Branch
\`worker-$i-[branch-name]\`

## Your Time
~254 hours (7 weeks)

## Daily Commands

**Pull latest**:
\`\`\`bash
cd /home/diddy/Desktop/PRISM-Worker-$i
git pull origin worker-$i-*
git merge parallel-development
\`\`\`

**Build & Test**:
\`\`\`bash
cargo build --features cuda
cargo test --lib [your_module]
\`\`\`

**Commit**:
\`\`\`bash
git add -A
git commit -m "feat: [description]"
git push origin worker-$i-*
\`\`\`

## Your Documentation

All in \`.worker-vault/\`:
- Constitution/ - Your rules
- Tasks/ - Your work
- Progress/ - Daily tracker
- Reference/ - Full plans

## Need Help?

- Kernel request → GitHub issue [KERNEL]
- Question → GitHub issue [QUESTION]  
- Blocker → GitHub issue [BLOCKER]

Check \`.worker-vault/Reference/8_WORKER_ENHANCED_PLAN.md\` for complete context.
QUICKREF

    echo "✅ Worker $i vault created"
done

echo ""
echo "✅ All 8 worker-specific vaults created"
