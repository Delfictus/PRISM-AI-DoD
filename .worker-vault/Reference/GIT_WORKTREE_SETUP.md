# Git Worktree Setup - 4 Independent Workspaces

**What This Enables**:
- 4 separate working directories on one computer
- Each tracks a different branch
- Each can push independently
- No file mixing or conflicts
- One shared .git repository

---

## SETUP COMMANDS

### Initial Setup (Run Once):

```bash
cd /home/diddy/Desktop/PRISM-AI-DoD

# Main repo stays on parallel-development
git checkout parallel-development

# Create worktree for Worker 1
git worktree add ../PRISM-Worker-1 worker-1-ai-core

# Create worktree for Worker 2
git worktree add ../PRISM-Worker-2 worker-2-gpu-infra

# Create worktree for Worker 3
git worktree add ../PRISM-Worker-3 worker-3-apps-domain1

# Create worktree for Worker 4
git worktree add ../PRISM-Worker-4 worker-4-apps-domain2
```

**Result**:
```
/home/diddy/Desktop/
├── PRISM-AI-DoD/           # Main repo (parallel-development)
├── PRISM-Worker-1/         # Worker 1 worktree (worker-1-ai-core)
├── PRISM-Worker-2/         # Worker 2 worktree (worker-2-gpu-infra)
├── PRISM-Worker-3/         # Worker 3 worktree (worker-3-apps-domain1)
└── PRISM-Worker-4/         # Worker 4 worktree (worker-4-apps-domain2)
```

---

## HOW IT WORKS

### Each Worktree is Independent:

**Worker 1 workspace**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-1
pwd  # /home/diddy/Desktop/PRISM-Worker-1
git branch  # * worker-1-ai-core
git status  # Only Worker 1's files
```

**Worker 2 workspace**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
pwd  # /home/diddy/Desktop/PRISM-Worker-2
git branch  # * worker-2-gpu-infra
git status  # Only Worker 2's files
```

**NO FILE MIXING** - Each workspace has its own file copies.

---

## DAILY WORKFLOW

### Morning - Pull Latest:

**Worker 1**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-1
git pull origin worker-1-ai-core
git merge parallel-development  # Get integration updates
```

**Worker 2**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
git pull origin worker-2-gpu-infra
git merge parallel-development
```

**Worker 3**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-3
git pull origin worker-3-apps-domain1
git merge parallel-development
```

**Worker 4**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-4
git pull origin worker-4-apps-domain2
git merge parallel-development
```

### Work Session - Completely Independent:

**Worker 1** (in PRISM-Worker-1/):
```bash
cd /home/diddy/Desktop/PRISM-Worker-1
# Edit files in src/orchestration/routing/, src/active_inference/, etc.
cargo build --features cuda
cargo test
```

**Worker 2** (in PRISM-Worker-2/):
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
# Edit files in src/gpu/, src/orchestration/local_llm/, etc.
cargo build --features cuda
cargo test
```

**NO INTERFERENCE** - Changes in Worker 1's directory don't affect Worker 2's directory.

### Evening - Push Independently:

**Worker 1**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-1
git add -A
git commit -m "feat: Implement KSG TE"
git push origin worker-1-ai-core
```

**Worker 2**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
git add -A
git commit -m "feat: Add Tensor Core matmul"
git push origin worker-2-gpu-infra
```

**Each push goes to its own branch on GitHub. NO MIXING.**

---

## INTEGRATION

### Creating PRs:

**Each worker** goes to GitHub and creates PR:
- **Worker 1**: PR from `worker-1-ai-core` → `parallel-development`
- **Worker 2**: PR from `worker-2-gpu-infra` → `parallel-development`
- **Worker 3**: PR from `worker-3-apps-domain1` → `parallel-development`
- **Worker 4**: PR from `worker-4-apps-domain2` → `parallel-development`

### Merging (Friday Integration):

```bash
# In main repo
cd /home/diddy/Desktop/PRISM-AI-DoD
git checkout parallel-development

# Merge in dependency order
git merge worker-2-gpu-infra      # Infrastructure first
git merge worker-1-ai-core        # AI uses GPU
git merge worker-3-apps-domain1   # Apps use AI + GPU
git merge worker-4-apps-domain2   # Final integration

# Test
cargo build --release --all-features
cargo test --all

# If pass, push
git push origin parallel-development

# Update all worktrees
cd /home/diddy/Desktop/PRISM-Worker-1 && git merge parallel-development
cd /home/diddy/Desktop/PRISM-Worker-2 && git merge parallel-development
cd /home/diddy/Desktop/PRISM-Worker-3 && git merge parallel-development
cd /home/diddy/Desktop/PRISM-Worker-4 && git merge parallel-development
```

---

## KEY ADVANTAGES

### 1. **Complete Isolation**:
```bash
# Worker 1 builds
cd PRISM-Worker-1
cargo build  # Only affects this directory

# Worker 2 builds simultaneously
cd PRISM-Worker-2
cargo build  # Independent build, no conflict
```

### 2. **Independent Cargo Targets**:
Each worktree has its own `target/` directory:
```
PRISM-Worker-1/target/  # Worker 1's build artifacts
PRISM-Worker-2/target/  # Worker 2's build artifacts
PRISM-Worker-3/target/  # Worker 3's build artifacts
PRISM-Worker-4/target/  # Worker 4's build artifacts
```

**NO SHARED BUILD CACHE CONFLICTS**

### 3. **Shared .git Database**:
All worktrees share ONE `.git` repository:
```
PRISM-AI-DoD/.git/  # Single source of truth
```

**Benefits**:
- Commits are instantly available across worktrees
- No duplicate repository storage
- Single remote configuration

### 4. **Easy Switching**:
```bash
# Want to test Worker 2's code while in Worker 1's directory?
cd /home/diddy/Desktop/PRISM-Worker-2
cargo test

# Back to Worker 1
cd /home/diddy/Desktop/PRISM-Worker-1
```

No branch switching, no stashing, just cd.

---

## VERIFICATION

### Check Worktrees:
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
git worktree list

# Output:
# /home/diddy/Desktop/PRISM-AI-DoD         (parallel-development)
# /home/diddy/Desktop/PRISM-Worker-1       (worker-1-ai-core)
# /home/diddy/Desktop/PRISM-Worker-2       (worker-2-gpu-infra)
# /home/diddy/Desktop/PRISM-Worker-3       (worker-3-apps-domain1)
# /home/diddy/Desktop/PRISM-Worker-4       (worker-4-apps-domain2)
```

### Verify Independence:
```bash
# Create test file in Worker 1
cd /home/diddy/Desktop/PRISM-Worker-1
echo "test" > test-worker-1.txt
git add test-worker-1.txt
git commit -m "Worker 1 test"
git push origin worker-1-ai-core

# Check Worker 2 - file NOT there
cd /home/diddy/Desktop/PRISM-Worker-2
ls test-worker-1.txt  # File does not exist ✅
```

---

## CLEANUP (When Done)

### Remove Worktrees:
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD

git worktree remove ../PRISM-Worker-1
git worktree remove ../PRISM-Worker-2
git worktree remove ../PRISM-Worker-3
git worktree remove ../PRISM-Worker-4
```

---

## ADVANTAGES FOR YOUR USE CASE

✅ **Work on multiple workers simultaneously**:
- Have 4 terminal windows open, one in each workspace
- Edit Worker 1's code, compile Worker 2's code, test Worker 3's code - all at once

✅ **No branch switching**:
- No `git checkout` (slow, requires clean working tree)
- Just `cd` between directories

✅ **Independent builds**:
- Build Worker 2's GPU code without affecting Worker 1's AI code
- No cargo conflicts

✅ **Easy comparison**:
- See all 4 implementations side by side
- Copy files between workers if needed

✅ **Testing integration**:
- Merge to parallel-development
- Test in main repo
- Each worktree still independent

---

This is EXACTLY what you need for managing 4 parallel development tracks on one machine.
