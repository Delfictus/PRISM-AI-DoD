# Quick Start - Parallel Development

**Two developers working simultaneously without conflicts**

---

## INITIAL SETUP (Both Workers)

### 1. Clone and Setup
```bash
cd /home/<user>/Desktop/PRISM-AI-DoD
git checkout parallel-development
git pull origin parallel-development
```

### 2. Create Your Branch

**Worker A** (Algorithm Developer):
```bash
git checkout -b worker-a-algorithms
echo "Worker A - Algorithm Track" > .worker-a-marker
git add .worker-a-marker
git commit -m "Setup Worker A branch"
git push -u origin worker-a-algorithms
```

**Worker B** (Infrastructure Developer):
```bash
git checkout -b worker-b-infrastructure
echo "Worker B - Infrastructure Track" > .worker-b-marker
git add .worker-b-marker
git commit -m "Setup Worker B branch"
git push -u origin worker-b-infrastructure
```

### 3. Build and Test
```bash
cd src-new
cargo build --release --features cuda
cargo test --lib --features cuda

# Verify GPU working
./target/release/test_gpu_kernel
./target/release/test_optimized_gpu
```

---

## DAILY WORKFLOW

### Morning Routine (Both Workers)

**9:00 AM** - Sync with latest:
```bash
# Get latest from integration branch
git checkout parallel-development
git pull origin parallel-development

# Merge into your branch
git checkout worker-a-algorithms  # or worker-b-infrastructure
git merge parallel-development

# Check for conflicts
git status

# If conflicts, resolve them
# Then push
git push origin worker-a-algorithms
```

### Work Session

**Worker A** - Work on your assigned files:
- `src/orchestration/routing/`
- `src/orchestration/thermodynamic/`
- `src/active_inference/`

**Worker B** - Work on your assigned files:
- `src/orchestration/local_llm/`
- `src/gpu/`
- `src/production/`
- `tests/`

### End of Day Routine (Both Workers)

**5:00 PM** - Commit and push:
```bash
# Save work
git add -A
git commit -m "WIP: [Task description] - [Your initials]"
git push origin worker-a-algorithms  # or worker-b

# Create PR on GitHub
# Title: "[WORKER A/B] - [Date] - [Tasks completed]"
# Merge to: parallel-development
```

**6:00 PM** - Merge PRs:
- Review each other's PR (quick check, not deep review)
- Merge both to `parallel-development`
- Resolve any integration issues

---

## PREVENTING CONFLICTS

### Rule 1: File Ownership

**Worker A NEVER edits**:
- `src/gpu/kernel_executor.rs` (read-only for you)
- `src/orchestration/local_llm/`
- `src/production/`

**Worker B NEVER edits**:
- `src/orchestration/routing/`
- `src/orchestration/thermodynamic/`
- `src/active_inference/` (except adding kernels)

### Rule 2: Shared Files

**IF you must edit a shared file**:
1. Check if other worker is editing it (ask in chat)
2. Coordinate time window
3. Edit in < 1 hour
4. Commit immediately
5. Notify other worker

**Shared Files**:
- `src/integration/mod.rs`
- `src/gpu/kernel_executor.rs`
- `Cargo.toml`

### Rule 3: Communication

**Before starting ANY task**:
```bash
# Update PARALLEL_WORK_GUIDE.md
# Change: - [ ] Task
# To: - [🔄] Task - Worker A/B - Started [date]

git add .obsidian-vault/PARALLEL_WORK_GUIDE.md
git commit -m "docs: Starting task [name]"
git push
```

---

## REQUESTING HELP

### Worker A Needs GPU Kernel from Worker B:

**Create GitHub Issue**:
```
Title: [KERNEL REQUEST] - [Kernel name]
Label: worker-b, gpu-kernel

Description:
- Kernel purpose: [what it computes]
- Input: [data types and shapes]
- Output: [what it returns]
- Math: [formula or algorithm]
- Priority: [high/medium/low]

Example:
I need a digamma function kernel for KSG TE computation.
Input: float* n (array of neighbor counts)
Output: float* psi_n (digamma values)
Formula: ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²)
Priority: HIGH (blocks TE implementation)
```

### Worker B Has Question for Worker A:

**Create GitHub Issue**:
```
Title: [QUESTION] - [Topic]
Label: worker-a, algorithm

Example:
What should be the default embedding dimension for TE?
What's the expected accuracy tolerance for TE validation?
```

---

## TESTING PROTOCOL

### Before Every Commit:

```bash
# Build
cargo build --features cuda

# Run relevant tests
cargo test --lib --features cuda -- [your_module]

# If adding GPU kernel, test it
cargo test --release --bin test_[your_feature] --features cuda
```

### Before Creating PR:

```bash
# Full build
cargo build --release --features cuda

# All tests
cargo test --lib --features cuda

# Check no regressions
./target/release/test_gpu_kernel
./target/release/test_optimized_gpu

# GPU still working
nvidia-smi
```

---

## MERGE CONFLICTS - HOW TO RESOLVE

**If you get merge conflicts**:

```bash
git merge parallel-development
# CONFLICT in src/some/file.rs

# Open file, look for:
<<<<<<< HEAD
your code
=======
their code
>>>>>>> parallel-development

# Choose correct version or combine
# Remove conflict markers
# Save file

git add src/some/file.rs
git commit -m "Merge parallel-development, resolved conflicts"
git push
```

**Prevention**: Daily merges catch conflicts early when they're small

---

## PROGRESS TRACKING

### Check Progress:

**Your Tasks**:
```bash
cat .obsidian-vault/WORKER_A_TASKS.md  # or WORKER_B_TASKS.md
```

**Overall Progress**:
```bash
cat .obsidian-vault/PARALLEL_WORK_GUIDE.md
```

**Production Plan**:
```bash
cat .obsidian-vault/PRODUCTION_UPGRADE_PLAN.md
```

### Update Progress:

**After completing a task**:
1. Mark ✅ in your WORKER_X_TASKS.md
2. Mark ✅ in PARALLEL_WORK_GUIDE.md
3. Commit both
4. Push

---

## EMERGENCY PROCEDURES

### Build Broken:

**If your build breaks**:
1. Try `cargo clean && cargo build`
2. Check if Worker B's latest kernel additions broke something
3. Pull latest `parallel-development`
4. Create GitHub issue if stuck

**If integration build breaks**:
1. Check which PR caused it
2. Revert that PR temporarily
3. Fix offline
4. Re-merge when fixed

### GPU Issues:

**If GPU tests fail**:
```bash
# Check GPU
nvidia-smi

# Reset CUDA
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Rebuild
cargo clean
cargo build --release --features cuda
```

### Merge Conflicts:

**If conflicts are complex**:
1. Create backup branch
2. Start fresh from `parallel-development`
3. Cherry-pick your commits
4. Skip conflicting commits
5. Redo them manually

---

## MILESTONE REVIEWS

### End of Week 1:
**Both Workers** meet to:
- Review progress
- Resolve any integration issues
- Adjust timeline if needed
- Demonstrate working features

### End of Week 2:
- Integration testing together
- Ensure Worker A algorithms use Worker B infrastructure
- Performance benchmarking

### End of Week 3:
- Full system integration
- End-to-end testing
- Identify remaining gaps

### End of Week 4:
- Final polish
- Documentation review
- Production readiness check
- Merge to `master`

---

## QUICK COMMAND REFERENCE

**Start Day**:
```bash
git checkout parallel-development && git pull
git checkout worker-a-algorithms && git merge parallel-development
```

**End Day**:
```bash
git add -A
git commit -m "feat: [what you did]"
git push origin worker-a-algorithms
# Create PR on GitHub
```

**Check GPU**:
```bash
nvidia-smi
./target/release/test_gpu_kernel
```

**Build & Test**:
```bash
cargo build --features cuda
cargo test --lib --features cuda
```

---

## SUCCESS CRITERIA

**After 3-4 weeks, you should have**:

**Worker A Delivers**:
- ✅ Full KSG Transfer Entropy (not proxy)
- ✅ Advanced thermodynamic schedules
- ✅ Replica exchange operational
- ✅ Hierarchical Active Inference

**Worker B Delivers**:
- ✅ GGUF model loading
- ✅ KV-cache working
- ✅ BPE tokenizer
- ✅ Tensor Core acceleration
- ✅ Production monitoring

**Together You Have**:
- ✅ Enterprise-grade LLM orchestration system
- ✅ Patent-worthy innovations fully implemented
- ✅ 10-100x performance optimizations
- ✅ Production-ready deployment

---

**REMEMBER**: Communication is key. Daily syncs prevent conflicts. GitHub is your coordination tool.