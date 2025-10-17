# CHERRY-PICK GUIDE FOR WORKERS

**Purpose**: Clear instructions on when and how to use git cherry-pick for deliverable publishing
**Status**: ACTIVE
**Applies To**: All Workers (1-8)

---

## WHAT IS CHERRY-PICKING?

**Cherry-picking** = Taking a specific commit from one branch and applying it to another branch

**In PRISM context**:
- You work on your branch: `worker-X-branch`
- You complete a feature and commit it
- You want to share it with other workers via `deliverables` branch
- You **cherry-pick** that specific commit to `deliverables`

**Why not just merge?**
- Merging brings ALL commits (including incomplete work)
- Cherry-picking brings ONLY completed features
- Keeps deliverables clean and tested

---

## WHEN TO CHERRY-PICK

### ✅ DO Cherry-Pick When:

1. **Feature is COMPLETE**
   - All code written and tested
   - Tests pass: `cargo test --lib <your_module>`
   - Build passes: `cargo check --lib --features cuda`
   - Documentation added

2. **Feature is STANDALONE**
   - Doesn't depend on uncommitted work
   - Other workers can use it independently
   - Has clear interface/API

3. **Other Workers Need It**
   - Another worker is blocked waiting for your feature
   - Feature is listed in dependency graph
   - You mentioned it in `.worker-deliverables.log`

4. **Milestone Reached**
   - Completed a week's worth of work
   - Hit a deliverable milestone from your constitution
   - Ready to publish progress

### ❌ DON'T Cherry-Pick When:

1. **Feature is INCOMPLETE**
   - Still WIP (work in progress)
   - Tests not written yet
   - Known bugs or issues

2. **Code Doesn't Build**
   - Compilation errors
   - Fails `cargo check`
   - Missing dependencies

3. **Tests Failing**
   - Unit tests fail
   - Integration tests fail
   - GPU tests fail

4. **Multiple Dependent Commits**
   - Feature spans 10+ commits that depend on each other
   - Better to cherry-pick as a squashed commit (see advanced section)

---

## HOW TO CHERRY-PICK (BASIC)

### Step 1: Identify the Commit to Share

```bash
# View your recent commits
git log --oneline -10

# Example output:
# a1b2c3d feat: Add Boltzmann schedule implementation
# e4f5g6h refactor: Optimize energy calculations
# i7j8k9l docs: Add thermodynamic schedule examples
# m1n2o3p fix: Correct temperature scaling
```

**Choose the commit** that represents a complete feature. In this example: `a1b2c3d`

### Step 2: Switch to Deliverables Branch

```bash
# Save any uncommitted work first
git status

# If you have uncommitted changes:
git stash  # Temporarily save changes

# Switch to deliverables
git checkout deliverables

# Pull latest from remote
git pull origin deliverables
```

### Step 3: Cherry-Pick the Commit

```bash
# Cherry-pick by commit hash
git cherry-pick a1b2c3d

# Git will apply that commit to deliverables branch
```

### Step 4: Verify and Push

```bash
# Verify the commit was applied
git log -1

# Push to deliverables
git push origin deliverables
```

### Step 5: Update Deliverables Log

```bash
# Switch back to your branch
git checkout worker-X-branch

# Restore any stashed changes
git stash pop  # If you stashed earlier

# Edit the deliverables log
nano /home/diddy/Desktop/PRISM-AI-DoD/.worker-deliverables.log

# Add entry:
# [2025-10-12 15:30] Worker X: Boltzmann Schedule - AVAILABLE
# Commit: a1b2c3d
# Files: src/orchestration/thermodynamic/boltzmann_schedule.rs
# Dependencies: None
```

### Step 6: Return to Your Work

```bash
# You're back on your branch
git status

# Continue development
```

---

## COMMON QUESTIONS & SCENARIOS

### Q1: "I have 5 commits that make up one feature. Do I cherry-pick all 5?"

**Option A: Cherry-pick all 5 individually** (if each is meaningful)

```bash
git checkout deliverables
git cherry-pick a1b2c3d  # Commit 1
git cherry-pick e4f5g6h  # Commit 2
git cherry-pick i7j8k9l  # Commit 3
git cherry-pick m1n2o3p  # Commit 4
git cherry-pick q4r5s6t  # Commit 5
git push origin deliverables
```

**Option B: Cherry-pick as squashed commit** (RECOMMENDED)

```bash
# On your branch, create a single combined commit
git log --oneline -5  # Note the commit BEFORE your 5 commits (e.g., z9y8x7w)

# Interactive rebase to squash
git rebase -i z9y8x7w  # The commit before your feature

# In the editor, change all but first commit from "pick" to "squash":
pick a1b2c3d feat: Start Boltzmann schedule
squash e4f5g6h feat: Add energy calculations
squash i7j8k9l feat: Add temperature scaling
squash m1n2o3p feat: Add sampling logic
squash q4r5s6t test: Add Boltzmann tests

# Save and exit. Git will combine into one commit with combined message.

# Now cherry-pick the single commit
git log --oneline -1  # Get new squashed commit hash (e.g., a1a1a1a)
git checkout deliverables
git cherry-pick a1a1a1a
git push origin deliverables
```

**When to use Option B**:
- Many small commits (refactoring, fixes, tweaks)
- Cleaner deliverables history
- Easier for other workers to understand

**When to use Option A**:
- Each commit is a distinct feature
- Other workers might want commits individually
- Clear logical separation

### Q2: "What if cherry-pick creates a conflict?"

**This happens when deliverables has changes to the same files.**

**Step-by-step resolution**:

```bash
git checkout deliverables
git cherry-pick a1b2c3d

# Output:
# error: could not apply a1b2c3d... feat: Add Boltzmann schedule
# hint: after resolving conflicts, mark them with 'git add'
# hint: and commit the result with 'git cherry-pick --continue'

# Check what's in conflict
git status

# Example:
# both modified: src/orchestration/thermodynamic/mod.rs

# Open the file and resolve conflicts
nano src/orchestration/thermodynamic/mod.rs

# You'll see conflict markers:
# <<<<<<< HEAD
# pub mod replica_exchange;  // From deliverables (another worker)
# =======
# pub mod boltzmann_schedule;  // Your change
# >>>>>>> a1b2c3d

# Resolution: KEEP BOTH (both are correct exports)
pub mod replica_exchange;  // Another worker's module
pub mod boltzmann_schedule;  // Your module

# Save and exit

# Mark as resolved
git add src/orchestration/thermodynamic/mod.rs

# Continue cherry-pick
git cherry-pick --continue

# Git will open editor for commit message - save and exit

# Push
git push origin deliverables
```

**Common conflict causes**:
- Multiple workers exporting modules in same mod.rs
- Multiple workers adding dependencies to Cargo.toml
- Shared file edits (see SHARED_FILE_COORDINATION_PROTOCOL.md)

**Resolution principle**: Usually BOTH changes are correct, merge them together

### Q3: "I cherry-picked the wrong commit. How do I undo?"

**If you haven't pushed yet**:

```bash
# Undo the cherry-pick
git reset --hard HEAD~1

# This removes the last commit from deliverables
```

**If you already pushed**:

```bash
# Revert the commit (creates a new commit that undoes it)
git revert <commit-hash>
git push origin deliverables

# Then cherry-pick the correct commit
git cherry-pick <correct-hash>
git push origin deliverables
```

**Best practice**: Always verify before pushing:
```bash
git log -1  # Check the commit
git diff HEAD~1  # Review what changed
# If correct: git push
# If wrong: git reset --hard HEAD~1
```

### Q4: "Should I cherry-pick every commit I make?"

**NO.** Only cherry-pick COMPLETE, TESTED features.

**Your development workflow**:

```bash
# Day 1: Start feature
git commit -m "WIP: Start Boltzmann schedule"  # DON'T cherry-pick

# Day 2: Continue
git commit -m "WIP: Add energy calculations"  # DON'T cherry-pick

# Day 3: Continue
git commit -m "WIP: Add sampling logic"  # DON'T cherry-pick

# Day 4: Complete and test
git commit -m "feat: Complete Boltzmann schedule with tests"  # ✅ Cherry-pick THIS

# Day 5: Fix bug
git commit -m "fix: Correct temperature scaling in Boltzmann"  # ✅ Cherry-pick if urgent
```

**Rule of thumb**:
- WIP commits: Stay on your branch
- Complete features: Cherry-pick to deliverables
- Bug fixes: Cherry-pick if other workers affected

### Q5: "Can I cherry-pick from deliverables to my branch?"

**Usually you don't need to.** The `worker_start.sh` script automatically merges integration-staging (which includes deliverables) into your branch.

**But if you need a specific feature RIGHT NOW**:

```bash
# On your branch
git checkout worker-X-branch

# Find the commit on deliverables
git log origin/deliverables --oneline | grep "feature-you-need"

# Cherry-pick it
git cherry-pick <commit-hash>

# Or just merge deliverables
git merge origin/deliverables
```

**Recommendation**: Just run `./worker_start.sh X` which does this automatically

### Q6: "What if my feature depends on uncommitted work?"

**Don't cherry-pick yet.** Wait until dependent work is committed.

**Example**:

```bash
# You have:
# Commit A: Add helper function (not committed yet, just in working directory)
# Commit B: Use helper in feature (committed)

# Can't cherry-pick B without A - it won't work

# Solution: Commit A first, then cherry-pick both
git add helper.rs
git commit -m "feat: Add thermodynamic helper functions"
git add feature.rs
git commit -m "feat: Add Boltzmann schedule using helpers"

# Now cherry-pick BOTH commits
git checkout deliverables
git cherry-pick <commit-A-hash>
git cherry-pick <commit-B-hash>
git push origin deliverables
```

### Q7: "My cherry-pick says 'empty commit' - what do I do?"

**This means the change already exists in deliverables.**

```bash
git cherry-pick a1b2c3d
# Output: The previous cherry-pick is now empty, possibly due to conflict resolution.
# nothing to commit, working tree clean

# Options:
# Option 1: Skip it (it's already there)
git cherry-pick --skip

# Option 2: Force it anyway (rare)
git cherry-pick --allow-empty --keep-redundant-commits a1b2c3d
```

**Usually Option 1 is correct** - means someone else already integrated your change or made equivalent change.

---

## STEP-BY-STEP CHECKLIST

Use this checklist every time you cherry-pick:

```
[ ] 1. Feature is complete and tested
[ ] 2. Build passes: cargo check --lib --features cuda
[ ] 3. Tests pass: cargo test --lib <module>
[ ] 4. Identify commit hash: git log --oneline
[ ] 5. Stash uncommitted work: git stash (if needed)
[ ] 6. Switch to deliverables: git checkout deliverables
[ ] 7. Pull latest: git pull origin deliverables
[ ] 8. Cherry-pick: git cherry-pick <hash>
[ ] 9. Resolve conflicts if any
[ ] 10. Verify: git log -1 and git diff HEAD~1
[ ] 11. Push: git push origin deliverables
[ ] 12. Return to branch: git checkout worker-X-branch
[ ] 13. Restore work: git stash pop (if stashed)
[ ] 14. Update log: Edit .worker-deliverables.log
[ ] 15. Continue development
```

---

## ALTERNATIVE: USING MERGE (NOT RECOMMENDED)

**Some workers ask: "Can I just merge instead of cherry-pick?"**

**Technically yes, but NOT recommended:**

```bash
# This brings ALL commits (not just completed features)
git checkout deliverables
git merge worker-X-branch
git push origin deliverables
```

**Problems with merging**:
- ❌ Brings incomplete WIP commits
- ❌ Brings experimental code
- ❌ Brings failed attempts
- ❌ Makes deliverables messy
- ❌ Other workers get your half-baked code

**Why cherry-pick is better**:
- ✅ Only completed features
- ✅ Clean deliverables history
- ✅ Other workers get tested code
- ✅ You control what's shared

**Exception**: If ALL your commits are clean and complete, merge is fine. But usually that's not the case during active development.

---

## ADVANCED: CHERRY-PICK MULTIPLE COMMITS

### Cherry-pick a range:

```bash
# Cherry-pick commits from A to E (inclusive)
git cherry-pick A^..E

# Example:
git checkout deliverables
git cherry-pick a1b2c3d^..q4r5s6t
# This picks commits a1b2c3d, e4f5g6h, i7j8k9l, m1n2o3p, q4r5s6t
```

### Cherry-pick with modifications:

```bash
# Cherry-pick but don't commit yet (so you can modify)
git cherry-pick -n a1b2c3d

# Make additional changes
nano src/file.rs

# Then commit
git add .
git commit -m "feat: Cherry-picked and adapted for deliverables"
```

### Cherry-pick from another worker's branch:

```bash
# If you need something Worker 2 did but hasn't published
git fetch origin worker-2-gpu-infra
git cherry-pick origin/worker-2-gpu-infra~5  # Pick their 5th-from-latest commit

# Better: Ask them to publish it to deliverables first
```

---

## TROUBLESHOOTING

### Problem: "I cherry-picked but my changes didn't appear"

**Diagnosis**:
```bash
git log deliverables -1  # Check if commit is there
git show <commit-hash>  # See what's in the commit
```

**Common causes**:
- Cherry-picked wrong commit
- Commit was empty
- Forgot to push: `git push origin deliverables`

### Problem: "Cherry-pick fails with 'bad object' error"

**Diagnosis**:
```bash
# The commit hash doesn't exist in deliverables' history
git log --all --oneline | grep a1b2c3d
```

**Solution**:
```bash
# Make sure you're cherry-picking from your branch, not from deliverables
git checkout worker-X-branch
git log --oneline -10  # Verify commit exists
git checkout deliverables
git cherry-pick a1b2c3d  # Use correct hash
```

### Problem: "Tons of conflicts when cherry-picking"

**Diagnosis**: You're trying to cherry-pick a commit that depends on many other commits not in deliverables.

**Solution**:
```bash
# Abort the cherry-pick
git cherry-pick --abort

# Option A: Cherry-pick all dependencies first
git cherry-pick <dep1> <dep2> <dep3> <main-feature>

# Option B: Create a single commit with all changes
git checkout worker-X-branch
git rebase -i HEAD~10  # Squash all related commits
git checkout deliverables
git cherry-pick <squashed-commit>
```

---

## QUICK REFERENCE COMMANDS

### Basic Cherry-Pick:
```bash
git checkout deliverables
git pull origin deliverables
git cherry-pick <commit-hash>
git push origin deliverables
git checkout worker-X-branch
```

### Multiple Commits:
```bash
git checkout deliverables
git cherry-pick <hash1> <hash2> <hash3>
git push origin deliverables
git checkout worker-X-branch
```

### Squash Then Cherry-Pick:
```bash
# On your branch
git rebase -i HEAD~5  # Squash 5 commits
git log -1  # Get new hash
git checkout deliverables
git cherry-pick <new-hash>
git push origin deliverables
git checkout worker-X-branch
```

### Resolve Conflicts:
```bash
# After conflict
nano <conflicted-file>  # Edit to resolve
git add <conflicted-file>
git cherry-pick --continue
git push origin deliverables
```

### Undo Cherry-Pick:
```bash
# Before push
git reset --hard HEAD~1

# After push
git revert <commit-hash>
git push origin deliverables
```

---

## BEST PRACTICES

### ✅ DO:

1. **Test before cherry-picking**
   - Run `cargo check --lib --features cuda`
   - Run `cargo test --lib <your-module>`
   - Verify feature works

2. **Use clear commit messages**
   - `feat: Add Boltzmann sampling schedule` ✅
   - `updated stuff` ❌

3. **Cherry-pick complete features**
   - Wait until feature is done
   - Don't cherry-pick WIP commits

4. **Update deliverables log**
   - Always document what you published
   - Other workers need to know

5. **Communicate**
   - If another worker is waiting, notify them
   - Comment on GitHub issues when dependency ready

### ❌ DON'T:

1. **Don't cherry-pick failing code**
   - No build errors
   - No test failures
   - No known bugs

2. **Don't cherry-pick experimental code**
   - Keep experiments on your branch
   - Only proven features to deliverables

3. **Don't cherry-pick others' commits without permission**
   - Ask first
   - Better: Ask them to publish

4. **Don't cherry-pick 50 commits at once**
   - Too many conflicts
   - Consider squashing first

5. **Don't forget to return to your branch**
   - Easy to accidentally continue work on deliverables
   - Always: `git checkout worker-X-branch` after pushing

---

## GOVERNANCE COMPLIANCE

### Cherry-picking and Governance:

- ✅ Cherry-picking completed features is ENCOURAGED
- ✅ Publishing to deliverables is part of your responsibilities
- ⚠️ Cherry-picking WIP code will show in governance (build failures)
- ⚠️ Cherry-picking others' code without permission violates file ownership

### Governance checks after cherry-pick:

```bash
# After publishing to deliverables, your branch is unchanged
# Governance checks YOUR branch, not deliverables
# So cherry-picking doesn't affect your governance status
```

---

## INTEGRATION WITH WORKER WORKFLOW

### Your Daily Workflow:

```
Morning:
1. ./worker_start.sh X  # Pulls integration-staging (includes deliverables)
2. Review dependencies (auto-sync reports what's new)
3. Start development

During Day:
4. Work on features (commit frequently to YOUR branch)
5. Test as you go

End of Day:
6. Complete feature
7. Test thoroughly
8. Cherry-pick to deliverables (if feature complete)
9. Update .worker-deliverables.log
10. Continue with next task
```

### Weekly:
- Review what you've published
- Ensure all completed work is in deliverables
- Verify other workers have what they need from you

---

## EXAMPLES FROM REAL SCENARIOS

### Example 1: Worker 5 Publishing Boltzmann Schedule

```bash
# Worker 5 completed Boltzmann schedule
cd /home/diddy/Desktop/PRISM-Worker-5

# View commits
git log --oneline -3
# Output:
# a1b2c3d feat: Complete Boltzmann schedule with tests
# e4f5g6h WIP: Add energy calculations
# i7j8k9l WIP: Start Boltzmann schedule

# Test it
cd 03-Source-Code
cargo test --lib thermodynamic::boltzmann_schedule
# All tests pass ✅

# Cherry-pick the complete commit (not WIP ones)
cd ..
git stash  # Save any uncommitted work
git checkout deliverables
git pull origin deliverables
git cherry-pick a1b2c3d

# Push
git push origin deliverables

# Update log
git checkout worker-5-te-advanced
git stash pop
nano /home/diddy/Desktop/PRISM-AI-DoD/.worker-deliverables.log

# Add:
# [2025-10-12 16:45] Worker 5: Boltzmann Schedule - AVAILABLE
# Commit: a1b2c3d
# Files: src/orchestration/thermodynamic/boltzmann_schedule.rs
# Dependencies: None
# Use case: Thermodynamic consensus with Boltzmann sampling

# Continue working
```

### Example 2: Worker 1 Publishing Time Series (Multiple Commits)

```bash
# Worker 1 has 3 related commits for time series
cd /home/diddy/Desktop/PRISM-Worker-1

git log --oneline -3
# a1a1a1a feat: Add ARIMA forecasting
# b2b2b2b feat: Add exponential smoothing
# c3c3c3c feat: Add time series preprocessing

# All are complete and related - squash them
git rebase -i HEAD~3

# In editor:
pick a1a1a1a feat: Add ARIMA forecasting
squash b2b2b2b feat: Add exponential smoothing
squash c3c3c3c feat: Add time series preprocessing

# Save. Git opens commit message editor:
feat: Complete time series forecasting module

Includes:
- ARIMA forecasting
- Exponential smoothing
- Time series preprocessing
- Comprehensive tests
- GPU acceleration

# Save. Now one commit:
git log --oneline -1
# d4d4d4d feat: Complete time series forecasting module

# Cherry-pick
git checkout deliverables
git pull origin deliverables
git cherry-pick d4d4d4d
git push origin deliverables

# Update log and notify Worker 5 (who's waiting)
git checkout worker-1-ai-core
```

### Example 3: Worker 2 Publishing GPU Kernel (With Conflict)

```bash
# Worker 2 publishing new GPU kernel
cd /home/diddy/Desktop/PRISM-Worker-2

git checkout deliverables
git pull origin deliverables
git cherry-pick e5e5e5e  # Boltzmann GPU kernel

# Conflict!
# CONFLICT (content): Merge conflict in src/gpu/kernel_executor.rs

nano src/gpu/kernel_executor.rs

# See:
<<<<<<< HEAD
pub fn execute_matrix_multiply(...) { }  // Existing
=======
pub fn execute_boltzmann_sample(...) { }  // New
>>>>>>> e5e5e5e

# Resolution: Keep both
pub fn execute_matrix_multiply(...) { }  // Keep
pub fn execute_boltzmann_sample(...) { }  // Keep

# Save, resolve
git add src/gpu/kernel_executor.rs
git cherry-pick --continue
git push origin deliverables

# Notify Worker 5
gh issue comment 42 -b "Boltzmann kernel published to deliverables (commit e5e5e5e)"

git checkout worker-2-gpu-infra
```

---

## SUMMARY

**Cherry-pick is your tool for publishing completed work to other workers.**

**Remember**:
1. Only cherry-pick COMPLETE, TESTED features
2. Resolve conflicts by keeping both changes (usually)
3. Update `.worker-deliverables.log` after publishing
4. Return to your branch after pushing
5. Cherry-pick is optional but encouraged for completed work

**When in doubt**:
- Test first
- Cherry-pick complete features
- Document what you shared
- Ask Worker 0-Alpha if unsure

---

**Questions about cherry-picking?**
- Review this guide
- Check INTEGRATION_PROTOCOL.md
- Ask in team chat
- Create GitHub issue with "coordination" label

---

**STATUS**: Guide ACTIVE - All workers authorized to cherry-pick completed features to deliverables.
