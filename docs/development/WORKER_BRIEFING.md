# WORKER BRIEFING: NEW INTEGRATION SYSTEM

**Date**: October 12, 2025
**Status**: ACTIVE
**Applies To**: Workers 1-8

---

## EXECUTIVE SUMMARY

The PRISM project now has a fully automated integration system with strict governance enforcement. This briefing explains:

1. Your new startup procedure
2. Auto-sync system for dependency management
3. Governance rules and enforcement
4. Deliverable publishing process
5. Your authorizations

---

## 1. STARTUP PROCEDURE

**BEFORE ALL WORK, RUN:**

```bash
./worker_start.sh <your-worker-number>
```

**What this does:**
1. Pulls latest from your branch
2. Merges integration-staging (shared progress)
3. Runs governance check (BLOCKS on violations)
4. Auto-syncs dependencies (pulls what you need)
5. Validates build

**Exit Codes:**
- `0` = Ready to work
- Non-zero = BLOCKED (fix issues and re-run)

---

## 2. AUTO-SYNC SYSTEM

### How It Works

The auto-sync system automatically:
- **Detects** dependencies you need
- **Checks** if they're available in `deliverables` branch
- **Pulls** them automatically if ready
- **Waits gracefully** if not ready (suggests alternative work)

### When You're Blocked

If dependencies aren't ready, auto-sync will:
- Tell you what's missing
- Tell you who's working on it
- Suggest work you CAN do in the meantime
- Exit with non-zero code

**You are authorized to:**
- Work on non-dependent features
- Re-run `./worker_start.sh X` anytime to re-check
- Continue when you receive prompt: "Worker X ready to continue"

### When Dependencies Arrive

Once dependencies are available:
- Auto-sync pulls them automatically
- Validates integration
- Reports "UNBLOCKED!"
- You can proceed with dependent work

---

## 3. GOVERNANCE ENFORCEMENT

### 7 STRICT RULES

The governance engine enforces these rules **BEFORE you can work**:

1. **File Ownership** - Only edit files assigned to you
2. **Dependencies** - Must have required dependencies available
3. **Integration Protocol** - Follow deliverable publishing process
4. **Build Hygiene** - Code must build before committing
5. **Commit Discipline** - Commit daily with proper messages
6. **Auto-Sync Compliance** - Must use auto-sync system
7. **GPU Mandate** - All compute code MUST use GPU

### Enforcement

- Governance check runs automatically in `worker_start.sh`
- **Violations = IMMEDIATE BLOCKING**
- You must fix violations and re-run startup
- No work allowed until governance passes

### Your Governance File

Located at: `.worker-vault/STRICT_GOVERNANCE_ENGINE.sh`

You can run it manually anytime:
```bash
./.worker-vault/STRICT_GOVERNANCE_ENGINE.sh <your-worker-number>
```

---

## 4. DELIVERABLE PUBLISHING

### When to Publish

Publish a deliverable when:
- Feature is complete and tested
- Build passes (`cargo check --features cuda`)
- Tests pass (`cargo test --lib <your_module>`)
- Other workers are waiting for it

### How to Publish

**Step 1: Commit your work**
```bash
git add <your-files>
git commit -m "Complete: <feature-name>"
```

**Step 2: Cherry-pick to deliverables**
```bash
git checkout deliverables
git cherry-pick <your-commit-hash>
git push origin deliverables
```

**Step 3: Log the deliverable**

Edit `.worker-deliverables.log` in main repo:
```
[2025-10-12 14:30] Worker X: <Feature Name> - AVAILABLE
Dependencies: Worker Y (if any)
Files: src/path/to/files.rs
```

**Step 4: Return to your branch**
```bash
git checkout worker-X-branch
```

### Publishing Authorization

**You are authorized to:**
- Push directly to `deliverables` branch (cherry-picked commits only)
- Update `.worker-deliverables.log`
- Pull from `deliverables` branch anytime (auto-sync does this)
- Merge from `integration-staging` (worker_start.sh does this)

**You are NOT authorized to:**
- Push to `production`, `staging`, or `integration-staging` directly
- Merge other workers' branches directly
- Modify files outside your ownership

---

## 5. BRANCH STRUCTURE

```
production              # Final SBIR deliverable (Worker 0-Alpha only)
    ↑
staging                 # Pre-production validation (Worker 0-Alpha)
    ↑
integration-staging     # Daily automated integration (Worker 0-Beta)
    ↑
deliverables            # YOUR SHARED EXCHANGE POINT
    ↑
parallel-development    # Coordination branch
    ↑
worker-X-branch        # YOUR BRANCH (you work here)
```

### Your Workflow

1. **Work** on your branch (`worker-X-branch`)
2. **Publish** complete features to `deliverables`
3. **Pull** from `deliverables` to get other workers' features (auto-sync)
4. **Merge** from `integration-staging` daily (worker_start.sh)
5. **Never** touch production, staging, or integration-staging directly

---

## 6. WORKER 0 ROLES

### Worker 0-Beta (Automated AI)

**What it does:**
- Runs daily at 6 PM (automated integration)
- Merges all worker branches into `integration-staging`
- Runs full build and test suite
- Creates GitHub issues on conflicts/failures
- Weekly validation and staging promotion

**Your relationship:**
- Provides you with integrated codebase daily
- Catches integration issues early
- You don't interact with it directly

### Worker 0-Alpha (Human Oversight)

**Who**: The project owner (you know who you are)

**What they do:**
- Reviews staging branch weekly
- Promotes to production when ready
- Handles emergency fixes
- Provides strategic direction

**Your relationship:**
- Reports to Worker 0-Alpha on blockers
- Receives final approval for production releases
- Escalate critical issues to Worker 0-Alpha

---

## 7. DAILY WORKFLOW

### Morning

```bash
cd /home/diddy/Desktop/PRISM-Worker-X
./worker_start.sh X
```

- Review startup output
- If BLOCKED: Work on suggested alternatives
- If READY: Review tasks and proceed

### During Work

- Edit only your assigned files
- Commit regularly with clear messages
- Run tests frequently
- Check build before committing

### End of Day

- Commit all work
- If feature complete: Publish to deliverables
- Update `.worker-vault/Progress/` with status
- Run final `cargo check` to ensure clean state

---

## 8. REFERENCE DOCUMENTS

All in your `.worker-vault/Reference/`:

- **INTEGRATION_SYSTEM.md** - Complete integration details
- **8_WORKER_ENHANCED_PLAN.md** - Overall project plan
- **GIT_WORKTREE_SETUP.md** - Worktree structure
- **PRODUCTION_UPGRADE_PLAN.md** - Production path

In main repo:
- **INTEGRATION_PROTOCOL.md** - Full protocol documentation
- **AUTO_SYNC_GUIDE.md** - Auto-sync detailed guide
- **DELIVERABLES.md** - Deliverable manifest
- **.worker-deliverables.log** - Real-time deliverable log

---

## 9. YOUR CONSTITUTION

Your worker-specific constitution is located at:
`.worker-vault/Constitution/WORKER_X_CONSTITUTION.md`

**Key additions:**
- **Article V**: Governance Enforcement (7 rules)
- **Article VI**: Auto-Sync Protocol (graceful dependency management)
- **Article VII**: Deliverable Publishing (when and how to share)

**Review it carefully** - it's your operational guide.

---

## 10. COMMON SCENARIOS

### Scenario A: I'm Blocked on Dependencies

**What you'll see:**
```
❌ Worker Y: <feature> NOT READY - BLOCKING
💡 Worker X Status: BLOCKED
   Meanwhile, can work on:
   • Alternative feature (no dependency)
```

**What to do:**
1. Work on suggested alternatives
2. When prompted "Worker X ready to continue", re-run `./worker_start.sh X`
3. Auto-sync will pull and unblock you

### Scenario B: Governance Check Failed

**What you'll see:**
```
❌ GOVERNANCE STATUS: BLOCKED
VIOLATIONS DETECTED:
  • Rule 4: Build check FAILED
```

**What to do:**
1. Fix the violation (e.g., fix build errors)
2. Re-run `./worker_start.sh X`
3. Proceed once governance passes

### Scenario C: Integration Conflict

**What you'll see:**
```
⚠️  Merge conflicts - manual resolution needed
   Run: git merge origin/integration-staging
```

**What to do:**
1. Resolve conflicts in your files (ONLY your files)
2. Run `git add <resolved-files>`
3. Run `git commit -m "Resolve integration conflicts"`
4. Re-run `./worker_start.sh X`

### Scenario D: Ready to Publish Deliverable

**What to do:**
1. Ensure tests pass: `cargo test --lib <your_module>`
2. Commit: `git commit -m "Complete: Feature X"`
3. Note commit hash: `git log -1 --oneline`
4. Checkout deliverables: `git checkout deliverables`
5. Cherry-pick: `git cherry-pick <hash>`
6. Push: `git push origin deliverables`
7. Update log: Edit `.worker-deliverables.log`
8. Return: `git checkout worker-X-branch`

---

## 11. AUTHORIZATIONS SUMMARY

### YOU ARE AUTHORIZED TO:

✅ **Read Access**
- All branches (read-only except your branch and deliverables)
- All worker vaults (read their reference docs)
- Integration logs and status files

✅ **Write Access**
- Your worker branch (`worker-X-branch`)
- Deliverables branch (cherry-picked commits only)
- Your worker vault (`.worker-vault/`)
- `.worker-deliverables.log` (append deliverables)

✅ **Pull/Merge**
- Pull from `deliverables` anytime (auto-sync does this)
- Merge from `integration-staging` daily (worker_start.sh does this)
- Pull from other workers' branches if needed (read-only)

✅ **Automation**
- Run `./worker_start.sh X` anytime
- Run `./worker_auto_sync.sh X` anytime
- Run `./.worker-vault/STRICT_GOVERNANCE_ENGINE.sh X` anytime

### YOU ARE NOT AUTHORIZED TO:

❌ **Prohibited Actions**
- Push to `production`, `staging`, or `integration-staging`
- Edit files outside your ownership
- Bypass governance checks
- Commit directly to main repo (use deliverables)
- Delete or force-push any branches

---

## 12. GETTING HELP

### Check Your Status

```bash
./worker_start.sh X        # Full status check
git status                  # Local changes
git log -5 --oneline       # Recent commits
```

### Check Dependencies

```bash
./worker_auto_sync.sh X    # Dependency check
cat .worker-deliverables.log | grep "Worker Y"  # Specific worker
```

### Check Governance

```bash
./.worker-vault/STRICT_GOVERNANCE_ENGINE.sh X
```

### Review Your Tasks

```bash
cat .worker-vault/Tasks/MY_TASKS.md
```

### Check Integration System

```bash
cat .worker-vault/Reference/INTEGRATION_SYSTEM.md
```

---

## 13. QUICK START

**First Time After This Briefing:**

1. Review your constitution:
   ```bash
   cat .worker-vault/Constitution/WORKER_X_CONSTITUTION.md
   ```

2. Run startup to validate setup:
   ```bash
   ./worker_start.sh X
   ```

3. Review your current tasks:
   ```bash
   cat .worker-vault/Tasks/MY_TASKS.md
   ```

4. Check if you have any dependencies:
   ```bash
   ./worker_auto_sync.sh X
   ```

5. If READY: Proceed with your assigned work
6. If BLOCKED: Work on suggested alternatives

**Every Day After:**

1. Run `./worker_start.sh X`
2. Follow the output instructions
3. Work on your tasks
4. Commit regularly
5. Publish deliverables when complete

---

## QUESTIONS?

- **Technical Issues**: Check `.worker-vault/Reference/INTEGRATION_SYSTEM.md`
- **Governance Questions**: Review `.worker-vault/Constitution/`
- **Dependency Issues**: Run `./worker_auto_sync.sh X` for status
- **Critical Blockers**: Report to Worker 0-Alpha

---

**STATUS**: You are cleared to proceed with your assigned tasks following this briefing.

**VALIDATION**: All workers have been verified to have:
- ✅ Correct branch checkout
- ✅ Updated worker vaults
- ✅ Governance engine installed
- ✅ Auto-sync scripts configured
- ✅ Updated constitutions
- ✅ Integration system documentation

**AUTHORIZATION**: You are authorized to use auto-sync, publish to deliverables, and pull dependencies as needed.

---

**Worker 0-Alpha Signature**: ________________
**Date**: ________________

(To be signed by project owner after review)
