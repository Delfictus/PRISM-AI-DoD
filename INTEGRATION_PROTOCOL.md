# PRISM-AI Integration Protocol
## 4-Tier Branch Strategy with Dual Worker 0 Management

**Last Updated**: 2025-10-12
**Integration Managers**: Worker 0-Alpha (human), Worker 0-Beta (automated)

---

## Branch Strategy Overview

```
production                  # SBIR/DoD final deliverable
    ↑ (Worker 0-Alpha approval)
staging                     # Pre-production validation
    ↑ (Weekly promotion if validation passes)
integration-staging         # Daily automated integration
    ↑ (Daily at 6 PM)
deliverables                # Shared deliverable exchange
    ↑ (Workers publish completed features)
parallel-development        # Worker coordination
    ↑ (Workers pull for general updates)
worker-X-branch            # Individual worker branches (8 total)
```

---

## Worker 0 Roles

### Worker 0-Alpha (Human Oversight)
**Worktree**: `/home/diddy/Desktop/PRISM-Worker-0-Alpha`
**Branch**: `staging`

**Responsibilities**:
- Review weekly validation reports from Worker 0-Beta
- Approve `staging` → `production` promotions
- Handle complex merge conflicts
- Make strategic decisions on priorities
- Final quality gate before production
- Tag production releases

**Tools**:
- Access to all worker branches
- Can manually test staging branch
- Reviews integration status dashboard

### Worker 0-Beta (Automated AI)
**Worktree**: `/home/diddy/Desktop/PRISM-Worker-0-Beta`
**Branch**: `integration-staging`

**Responsibilities**:
- Daily integration builds (6 PM)
- Merge all worker branches in dependency order
- Run incremental build validation
- Run unit test suite
- Create GitHub issues for failures
- Weekly full validation (Friday)
- Promote to `staging` if weekly validation passes
- Generate integration reports

**Automation**:
- `worker_0_beta_daily.sh` - runs daily
- `worker_0_beta_weekly.sh` - runs weekly (Friday)

---

## Daily Workflow for Workers

### Morning (9 AM) - Pull Latest Integration

```bash
cd /home/diddy/Desktop/PRISM-Worker-<X>

# Pull your branch
git pull origin worker-<X>-<branch>

# Merge latest integration (if needed)
git fetch origin integration-staging
git merge origin/integration-staging  # Get yesterday's integration

# Build and test
cargo build --features cuda
cargo test --lib <your_module>
```

### During Day - Check Dependencies

```bash
# Check if you're blocked by dependencies
./check_dependencies.sh <your-worker-number>

# Example for Worker 5:
./check_dependencies.sh 5
# Shows: Worker 1 time series status
```

### Afternoon - Publish Deliverables (When Feature Complete)

```bash
# 1. Test your feature
cargo test --lib <your_module>

# 2. Commit to your branch
git add <files>
git commit -m "feat: <description>"
git push origin worker-<X>-<branch>

# 3. Publish to deliverables
git fetch origin deliverables
git checkout deliverables
git cherry-pick <commit-hash>

# 4. Update tracking
echo "✅ Worker <X>: <feature> (Week <Y>) - AVAILABLE" >> .worker-deliverables.log
# Edit DELIVERABLES.md to change status to ✅ AVAILABLE

# 5. Commit and push deliverables
git add .worker-deliverables.log DELIVERABLES.md
git commit -m "Worker <X> deliverable: <feature>"
git push origin deliverables

# 6. Return to your branch
git checkout worker-<X>-<branch>
```

### Evening (5 PM) - Daily Commit

```bash
# Commit daily progress
git add -A
git commit -m "WIP: <what you worked on today>"
git push origin worker-<X>-<branch>
```

### Evening (6 PM) - Worker 0-Beta Daily Integration
- **Automated** - Worker 0-Beta runs daily integration
- Workers will be notified via GitHub issues if conflicts occur
- Check email/GitHub for integration status

---

## Weekly Workflow

### Friday - Worker 0-Beta Full Validation

**Automated Process**:
1. Worker 0-Beta runs comprehensive validation:
   - Full release build
   - All tests
   - GPU validation
   - Integration tests
   - Performance benchmarks

2. If all pass → promotes to `staging`
3. Creates GitHub issue for Worker 0-Alpha review

### Friday Evening - Worker 0-Alpha Review

**Worker 0-Alpha (Human)**:
1. Receives notification from Worker 0-Beta
2. Reviews weekly validation report
3. Optionally tests `staging` branch manually
4. If satisfied → approves for production consideration

---

## Consuming Deliverables from Other Workers

### Scenario: Worker 5 needs Worker 1's time series module

```bash
cd /home/diddy/Desktop/PRISM-Worker-5

# 1. Check if dependency is ready
./check_dependencies.sh 5
# Output shows: ✅ Worker 1 time series AVAILABLE

# 2. Pull from deliverables
git fetch origin deliverables
git merge origin/deliverables

# 3. Verify integration
cargo check --features cuda

# 4. Test the dependency
cargo test --lib time_series

# 5. Now use it in your code
# Worker 1's time series module is now available!
```

---

## Emergency Procedures

### If Daily Integration Build Fails

**Worker 0-Beta automatically**:
1. Identifies which worker's code caused the failure
2. Creates GitHub issue assigned to that worker
3. Aborts integration for that worker
4. Continues with other workers

**Worker Action**:
1. Check GitHub for assigned issue
2. Fix the build error in your branch
3. Re-push your branch
4. Worker 0-Beta will retry next day

### If You're Blocked by a Dependency

```bash
# 1. Check dependency status
./check_dependencies.sh <your-worker-number>

# 2. If blocked, work on non-dependent features
# See DELIVERABLES.md for alternatives

# 3. Or notify Worker 0-Alpha for prioritization
# Create GitHub issue: "Blocked: Need <dependency> from Worker X"
```

### If Merge Conflict in Your Branch

**During daily integration**:
1. Worker 0-Beta will create GitHub issue
2. Issue will contain conflict details
3. **Your action**:
   ```bash
   cd /home/diddy/Desktop/PRISM-Worker-<X>

   git fetch origin integration-staging
   git merge origin/integration-staging
   # Resolve conflicts manually

   git add <resolved-files>
   git commit -m "fix: resolve integration conflicts"
   git push origin worker-<X>-<branch>
   ```

---

## File Ownership Rules

### Each Worker Owns Specific Directories

**Worker 1**:
- `src/active_inference/`
- `src/orchestration/routing/`
- `src/time_series/`
- `src/information_theory/`

**Worker 2**:
- `src/gpu/`
- `src/orchestration/local_llm/` (GPU parts)
- `src/production/`
- All `.cu` files

**Worker 3**:
- `src/pwsa/`
- `src/finance/` (portfolio optimization)

**Worker 4**:
- `src/telecom/`
- `src/robotics/` (motion planning)

**Worker 5**:
- `src/orchestration/thermodynamic/`
- `src/orchestration/routing/` (advanced TE)

**Worker 6**:
- `src/orchestration/local_llm/` (transformer logic)

**Worker 7**:
- `src/drug_discovery/`
- `src/robotics/` (advanced features)

**Worker 8**:
- `src/api_server/`
- `deployment/`
- `docs/`

### Shared Files (Coordinate Before Editing)

**These require coordination**:
- `src/lib.rs`
- `src/integration/mod.rs`
- `Cargo.toml`
- `src/gpu/kernel_executor.rs` (Worker 2 ONLY adds kernels)

**Protocol**:
1. Post in team chat: "Editing <file> for <reason>"
2. Wait 5 minutes for objections
3. Edit quickly (< 30 minutes)
4. Commit and push immediately
5. Notify when done

---

## Success Metrics

### Daily Integration (Worker 0-Beta)
- ✅ All 8 workers merged without conflicts
- ✅ Build check passes
- ✅ Unit tests pass
- → integration-staging pushed to GitHub

### Weekly Validation (Worker 0-Beta)
- ✅ Release build passes
- ✅ All tests pass
- ✅ GPU validation passes
- → staging promoted

### Production Release (Worker 0-Alpha)
- ✅ Worker 0-Alpha manual approval
- ✅ All 8 workers' features complete
- ✅ Documentation complete
- ✅ Deployment tested
- → production tagged and released

---

## Tools and Scripts

### For Workers
- `./check_dependencies.sh <worker-number>` - Check if dependencies ready
- `.worker-deliverables.log` - See recent deliverables
- `DELIVERABLES.md` - Full deliverables manifest

### For Worker 0-Beta (Automated)
- `/home/diddy/Desktop/PRISM-Worker-0-Beta/worker_0_beta_daily.sh`
- `/home/diddy/Desktop/PRISM-Worker-0-Beta/worker_0_beta_weekly.sh`

### For Worker 0-Alpha (Human)
- Access to `/home/diddy/Desktop/PRISM-Worker-0-Alpha` (staging branch)
- GitHub notifications for weekly reports
- Integration status dashboard (future)

---

## Timeline

### Week 1-2: Foundation
- Workers 1, 2 deliver base infrastructure
- Other workers pull and begin integration

### Week 3-4: Core Features
- Workers 3-7 deliver main features
- Daily integration catches issues early

### Week 5-6: Advanced Features & Integration
- All workers deliver advanced features
- Worker 8 begins deployment preparation

### Week 7: Production Ready
- Worker 8 completes deployment
- Worker 0-Alpha approves production release
- SBIR/DoD deliverable ready

---

## Key Principles

1. **Early Integration** - Daily builds catch conflicts early
2. **Incremental Validation** - Don't wait until the end
3. **Clear Ownership** - Each worker owns specific files
4. **Unblock Quickly** - Use deliverables branch to share code
5. **Automate Everything** - Worker 0-Beta handles routine integration
6. **Human Oversight** - Worker 0-Alpha makes strategic decisions

---

## Questions?

- **Blocked by dependency?** → Check `./check_dependencies.sh <worker-number>`
- **Build failing?** → Check GitHub issues for your worker
- **Merge conflict?** → See "Emergency Procedures" above
- **Need guidance?** → Contact Worker 0-Alpha (human)

---

**Remember**: This is a **parallel development** system. Workers should rarely be blocked. Use `deliverables` branch to share code immediately when features complete.

**The goal**: 8 workers × 254 hours = production system in 7 weeks
