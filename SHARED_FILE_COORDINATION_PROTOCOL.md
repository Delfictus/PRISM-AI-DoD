# SHARED FILE COORDINATION PROTOCOL

**Status**: ACTIVE
**Last Updated**: October 12, 2025
**Applies To**: All Workers (1-8)

---

## PURPOSE

This document defines the protocol for editing shared files that multiple workers may need to modify. Shared files require coordination to prevent conflicts and maintain code integrity.

---

## SHARED FILES IDENTIFIED

### 1. Module Export Files (mod.rs)

**Files**:
- `03-Source-Code/src/lib.rs` - Root library exports
- `03-Source-Code/src/orchestration/mod.rs` - Orchestration module exports
- `03-Source-Code/src/orchestration/thermodynamic/mod.rs` - Thermodynamic submodule exports
- Other `mod.rs` files in shared module hierarchies

**Purpose**: Export submodules and make them accessible to parent modules

**Who Can Edit**:
- The worker who owns the submodules being exported
- Example: Worker 5 owns thermodynamic modules, so can edit `thermodynamic/mod.rs` to export them

**Coordination Required**:
- ⚠️ Governance will issue WARNING (not blocking)
- Proceed if you're only exporting YOUR owned modules
- Coordinate with other workers if editing exports for modules you don't own

### 2. Cargo Configuration

**File**: `03-Source-Code/Cargo.toml`

**Purpose**: Project dependencies, features, metadata

**Who Can Edit**:
- Any worker adding dependencies for their features
- Worker 8 (deployment) has primary ownership for CI/CD configuration

**Coordination Required**:
- ⚠️ Governance will issue WARNING
- Add dependencies in alphabetical order
- Document why dependency is needed
- Check for duplicates before adding
- Notify in team chat if adding major dependencies (>1MB)

### 3. GPU Kernel Executor

**File**: `03-Source-Code/src/gpu/kernel_executor.rs`

**Purpose**: Central GPU kernel execution interface

**Who Can Edit**:
- Worker 2 (GPU Infrastructure) - primary owner
- Other workers may add wrapper functions for their specific kernels

**Coordination Required**:
- ⚠️ Governance will issue WARNING
- Workers other than Worker 2 should create GitHub issue first
- Worker 2 must review and approve kernel additions
- Alternative: Create your own kernel wrapper in your owned directory

### 4. Integration Module

**File**: `03-Source-Code/src/integration/mod.rs`

**Purpose**: Cross-module integration glue code

**Who Can Edit**:
- Worker 8 (deployment) - primary owner
- Other workers for specific integration functions

**Coordination Required**:
- ⚠️ Governance will issue WARNING
- Create integration functions specific to your module
- Don't modify existing integration code without coordination
- Document integration dependencies clearly

---

## PROTOCOL FOR EDITING SHARED FILES

### Step 1: Identify the Edit Type

**A. Simple Export Addition** (Low Risk)
- You're adding exports for modules YOU created
- Example: `pub mod my_new_thermodynamic_schedule;`
- **Action**: Proceed with WARNING acknowledgment

**B. Dependency Addition** (Medium Risk)
- Adding a new crate dependency
- Modifying feature flags
- **Action**: Document rationale, check for duplicates, proceed with WARNING

**C. Modifying Existing Code** (High Risk)
- Changing existing exports
- Modifying shared functions
- Refactoring shared structures
- **Action**: Coordination REQUIRED (see Step 2)

### Step 2: Coordination Process

**For High Risk Edits:**

1. **Create GitHub Issue**:
   ```bash
   gh issue create \
       --title "Coordination: Editing shared file [FILE_NAME]" \
       --body "I need to edit shared file: [FILE_PATH]

   Reason: [Why you need to edit this file]

   Proposed changes:
   - [List specific changes]

   Impact:
   - Workers affected: [List workers]
   - Risk level: [High/Medium/Low]

   Requesting approval from:
   - @worker-X (if specific worker owns the file)
   - @worker-0-alpha (for high-risk changes)

   " \
       --label "coordination,shared-file,worker-Y"
   ```

2. **Wait for Approval**:
   - Worker 0-Alpha or file owner will review
   - Response expected within 24 hours
   - Proceed only after approval

3. **Make Changes**:
   - Edit the shared file as approved
   - Commit with reference to issue: `Shared file edit: <description> (issue #X)`

4. **Notify**:
   - Comment on issue when complete
   - Update `.worker-deliverables.log` if publishing

### Step 3: Governance Handling

**When you edit a shared file**, governance will:

```
⚠️  WARNING: Editing shared file: 03-Source-Code/src/orchestration/thermodynamic/mod.rs
   → Shared module files (mod.rs) are allowed with coordination
   → Ensure exports are for your owned modules only
```

**This is a WARNING, not a BLOCKING violation.**

**You can proceed if**:
- You're exporting modules you own
- You've documented the change
- The change is low risk (simple export addition)

**You should coordinate if**:
- You're modifying existing exports
- You're changing shared function signatures
- Multiple workers are affected

---

## SPECIFIC WORKER GUIDELINES

### Worker 1 (AI Core)

**Can edit without coordination**:
- `src/active_inference/mod.rs` (owns this)
- `src/orchestration/routing/mod.rs` (owns this)
- `src/time_series/mod.rs` (owns this)

**Must coordinate**:
- `src/lib.rs` (shared root)
- `src/orchestration/mod.rs` (shared with Worker 5)

### Worker 2 (GPU Infrastructure)

**Can edit without coordination**:
- `src/gpu/mod.rs` (owns this)
- Any `.cu` kernel files (owns all)

**Must coordinate**:
- `src/gpu/kernel_executor.rs` (other workers may add wrappers)
- `Cargo.toml` (for CUDA dependencies)

### Worker 3 (PWSA + Finance Apps)

**Can edit without coordination**:
- `src/pwsa/mod.rs` (owns this)
- `src/finance/portfolio/mod.rs` (owns this)

**Must coordinate**:
- `src/lib.rs` (if adding top-level exports)
- `Cargo.toml` (for domain-specific dependencies)

### Worker 4 (Telecom + Robotics Apps)

**Can edit without coordination**:
- `src/telecom/mod.rs` (owns this)
- `src/robotics/motion/mod.rs` (owns this)

**Must coordinate**:
- `src/lib.rs` (if adding top-level exports)
- `src/robotics/mod.rs` (shared with Worker 7)

### Worker 5 (Time Exchange Advanced)

**Can edit without coordination**:
- `src/orchestration/thermodynamic/mod.rs` (owns this)
- `src/orchestration/routing/advanced/mod.rs` (owns advanced routing)

**Must coordinate**:
- `src/orchestration/mod.rs` (shared with Worker 1)
- `src/lib.rs` (if adding orchestration exports)

### Worker 6 (LLM Advanced)

**Can edit without coordination**:
- `src/orchestration/local_llm/transformer/mod.rs` (owns transformer)

**Must coordinate**:
- `src/orchestration/local_llm/mod.rs` (shared with Worker 2)
- GPU kernel requests (Worker 2 owns GPU infra)

### Worker 7 (Drug Discovery + Robotics)

**Can edit without coordination**:
- `src/drug_discovery/mod.rs` (owns this)
- `src/robotics/advanced/mod.rs` (owns advanced robotics)

**Must coordinate**:
- `src/robotics/mod.rs` (shared with Worker 4)
- `src/lib.rs` (if adding top-level exports)

### Worker 8 (Finance Deploy + CI/CD)

**Can edit without coordination**:
- `deployment/` directory (owns all)
- `docs/` directory (owns all)
- `src/api_server/` (owns all)
- `Cargo.toml` (primary owner for CI/CD config)

**Must coordinate**:
- Less coordination needed (owns deployment layer)
- Should notify if changing build configuration that affects other workers

---

## GOVERNANCE WARNINGS VS VIOLATIONS

### WARNING (Caution - Can Proceed)

```
⚠️  GOVERNANCE STATUS: CAUTION

WARNINGS DETECTED:
  • You have X warnings
  • These should be addressed but don't block work

✅ PROCEEDING WITH CAUTION
```

**What to do**:
- Review the warning
- Ensure your edit is within protocol
- Proceed with your work
- Address warning when convenient

### VIOLATION (Blocked - Cannot Proceed)

```
❌ GOVERNANCE STATUS: BLOCKED

VIOLATIONS DETECTED:
  • You have X rule violations
  • These MUST be fixed before proceeding
```

**What to do**:
- Stop immediately
- Review violations
- Fix issues
- Re-run governance check
- Only proceed when violations are cleared

---

## SHARED FILE BEST PRACTICES

### 1. Module Exports (mod.rs)

**Good Practice**:
```rust
// Worker 5: Adding exports for thermodynamic modules I own
pub mod boltzmann_schedule;
pub mod replica_exchange_schedule;
pub mod simulated_annealing_schedule;
pub mod thermodynamic_ensemble;
pub mod parallel_tempering_schedule;
pub mod optimized_thermodynamic_consensus;
```

**Why Good**: Only exporting modules Worker 5 created

**Bad Practice**:
```rust
// Worker 5: DON'T DO THIS
pub mod routing;  // This is Worker 1's module
pub mod gpu_kernels;  // This is Worker 2's module
```

**Why Bad**: Exporting modules owned by other workers

### 2. Cargo.toml Dependencies

**Good Practice**:
```toml
# Worker 5: Adding thermodynamic-specific dependency
rand = "0.8"  # For Boltzmann sampling
rand_distr = "0.4"  # For statistical distributions

# Reason: Thermodynamic schedules require statistical sampling
# Used in: src/orchestration/thermodynamic/boltzmann_schedule.rs
```

**Why Good**: Documented reason, specific to Worker 5's needs

**Bad Practice**:
```toml
# Worker 5: DON'T DO THIS
tokio = { version = "1.0", features = ["full"] }  # Duplicates existing
serde = "1.0"  # Already in dependencies
```

**Why Bad**: Adding duplicate dependencies without checking

### 3. Git Commit Messages

**Good Practice**:
```bash
git commit -m "feat: Export 6 new thermodynamic schedule modules

Added exports to orchestration/thermodynamic/mod.rs for:
- Boltzmann schedule
- Replica exchange schedule
- Simulated annealing schedule
- Thermodynamic ensemble
- Parallel tempering schedule
- Optimized consensus

Shared file: mod.rs (coordination: exports for owned modules only)
Worker: 5
Ref: Week 1 & 2 tasks"
```

**Why Good**: Clear description, lists specific changes, notes shared file

**Bad Practice**:
```bash
git commit -m "update mod.rs"
```

**Why Bad**: No context, unclear what was changed or why

---

## CONFLICT RESOLUTION

### If Multiple Workers Edit Same Shared File

**Scenario**: Worker 5 and Worker 1 both edit `src/orchestration/mod.rs`

**Resolution Process**:

1. **First to commit**: Worker 5 commits and pushes to deliverables
2. **Second worker**: Worker 1 pulls and sees conflict
3. **Worker 1 resolves**:
   ```bash
   git pull origin deliverables
   # Conflict in src/orchestration/mod.rs

   # Open file, see:
   <<<<<<< HEAD
   pub mod routing;  // Worker 1's modules
   =======
   pub mod thermodynamic;  // Worker 5's modules
   >>>>>>> origin/deliverables

   # Resolution: BOTH are correct, merge both:
   pub mod routing;  // Worker 1
   pub mod thermodynamic;  // Worker 5

   git add src/orchestration/mod.rs
   git commit -m "Resolve: Merge orchestration exports (Worker 1 + Worker 5)"
   ```

4. **Verification**: Both workers' exports are preserved
5. **Notification**: Update integration log

---

## VAULT FILE POLICY

### Progress Tracking Files

**Status**: ✅ ALLOWED (no coordination needed)

**Files**:
- `.worker-vault/Progress/DAILY_PROGRESS.md`
- `.worker-vault/Progress/WEEKLY_SUMMARY.md`
- Any files in `.worker-vault/Progress/`

**Policy**:
- Workers maintain their own progress files
- No governance blocking for progress edits
- Update daily to track your work

### Constitution Files

**Status**: ✅ ALLOWED (read-only reference, but editable for notes)

**File**: `.worker-vault/Constitution/WORKER_X_CONSTITUTION.md`

**Policy**:
- Workers can reference and annotate their constitution
- Don't modify the core articles (I-VII)
- Can add personal notes or clarifications
- No governance blocking

### Reference Files

**Status**: 🔒 READ-ONLY (managed by Worker 0-Beta)

**Files**:
- `.worker-vault/Reference/INTEGRATION_SYSTEM.md`
- `.worker-vault/Reference/8_WORKER_ENHANCED_PLAN.md`
- Other reference documents

**Policy**:
- Do not edit reference files
- They are maintained by Worker 0-Beta/0-Alpha
- Request updates via GitHub issue if needed

---

## SUMMARY CHECKLIST

Before editing a shared file, ask:

- [ ] Do I own the modules I'm exporting? (mod.rs)
- [ ] Is this dependency already in Cargo.toml? (Cargo.toml)
- [ ] Am I only adding, not modifying existing code? (Low risk)
- [ ] Have I documented why this change is needed?
- [ ] If high risk, have I created coordination issue?
- [ ] Will this change affect other workers?
- [ ] Is my commit message clear and descriptive?

If YES to ownership and documentation questions: **Proceed with WARNING**
If NO or UNSURE: **Create coordination issue first**

---

## GOVERNANCE ENGINE BEHAVIOR

### Current Implementation (Fixed)

**File Path Patterns**: Updated to include `03-Source-Code/` prefix
```bash
# Before (BROKEN):
if [[ "$file" =~ ^src/orchestration/thermodynamic/ ]]; then

# After (FIXED):
if [[ "$file" =~ ^03-Source-Code/src/orchestration/thermodynamic/ ]]; then
```

**Build Check**: Updated to use `--lib` for library workers
```bash
# Before (BROKEN):
cargo check --features cuda  # Checks bins too

# After (FIXED):
cargo check --lib --features cuda  # Only checks library
```

**Vault Files**: Exception added for progress tracking
```bash
# NEW: Allow progress file edits
if [[ "$file" =~ ^\.worker-vault/Progress/ ]]; then
    OWNS_FILE=true
fi
```

**Shared Files**: mod.rs recognized as coordination (WARNING, not VIOLATION)
```bash
# NEW: mod.rs files treated as shared with coordination
if [[ "$file" =~ ^03-Source-Code/src/(lib\.rs|orchestration/thermodynamic/mod\.rs|orchestration/mod\.rs|gpu/kernel_executor\.rs)$ ]]; then
    echo "⚠️  WARNING: Editing shared file: $file"
    echo "   → Shared module files (mod.rs) are allowed with coordination"
    ((WARNINGS++))  # WARNING, not VIOLATION
fi
```

---

## QUESTIONS?

**Technical Questions**: Review this document and INTEGRATION_SYSTEM.md
**Coordination Questions**: Create GitHub issue with "coordination" label
**Governance Questions**: Review `.worker-vault/Constitution/`
**Critical Blockers**: Escalate to Worker 0-Alpha

---

**STATUS**: Protocol ACTIVE - All workers authorized to follow this protocol for shared file coordination.

**Worker 0-Alpha Approval**: ________________ (To be signed)
