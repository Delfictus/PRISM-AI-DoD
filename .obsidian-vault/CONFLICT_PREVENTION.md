# Conflict Prevention Matrix

**How to ensure Worker A and Worker B never edit the same file simultaneously**

---

## FILE OWNERSHIP MAP

### ✅ WORKER A - EXCLUSIVE OWNERSHIP

**Can Edit Freely** (No coordination needed):
```
src/orchestration/routing/
├── te_embedding_gpu.rs (CREATE)
├── gpu_kdtree.rs (CREATE)
├── ksg_transfer_entropy_gpu.rs (CREATE)
└── gpu_transfer_entropy_router.rs (MODIFY - owns this)

src/orchestration/thermodynamic/
├── advanced_energy.rs (CREATE)
├── temperature_schedules.rs (CREATE)
├── replica_exchange.rs (CREATE)
├── bayesian_learning.rs (CREATE)
└── gpu_thermodynamic_consensus.rs (MODIFY - owns this)

src/active_inference/
├── hierarchical_inference_gpu.rs (CREATE)
├── policy_search_gpu.rs (CREATE)
└── [existing files] (MODIFY as needed)

src/information_theory/
└── [all files] (MODIFY as needed)
```

**READ-ONLY** (Cannot modify):
```
src/gpu/kernel_executor.rs (read to use kernels, don't edit)
src/orchestration/local_llm/ (Worker B's area)
src/production/ (Worker B's area)
```

---

### ✅ WORKER B - EXCLUSIVE OWNERSHIP

**Can Edit Freely**:
```
src/orchestration/local_llm/
├── gguf_loader.rs (CREATE)
├── kv_cache.rs (CREATE)
├── bpe_tokenizer.rs (CREATE)
├── mixed_precision.rs (CREATE)
├── gpu_transformer.rs (MODIFY - owns this)
└── gpu_llm_inference.rs (MODIFY - owns this)

src/gpu/
├── kernel_executor.rs (OWNS - adds kernels here)
├── tensor_core_ops.rs (CREATE)
├── async_executor.rs (CREATE)
├── kernel_fusion_advanced.rs (CREATE)
└── [all other gpu files] (MODIFY as needed)

src/production/
├── error_handling.rs (CREATE)
├── monitoring.rs (CREATE)
└── config.rs (CREATE)

src/quantum/src/
├── gpu_k_opt.rs (MODIFY - owns GPU parts)
└── [GPU-related files]

tests/
└── [all test files] (CREATE/MODIFY)
```

**READ-ONLY**:
```
src/orchestration/routing/ (Worker A's area)
src/orchestration/thermodynamic/ (Worker A's area)
src/active_inference/ (Worker A's area, except when adding kernels)
```

---

### ⚠️  SHARED FILES (Coordinate Before Editing)

**These files both workers might need to edit**:

#### `src/integration/mod.rs`
**Conflict Risk**: MEDIUM
**Protocol**:
- Worker A: Only add exports for routing/thermodynamic
- Worker B: Only add exports for local_llm/production
- Both: Announce in chat before editing

#### `src/gpu/kernel_executor.rs`
**Conflict Risk**: HIGH
**Protocol**:
- **Worker B ONLY** adds kernels
- **Worker A** requests kernels via GitHub issues
- Worker A: Read-only, use existing methods
- Worker B: Add to `kernels` module, register, add methods

#### `Cargo.toml` (main)
**Conflict Risk**: MEDIUM
**Protocol**:
- Worker A: Unlikely to need
- Worker B: May add dependencies for tokenizer, etc.
- Both: Announce dependency changes

#### `src/lib.rs`
**Conflict Risk**: LOW
**Protocol**:
- Both: Only add module declarations
- Coordinate if adding features

---

## COORDINATION WORKFLOW

### Before Editing Shared File:

**Step 1**: Check GitHub
```bash
# See if file is being edited
git log -1 --oneline -- path/to/shared/file.rs
```

**Step 2**: Announce
- Post in team chat: "Editing [file] for [reason], will be done in [time]"
- Wait 5 minutes for objection

**Step 3**: Edit quickly
- Make minimal changes
- Finish in < 30 minutes
- Commit immediately

**Step 4**: Notify
- Push immediately
- Notify in chat: "[file] updated, pull and merge"

### Kernel Request Protocol (Worker A → Worker B):

**Worker A Creates GitHub Issue**:
```markdown
Title: [KERNEL] Digamma function for KSG TE
Label: kernel-request, priority-high

### Purpose
Compute digamma function ψ(x) for Transfer Entropy

### Signature
```rust
pub fn digamma_gpu(&self, n: &[f32]) -> Result<Vec<f32>>
```

### Math
ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²) for x > 10
(Use series expansion for x < 10)

### Priority
HIGH - blocks KSG TE implementation (1.3.3)

### Tests
Test cases:
- ψ(1) ≈ -0.5772 (Euler constant)
- ψ(10) ≈ 2.2518
```

**Worker B Implements**:
```bash
# Add to kernel_executor.rs
# Implement kernel
# Register
# Add method
# Test
# Commit with issue reference

git commit -m "feat: Add digamma GPU kernel

Implements digamma function on GPU for KSG Transfer Entropy.

Closes #123"
git push

# Comment on issue: "Done, available in commit abc123"
```

---

## INTEGRATION POINTS

### Worker A Uses Worker B's Kernels:

```rust
// Worker A code
use crate::gpu::GpuKernelExecutor;

pub fn compute_te_gpu(&self) -> Result<f64> {
    let executor = self.executor.lock().unwrap();

    // Use kernels that Worker B implemented
    let embedded = executor.time_delayed_embedding(&data_gpu)?;
    let digamma_vals = executor.digamma_gpu(&neighbor_counts)?;
    let te = executor.ksg_te_formula(&digamma_vals)?;

    Ok(te)
}
```

**Worker A doesn't**:
- Edit kernel_executor.rs
- Implement kernels
- Register kernels

**Worker A does**:
- Use kernel methods
- Report bugs
- Request new kernels

### Worker B Provides Kernels:

**Checklist for new kernel**:
- [ ] Implement CUDA kernel code
- [ ] Add to `kernels` module
- [ ] Register in `register_standard_kernels()`
- [ ] Add public method to `GpuKernelExecutor`
- [ ] Add unit test
- [ ] Document in code comments
- [ ] Update kernel count
- [ ] Notify Worker A

---

## MERGE STRATEGY

### Daily Integration:

```
worker-a-algorithms ──┐
                      ├──> parallel-development
worker-b-infrastructure ──┘
```

**Every evening**:
1. Both workers create PRs to `parallel-development`
2. Both review (quick check)
3. Merge Worker B first (infrastructure)
4. Then merge Worker A (uses Worker B's kernels)
5. Run integration tests
6. Fix any issues immediately

### Weekly Integration to Master:

**Friday evening**:
```bash
# After both workers merge to parallel-development

git checkout parallel-development
cargo build --release --features cuda
cargo test --all --features cuda

# If all tests pass
git checkout master
git merge parallel-development
git push origin master
```

---

## BLOCKERS & DEPENDENCIES

### Worker A Blocked By Worker B:

**Scenario**: Worker A needs kernel that doesn't exist yet

**Solution**:
1. Create GitHub issue (kernel request)
2. Mark task as blocked: ❌ BLOCKED: waiting for kernel #123
3. Switch to different task
4. Worker B implements kernel
5. Worker A unblocks and continues

### Worker B Blocked By Worker A:

**Scenario**: Worker B doesn't understand algorithm requirements

**Solution**:
1. Create GitHub issue (question)
2. Worker A provides specification
3. Worker B implements
4. Worker A validates

---

## EXAMPLES OF GOOD COORDINATION

### Example 1: Adding Digamma Kernel

**Day 1, 9:00 AM** - Worker A:
```
GitHub Issue #125: [KERNEL] Digamma function
Priority: HIGH, blocks KSG TE
```

**Day 1, 2:00 PM** - Worker B:
```
Commit: "feat: Add digamma GPU kernel (closes #125)"
Comment on issue: "Done, available now"
```

**Day 1, 3:00 PM** - Worker A:
```
git pull origin parallel-development
git merge parallel-development
# Now has digamma kernel
# Continues with KSG TE implementation
```

### Example 2: Coordinating Shared File Edit

**Day 2, 10:00 AM** - Worker A needs to add export to `integration/mod.rs`:
```
[Chat]: "Need to add TE router export to integration/mod.rs, will be done in 5 min"
```

**Day 2, 10:05 AM** - Worker B:
```
[Chat]: "Go ahead, not touching that file today"
```

**Day 2, 10:10 AM** - Worker A:
```
git add src/integration/mod.rs
git commit -m "feat: Export TE router from integration"
git push
[Chat]: "integration/mod.rs updated, safe to use now"
```

---

## TROUBLESHOOTING

### "My build is broken"

**Check**:
1. Did Worker B add new kernels? Pull and rebuild
2. Did you modify Worker B's files? Revert
3. Are you on latest `parallel-development`? Merge it

### "Tests are failing"

**Check**:
1. Run just YOUR tests: `cargo test --lib [your_module]`
2. If YOUR tests pass, but integration fails → coordinate with Worker B
3. If YOUR tests fail → fix before pushing

### "Merge conflicts"

**Check**:
1. Are you editing files outside your ownership? → DON'T
2. Daily merges help → do them every morning
3. Complex conflicts → ask for help

---

## FINAL CHECKLIST

**Worker A - Before Each Commit**:
- [ ] Only edited files in `routing/`, `thermodynamic/`, `active_inference/`
- [ ] Used GPU kernels (didn't implement them)
- [ ] Tests pass
- [ ] Updated task checklist

**Worker B - Before Each Commit**:
- [ ] Only edited files in `local_llm/`, `gpu/`, `production/`
- [ ] New kernels are registered and tested
- [ ] Notified Worker A of new kernels
- [ ] Tests pass
- [ ] Updated task checklist

**Both - Before Daily PR**:
- [ ] Merged latest `parallel-development`
- [ ] Build succeeds
- [ ] GPU tests pass
- [ ] No conflicts
- [ ] Descriptive commit messages

---

**GOAL**: 255 hours of work in 3-4 weeks with ZERO conflicts and MAXIMUM efficiency