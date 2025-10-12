# CPU Fallback Elimination Plan

**Status**: IN PROGRESS
**Violations Detected**: 93 instances
**Files Affected**: 24 files
**Priority**: CRITICAL - Blocking all other tasks

## Strategy

### Phase 1: Delete Old/Disabled Files (Immediate)
These files are not used and contain CPU fallback:
- `src/gpu/gpu_enabled_old.rs` - OLD, replaced by gpu_enabled.rs
- `src/gpu/simple_gpu_v2.rs` - Unused variant
- `src/gpu/gpu_real.rs` - Superseded

### Phase 2: Fix Core Infrastructure (High Priority)
1. `src/integration/adapters.rs` - 14 violations
2. `src/gpu/tensor_ops.rs` - 10 violations
3. `src/gpu/kernel_launcher.rs` - 10 violations
4. `src/gpu/memory_manager.rs` - 8 violations
5. `src/gpu/memory_simple.rs` - 7 violations

### Phase 3: Fix PWSA Modules (High Priority)
1. `src/pwsa/gpu_kernels.rs` - 6 violations
2. `src/pwsa/active_inference_classifier.rs` - 1 violation

### Phase 4: Fix Active Inference (High Priority)
1. `src/active_inference/gpu_policy_eval.rs` - 6 violations
2. `src/active_inference/gpu_inference.rs` - 2 violations

### Phase 5: Fix Remaining Modules
- Statistical mechanics
- Quantum simulation
- Transfer entropy
- CMA integration
- Others

## Approach

For each file:
1. **Remove** `#[cfg(not(feature = "cuda"))]` blocks entirely
2. **Delete** CPU fallback implementations
3. **Require** GPU with `.expect("GPU REQUIRED")`
4. **Test** compilation with `--features cuda`
5. **Verify** no CPU code remains

## Completion Criteria

- [ ] Zero `#[cfg(not(feature = "cuda"))]` patterns
- [ ] Governance engine passes
- [ ] Compiles with `--features cuda` only
- [ ] All tests pass
- [ ] System fails gracefully without GPU (not fallback)

---

**STARTING ELIMINATION NOW**