# PRISM-AI GPU Governance Vault

**Purpose**: Enforce GPU-only implementation through automated governance
**Authority**: SUPREME - NO OVERRIDES
**Status**: ACTIVE

## Structure

```
.obsidian-vault/
├── Constitution/          # Immutable laws
│   └── GPU_CONSTITUTION.md
├── Enforcement/          # Governance engine
│   ├── governance_engine.sh
│   └── compliance_report_*.md
├── Progress/             # Status tracking
│   └── CURRENT_STATUS.md
├── Tasks/                # Task templates
│   └── *.md
└── README.md             # This file
```

## Quick Start

### Run Governance Engine

```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
./.obsidian-vault/Enforcement/governance_engine.sh
```

### Run with Auto-Commit

```bash
./.obsidian-vault/Enforcement/governance_engine.sh --commit
```

### Check Current Status

```bash
cat .obsidian-vault/Progress/CURRENT_STATUS.md
```

## Governance Engine Features

The engine automatically:

✅ **Enforces Constitution**
- Scans for prohibited CPU fallback patterns
- Blocks commits with violations
- Requires GPU-only code

✅ **Builds & Tests**
- Compiles with `--features cuda` only
- Runs GPU verification tests
- Verifies performance standards

✅ **Tracks Progress**
- Updates task completion status
- Calculates progress percentages
- Generates compliance reports

✅ **Commits & Pushes**
- Auto-commits compliant code
- Pushes to repository
- Maintains audit trail

## Constitutional Rules

### ❌ FORBIDDEN
- CPU fallback code
- `#[cfg(not(feature = "cuda"))]` blocks
- Placeholder CPU implementations
- Optional GPU patterns with CPU else branches

### ✅ REQUIRED
- All computations on GPU
- GPU context initialization
- Direct kernel execution
- Performance >1 GFLOPS

## Compliance Workflow

1. **Write GPU code** - Use actual kernels
2. **Run governance engine** - `./governance_engine.sh`
3. **Fix violations** - If any detected
4. **Verify tests pass** - GPU execution confirmed
5. **Auto-commit** - `./governance_engine.sh --commit`

## Progress Tracking

Current progress is automatically maintained in:
- `Progress/CURRENT_STATUS.md` - Live status
- `Enforcement/compliance_report_*.md` - Historical reports

## Task Completion Criteria

A task is **COMPLETE** when:
1. ✅ ALL CPU code replaced with GPU kernels
2. ✅ Compiles with `cargo build --features cuda`
3. ✅ Tests pass with `cargo test --features cuda`
4. ✅ NO prohibited patterns remain
5. ✅ GPU kernel execution verified
6. ✅ Performance meets >1 GFLOPS threshold

## Integration with Development

### Pre-Commit Hook (Optional)

```bash
#!/bin/bash
# .git/hooks/pre-commit
/home/diddy/Desktop/PRISM-AI-DoD/.obsidian-vault/Enforcement/governance_engine.sh
```

### CI/CD Integration

```yaml
# .github/workflows/governance.yml
name: GPU Governance
on: [push, pull_request]
jobs:
  enforce:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v2
      - name: Run Governance Engine
        run: ./.obsidian-vault/Enforcement/governance_engine.sh
```

## Reporting

### View Latest Compliance Report

```bash
ls -t .obsidian-vault/Enforcement/compliance_report_*.md | head -1 | xargs cat
```

### Check Logs

```bash
tail -f .obsidian-vault/Enforcement/compliance.log
```

## Constitution Authority

The GPU Constitution has **ABSOLUTE AUTHORITY**:
- Cannot be overridden
- Cannot be disabled
- Cannot be bypassed
- Enforcement is mandatory

See: `Constitution/GPU_CONSTITUTION.md`

## Emergency Procedures

### If Governance Blocks You

1. **DO NOT** try to bypass
2. **FIX** the violation
3. **RUN** governance engine again
4. **VERIFY** compliance

### If Tests Fail

1. Check GPU availability: `nvidia-smi`
2. Verify CUDA: `nvcc --version`
3. Rebuild: `cargo clean && cargo build --features cuda`
4. Re-run governance engine

## Architecture

The governance engine is **self-enforcing**:

```
User Code Change
       ↓
Governance Engine Scan
       ↓
   Violations? → YES → BLOCK & REPORT
       ↓ NO
Compile (CUDA only)
       ↓
   Success? → NO → BLOCK & REPORT
       ↓ YES
Run GPU Tests
       ↓
   Pass? → NO → BLOCK & REPORT
       ↓ YES
Update Progress
       ↓
Generate Report
       ↓
[Optional] Commit & Push
       ↓
✅ APPROVED
```

## Key Principles

1. **GPU-ONLY** - No CPU fallback ever
2. **Automatic** - No manual intervention needed
3. **Enforced** - Cannot be bypassed
4. **Transparent** - All actions logged
5. **Comprehensive** - Checks everything

## Support

The governance engine is **self-documenting** and **self-enforcing**.

If you need to modify enforcement rules:
1. Read the Constitution first
2. Propose changes via governance engine modifications
3. Ensure changes strengthen (not weaken) GPU requirements

---

**GPU-ONLY. NO EXCEPTIONS. NO COMPROMISES.**

*Governance Engine v1.0*
*Adopted: 2025-10-11*