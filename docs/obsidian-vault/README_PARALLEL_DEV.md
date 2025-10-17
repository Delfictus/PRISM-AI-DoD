# Parallel Development Setup - Quick Reference

**Two developers, zero conflicts, maximum efficiency**

---

## 🚀 QUICK START

### Worker A (Algorithm Developer):
```bash
cd /home/<user>/Desktop/PRISM-AI-DoD
git checkout parallel-development
git checkout -b worker-a-algorithms
cat .obsidian-vault/WORKER_A_TASKS.md  # Your task list
```

### Worker B (Infrastructure Developer):
```bash
cd /home/<user>/Desktop/PRISM-AI-DoD
git checkout parallel-development
git checkout -b worker-b-infrastructure
cat .obsidian-vault/WORKER_B_TASKS.md  # Your task list
```

---

## 📋 DOCUMENTATION

1. **PARALLEL_WORK_GUIDE.md** - Complete work division and workflow
2. **WORKER_A_TASKS.md** - Detailed tasks for Algorithm Developer
3. **WORKER_B_TASKS.md** - Detailed tasks for Infrastructure Developer
4. **QUICK_START_PARALLEL.md** - Daily workflow and commands
5. **CONFLICT_PREVENTION.md** - File ownership and coordination
6. **PRODUCTION_UPGRADE_PLAN.md** - Overall technical plan

---

## 📊 WORK DIVISION

**Worker A** - 125 hours:
- Transfer Entropy Router (40h) - Full KSG implementation
- Thermodynamic Consensus (35h) - Advanced energy + replica exchange
- Active Inference (30h) - Hierarchical + policy search
- Documentation (20h)

**Worker B** - 130 hours:
- Local LLM (80h) - GGUF, KV-cache, BPE, Tensor Cores
- GPU Optimization (30h) - Fused kernels, async streams
- Production Features (40h) - Testing, monitoring, docs

---

## 🎯 FILE OWNERSHIP (CRITICAL)

**Worker A Owns**:
- `src/orchestration/routing/`
- `src/orchestration/thermodynamic/`
- `src/active_inference/`

**Worker B Owns**:
- `src/orchestration/local_llm/`
- `src/gpu/`
- `src/production/`
- `tests/`

**Shared** (coordinate):
- `src/integration/mod.rs`
- `src/gpu/kernel_executor.rs` (Worker B adds, Worker A uses)
- `Cargo.toml`

---

## 🔄 DAILY WORKFLOW

**Morning** (9:00 AM):
```bash
git checkout parallel-development && git pull
git checkout worker-a-algorithms  # or worker-b
git merge parallel-development
```

**Evening** (5:00 PM):
```bash
git add -A
git commit -m "feat: [what you did]"
git push origin worker-a-algorithms
# Create PR to parallel-development
```

**Merge PRs**: After both workers push, merge to `parallel-development`

---

## ⚡ COORDINATION

**Worker A needs GPU kernel** → Create GitHub issue with [KERNEL REQUEST]
**Worker B needs algorithm spec** → Create GitHub issue with [QUESTION]
**Shared file edit** → Announce in chat first
**Blocked** → Mark in task list, switch tasks

---

## ✅ SUCCESS METRICS

**After 3-4 weeks**:
- Full KSG Transfer Entropy ✅
- Advanced thermodynamic consensus ✅
- Production-ready LLM inference ✅
- Tensor Core optimization ✅
- Complete test coverage ✅
- Enterprise deployment ready ✅

---

## 📞 GETTING STARTED

**Read in order**:
1. This file (README_PARALLEL_DEV.md)
2. QUICK_START_PARALLEL.md
3. Your worker file (WORKER_A or WORKER_B)
4. CONFLICT_PREVENTION.md
5. Start working!

---

**Branch**: `parallel-development`
**Your Branch**: `worker-a-algorithms` or `worker-b-infrastructure`
**Integration**: Daily PRs to `parallel-development`
**Production**: Weekly merges to `master`

**Questions?** Check QUICK_START_PARALLEL.md or create GitHub issue.