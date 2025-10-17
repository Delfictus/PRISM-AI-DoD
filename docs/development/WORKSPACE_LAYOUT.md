# PRISM-AI Workspace Layout
**Worker Directories, Branches, and File Locations**

---

## Complete Workspace Structure

### Worker 0-Alpha (Integration Manager)

**Directory**: `/home/diddy/Desktop/PRISM-AI-DoD`
**Branch**: `deliverables`
**Role**: Runs orchestrator, performs all merges to deliverables branch

**What Worker 0-Alpha Does**:
- ✅ Runs the integration orchestrator script
- ✅ Merges all worker branches into `deliverables`
- ✅ Runs builds and tests in `03-Source-Code/`
- ✅ Creates dashboard and logs in `/home/diddy/Desktop/PRISM-Worker-8/`
- ✅ Does NOT work in worker directories or branches

**Terminal 1 Commands**:
```bash
# Start location
cd /home/diddy/Desktop/PRISM-AI-DoD

# Should be on deliverables branch
git branch --show-current  # Should show: deliverables

# Run orchestrator
./00-Integration-Management/integration_orchestrator.sh
```

---

### Worker 1 (Time Series/LSTM)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-1`
**Branch**: `worker-1-te-thermo` (or `worker-1-ai-core`)
**Worktree**: Linked to main repo via git worktree

**What Worker 1 Works On**:
- ✅ Files: `03-Source-Code/src/time_series/*`
- ✅ Files: `03-Source-Code/src/active_inference/*`
- ✅ Tests: `03-Source-Code/tests/*lstm*`, `*arima*`, `*active_inference*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-1/NOTIFICATION.md`

**When Worker 1 Is Needed**:
- Phase 2 merge conflicts
- LSTM/ARIMA performance issues
- Time-series integration questions

**Worker 1 Pushes To**: `origin/worker-1-te-thermo` (or `worker-1-ai-core`)
**Worker 1 Does NOT**: Touch deliverables branch directly

---

### Worker 2 (GPU Infrastructure)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-2`
**Branch**: `worker-2-gpu-infra`
**Worktree**: Linked to main repo via git worktree

**What Worker 2 Works On**:
- ✅ Files: `03-Source-Code/src/gpu/*`
- ✅ Files: `03-Source-Code/src/cuda/*`
- ✅ Tests: `03-Source-Code/tests/*gpu*`, `*cuda*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-2/NOTIFICATION.md`

**When Worker 2 Is Needed**:
- Phase 1-2 GPU kernel issues
- CUDA compilation errors
- GPU utilization problems

**Worker 2 Pushes To**: `origin/worker-2-gpu-infra`
**Worker 2 Does NOT**: Touch deliverables branch directly

---

### Worker 3 (PWSA/Applications)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-3`
**Branch**: `worker-3-apps-domain1`
**Worktree**: Linked to main repo via git worktree

**What Worker 3 Works On**:
- ✅ Files: `03-Source-Code/src/applications/pwsa/*`
- ✅ Files: `03-Source-Code/src/applications/cybersecurity/*`
- ✅ Tests: `03-Source-Code/tests/*pwsa*`, `*cybersecurity*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-3/NOTIFICATION.md`

**When Worker 3 Is Needed**:
- Phase 3 PWSA merge conflicts
- PWSA latency issues (<5ms target)
- Cybersecurity domain questions

**Worker 3 Pushes To**: `origin/worker-3-apps-domain1`
**Worker 3 Does NOT**: Touch deliverables branch directly

---

### Worker 4 (Finance/GNN)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-4`
**Branch**: `worker-4-apps-domain2`
**Worktree**: Linked to main repo via git worktree

**What Worker 4 Works On**:
- ✅ Files: `03-Source-Code/src/applications/finance/*`
- ✅ Files: `03-Source-Code/src/gnn/*`
- ✅ Tests: `03-Source-Code/tests/*finance*`, `*gnn*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-4/NOTIFICATION.md`

**When Worker 4 Is Needed**:
- Phase 3 finance merge conflicts
- GNN solver issues (10-100× speedup target)
- Portfolio optimization questions

**Worker 4 Pushes To**: `origin/worker-4-apps-domain2`
**Worker 4 Does NOT**: Touch deliverables branch directly

---

### Worker 5 (Mission Charlie/TE Advanced)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-5`
**Branch**: `worker-5-te-advanced`
**Worktree**: Linked to main repo via git worktree

**What Worker 5 Works On**:
- ✅ Files: `03-Source-Code/src/mission_charlie/*`
- ✅ Files: `03-Source-Code/src/thermodynamic/*`
- ✅ Tests: `03-Source-Code/tests/*mission_charlie*`, `*thermodynamic*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-5/NOTIFICATION.md`

**When Worker 5 Is Needed**:
- Phase 3 Mission Charlie merge conflicts
- Thermodynamic engine issues
- LLM integration with Mission Charlie

**Worker 5 Pushes To**: `origin/worker-5-te-advanced`
**Worker 5 Does NOT**: Touch deliverables branch directly

---

### Worker 6 (LLM Advanced)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-6`
**Branch**: `worker-6-llm-advanced`
**Worktree**: Linked to main repo via git worktree

**What Worker 6 Works On**:
- ✅ Files: `03-Source-Code/src/llm/*`
- ✅ Files: `03-Source-Code/src/llm_router/*`
- ✅ Tests: `03-Source-Code/tests/*llm*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-6/NOTIFICATION.md`

**When Worker 6 Is Needed**:
- Phase 4 LLM merge conflicts
- Multi-provider routing issues
- LLM cache efficiency problems

**Worker 6 Pushes To**: `origin/worker-6-llm-advanced`
**Worker 6 Does NOT**: Touch deliverables branch directly

---

### Worker 7 (Drug Discovery/QA Lead)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-7`
**Branch**: `worker-7-drug-robotics`
**Worktree**: Linked to main repo via git worktree

**What Worker 7 Works On**:
- ✅ Files: `03-Source-Code/src/drug_discovery/*`
- ✅ Files: `03-Source-Code/src/robotics/*`
- ✅ Tests: `03-Source-Code/tests/*` (all tests - QA lead)
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-7/NOTIFICATION.md`

**When Worker 7 Is Needed**:
- Phase 5 drug discovery merge conflicts
- Phase 6 full QA validation
- Any cross-domain issues (QA lead role)

**Worker 7 Pushes To**: `origin/worker-7-drug-robotics`
**Worker 7 Does NOT**: Touch deliverables branch directly

---

### Worker 8 (API/Deployment)

**Directory**: `/home/diddy/Desktop/PRISM-Worker-8`
**Branch**: `worker-8-finance-deploy`
**Worktree**: Linked to main repo via git worktree

**What Worker 8 Works On**:
- ✅ Files: `03-Source-Code/src/api/*`
- ✅ Files: `03-Source-Code/src/server/*`
- ✅ Tests: `03-Source-Code/tests/*api*`, `*server*`
- ✅ Notification: `/home/diddy/Desktop/PRISM-Worker-8/NOTIFICATION.md`
- ✅ **Special**: Dashboard and logs stored here

**When Worker 8 Is Needed**:
- Phase 5 API server merge conflicts
- Deployment configuration issues
- Endpoint exposure problems

**Worker 8 Pushes To**: `origin/worker-8-finance-deploy`
**Worker 8 Does NOT**: Touch deliverables branch directly

**Special Files Created By Orchestrator**:
- `/home/diddy/Desktop/PRISM-Worker-8/INTEGRATION_DASHBOARD.md`
- `/home/diddy/Desktop/PRISM-Worker-8/integration_orchestrator.log`

---

## Critical Rules for Worker File Management

### ✅ DO: Workers Work in Their Own Directories

**Example - Worker 4 fixing a conflict**:

```bash
# Worker 4 Terminal (Terminal 5)
cd /home/diddy/Desktop/PRISM-Worker-4

# Check current branch
git branch --show-current  # Should show: worker-4-apps-domain2

# Fix conflict in their files
vim 03-Source-Code/src/applications/finance/advanced_finance.rs

# Commit to THEIR branch
git add 03-Source-Code/src/applications/finance/advanced_finance.rs
git commit -m "fix: Resolve merge conflict in advanced finance"

# Push to THEIR branch
git push origin worker-4-apps-domain2
```

### ❌ DON'T: Workers Never Touch Deliverables Branch

**Workers should NEVER**:
- ❌ `cd /home/diddy/Desktop/PRISM-AI-DoD` (that's Worker 0-Alpha's domain)
- ❌ `git checkout deliverables` (only Worker 0-Alpha uses this)
- ❌ `git merge` anything to deliverables (orchestrator does this)
- ❌ Work in another worker's directory

### ✅ DO: Worker 0-Alpha Operates from Main Repo

**Worker 0-Alpha (Integration Manager)**:

```bash
# Worker 0-Alpha Terminal (Terminal 1)
cd /home/diddy/Desktop/PRISM-AI-DoD

# ALWAYS on deliverables branch
git branch --show-current  # Should show: deliverables

# Runs orchestrator which merges worker branches
./00-Integration-Management/integration_orchestrator.sh

# Orchestrator does:
# - git checkout deliverables
# - git merge worker-1-te-thermo
# - git merge worker-2-gpu-infra
# - git merge worker-3-apps-domain1
# - etc.
```

### ❌ DON'T: Worker 0-Alpha Never Works in Worker Directories

**Worker 0-Alpha should NEVER**:
- ❌ `cd /home/diddy/Desktop/PRISM-Worker-X`
- ❌ Work on worker branches directly
- ❌ Fix conflicts in worker directories (workers do that)

---

## Integration Workflow: Where Files Go

### Phase 2 Example: Worker 1 Merge

**Before Integration**:
```
/home/diddy/Desktop/PRISM-Worker-1/03-Source-Code/src/time_series/lstm_gpu.rs
├── Branch: worker-1-te-thermo
└── Has Worker 1's LSTM implementation

/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/src/time_series/
├── Branch: deliverables
└── Does NOT have Worker 1's changes yet
```

**After Orchestrator Merge**:
```
/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/src/time_series/lstm_gpu.rs
├── Branch: deliverables
└── NOW has Worker 1's LSTM implementation (merged from worker-1-te-thermo)

/home/diddy/Desktop/PRISM-Worker-1/03-Source-Code/src/time_series/lstm_gpu.rs
├── Branch: worker-1-te-thermo
└── Still has Worker 1's implementation (unchanged)
```

**Result**: Worker 1's code now in `deliverables` branch at main repo.

### If Conflict Occurs

**Orchestrator detects conflict**:
```
[ERROR] Merge conflict in 03-Source-Code/src/time_series/lstm_gpu.rs
[INFO] Notifying Worker 1...
[PAUSE] Waiting for Worker 1 resolution
```

**YOU (human coordinator)**:
1. Switch to Terminal 2 (Worker 1)
2. Prompt Worker 1 to fix conflict

**Worker 1 resolves IN THEIR DIRECTORY**:
```bash
# Worker 1 Terminal
cd /home/diddy/Desktop/PRISM-Worker-1

# Fix conflict in THEIR copy
vim 03-Source-Code/src/time_series/lstm_gpu.rs

# Commit to THEIR branch
git add 03-Source-Code/src/time_series/lstm_gpu.rs
git commit -m "fix: Resolve integration conflict"
git push origin worker-1-te-thermo
```

**YOU tell Worker 0-Alpha**:
"Worker 1 resolved the conflict. Continue integration."

**Orchestrator retries merge**:
```bash
# Back in main repo (Worker 0-Alpha's domain)
cd /home/diddy/Desktop/PRISM-AI-DoD
git merge worker-1-te-thermo  # Now succeeds
```

---

## Output Files: Where They Go

### Logs and Dashboard (Worker 8 Directory)

**Created by orchestrator**:
```
/home/diddy/Desktop/PRISM-Worker-8/INTEGRATION_DASHBOARD.md
/home/diddy/Desktop/PRISM-Worker-8/integration_orchestrator.log
```

**Why Worker 8?**: Convention - Worker 8 is deployment/integration lead

### Notification Files (Each Worker Directory)

**Created by orchestrator when assistance needed**:
```
/home/diddy/Desktop/PRISM-Worker-1/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-2/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-3/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-4/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-5/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-6/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-7/NOTIFICATION.md
/home/diddy/Desktop/PRISM-Worker-8/NOTIFICATION.md
```

### Build Artifacts (Main Repo Only)

**Build/test happens in main repo**:
```
/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/target/
├── Generated by: cargo build (orchestrator runs this)
├── Location: Main repo only (not worker directories)
└── Branch: deliverables
```

**Workers do NOT build** - orchestrator does all builds in main repo.

---

## Summary: Who Works Where

| Entity | Directory | Branch | What They Do |
|--------|-----------|--------|--------------|
| **Worker 0-Alpha** | `/home/diddy/Desktop/PRISM-AI-DoD` | `deliverables` | Runs orchestrator, merges branches, builds/tests |
| **Worker 1** | `/home/diddy/Desktop/PRISM-Worker-1` | `worker-1-te-thermo` | Fixes conflicts in time-series code |
| **Worker 2** | `/home/diddy/Desktop/PRISM-Worker-2` | `worker-2-gpu-infra` | Fixes conflicts in GPU code |
| **Worker 3** | `/home/diddy/Desktop/PRISM-Worker-3` | `worker-3-apps-domain1` | Fixes conflicts in PWSA code |
| **Worker 4** | `/home/diddy/Desktop/PRISM-Worker-4` | `worker-4-apps-domain2` | Fixes conflicts in finance code |
| **Worker 5** | `/home/diddy/Desktop/PRISM-Worker-5` | `worker-5-te-advanced` | Fixes conflicts in Mission Charlie code |
| **Worker 6** | `/home/diddy/Desktop/PRISM-Worker-6` | `worker-6-llm-advanced` | Fixes conflicts in LLM code |
| **Worker 7** | `/home/diddy/Desktop/PRISM-Worker-7` | `worker-7-drug-robotics` | Fixes conflicts in drug discovery, QA validation |
| **Worker 8** | `/home/diddy/Desktop/PRISM-Worker-8` | `worker-8-finance-deploy` | Fixes conflicts in API code; hosts logs |

---

## Quick Verification Commands

### Worker 0-Alpha (Terminal 1)
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
pwd  # Should show: /home/diddy/Desktop/PRISM-AI-DoD
git branch --show-current  # Should show: deliverables
```

### Worker 1 (Terminal 2)
```bash
cd /home/diddy/Desktop/PRISM-Worker-1
pwd  # Should show: /home/diddy/Desktop/PRISM-Worker-1
git branch --show-current  # Should show: worker-1-te-thermo (or worker-1-ai-core)
```

### Worker 2 (Terminal 3)
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
pwd  # Should show: /home/diddy/Desktop/PRISM-Worker-2
git branch --show-current  # Should show: worker-2-gpu-infra
```

**Repeat pattern for Workers 3-8...**

---

## Answer to Your Question

**Q**: "Does this document specify what worktree and branch worker 0 alpha should be in and where he and other workers should be and where they should create or push files they work on?"

**A**:

### Worker 0-Alpha:
- **Directory**: `/home/diddy/Desktop/PRISM-AI-DoD`
- **Branch**: `deliverables`
- **Pushes to**: `origin/deliverables` (orchestrator does this automatically)
- **Works on**: Orchestration only (no code editing, just merging/building/testing)

### Workers 1-8:
- **Directory**: `/home/diddy/Desktop/PRISM-Worker-X`
- **Branch**: `worker-X-branch-name` (their specialized branch)
- **Push to**: `origin/worker-X-branch-name`
- **Work on**: Only their domain-specific code in their directory
- **Do NOT**: Touch deliverables branch or other worker directories

### The Orchestrator Script:
- **Runs from**: `/home/diddy/Desktop/PRISM-AI-DoD`
- **Operates on**: `deliverables` branch
- **Merges from**: All worker branches (`worker-1-te-thermo`, etc.)
- **Builds in**: `/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/`
- **Logs to**: `/home/diddy/Desktop/PRISM-Worker-8/`

---

**Everything is specified and ready!** Worker 0-Alpha just needs to run:

```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
./00-Integration-Management/integration_orchestrator.sh
```

And the orchestrator handles all the directory/branch management automatically.
