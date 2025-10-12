# Worker Auto-Sync System - Complete Guide

**Feature**: Workers automatically pull dependencies when available and gracefully wait when blocked

**Status**: ✅ OPERATIONAL

---

## What Is Auto-Sync?

The Auto-Sync System allows workers to:
- ✅ **Automatically detect** when dependencies are available
- ✅ **Automatically pull** required deliverables from other workers
- ✅ **Gracefully wait** when dependencies not ready
- ✅ **Validate integration** after pulling dependencies
- ✅ **Provide clear status** on what's ready and what's blocking

**Result**: Workers never need to manually track dependencies - the system handles it automatically!

---

## How It Works

### **When You Start a Worker Session**

```bash
cd /home/diddy/Desktop/PRISM-Worker-<X>
./worker_start.sh <worker-number>
```

**The script automatically**:
1. ✅ Pulls latest from worker's own branch
2. ✅ Merges latest integration (from integration-staging)
3. ✅ **Checks dependencies** (are they available?)
4. ✅ **Auto-pulls** any available dependencies
5. ✅ Validates build after integration
6. ✅ Reports status: READY, WAITING, or BLOCKED

### **If Dependencies Available**
```
🎉 Status: READY TO WORK

Your environment is ready:
  ✅ Worker branch up to date
  ✅ Integration merged
  ✅ Dependencies synced
  ✅ Build validated

📋 Next steps:
  1. Review your tasks
  2. Start development
  3. Run tests
```

### **If Dependencies Not Available**
```
⏳ Status: WAITING FOR DEPENDENCIES

Some dependencies are not yet available.

📋 What to do:
  1. Work on non-dependent features (suggestions provided)
  2. When ready to check again, run: ./worker_start.sh <X>
  3. Or prompt: 'Worker <X> ready to continue'

💡 Your dependencies will auto-pull when available!
```

---

## Usage Examples

### **Example 1: Worker 5 (Blocked, then Unblocked)**

#### **Scenario**: Worker 5 needs Worker 1's time series module

**First attempt (Week 2 - dependency not ready)**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-5
./worker_start.sh 5
```

**Output**:
```
🔍 Worker 5 Dependencies:

  ❌ Worker 1: Time series module NOT READY - BLOCKING
  → Expected: Worker 1 Week 3 completion

💡 Worker 5 Status: BLOCKED
   Meanwhile, can work on:
   • Replica exchange (no dependency)
   • Advanced energy functions
   • Bayesian learning

⏳ Waiting for Worker 1 time series...
   I'll automatically pull when available - prompt me when ready!
```

**Worker 5 works on alternative features...**

**Later attempt (Week 3 - dependency now available)**:
```bash
./worker_start.sh 5
```

**Output**:
```
🔍 Worker 5 Dependencies:

  ✅ Worker 1: Time series forecasting AVAILABLE
  📥 Auto-pulling time series module...
  ✅ Deliverables merged successfully
  ✅ Build validation PASSED

🎉 UNBLOCKED! Time series module integrated!

🚀 Worker 5 Status: READY
   Can now implement:
   • LLM cost forecasting
   • Proactive model selection
   • Time series integration with thermodynamic consensus

🎉 Status: READY TO WORK
```

**Result**: Worker 5 is now unblocked and can proceed!

---

### **Example 2: Worker 3 (Waiting for Pixel Kernels)**

```bash
cd /home/diddy/Desktop/PRISM-Worker-3
./worker_start.sh 3
```

**If pixel kernels not ready**:
```
🔍 Worker 3 Dependencies:

  ⏳ Worker 2: Pixel kernels NOT READY
  → Expected Week 3

💡 Worker 3 Status: WAITING
   Meanwhile, can work on:
   • PWSA frame-level processing (no pixel dependency)
   • Finance portfolio optimization
   • Prepare pixel integration code

   I'll auto-pull pixel kernels when available - just prompt when ready!
```

**When pixel kernels available** (you run `./worker_start.sh 3` again):
```
  ✅ Worker 2: Pixel kernels AVAILABLE
  📥 Auto-pulling pixel kernels...
  ✅ Deliverables merged successfully
  ✅ Build validation PASSED

🚀 Worker 3 Status: READY
   Can proceed with pixel processing integration!
```

---

### **Example 3: Worker 2 (No Dependencies)**

```bash
cd /home/diddy/Desktop/PRISM-Worker-2
./worker_start.sh 2
```

**Output**:
```
✅ Worker 2 has NO dependencies
   GPU infrastructure is the foundation

🚀 Worker 2 Status: READY - can proceed with all GPU work

🎉 Status: READY TO WORK
```

**Result**: Worker 2 never blocked - can always proceed!

---

## Worker Dependency Matrix

| Worker | Depends On | Auto-Pull Behavior |
|--------|------------|-------------------|
| **Worker 1** | Worker 2 (GPU kernels) | ✅ Auto-pulls base + time series kernels |
| **Worker 2** | Nothing | ✅ Always READY (foundation) |
| **Worker 3** | Worker 2 (Pixel kernels) | ✅ Auto-pulls when available, suggests alternatives if waiting |
| **Worker 4** | Workers 1, 2 (base) | ✅ Auto-pulls when both ready |
| **Worker 5** | Worker 1 (Time series) | ✅ Auto-pulls when available, suggests alternatives, **BLOCKING** |
| **Worker 6** | Worker 2 (GPU) | ✅ Auto-pulls base GPU kernels |
| **Worker 7** | Worker 1 (Time series) | ✅ Auto-pulls when available, suggests alternatives, **BLOCKING** |
| **Worker 8** | Workers 1-7 (most) | ✅ Auto-pulls when 2+ core workers ready |

---

## How to Use As User (You)

### **Starting a Worker**

When you want to work on a specific worker or prompt them to continue:

```bash
# Option 1: Run startup script
cd /home/diddy/Desktop/PRISM-Worker-<X>
./worker_start.sh <worker-number>

# Option 2: Just prompt me
"Worker 5, ready to continue work"
"Start Worker 3 with latest dependencies"
```

**The system will**:
1. Check if dependencies available
2. Auto-pull if available
3. Tell worker if READY or WAITING
4. Suggest alternative work if blocked

### **When Worker is Blocked**

**You don't need to do anything special!**

The worker will:
- ✅ Tell you they're waiting
- ✅ Suggest alternative features to work on
- ✅ Automatically pull dependencies when available next time you prompt

**Example prompt**:
```
You: "Worker 5, continue your work"

Worker 5: "I'm currently blocked waiting for Worker 1's time series module.
           Meanwhile, I can work on replica exchange or advanced energy functions.

           Would you like me to:
           1. Work on replica exchange (no dependency)?
           2. Wait until you tell me Worker 1 is complete?
           3. Check now if Worker 1 deliverable is available?"
```

### **When Dependency Becomes Available**

**Option 1: Worker checks automatically**
```
You: "Worker 1 completed time series - Worker 5, continue"

System runs: ./worker_start.sh 5
- ✅ Detects time series available
- ✅ Auto-pulls time series module
- ✅ Validates integration
- ✅ Worker 5 now READY

Worker 5: "Great! Time series module integrated.
           I can now implement LLM cost forecasting!"
```

**Option 2: Manual notification**
```
You: "Worker 5, Worker 1 published time series deliverable"

Worker 5: [runs auto-sync]
"Checking dependencies... ✅ Available! Pulling now..."
[auto-merges deliverables]
"Integration complete! Ready to implement cost forecasting."
```

---

## Scripts Reference

### **`worker_start.sh <worker-number>`**
**Location**: Each worker directory
**Purpose**: Complete worker startup with auto-sync

**What it does**:
1. Pull latest from worker branch
2. Merge integration-staging
3. Run auto-sync (check/pull dependencies)
4. Validate build
5. Report status

**Usage**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-5
./worker_start.sh 5
```

### **`worker_auto_sync.sh <worker-number>`**
**Location**: Each worker directory
**Purpose**: Check and pull dependencies only

**What it does**:
1. Check if dependencies available
2. Auto-pull from deliverables branch
3. Validate integration
4. Report READY, WAITING, or BLOCKED

**Usage** (standalone):
```bash
cd /home/diddy/Desktop/PRISM-Worker-5
./worker_auto_sync.sh 5
```

### **`check_dependencies.sh <worker-number>`**
**Location**: Each worker directory
**Purpose**: Check dependency status without pulling

**What it does**:
1. Check if dependencies available
2. Show status and suggestions
3. **Does NOT auto-pull** (read-only)

**Usage**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-5
./check_dependencies.sh 5
```

---

## Integration with Your Workflow

### **As User Prompting Workers**

**Before Auto-Sync**:
```
You: "Worker 5, implement cost forecasting"
Worker 5: "I need Worker 1's time series module first.
           Please publish that dependency."
You: [manually check if Worker 1 done]
You: [manually tell Worker 5 when ready]
You: "Worker 5, Worker 1 is done, pull dependencies"
Worker 5: [you manually run git commands]
```

**With Auto-Sync**:
```
You: "Worker 5, implement cost forecasting"
Worker 5: [auto-runs startup script]
         "Time series module not available yet.
          Meanwhile working on replica exchange.
          I'll auto-check dependencies when you prompt me next."

[Later...]
You: "Worker 5, continue"
Worker 5: [auto-runs startup script]
         [auto-detects time series available]
         [auto-pulls dependencies]
         "Time series integrated! Implementing cost forecasting now."
```

### **Worker AI Behavior**

**When you prompt a worker, they will**:
1. ✅ Run `./worker_start.sh <X>` automatically
2. ✅ Check dependencies
3. ✅ Auto-pull if available
4. ✅ Report status to you
5. ✅ Either proceed with work OR suggest alternatives if blocked

**You never need to**:
- ❌ Manually check if dependencies ready
- ❌ Manually tell workers to pull deliverables
- ❌ Run git commands yourself
- ❌ Track which worker is blocked by which dependency

**The system handles all of this automatically!**

---

## Troubleshooting

### **"Merge conflicts during auto-pull"**

**Cause**: Deliverable conflicts with worker's code

**Solution**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-<X>

# Auto-sync detected conflict and aborted
# Manual merge needed
git merge origin/deliverables

# Resolve conflicts in affected files
# Then:
git add <resolved-files>
git commit -m "fix: resolve deliverable merge conflicts"
```

**Prevention**: Workers should coordinate on shared files (see INTEGRATION_PROTOCOL.md)

### **"Build validation failed after auto-pull"**

**Cause**: Dependency has integration issues

**Solution**:
```bash
# Check build errors
cd 03-Source-Code
cargo check --features cuda

# If dependency issue, report to dependency worker
# If your code issue, fix and re-test
```

### **"Worker says WAITING but dependency should be available"**

**Cause**: Deliverable not published yet or log not updated

**Solution**:
```bash
# Check deliverables log manually
cat /home/diddy/Desktop/PRISM-AI-DoD/.worker-deliverables.log

# If dependency truly available, publishing worker should:
cd /home/diddy/Desktop/PRISM-Worker-<X>
git checkout deliverables
# Cherry-pick commits
# Update log
# Push
```

---

## Summary

### **What Auto-Sync Gives You**

✅ **Zero manual dependency tracking** - System handles it
✅ **Automatic unblocking** - Workers pull dependencies when ready
✅ **Clear status** - Always know if READY, WAITING, or BLOCKED
✅ **Suggested alternatives** - Workers never idle when blocked
✅ **Build validation** - Catch integration issues immediately
✅ **Graceful waiting** - Workers work on alternatives until unblocked

### **How to Use**

1. **Start worker**: `./worker_start.sh <X>` (or just prompt me)
2. **System auto-syncs** dependencies
3. **Worker reports** READY or WAITING
4. **If READY** → proceed with work
5. **If WAITING** → work on alternatives, auto-checks next prompt

### **Key Commands**

- `./worker_start.sh <X>` - Complete startup with auto-sync
- `./worker_auto_sync.sh <X>` - Just check/pull dependencies
- `./check_dependencies.sh <X>` - Check status without pulling

---

**Status**: ✅ **AUTO-SYNC SYSTEM OPERATIONAL**

Workers will now automatically manage their dependencies!

---

Generated by Integration System | Auto-Sync Module
