# Integration Coordinator Cheat Sheet
**Quick Reference for Human Coordinator** | Keep This Open During Integration

---

## Your Role: Message Bus Between 8 AI Workers

You coordinate between specialized Claude Code instances by:
1. Monitoring orchestrator (Terminal 1)
2. Reading notifications when created
3. Switching to relevant worker terminal
4. Prompting that Claude instance
5. Relaying responses back to orchestrator

---

## Quick Workflow

```
Orchestrator pauses → YOU read notification → Switch to Worker X terminal
→ Prompt Worker X → Worker X resolves → Switch back to orchestrator
→ Tell orchestrator to continue
```

**Time per issue**: 15-60 minutes (depending on complexity)

---

## Terminal Layout

```
T1: Worker 0-Alpha (Orchestrator)    T6: Worker 5 (Mission Charlie)
T2: Worker 1 (Time Series/LSTM)      T7: Worker 6 (LLM Advanced)
T3: Worker 2 (GPU Infrastructure)    T8: Worker 7 (Drug/QA Lead)
T4: Worker 3 (PWSA/Apps)             T9: Worker 8 (API/Deployment)
T5: Worker 4 (Finance/GNN)           T10: Dashboard (watch)
```

---

## Prompt Templates (Copy-Paste Ready)

### Generic Template
```
"Worker [X], check your notification file at
/home/diddy/Desktop/PRISM-Worker-[X]/NOTIFICATION.md and resolve
the [issue type] described. Use your [domain] expertise to ensure
[critical requirement] is preserved."
```

### Worker 1 (Time Series/LSTM)
```
"Worker 1, check your notification file. There's an issue with
[LSTM/ARIMA/time-series]. Resolve using your time-series expertise.
Ensure 50-100× LSTM speedup is preserved."
```

### Worker 2 (GPU Infrastructure)
```
"Worker 2, check your notification file. There's a [CUDA/GPU/kernel]
issue. Debug and fix using your GPU expertise. Ensure 80%+ GPU
utilization is maintained."
```

### Worker 3 (PWSA/Applications)
```
"Worker 3, check your notification file. There's an issue with
[PWSA/cybersecurity/applications]. Resolve using your domain expertise.
Ensure <5ms PWSA latency is preserved."
```

### Worker 4 (Finance/GNN)
```
"Worker 4, check your notification file. There's an issue with
[finance/GNN/portfolio]. Resolve using your finance expertise.
Ensure 10-100× GNN speedup is preserved."
```

### Worker 5 (Mission Charlie)
```
"Worker 5, check your notification file. There's an issue with
[Mission Charlie/thermodynamic engine/LLM]. Resolve using your
expertise. Ensure Mission Charlie functionality is preserved."
```

### Worker 6 (LLM Advanced)
```
"Worker 6, check your notification file. There's an issue with
[multi-provider LLM/routing/caching]. Resolve using your LLM
infrastructure expertise. Ensure cache efficiency is maintained."
```

### Worker 7 (Drug Discovery/QA)
```
"Worker 7, check your notification file. There's an issue with
[drug discovery/robotics/QA]. Resolve using your domain expertise.
Also perform QA validation of the integrated system."
```

### Worker 8 (API/Deployment)
```
"Worker 8, check your notification file. There's an issue with
[API server/deployment/endpoints]. Resolve using your deployment
expertise. Ensure all endpoints are properly exposed."
```

### Orchestrator Continue
```
"[Worker X] has resolved the [issue]. [1-2 sentence summary].
Continue the integration."
```

### Cross-Worker Relay
```
"Worker [Y], Worker [X] analyzed the issue and found: [paste analysis].
Resolve this from your [domain] perspective, considering Worker [X]'s
requirements."
```

---

## When Orchestrator Pauses

**Step-by-step**:
1. ✅ Read orchestrator output (Terminal 1)
2. ✅ Note which worker(s) involved
3. ✅ Check notification file if needed: `cat /home/diddy/Desktop/PRISM-Worker-X/NOTIFICATION.md`
4. ✅ Switch to Worker X terminal (Terminal 2-9)
5. ✅ Prompt Worker X using template above
6. ✅ Wait for Worker X to resolve and report
7. ✅ Switch back to Terminal 1 (orchestrator)
8. ✅ Tell orchestrator to continue using template above

---

## Issue Type → Worker Mapping

| Issue Type | Primary Worker | Support Workers |
|------------|----------------|-----------------|
| LSTM/ARIMA/Time-Series | Worker 1 | Worker 2 (GPU) |
| CUDA/GPU Kernels | Worker 2 | Worker 1 (LSTM) |
| PWSA/Cybersecurity | Worker 3 | Worker 7 (QA) |
| Finance/GNN/Portfolio | Worker 4 | Worker 3 (PWSA) |
| Mission Charlie/TE | Worker 5 | Worker 6 (LLM) |
| LLM Routing/Caching | Worker 6 | Worker 5 (MC) |
| Drug Discovery/Robotics | Worker 7 | - |
| API/Deployment | Worker 8 | Worker 7 (QA) |
| Multi-domain/Unknown | Worker 7 (QA Lead) | Analyze first |

---

## Key Files (Quick Access)

```bash
# Dashboard (real-time status)
cat /home/diddy/Desktop/PRISM-Worker-8/INTEGRATION_DASHBOARD.md

# Log (detailed output)
tail -50 /home/diddy/Desktop/PRISM-Worker-8/integration_orchestrator.log

# Notification for Worker X
cat /home/diddy/Desktop/PRISM-Worker-X/NOTIFICATION.md

# Workload plan (detailed tasks)
cat /home/diddy/Desktop/PRISM-AI-DoD/00-Integration-Management/MASTER_INTEGRATION_WORKLOAD_PLAN.md
```

---

## Emergency Commands

```bash
# Rollback last merge
cd /home/diddy/Desktop/PRISM-AI-DoD
./00-Integration-Management/integration_orchestrator.sh --rollback

# Restart from specific phase
./00-Integration-Management/integration_orchestrator.sh --phase 3

# Check orchestrator status
./00-Integration-Management/integration_orchestrator.sh --status
```

---

## Phase-Specific Expectations

| Phase | Workers On-Call | Expected Issues | Your Time |
|-------|-----------------|-----------------|-----------|
| **Phase 2** | W1, W2 | 2-3 LSTM/GPU conflicts | ~3h |
| **Phase 3** | W3, W4, W5 | 4-6 app domain conflicts | ~6h |
| **Phase 4** | W6, W5 | API keys + 3-4 LLM issues | ~5h |
| **Phase 5** | W7, W8 | 3-5 API/deployment issues | ~6h |
| **Phase 6** | W7 (QA) | Full validation + approval | ~12h |

**Total**: ~33 hours over 18 days = ~2 hours/day

---

## Critical Performance Targets (Remind Workers)

- **LSTM Speedup**: 50-100× (Worker 1)
- **GPU Utilization**: >80% (Worker 2)
- **PWSA Latency**: <5ms (Worker 3)
- **GNN Speedup**: 10-100× (Worker 4)
- **Mission Charlie**: Functional (Worker 5)
- **LLM Cache**: Efficient (Worker 6)
- **Drug APIs**: Working (Worker 7)
- **Deployment**: Production-ready (Worker 8)

---

## Troubleshooting Quick Fixes

### Worker lacks context
**Fix**: Provide full context in your prompt including file paths, error messages, and what changed

### Multiple workers needed
**Fix**: Start with Worker 7 (QA Lead) to analyze and recommend specialists

### Orchestrator stuck
**Fix**: Check log for last action, may need to restart from that phase

### Worker solution unclear
**Fix**: Ask worker to explain their resolution in 2-3 sentences before continuing

### Performance regression
**Fix**: Identify which merge caused it (check log timestamps), prompt that worker to investigate

---

## Success Indicators

✅ **Per-task**: Worker resolves issue in <30 min, all tests pass, performance preserved
✅ **Per-phase**: Dashboard shows "✅ COMPLETE", 0 build errors, all metrics met
✅ **Overall**: 108 hours complete in 18 days, all 8 workers merged, production-ready

---

## Your Daily Workflow

**Morning** (30 min):
- Check dashboard for overnight progress
- Review log for any paused issues
- Resume orchestrator if paused

**Midday** (1 hour):
- Actively coordinate between workers
- Resolve 2-3 conflicts/issues
- Monitor orchestrator progress

**Afternoon** (30 min):
- Final check before end of day
- Ensure no workers blocked
- Review tomorrow's expected phase

**Total**: ~2 hours/day active coordination

---

## Quick Mental Model

```
YOU = Message Bus
Orchestrator = Automation Engine (70% of work)
Workers = Specialized Experts (resolve issues you route to them)

Your value: Context switching + routing + relaying
Orchestrator value: Repetitive tasks + validation + logging
Worker value: Domain expertise + complex resolution
```

---

## Start Command (Terminal 1)

```
"Start the integration orchestrator. I'll monitor and coordinate
between specialized workers as needed."
```

Then Worker 0-Alpha runs:
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
./00-Integration-Management/integration_orchestrator.sh
```

---

**Keep this cheat sheet open in a browser/editor while coordinating!**

🚀 **You've got this!**
