# Worker 2 Development Constitution

**Immutable Laws for Worker 2**

## Article I: File Ownership

YOU SHALL:
- Only edit files assigned to Worker 2
- Request kernels from Worker 2 via GitHub issues
- Coordinate shared file edits
- Follow daily sync protocol

YOU SHALL NOT:
- Edit files owned by other workers
- Modify kernel_executor.rs (Worker 2 only)
- Skip daily merges from parallel-development

## Article II: GPU Acceleration

ALL computational code SHALL:
- Use GPU kernels (not CPU loops)
- Request new kernels if needed (don't implement yourself unless you're Worker 2)
- Verify GPU execution with tests
- Maintain 95%+ GPU utilization for compute

## Article III: Testing

YOU SHALL:
- Test after every significant change
- Maintain 90%+ coverage for your modules
- Run full build before pushing
- Fix broken tests before EOD

## Article IV: Daily Protocol

MORNING (9 AM):
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
git pull origin worker-2-[branch]
git merge parallel-development
cargo build --features cuda
```

EVENING (5 PM):
```bash
git add -A
git commit -m "feat: [your work]"
git push origin worker-2-[branch]
```


## Article V: Governance Enforcement

BEFORE ALL WORK:
- Governance engine runs automatically via worker_start.sh
- Checks file ownership, dependencies, build hygiene
- **VIOLATION = IMMEDIATE BLOCKING** until resolved

YOU SHALL:
- Accept governance verdicts
- Fix violations immediately
- Not bypass governance checks
- Report issues to Worker 0-Alpha if governance incorrect

GOVERNANCE RULES:
1. ✅ Only edit files you own
2. ✅ Have required dependencies before proceeding
3. ✅ Code must build before committing
4. ✅ Use GPU for all compute
5. ✅ Commit daily with proper messages
6. ✅ Follow integration protocol
7. ✅ Use auto-sync system

## Article VI: Auto-Sync System

YOU SHALL:
- Use `./worker_start.sh 2` to begin each session
- Allow automatic dependency pulling
- Wait gracefully when dependencies not ready
- Work on alternative features when blocked

YOU SHALL NOT:
- Manually track dependencies
- Skip auto-sync checks
- Proceed when governance blocks
- Bypass automatic integration

## Article VII: Deliverable Publishing

WHEN FEATURES COMPLETE:
- Publish to deliverables branch immediately
- Update .worker-deliverables.log
- Update DELIVERABLES.md status
- Notify dependent workers

WORKER 2 SPECIFIC:
- **Week 2**: Time series kernels published (CRITICAL - unblocks Worker 1)
- **Week 3**: Pixel kernels published (CRITICAL - unblocks Worker 3)
