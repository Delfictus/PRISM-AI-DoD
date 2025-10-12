# Worker 1 Development Constitution

**Immutable Laws for Worker 1**

## Article I: File Ownership

YOU SHALL:
- Only edit files assigned to Worker 1
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
cd /home/diddy/Desktop/PRISM-Worker-1
git pull origin worker-1-[branch]
git merge parallel-development
cargo build --features cuda
```

EVENING (5 PM):
```bash
git add -A
git commit -m "feat: [your work]"
git push origin worker-1-[branch]
```

