# Worker 2 Quick Reference

## Your Worktree
`/home/diddy/Desktop/PRISM-Worker-2`

## Your Branch
`worker-2-[branch-name]`

## Your Time
~254 hours (7 weeks)

## Daily Commands

**Pull latest**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-2
git pull origin worker-2-*
git merge parallel-development
```

**Build & Test**:
```bash
cargo build --features cuda
cargo test --lib [your_module]
```

**Commit**:
```bash
git add -A
git commit -m "feat: [description]"
git push origin worker-2-*
```

## Your Documentation

All in `.worker-vault/`:
- Constitution/ - Your rules
- Tasks/ - Your work
- Progress/ - Daily tracker
- Reference/ - Full plans

## Need Help?

- Kernel request → GitHub issue [KERNEL]
- Question → GitHub issue [QUESTION]  
- Blocker → GitHub issue [BLOCKER]

Check `.worker-vault/Reference/8_WORKER_ENHANCED_PLAN.md` for complete context.
