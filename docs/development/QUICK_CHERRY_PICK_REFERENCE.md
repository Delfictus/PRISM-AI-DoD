# QUICK CHERRY-PICK REFERENCE

**For workers who just need the commands fast**

---

## When to Cherry-Pick

✅ **DO** cherry-pick when:
- Feature is complete and tested
- Build passes: `cargo check --lib --features cuda`
- Tests pass: `cargo test --lib <module>`
- Other workers need it

❌ **DON'T** cherry-pick when:
- WIP (work in progress)
- Tests failing
- Code doesn't build
- Experimental code

---

## The 5-Step Process

### 1. Find Your Commit
```bash
git log --oneline -10
# Pick the commit hash (e.g., a1b2c3d)
```

### 2. Switch to Deliverables
```bash
git checkout deliverables
git pull origin deliverables
```

### 3. Cherry-Pick
```bash
git cherry-pick a1b2c3d
```

### 4. Push
```bash
git push origin deliverables
```

### 5. Return to Your Branch
```bash
git checkout worker-X-branch
```

**Done!** ✅

---

## If You Get Conflicts

```bash
# Edit the conflicting file
nano <file>

# Usually keep BOTH changes (yours + existing)

# Mark as resolved
git add <file>

# Continue
git cherry-pick --continue

# Push
git push origin deliverables
```

---

## If You Need to Undo

**Before pushing**:
```bash
git reset --hard HEAD~1
```

**After pushing**:
```bash
git revert <commit-hash>
git push origin deliverables
```

---

## Multiple Commits?

### Option A: Pick them all
```bash
git cherry-pick hash1 hash2 hash3
```

### Option B: Squash first (recommended)
```bash
# On your branch
git rebase -i HEAD~3  # Change "pick" to "squash" for commits 2-3
git log -1  # Get new hash
git checkout deliverables
git cherry-pick <new-hash>
git push origin deliverables
git checkout worker-X-branch
```

---

## Don't Forget!

After publishing, update the log:
```bash
nano /home/diddy/Desktop/PRISM-AI-DoD/.worker-deliverables.log
```

Add:
```
[2025-10-12 15:30] Worker X: Feature Name - AVAILABLE
Commit: a1b2c3d
Files: src/path/to/file.rs
```

---

## Full Guide

For detailed explanations, troubleshooting, and examples:
**See: `/home/diddy/Desktop/PRISM-AI-DoD/CHERRY_PICK_GUIDE.md`**

---

## Common Questions

**Q: Do I cherry-pick every commit?**
A: No, only COMPLETE features

**Q: What if it conflicts?**
A: Edit the file, keep both changes, `git add`, `git cherry-pick --continue`

**Q: Can I undo?**
A: Yes, `git reset --hard HEAD~1` (before push) or `git revert <hash>` (after push)

**Q: Should I merge instead?**
A: No, cherry-pick is cleaner (only completed features)

**Q: I have 10 commits for one feature?**
A: Squash them first with `git rebase -i HEAD~10`

---

**Remember**: Test before cherry-picking! ✅
