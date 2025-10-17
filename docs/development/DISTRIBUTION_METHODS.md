# PRISM-AI Private Distribution Methods

## ⚠️ SECURITY NOTE
- `publish = false` is set in all Cargo.toml files
- This PREVENTS accidental publication to crates.io
- All distribution methods below are PRIVATE

---

## METHOD 1: LOCAL PATH DEPENDENCY

### For Other Projects on Same Machine

**In consuming project's Cargo.toml:**
```toml
[dependencies]
prism-ai = { path = "/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code" }
```

**With specific features:**
```toml
[dependencies]
prism-ai = {
    path = "/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code",
    features = ["cuda", "mission_charlie"]
}
```

**Usage in code:**
```rust
use prism_ai::api::{PrismApi, PrismRequest, PrismResponse};

fn main() {
    let api = PrismApi::new();
    // Use PRISM-AI capabilities
}
```

---

## METHOD 2: GIT REPOSITORY DEPENDENCY

### For Team/Private Distribution

**Step 1: Push to Private Git Repository**
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
git remote add origin git@github.com:YOUR_PRIVATE_REPO/PRISM-AI.git
git push -u origin main
```

**Step 2: In consuming project's Cargo.toml:**
```toml
[dependencies]
# Via SSH (recommended for private repos)
prism-ai = {
    git = "ssh://git@github.com/YOUR_PRIVATE_REPO/PRISM-AI.git",
    branch = "main"
}

# Or via HTTPS with credentials
prism-ai = {
    git = "https://github.com/YOUR_PRIVATE_REPO/PRISM-AI.git",
    branch = "main"
}

# With specific commit (most secure - locks version)
prism-ai = {
    git = "ssh://git@github.com/YOUR_PRIVATE_REPO/PRISM-AI.git",
    rev = "d0cf1fc"  # Use your commit hash
}
```

**For GitHub Private Repo Access:**
- Ensure SSH keys are configured: `ssh -T git@github.com`
- Or use GitHub Personal Access Token for HTTPS

---

## METHOD 3: BINARY DISTRIBUTION

### Distribute Compiled Library File

**Step 1: Build the Library**
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code

# Build optimized release library
cargo build --release --lib --features cuda

# Library location:
# target/release/libprism_ai.rlib (47MB)
```

**Step 2: Package for Distribution**
```bash
# Create distribution package
mkdir -p /home/diddy/Desktop/prism-ai-binary-dist
cd /home/diddy/Desktop/prism-ai-binary-dist

# Copy library file
cp /home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/target/release/libprism_ai.rlib .

# Copy dependencies info
cp /home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/target/release/deps/*.rlib .

# Create README
cat > README.md << 'EOF'
# PRISM-AI Binary Distribution

## Files
- libprism_ai.rlib - Main PRISM-AI library
- deps/*.rlib - Required dependencies

## Usage
Place these files in your project and reference in build:
rustc --extern prism_ai=libprism_ai.rlib -L . your_code.rs
EOF

# Create tarball for distribution
tar -czf prism-ai-binary-v0.1.0.tar.gz *.rlib README.md
```

**Step 3: Using Binary Distribution**

Option A - Direct rustc:
```bash
rustc --edition 2021 \
      --extern prism_ai=/path/to/libprism_ai.rlib \
      -L /path/to/deps \
      main.rs
```

Option B - In Cargo project:
```toml
# .cargo/config.toml
[build]
rustflags = ["-L", "/path/to/prism-ai-libs", "--extern", "prism_ai=/path/to/libprism_ai.rlib"]
```

---

## SECURITY RECOMMENDATIONS

### For Maximum Security:

1. **Path Dependency** (MOST SECURE)
   - No network access required
   - Full source control
   - Direct file system access only

2. **Git Repository** (SECURE WITH PROPER SETUP)
   - Use private repository
   - Enable 2FA on GitHub
   - Use SSH keys, not HTTPS
   - Consider self-hosted Git (GitLab, Gitea)
   - Use specific commit hashes in dependencies

3. **Binary Distribution** (SECURE FOR DEPLOYMENT)
   - No source code exposure
   - Sign the binary with GPG
   - Use checksums for verification
   - Distribute via secure channels only

### Additional Security Measures

**Add to all Cargo.toml files:**
```toml
[package]
publish = false  # Already added - prevents crates.io publication

[package.metadata]
private = true   # Additional marker
distribution = "private-only"
```

**For Git Repository:**
```bash
# Add .gitignore entries
echo "target/" >> .gitignore
echo "*.key" >> .gitignore
echo ".env" >> .gitignore

# Create private distribution notice
cat > DISTRIBUTION_NOTICE.md << 'EOF'
# ⚠️ PRIVATE DISTRIBUTION ONLY

This repository contains proprietary technology.
DO NOT make this repository public.
DO NOT share with unauthorized parties.
EOF
```

---

## VERIFICATION COMMANDS

### Verify Publication is Blocked:
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code
cargo publish --dry-run
# Should output: error: `prism-ai` cannot be published
```

### Verify Library Builds:
```bash
cargo build --release --lib
ls -lh target/release/libprism_ai.rlib
```

### Test Local Path Dependency:
```bash
# In test project
cargo new test-prism --bin
cd test-prism
echo 'prism-ai = { path = "../PRISM-AI-DoD/03-Source-Code" }' >> Cargo.toml
cargo check
```

---

## DISTRIBUTION CHECKLIST

- [x] `publish = false` added to all Cargo.toml files
- [x] Library builds successfully
- [x] Path dependency method documented
- [x] Git repository method documented
- [x] Binary distribution method documented
- [ ] Choose distribution method(s) to use
- [ ] Set up chosen method(s)
- [ ] Test with consuming project
- [ ] Document access credentials/paths for team

---

## REMEMBER

**NEVER** run `cargo publish` without the `--dry-run` flag
**NEVER** make the Git repository public
**ALWAYS** verify `publish = false` is in Cargo.toml