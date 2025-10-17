# Contributing to PRISM-AI DoD

## Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-org/PRISM-AI-DoD.git
   cd PRISM-AI-DoD
   ```

2. **Build and Test**
   ```bash
   cd src
   cargo build --release --features constitutional_validation
   cargo test --all-features
   ```

3. **Run Governance Checks**
   ```bash
   cargo run --bin governance_validator
   ```

## Governance Enforcement

### Build-Time Requirements
- Mandatory trait implementation
- 95% test coverage requirement
- Memory limit validation

### Runtime Constraints
- Entropy non-decreasing check
- Convergence monitoring
- GPU utilization >80%
- Automatic violation handling

### Deployment Gates
- Performance benchmarks must pass
- Security audit required
- Constraint validation mandatory

## Code Standards

- Follow Rust conventions
- Document all public APIs
- Include unit tests for new features
- Ensure GPU acceleration compatibility

## Testing

See [Testing Guide](testing.md) for comprehensive testing procedures.