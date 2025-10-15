# WEEK 2: INTEGRATION ERROR FIXING - ACTION PLAN

**Status**: Pulled latest, 162 errors confirmed, beginning systematic fixes

## ERROR BREAKDOWN

Total: **162 errors**

### Priority Categories:
1. **E0599 - Method Not Found**: 75 errors (46%) - BIGGEST CATEGORY
2. **E0432 - Unresolved Imports**: 22 errors (14%)
3. **E0560 - Struct Fields**: 17 errors (10%)
4. **E0277 - Trait Bounds**: 12 errors (7%)
5. **E0609 - Field Not Found**: 9 errors (6%)
6. **Others**: 27 errors (17%)

## CRITICAL IMPORT ERRORS TO FIX FIRST (E0432 - 22 errors)

### Cluster 1: CudaDevice API Changes (4 occurrences)
- `cudarc::driver::CudaDevice` → Likely changed to `CudaContext` or removed
- `cudarc::driver::LaunchAsync` → API change
- **Files**: gpu_neuromorphic.rs, gpu_monitoring.rs, gpu_docking.rs

### Cluster 2: Missing Local LLM Types (2 occurrences)
- `LLMMetrics`, `AttentionAnalyzer`, `TransferEntropyLLM` not exported
- **Files**: gpu_transformer.rs (both assistant & orchestration)
- **FIX**: Export from local_llm/mod.rs

### Cluster 3: Worker 4 Modules Not Found (4 occurrences)
- `QuantumApproximateCache` - quantum_cache module
- `GeometricManifoldOptimizer` - geometric_manifold module
- `QuantumVotingConsensus` - wrong path
- `TransferEntropyRouter` - wrong export name

### Cluster 4: Worker 3 Consensus (2 occurrences)
- `ThermodynamicConsensus` not exported properly

### Cluster 5: Other Missing (10 occurrences)
- `chemcore::prelude` - chemistry crate
- `rand::distributions::Poisson` → Use `rand_distr::Poisson`
- `ConditionalTe` vs `ConditionalTE` (case mismatch)
- `hamiltonian` module missing
- quantum_mlir types missing
- `SimpleTokenizer` not exported

## SYSTEMATIC FIX PLAN

### Phase 1: Quick Wins (Est: -15 errors, 1 hour)
1. ✅ Fix case mismatch: `ConditionalTe` → `ConditionalTE`
2. ✅ Fix rand imports: `rand::distributions` → `rand_distr::`
3. ✅ Export RedundancyMeasure (already flagged)
4. ✅ Fix doc comment error in gpu_llm_inference.rs

### Phase 2: Export Missing Types (Est: -20 errors, 2 hours)
5. Export LLMMetrics, AttentionAnalyzer, TransferEntropyLLM
6. Export ThermodynamicConsensus properly
7. Export QuantumVotingConsensus from correct location
8. Export TransferEntropyRouter (vs TransferEntropy)

### Phase 3: Create Missing Modules (Est: -10 errors, 2-3 hours)
9. Create geometric_manifold module stub (if Worker 4 didn't deliver)
10. Check quantum_cache module (Worker 4 should have this)
11. Create hamiltonian module stub if missing
12. Handle chemcore prelude issue

### Phase 4: Fix CudaDevice API (Est: -5 errors, 1 hour)
13. Replace CudaDevice with correct cudarc API
14. Fix LaunchAsync usage

### Phase 5: Method Stubs (Est: -40 errors, 4-6 hours)
15. Analyze E0599 method errors
16. Create stub implementations for missing methods
17. Wire integration points

### Phase 6: Struct Fields (Est: -26 errors, 3-4 hours)
18. Add missing struct fields (E0560, E0609)
19. Match worker expectations

### Phase 7: Trait Bounds (Est: -12 errors, 2-3 hours)
20. Fix trait constraint issues
21. Implement missing trait impls

### Phase 8: Final Cleanup (Est: -34 errors, 3-4 hours)
22. Type mismatches
23. Ownership issues
24. Final build and test

## ESTIMATED TIMELINE

- **Day 1 (Today)**: Phases 1-3 → 162 to ~120 errors (-42)
- **Day 2**: Phases 4-5 → 120 to ~80 errors (-40)
- **Day 3**: Phases 6-7 → 80 to ~40 errors (-40)
- **Day 4**: Phase 8 → 40 to 0 errors (-40)

**Target**: Clean build in 4 days

## SAFETY RULES (STRICTLY FOLLOWED)

❌ NO commenting out worker code
❌ NO deleting implementations
❌ NO removing imports
✅ YES creating missing types
✅ YES exporting needed types
✅ YES implementing stubs
✅ YES asking when confused

---
Generated: $(date)
Status: Ready to begin Phase 1
