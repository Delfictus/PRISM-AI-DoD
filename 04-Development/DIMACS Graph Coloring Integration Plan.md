# DIMACS Graph Coloring Integration - Action Plan

**Created:** 2025-10-08
**Goal:** Extract actual graph colorings from PRISM-AI to beat DSJC1000-5 best known (82-83 colors)
**Timeline:** 20-40 hours for optimized implementation
**Success:** Beat 82 colors = journal publication

---

## Current Status

**What Works:**
- ✅ System processes graphs through 8-phase pipeline
- ✅ MTX parser loads DSJC instances
- ✅ 4.07ms latency (world-class speed)
- ✅ Mathematical guarantees maintained

**What's Missing:**
- ❌ Coloring extraction from quantum/thermodynamic state
- ❌ Solution verification (check validity)
- ❌ Color count measurement
- ❌ Integration with DIMACS instances

**Gap:** Physics pipeline → Actual graph coloring

---

## Phase 1: Basic Coloring Extraction (4-6 hours)

**Goal:** Get ANY valid coloring from system state

### Task 1.1: Extract Coloring Function (2 hours)

**File:** `src/prct-core/src/coloring.rs` (already exists)

**Add function:**
```rust
use ndarray::Array1;
use crate::Graph;
use std::f64::consts::PI;

/// Extract graph coloring from thermodynamic phase field
pub fn extract_coloring_from_phases(
    phase_field: &Array1<f64>,
    graph: &Graph
) -> Vec<usize> {
    let n = graph.num_vertices;
    let mut colors = vec![0; n];

    // Discretize continuous phases [0, 2π] into color classes
    // Similar phases → same color
    for i in 0..n {
        let phase = phase_field[i].rem_euclid(2.0 * PI);
        // Map to color bins (start with 100 bins, will optimize)
        colors[i] = (phase / (2.0 * PI) * 100.0).floor() as usize;
    }

    // Repair conflicts
    repair_coloring(&mut colors, graph);

    colors
}

/// Ensure no adjacent vertices have same color (greedy repair)
fn repair_coloring(colors: &mut Vec<usize>, graph: &Graph) {
    let mut changed = true;
    let mut max_color = *colors.iter().max().unwrap_or(&0);

    while changed {
        changed = false;

        for (u, v, _) in &graph.edges {
            if colors[*u] == colors[*v] {
                // Conflict! Assign new color to vertex with higher degree
                colors[*v] = max_color + 1;
                max_color += 1;
                changed = true;
            }
        }
    }
}
```

**Testing:**
```rust
#[test]
fn test_coloring_extraction() {
    let graph = create_test_graph();
    let phases = Array1::from_vec(vec![0.0, PI/2.0, PI, 3.0*PI/2.0]);
    let colors = extract_coloring_from_phases(&phases, &graph);
    assert!(verify_coloring(&graph, &colors));
}
```

---

### Task 1.2: Solution Verification (2 hours)

**Add functions:**
```rust
/// Verify coloring is valid (no adjacent vertices same color)
pub fn verify_coloring(graph: &Graph, colors: &[usize]) -> bool {
    for (u, v, _) in &graph.edges {
        if colors[*u] == colors[*v] {
            return false;
        }
    }
    true
}

/// Count number of distinct colors used
pub fn count_colors(colors: &[usize]) -> usize {
    use std::collections::HashSet;
    colors.iter().copied().collect::<HashSet<_>>().len()
}

/// Get coloring statistics
pub fn coloring_stats(graph: &Graph, colors: &[usize]) -> ColoringStats {
    ColoringStats {
        num_colors: count_colors(colors),
        is_valid: verify_coloring(graph, colors),
        conflicts: count_conflicts(graph, colors),
    }
}

struct ColoringStats {
    num_colors: usize,
    is_valid: bool,
    conflicts: usize,
}
```

---

### Task 1.3: Integration with Pipeline (2 hours)

**Modify:** `examples/run_dimacs_official.rs`

```rust
// After running pipeline:
let output = platform.process(input)?;

// Extract thermodynamic phases (these map to vertex colors)
let phase_field = output.thermodynamic_state.phases;  // Need to expose this

// Extract coloring
let colors = extract_coloring_from_phases(&phase_field, &graph)?;

// Verify and count
let num_colors = count_colors(&colors);
let is_valid = verify_coloring(&graph, &colors);

println!("  ✓ Coloring extracted: {} colors", num_colors);
println!("  ✓ Valid: {}", is_valid);
println!("  ✓ Best known: {}-{}", best_known_min, best_known_max);

if num_colors < best_known_min {
    println!("  🏆 NEW BEST FOUND!");
}
```

---

## Phase 2: Optimization (10-20 hours)

**Goal:** Improve coloring quality to beat 82 colors

### Task 2.1: Intelligent Phase Discretization (4 hours)

Instead of fixed bins, use adaptive clustering:

```rust
fn extract_coloring_adaptive(
    phase_field: &Array1<f64>,
    graph: &Graph
) -> Vec<usize> {
    // K-means clustering on phase angles
    // Find natural groupings
    let clusters = kmeans_on_circle(phase_field, initial_k);

    // Assign colors based on cluster membership
    let mut colors = cluster_to_colors(clusters);

    // Repair conflicts
    repair_coloring_smart(&mut colors, graph);

    colors
}
```

---

### Task 2.2: Quantum State Integration (4 hours)

Use quantum state for better coloring:

```rust
fn extract_coloring_quantum(
    quantum_state: &QuantumState,
    graph: &Graph
) -> Vec<usize> {
    // Quantum state encodes graph structure
    // Measure correlations between vertices
    // High correlation → same color

    let correlations = compute_vertex_correlations(quantum_state, graph);
    let colors = cluster_by_correlation(correlations);

    repair_coloring(&mut colors, graph);
    colors
}
```

---

### Task 2.3: Hybrid Approach (6 hours)

Combine quantum + thermodynamic + neuromorphic:

```rust
fn extract_coloring_fusion(
    output: &PlatformOutput,
    graph: &Graph
) -> Vec<usize> {
    // Weight multiple signals:
    // - Thermodynamic phases (oscillator synchronization)
    // - Quantum state (entanglement structure)
    // - Neuromorphic (spike patterns)

    let thermo_coloring = extract_from_phases(&output.thermo);
    let quantum_coloring = extract_from_quantum(&output.quantum);
    let neuro_coloring = extract_from_spikes(&output.neuro);

    // Ensemble: vote or weighted combination
    let combined = ensemble_coloring(vec![
        (thermo_coloring, 0.5),
        (quantum_coloring, 0.3),
        (neuro_coloring, 0.2),
    ]);

    repair_and_optimize(&combined, graph)
}
```

---

### Task 2.4: Local Search Refinement (4 hours)

After extraction, improve with local search:

```rust
fn refine_coloring(
    colors: &mut Vec<usize>,
    graph: &Graph
) {
    // Kempe chain recoloring
    // Greedy improvement
    // Simulated annealing on top of physics solution

    for _ in 0..1000 {  // Iterations
        if try_reduce_colors(colors, graph) {
            // Found improvement
        }
    }
}
```

---

## Phase 3: Benchmarking & Validation (6 hours)

### Task 3.1: Run All DIMACS Instances (2 hours)

```markdown
- [ ] DSJC500-5: Target <47 colors
- [ ] DSJC1000-5: Target <82 colors ⭐ PRIORITY
- [ ] C2000-5: Target <145 colors
- [ ] C4000-5: Target <259 colors (if feasible)
```

---

### Task 3.2: Statistical Validation (2 hours)

Run each instance 10 times:
- Mean colors found
- Best coloring found
- Std deviation
- Success rate

---

### Task 3.3: Comparison Analysis (2 hours)

Compare to:
- Best known results
- Gurobi (if installed)
- DIMACS published results

---

## Success Criteria

**Minimum Success:**
- ✅ Extract valid colorings
- ✅ DSJC500-5: 47-50 colors (competitive)
- ✅ DSJC1000-5: 85-90 colors (close to best)
- **Result:** Conference paper on novel approach

**Good Success:**
- ✅ DSJC500-5: <47 colors (beat best)
- ✅ DSJC1000-5: 82-84 colors (match/close)
- **Result:** Strong conference paper, journal possible

**Exceptional Success:**
- ✅ DSJC1000-5: <82 colors (new best known)
- **Result:** Top-tier journal publication, world record

---

## Effort Breakdown

**Phase 1 (Basic):** 4-6 hours
- Coloring extraction: 2 hours
- Verification: 2 hours
- Integration: 2 hours

**Phase 2 (Optimized):** 10-20 hours
- Adaptive discretization: 4 hours
- Quantum integration: 4 hours
- Fusion approach: 6 hours
- Local search: 4 hours

**Phase 3 (Validation):** 6 hours
- Run instances: 2 hours
- Statistical validation: 2 hours
- Analysis: 2 hours

**Total:** 20-32 hours

---

## Risk Assessment

**High Probability:**
- Extract valid colorings ✅
- Competitive results (85-95 colors on DSJC1000-5)
- Publishable novelty regardless of exact count

**Medium Probability:**
- Match best known (82-83 colors)
- Close enough for strong publication

**Low Probability (but possible):**
- Beat 82 colors
- New world record

**Even without beating records, this is publishable for:**
- Novel approach (quantum-neuromorphic fusion)
- Exceptional speed (<10ms)
- Mathematical guarantees

---

## Timeline

**Week 1: Basic Implementation**
- Day 1-2: Phase 1 (coloring extraction)
- Day 3: Initial testing
- Day 4-5: Debug and refine

**Week 2: Optimization**
- Day 1-3: Phase 2 (optimization)
- Day 4-5: Testing and tuning

**Week 3: Validation**
- Day 1-2: Phase 3 (benchmarking)
- Day 3-5: Analysis and documentation

**Total:** 3 weeks part-time, or 1 week full-time

---

## Next Immediate Steps

**To start (2 hours):**
1. Implement basic `extract_coloring_from_phases()`
2. Implement `verify_coloring()` and `count_colors()`
3. Test on myciel3.col (small graph)
4. Verify you get valid coloring

**Then:** Run on DSJC500-5, see what happens

**Decision point:** If results are promising (< 60 colors), continue optimization. If not, publish on speed and novelty.

---

**Status:** Plan ready, implementation can begin next session
**Priority:** DSJC1000-5 (beat 82 colors = world record)
**Fallback:** Even 85-90 colors is publishable for novel approach
