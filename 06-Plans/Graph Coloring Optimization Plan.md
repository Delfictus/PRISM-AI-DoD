# Graph Coloring Optimization Plan - Path to World Records

**Created:** 2025-10-08
**Goal:** Beat or match best-known results on DIMACS benchmarks
**Target:** DSJC500-5 from 72 → <48 colors (world record territory)

---

## 🎯 Executive Summary

**Current State:**
- Valid colorings with 0 conflicts ✅
- ~53% above best known (consistent across scales)
- Fast execution (35ms - 3.1s depending on size)

**Optimization Strategy:**
Progressive improvement through 3 phases, each building on the previous:
1. **Quick Wins** (6-8 hours) → Target: 60-65 colors
2. **Advanced Techniques** (10-15 hours) → Target: 50-55 colors
3. **Competitive Push** (20-30 hours) → Target: <48 colors (world record)

**Expected Outcome:**
- Reasonable chance of matching/beating best known on DSJC500-5
- Novel quantum-inspired approach validated
- Publishable results

---

## 📊 Current Baseline

### Results (2025-10-08)

| Instance | Vertices | Best Known | Current | Gap | Gap % |
|----------|----------|------------|---------|-----|-------|
| DSJC500-5 | 500 | 47-48 | 72 | +24-25 | +53% |
| DSJC1000-5 | 1,000 | 82-83 | 126 | +43-44 | +52% |
| C2000-5 | 2,000 | 145 | 223 | +78 | +54% |
| C4000-5 | 4,000 | 259 | 401 | +142 | +55% |

### Analysis

**Strengths:**
- Consistent quality (~53%) across all scales
- Zero conflicts (always valid)
- Fast execution
- Deterministic (given phase state)

**Weaknesses:**
- Single run (no exploration of solution space)
- Simple expansion (only 3 iterations)
- Greedy coloring (no look-ahead or backtracking)
- Fixed parameters (no adaptation)

**Opportunities:**
- 20+ colors improvement seems feasible
- Multiple techniques available
- Room for hybrid approaches
- Parallelization potential

---

## 🚀 Phase 1: Quick Wins (Target: 60-65 colors)

**Duration:** 6-8 hours
**Expected Gain:** 7-12 colors on DSJC500-5
**Risk:** Low (all proven techniques)

### 1.A: Better Phase Expansion (2-3 hours)

**Problem:** Current expansion uses only 3 iterations of averaging
**Impact:** Phase information doesn't fully propagate across graph

**Implementation:**
```rust
// In examples/run_dimacs_official.rs:expand_phase_field()

// Current: 3 iterations
for iteration in 0..3 { ... }

// Improvement: Adaptive iterations based on graph diameter
let n_iterations = (n_vertices as f64).log2().ceil() as usize * 2;
let n_iterations = n_iterations.clamp(10, 30);

for iteration in 0..n_iterations {
    let mut new_phases = expanded_phases.clone();

    for v in 0..n_vertices {
        // Degree-weighted averaging
        let degree = (0..n_vertices)
            .filter(|&u| graph.adjacency[v * n_vertices + u])
            .count() as f64;

        let mut sum_phase = expanded_phases[v] * degree;
        let mut total_weight = degree;

        // Average with neighbors
        for u in 0..n_vertices {
            if graph.adjacency[v * n_vertices + u] {
                sum_phase += expanded_phases[u];
                total_weight += 1.0;
            }
        }

        new_phases[v] = sum_phase / total_weight;
    }

    // Add momentum for faster convergence
    for v in 0..n_vertices {
        expanded_phases[v] = 0.7 * new_phases[v] + 0.3 * expanded_phases[v];
    }
}
```

**Expected Gain:** 5-8 colors
**Why:** Better phase propagation → more meaningful coherence → better color choices

### 1.B: Multiple Random Seeds (1 hour)

**Problem:** Single deterministic run explores only one solution
**Impact:** Missing potentially better colorings

**Implementation:**
```rust
// Add before coloring section in run_dimacs_official.rs

fn try_multiple_seeds(
    graph: &Graph,
    phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
    target_colors: usize,
    n_attempts: usize,
) -> ColoringSolution {
    let mut best_solution = None;
    let mut best_colors = usize::MAX;

    for seed in 0..n_attempts {
        // Perturb phases slightly
        let mut perturbed_pf = phase_field.clone();
        let mut rng = rand::thread_rng();
        let perturbation = 0.1 * (seed as f64 / n_attempts as f64);

        for phase in &mut perturbed_pf.phases {
            let noise: f64 = rng.gen_range(-perturbation..perturbation);
            *phase += noise;
        }

        // Try coloring with perturbed state
        if let Ok(solution) = phase_guided_coloring(
            graph,
            &perturbed_pf,
            kuramoto_state,
            target_colors
        ) {
            if solution.conflicts == 0 && solution.chromatic_number < best_colors {
                best_colors = solution.chromatic_number;
                best_solution = Some(solution);
            }
        }
    }

    best_solution.expect("No valid coloring found")
}

// Use: let solution = try_multiple_seeds(&graph, &expanded_phase_field,
//                                        &expanded_kuramoto, target_colors, 20);
```

**Expected Gain:** 5-10 colors
**Why:** Explores different regions of solution space, takes best result

### 1.C: Smarter Greedy Selection (3-4 hours)

**Problem:** Current greedy picks first available color with max coherence
**Impact:** May make locally optimal but globally suboptimal choices

**Implementation:**
```rust
// Modify src/prct-core/src/coloring.rs:find_phase_coherent_color()

fn find_best_color_with_lookahead(
    vertex: usize,
    coloring: &[usize],
    adjacency: &Array2<bool>,
    phase_field: &PhaseField,
    max_colors: usize,
    lookahead_depth: usize,
) -> Result<usize> {
    let n = coloring.len();

    // Get forbidden colors
    let forbidden: HashSet<usize> = (0..n)
        .filter(|&u| adjacency[[vertex, u]] && coloring[u] != usize::MAX)
        .map(|u| coloring[u])
        .collect();

    // Try top 3 colors with look-ahead
    let mut candidates = Vec::new();

    for color in 0..max_colors {
        if forbidden.contains(&color) {
            continue;
        }

        let base_score = compute_color_coherence_score(vertex, color, coloring, phase_field);

        if lookahead_depth > 0 {
            // Look ahead: how many future vertices would be constrained?
            let constraint_score = estimate_future_constraints(
                vertex, color, coloring, adjacency, phase_field
            );

            candidates.push((color, base_score - 0.3 * constraint_score));
        } else {
            candidates.push((color, base_score));
        }
    }

    if candidates.is_empty() {
        return Err(PRCTError::ColoringFailed(format!(
            "Vertex {} has no available colors", vertex
        )));
    }

    // Pick color with best combined score
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    Ok(candidates[0].0)
}

fn estimate_future_constraints(
    vertex: usize,
    color: usize,
    coloring: &[usize],
    adjacency: &Array2<bool>,
    phase_field: &PhaseField,
) -> f64 {
    let n = coloring.len();
    let mut constraint_count = 0.0;

    // Check uncolored neighbors
    for u in 0..n {
        if coloring[u] == usize::MAX && adjacency[[vertex, u]] {
            // This neighbor would be forbidden from using this color
            let coherence = get_phase_coherence(phase_field, u, vertex);
            if coherence > 0.7 {
                // High coherence = this color would have been good choice
                constraint_count += coherence;
            }
        }
    }

    constraint_count
}
```

**Expected Gain:** 3-5 colors
**Why:** Avoids decisions that limit future options for difficult vertices

### Phase 1 Expected Outcome

**Conservative:** 72 → 65 colors (-10%)
**Realistic:** 72 → 62 colors (-14%)
**Optimistic:** 72 → 60 colors (-17%)

**All techniques are low-risk and independently valuable.**

---

## 🔬 Phase 2: Advanced Techniques (Target: 50-55 colors)

**Duration:** 10-15 hours
**Expected Gain:** 7-12 additional colors
**Risk:** Medium (requires more complex implementation)

### 2.D: Adaptive Color Budget with Binary Search (2-3 hours)

**Problem:** Fixed target_colors may be too high or too low
**Impact:** Algorithm doesn't find minimum chromatic number

**Implementation:**
```rust
fn find_minimum_colors_binary_search(
    graph: &Graph,
    phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
    lower_bound: usize,
    upper_bound: usize,
) -> ColoringSolution {
    let mut best_valid = None;
    let mut low = lower_bound;
    let mut high = upper_bound;

    while low <= high {
        let mid = (low + high) / 2;
        println!("  🔍 Trying {} colors...", mid);

        match phase_guided_coloring(graph, phase_field, kuramoto_state, mid) {
            Ok(solution) if solution.conflicts == 0 => {
                // Valid coloring found
                best_valid = Some(solution.clone());
                high = mid - 1;  // Try fewer colors
                println!("  ✓ Success with {} colors, trying lower...", mid);
            }
            _ => {
                // Failed or has conflicts
                low = mid + 1;  // Need more colors
                println!("  ✗ Failed with {} colors, trying higher...", mid);
            }
        }
    }

    best_valid.expect("No valid coloring in range")
}

// Use after Phase 1 improvements give good solution
let initial_solution = try_multiple_seeds(...);  // From Phase 1
let lower = initial_solution.chromatic_number * 7 / 10;  // 30% below current
let upper = initial_solution.chromatic_number;

let final_solution = find_minimum_colors_binary_search(
    &graph, &expanded_phase_field, &expanded_kuramoto, lower, upper
);
```

**Expected Gain:** 5-10 colors
**Why:** Systematically finds minimum feasible chromatic number

### 2.E: Local Search Refinement (4-6 hours)

**Problem:** Greedy solution may have mergeable color classes
**Impact:** Using more colors than necessary

**Implementation:**
```rust
fn local_search_refinement(
    graph: &Graph,
    initial_solution: &ColoringSolution,
    max_iterations: usize,
) -> ColoringSolution {
    let mut current_colors = initial_solution.colors.clone();
    let mut current_chromatic = initial_solution.chromatic_number;

    for iter in 0..max_iterations {
        // Try to merge color classes
        let mut improved = false;

        for c1 in 0..current_chromatic {
            for c2 in (c1+1)..current_chromatic {
                // Try merging c2 into c1
                if can_merge_colors(graph, &current_colors, c1, c2) {
                    merge_colors(&mut current_colors, c1, c2);
                    current_chromatic -= 1;
                    improved = true;
                    println!("  🔄 Merged colors {} and {}, now using {} colors",
                             c2, c1, current_chromatic);
                    break;
                }
            }
            if improved { break; }
        }

        if !improved {
            // Try recoloring individual vertices
            improved = try_recolor_vertices(graph, &mut current_colors);
        }

        if !improved {
            break;  // No more improvements possible
        }
    }

    ColoringSolution {
        colors: current_colors.clone(),
        chromatic_number: compute_chromatic_number(&current_colors),
        conflicts: count_conflicts(&current_colors, graph),
        quality_score: 1.0,  // Recalculate
        computation_time_ms: 0.0,
    }
}

fn can_merge_colors(graph: &Graph, coloring: &[usize], c1: usize, c2: usize) -> bool {
    let n = graph.num_vertices;

    // Check if any vertex with color c2 is adjacent to vertex with color c1
    for v in 0..n {
        if coloring[v] == c2 {
            for u in 0..n {
                if coloring[u] == c1 && graph.adjacency[v * n + u] {
                    return false;  // Conflict would occur
                }
            }
        }
    }

    true
}

fn try_recolor_vertices(graph: &Graph, coloring: &mut [usize]) -> bool {
    let n = graph.num_vertices;
    let max_color = *coloring.iter().max().unwrap();

    // Try recoloring vertices to reduce max color used
    for v in 0..n {
        if coloring[v] == max_color {
            // Try to recolor to lower color
            for new_color in 0..max_color {
                if is_color_valid(graph, coloring, v, new_color) {
                    coloring[v] = new_color;
                    return true;
                }
            }
        }
    }

    false
}
```

**Expected Gain:** 5-8 colors
**Why:** Refines greedy solution by merging compatible color classes

### 2.F: Increase Pipeline Dimensions (3-4 hours)

**Problem:** Only 20D phase state limits expressiveness
**Impact:** Less information for guiding coloring decisions

**Implementation:**
```rust
// Modify examples/run_dimacs_official.rs

// Current:
let dims = graph.num_vertices.min(20);

// Improvement:
let dims = graph.num_vertices.min(100);  // or even 200

// This requires more GPU memory but provides richer phase information
// May need to tune other parameters (reservoir size, etc.)
```

**Alternative:** Run pipeline multiple times with different parameters, combine results

**Expected Gain:** 3-5 colors
**Why:** Richer phase information → better coherence discrimination

### Phase 2 Expected Outcome

**From Phase 1:** 60-65 colors
**After Phase 2:** 50-55 colors
**Total Improvement:** 72 → 50-55 (-24% to -31%)

---

## 🏆 Phase 3: Competitive Push (Target: <48 colors - World Record)

**Duration:** 20-30 hours
**Expected Gain:** 2-7 additional colors (the hardest improvements)
**Risk:** High (may not reach world record, but valuable insights)

### 3.G: Hybrid with Classical Algorithms (8-10 hours)

**Strategy:** Use best of both quantum-inspired and classical worlds

**Implementation:**
```rust
// Implement DSATUR (classical greedy with saturation degree)
fn dsatur_coloring(graph: &Graph) -> ColoringSolution { ... }

// Implement TabuCol (tabu search)
fn tabucol_refinement(graph: &Graph, initial: &ColoringSolution) -> ColoringSolution { ... }

// Hybrid approach
fn hybrid_coloring(
    graph: &Graph,
    phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
) -> ColoringSolution {
    // 1. Get initial solution from both approaches
    let quantum_sol = phase_guided_coloring(graph, phase_field, kuramoto_state, 100)?;
    let classical_sol = dsatur_coloring(graph);

    // 2. Take better one
    let mut best = if quantum_sol.chromatic_number < classical_sol.chromatic_number {
        quantum_sol
    } else {
        classical_sol
    };

    // 3. Refine with tabu search
    best = tabucol_refinement(graph, &best);

    // 4. Use phase guidance for difficult vertices
    best = phase_guided_refinement(graph, &best, phase_field);

    best
}
```

**Expected Gain:** 3-5 colors
**Why:** Combines strengths of different approaches

### 3.H: Problem-Specific Analysis (6-8 hours)

**Strategy:** Analyze graph structure, adapt algorithm accordingly

**Implementation:**
```rust
fn analyze_graph_structure(graph: &Graph) -> GraphProperties {
    GraphProperties {
        density: compute_density(graph),
        clustering_coefficient: compute_clustering(graph),
        degree_distribution: compute_degree_dist(graph),
        communities: detect_communities(graph),
        clique_number: estimate_clique_number(graph),
    }
}

fn adaptive_coloring_strategy(
    graph: &Graph,
    properties: &GraphProperties,
    phase_field: &PhaseField,
) -> ColoringSolution {
    if properties.has_clear_communities {
        // Color each community separately, then merge
        color_by_communities(graph, &properties.communities, phase_field)
    } else if properties.density > 0.8 {
        // Dense graph: focus on finding large independent sets
        color_dense_graph(graph, phase_field)
    } else {
        // Sparse graph: standard approach works well
        standard_phase_guided_coloring(graph, phase_field)
    }
}
```

**Expected Gain:** 2-4 colors
**Why:** Exploits specific graph properties for better results

### 3.I: Iterative Multi-Stage Refinement (6-12 hours)

**Strategy:** Multiple passes with different strategies

**Implementation:**
```rust
fn multi_stage_coloring(
    graph: &Graph,
    phase_field: &PhaseField,
    kuramoto_state: &KuramotoState,
) -> ColoringSolution {
    // Stage 1: Quick greedy for easy vertices
    let stage1 = greedy_coloring_easy_vertices(graph, phase_field);

    // Stage 2: Phase-guided for medium difficulty
    let stage2 = refine_with_phase_guidance(graph, &stage1, phase_field);

    // Stage 3: Intensive search for hardest vertices
    let stage3 = intensive_search_hard_vertices(graph, &stage2, phase_field);

    // Stage 4: Global optimization
    let stage4 = global_optimization(graph, &stage3);

    stage4
}

fn greedy_coloring_easy_vertices(graph: &Graph, phase_field: &PhaseField) -> PartialColoring {
    // Color vertices with high phase coherence first
    // These are "easy" - clear color choices
    ...
}

fn intensive_search_hard_vertices(
    graph: &Graph,
    partial: &PartialColoring,
    phase_field: &PhaseField,
) -> ColoringSolution {
    // For remaining uncolored vertices:
    // - Try all color combinations for small sets
    // - Use backtracking
    // - Phase guidance for tie-breaking
    ...
}
```

**Expected Gain:** 2-5 colors
**Why:** Applies appropriate level of effort to each vertex based on difficulty

### Phase 3 Expected Outcome

**From Phase 2:** 50-55 colors
**After Phase 3:** 47-52 colors
**Best Case:** <48 colors (world record territory!)

---

## 📊 Projected Timeline & Results

### Cumulative Progress

| Phase | Duration | DSJC500-5 Target | Confidence |
|-------|----------|------------------|------------|
| Baseline | - | 72 colors | ✅ 100% |
| Phase 1 | 6-8 hours | 60-65 colors | 🟢 90% |
| Phase 2 | 10-15 hours | 50-55 colors | 🟡 70% |
| Phase 3 | 20-30 hours | <48 colors | 🟠 40% |

### Risk Assessment

**Phase 1: Low Risk**
- All techniques proven
- Independent improvements
- Minimal complexity

**Phase 2: Medium Risk**
- More complex implementations
- May hit diminishing returns
- Still well-understood techniques

**Phase 3: High Risk**
- Approaching theoretical limits
- May not reach world record
- Valuable even if <48 not achieved
- Novel insights regardless of outcome

---

## 🎯 Success Criteria

### Minimum Success (Phase 1 Complete)
- [ ] <65 colors on DSJC500-5
- [ ] Maintained 0 conflicts
- [ ] Sub-second performance
- [ ] All techniques documented

### Target Success (Phase 2 Complete)
- [ ] <55 colors on DSJC500-5
- [ ] <100 colors on DSJC1000-5
- [ ] Competitive with published results
- [ ] Novel quantum-inspired insights

### Breakthrough Success (Phase 3 Complete)
- [ ] <48 colors on DSJC500-5 (world record potential)
- [ ] Validated novel approach
- [ ] Publishable results
- [ ] Clear path to further improvements

---

## 📝 Implementation Priority

### Sprint 1 (Week 1): Phase 1 Quick Wins
1. Better expansion (2-3h) - Do first, highest impact/effort ratio
2. Multiple seeds (1h) - Easy parallelization
3. Smarter greedy (3-4h) - Most complex of Phase 1

### Sprint 2 (Week 2): Phase 2 Advanced
4. Adaptive color budget (2-3h) - Builds on Phase 1
5. Increase pipeline dimensions (3-4h) - Infrastructure change
6. Local search (4-6h) - Most complex of Phase 2

### Sprint 3 (Weeks 3-4): Phase 3 Competitive
7. Hybrid approaches (8-10h) - New implementations needed
8. Problem-specific (6-8h) - Analysis + adaptation
9. Multi-stage refinement (6-12h) - Orchestration layer

---

## 🔬 Experimental Protocol

### For Each Optimization

1. **Baseline**: Run current version 10× on DSJC500-5, record results
2. **Implement**: Add new technique
3. **Test**: Run new version 10× on DSJC500-5
4. **Compare**: Statistical significance test (t-test)
5. **Validate**: Verify on DSJC1000-5, C2000-5
6. **Document**: Record results, insights, parameters
7. **Commit**: Git commit with clear message

### Metrics to Track

- **Chromatic number** (primary)
- Conflicts (must stay 0)
- Computation time
- Memory usage
- Success rate (% runs that complete)
- Standard deviation across runs

---

## 📚 References

### Classical Algorithms to Study
- DSATUR (Brélaz 1979)
- TabuCol (Hertz & de Werra 1987)
- RLF (Recursive Largest First)
- Iterated Greedy
- Simulated Annealing for coloring

### DIMACS Resources
- Benchmark instances: http://mat.gsia.cmu.edu/COLOR/instances.html
- Best known results: http://www.info.univ-angers.fr/pub/porumbel/graphs/
- Competition results: COLOR02/COLOR04

### Our Codebase
- Current coloring: `src/prct-core/src/coloring.rs`
- Expansion: `examples/run_dimacs_official.rs:12-148`
- Pipeline: `src/integration/unified_platform.rs`
- Results: `DIMACS_RESULTS.md`

---

**Status:** 📋 Ready to Execute
**Next Action:** Begin Phase 1 Sprint 1 (Better Expansion)
**Expected Completion:** Phase 1 within 1 week, Phase 2 within 2 weeks, Phase 3 within 4-5 weeks
**Ultimate Goal:** Match or beat 47-48 colors on DSJC500-5
