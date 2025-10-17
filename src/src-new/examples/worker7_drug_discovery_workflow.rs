//! Worker 7: End-to-End Drug Discovery Workflow
//!
//! Production-ready example demonstrating Active Inference-based molecular optimization
//! using Worker 7's drug discovery capabilities.
//!
//! This example shows:
//! 1. Molecular library initialization and descriptor calculation
//! 2. Chemical space exploration using information theory
//! 3. Active learning for experiment design
//! 4. Binding affinity optimization
//! 5. Expected Information Gain (EIG) calculation for next experiments
//!
//! Constitution: Worker 7 - Drug Discovery & Robotics
//! Time: Production example (5 hours quality enhancement allocation)

use prism_ai::applications::{
    DrugDiscoveryController,
    DrugDiscoveryConfig,
    MolecularInformationMetrics,
    information_metrics_optimized::OptimizedExperimentInformationMetrics,
};
use ndarray::{Array1, Array2};
use anyhow::Result;
use std::time::Instant;

/// Represents a molecular candidate in our drug discovery pipeline
#[derive(Clone, Debug)]
struct MolecularCandidate {
    id: String,
    descriptors: Array1<f64>,
    binding_affinity: Option<f64>,
    synthesis_cost: f64,
}

impl MolecularCandidate {
    fn new(id: String, descriptors: Array1<f64>, synthesis_cost: f64) -> Self {
        Self {
            id,
            descriptors,
            binding_affinity: None,
            synthesis_cost,
        }
    }

    fn with_affinity(mut self, affinity: f64) -> Self {
        self.binding_affinity = Some(affinity);
        self
    }
}

/// Main drug discovery workflow
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Worker 7: End-to-End Drug Discovery Workflow                   ║");
    println!("║  Active Inference-Based Molecular Optimization                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1: Initialize Molecular Library
    // ═══════════════════════════════════════════════════════════════════════════

    println!("📚 Phase 1: Initializing Molecular Library");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let library_size = 1000;
    let descriptor_dim = 256; // Typical molecular fingerprint size

    let mut molecular_library = generate_molecular_library(library_size, descriptor_dim)?;

    println!("✓ Generated {} molecular candidates", library_size);
    println!("✓ Descriptor dimensionality: {}", descriptor_dim);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2: Chemical Space Analysis
    // ═══════════════════════════════════════════════════════════════════════════

    println!("🔬 Phase 2: Chemical Space Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mol_metrics = MolecularInformationMetrics::new();

    // Calculate chemical space entropy
    let descriptors_matrix = stack_descriptors(&molecular_library)?;
    let start = Instant::now();
    let chemical_space_entropy = mol_metrics.chemical_space_entropy(&descriptors_matrix)?;
    let duration = start.elapsed();

    println!("✓ Chemical space entropy: {:.4} nats", chemical_space_entropy);
    println!("✓ Calculation time: {:.2?}", duration);
    println!("  → High entropy indicates diverse molecular library");
    println!();

    // Calculate pairwise molecular similarities
    println!("📊 Computing pairwise molecular similarities...");
    let similarity_matrix = compute_similarity_matrix(&molecular_library, &mol_metrics)?;

    let avg_similarity = similarity_matrix.iter().sum::<f64>() / (library_size * library_size) as f64;
    println!("✓ Average molecular similarity: {:.4}", avg_similarity);
    println!("  → Lower average indicates more diversity");
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3: Initial Screening (Random Sampling)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("🎯 Phase 3: Initial Screening (Random Sampling)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let initial_screening_size = 50; // 5% of library
    let screened_indices = random_sample(library_size, initial_screening_size);

    // Simulate experimental measurements (in real application, these would be actual assay results)
    for &idx in &screened_indices {
        let affinity = simulate_binding_affinity(&molecular_library[idx].descriptors);
        molecular_library[idx].binding_affinity = Some(affinity);
    }

    let screened_molecules: Vec<_> = screened_indices.iter()
        .map(|&i| &molecular_library[i])
        .collect();

    let best_initial = screened_molecules.iter()
        .max_by(|a, b| {
            a.binding_affinity.unwrap()
                .partial_cmp(&b.binding_affinity.unwrap())
                .unwrap()
        })
        .unwrap();

    println!("✓ Screened {} molecules ({}% of library)", initial_screening_size,
        (initial_screening_size as f64 / library_size as f64) * 100.0);
    println!("✓ Best initial candidate: {} (affinity: {:.4})",
        best_initial.id, best_initial.binding_affinity.unwrap());
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 4: Active Learning Optimization
    // ═══════════════════════════════════════════════════════════════════════════

    println!("🧠 Phase 4: Active Learning Optimization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = DrugDiscoveryConfig {
        descriptor_dim,
        population_size: 50,
        num_generations: 20,
        mutation_rate: 0.1,
        target_binding_affinity: 0.9,
    };

    let controller = DrugDiscoveryController::new(config)?;
    println!("✓ Initialized Active Inference controller");

    // Active learning loop: iteratively select most informative molecules
    let num_active_rounds = 10;
    let molecules_per_round = 10;

    let opt_metrics = OptimizedExperimentInformationMetrics::new()?;

    for round in 1..=num_active_rounds {
        println!("\n📍 Active Learning Round {}/{}", round, num_active_rounds);
        println!("   ─────────────────────────────────────");

        // Get descriptors of screened molecules
        let screened_descriptors = get_screened_descriptors(&molecular_library)?;

        // Calculate Expected Information Gain for each unscreened molecule
        let start = Instant::now();
        let eig_scores = calculate_eig_scores(
            &molecular_library,
            &screened_descriptors,
            &opt_metrics
        )?;
        let eig_duration = start.elapsed();

        // Select top molecules by EIG
        let mut eig_with_indices: Vec<_> = eig_scores.iter()
            .enumerate()
            .filter(|(i, _)| molecular_library[*i].binding_affinity.is_none())
            .collect();

        eig_with_indices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let selected_indices: Vec<_> = eig_with_indices.iter()
            .take(molecules_per_round)
            .map(|(i, _)| *i)
            .collect();

        // "Measure" selected molecules
        for &idx in &selected_indices {
            let affinity = simulate_binding_affinity(&molecular_library[idx].descriptors);
            molecular_library[idx].binding_affinity = Some(affinity);
        }

        // Find current best
        let current_best = molecular_library.iter()
            .filter_map(|mol| mol.binding_affinity.map(|a| (mol, a)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!("   ✓ EIG calculation: {:.2?} (optimized)", eig_duration);
        println!("   ✓ Selected {} molecules with highest EIG", molecules_per_round);
        println!("   ✓ Current best: {} (affinity: {:.4})",
            current_best.0.id, current_best.1);
        println!("   ✓ Total molecules screened: {}",
            molecular_library.iter().filter(|m| m.binding_affinity.is_some()).count());
    }

    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 5: Molecular Optimization with Active Inference
    // ═══════════════════════════════════════════════════════════════════════════

    println!("🚀 Phase 5: Molecular Optimization (Active Inference)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Get best candidate from active learning
    let seed_molecule = molecular_library.iter()
        .filter_map(|mol| mol.binding_affinity.map(|a| (mol, a)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap().0;

    println!("✓ Starting from: {} (affinity: {:.4})",
        seed_molecule.id, seed_molecule.binding_affinity.unwrap());

    // Run Active Inference-based optimization
    let start = Instant::now();
    let optimized = controller.optimize_molecule(&seed_molecule.descriptors).await?;
    let optimization_duration = start.elapsed();

    println!("✓ Optimization complete: {:.2?}", optimization_duration);
    println!("✓ Optimized binding affinity: {:.4}", optimized.binding_affinity);
    println!("✓ Improvement: {:.2}%",
        ((optimized.binding_affinity - seed_molecule.binding_affinity.unwrap())
            / seed_molecule.binding_affinity.unwrap()) * 100.0);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 6: Results Summary
    // ═══════════════════════════════════════════════════════════════════════════

    println!("📋 Phase 6: Results Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let total_screened = molecular_library.iter()
        .filter(|m| m.binding_affinity.is_some())
        .count();

    let screening_percentage = (total_screened as f64 / library_size as f64) * 100.0;

    println!("Library Statistics:");
    println!("  • Total library size: {} molecules", library_size);
    println!("  • Molecules screened: {} ({:.1}%)", total_screened, screening_percentage);
    println!("  • Chemical space entropy: {:.4} nats", chemical_space_entropy);
    println!();

    println!("Optimization Results:");
    println!("  • Initial best (random): {:.4}", best_initial.binding_affinity.unwrap());
    println!("  • Active learning best: {:.4}", seed_molecule.binding_affinity.unwrap());
    println!("  • Final optimized: {:.4}", optimized.binding_affinity);
    println!("  • Total improvement: {:.2}%",
        ((optimized.binding_affinity - best_initial.binding_affinity.unwrap())
            / best_initial.binding_affinity.unwrap()) * 100.0);
    println!();

    println!("Efficiency Metrics:");
    println!("  • Molecules tested: {} / {} ({:.1}% efficiency)",
        total_screened, library_size, screening_percentage);
    println!("  • Optimization time: {:.2?}", optimization_duration);
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  🎉 Drug Discovery Workflow Complete!                            ║");
    println!("║  Worker 7 successfully identified and optimized lead compound    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Generate synthetic molecular library with random descriptors
fn generate_molecular_library(size: usize, dim: usize) -> Result<Vec<MolecularCandidate>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut library = Vec::with_capacity(size);

    for i in 0..size {
        let descriptors = Array1::from_vec(
            (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect()
        );

        // Synthesis cost varies with molecular complexity
        let complexity = descriptors.iter().sum::<f64>() / dim as f64;
        let synthesis_cost = 1000.0 + complexity * 9000.0; // $1K - $10K

        library.push(MolecularCandidate::new(
            format!("MOL-{:05}", i + 1),
            descriptors,
            synthesis_cost,
        ));
    }

    Ok(library)
}

/// Stack molecular descriptors into a matrix for bulk operations
fn stack_descriptors(library: &[MolecularCandidate]) -> Result<Array2<f64>> {
    let n = library.len();
    let d = library[0].descriptors.len();

    let mut matrix = Array2::zeros((n, d));

    for (i, mol) in library.iter().enumerate() {
        for (j, &val) in mol.descriptors.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }

    Ok(matrix)
}

/// Compute pairwise molecular similarity matrix
fn compute_similarity_matrix(
    library: &[MolecularCandidate],
    metrics: &MolecularInformationMetrics,
) -> Result<Array2<f64>> {
    let n = library.len();
    let mut similarity_matrix = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            let sim = if i == j {
                1.0
            } else {
                metrics.molecular_similarity(
                    &library[i].descriptors,
                    &library[j].descriptors,
                )
            };

            similarity_matrix[[i, j]] = sim;
            similarity_matrix[[j, i]] = sim;
        }
    }

    Ok(similarity_matrix)
}

/// Random sampling without replacement
fn random_sample(total: usize, count: usize) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut rng);

    indices.into_iter().take(count).collect()
}

/// Simulate binding affinity measurement (placeholder for actual assay)
///
/// In production, this would call an actual experimental assay or
/// docking simulation. Here we use a synthetic function for demonstration.
fn simulate_binding_affinity(descriptors: &Array1<f64>) -> f64 {
    // Synthetic binding affinity based on descriptor pattern
    // Real applications would use actual experimental data

    let n = descriptors.len();

    // Features favoring binding
    let hydrophobicity = descriptors.iter().take(n / 4).sum::<f64>() / (n as f64 / 4.0);
    let aromatic_content = descriptors.iter().skip(n / 4).take(n / 4).sum::<f64>() / (n as f64 / 4.0);
    let h_bond_donors = descriptors.iter().skip(n / 2).take(n / 4).sum::<f64>() / (n as f64 / 4.0);
    let molecular_weight = descriptors.iter().skip(3 * n / 4).sum::<f64>() / (n as f64 / 4.0);

    // Nonlinear combination (simplified drug-likeness)
    let base_affinity = (0.4 * hydrophobicity + 0.3 * aromatic_content
        + 0.2 * h_bond_donors + 0.1 * molecular_weight).min(1.0);

    // Add some noise
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let noise = rng.gen_range(-0.05..0.05);

    (base_affinity + noise).clamp(0.0, 1.0)
}

/// Get descriptors of all screened molecules
fn get_screened_descriptors(library: &[MolecularCandidate]) -> Result<Array2<f64>> {
    let screened: Vec<_> = library.iter()
        .filter(|m| m.binding_affinity.is_some())
        .collect();

    if screened.is_empty() {
        anyhow::bail!("No molecules have been screened yet");
    }

    let n = screened.len();
    let d = screened[0].descriptors.len();

    let mut matrix = Array2::zeros((n, d));

    for (i, mol) in screened.iter().enumerate() {
        for (j, &val) in mol.descriptors.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }

    Ok(matrix)
}

/// Calculate Expected Information Gain scores for all unscreened molecules
fn calculate_eig_scores(
    library: &[MolecularCandidate],
    screened_descriptors: &Array2<f64>,
    metrics: &OptimizedExperimentInformationMetrics,
) -> Result<Vec<f64>> {
    let mut eig_scores = Vec::with_capacity(library.len());

    for mol in library.iter() {
        if mol.binding_affinity.is_some() {
            // Already screened
            eig_scores.push(0.0);
            continue;
        }

        // Simulate posterior distribution if this molecule were screened
        // In real application, this would be based on a probabilistic model
        let posterior_descriptors = simulate_posterior_with_molecule(
            screened_descriptors,
            &mol.descriptors,
        )?;

        // Calculate EIG as reduction in entropy
        let eig = metrics.expected_information_gain(
            screened_descriptors,
            &posterior_descriptors,
        )?;

        eig_scores.push(eig);
    }

    Ok(eig_scores)
}

/// Simulate posterior distribution after adding a new molecule
fn simulate_posterior_with_molecule(
    prior: &Array2<f64>,
    new_mol: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n = prior.nrows();
    let d = prior.ncols();

    let mut posterior = Array2::zeros((n + 1, d));

    // Copy prior samples
    for i in 0..n {
        for j in 0..d {
            posterior[[i, j]] = prior[[i, j]];
        }
    }

    // Add new molecule
    for j in 0..d {
        posterior[[n, j]] = new_mol[j];
    }

    Ok(posterior)
}
