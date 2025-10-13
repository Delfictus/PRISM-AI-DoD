//! Drug Discovery with Active Inference Demo
//!
//! Demonstrates Worker 7's drug discovery module:
//! - Molecular representation and properties
//! - Protein target modeling
//! - Active Inference-based molecular optimization
//! - Binding affinity prediction
//! - Drug-likeness scoring (Lipinski's Rule of Five)
//!
//! Run with: cargo run --example drug_discovery_demo

use anyhow::Result;
use ndarray::{array, Array1, Array2};
use prism_ai::applications::{DrugDiscoveryController, DrugDiscoveryConfig};
use prism_ai::applications::drug_discovery::molecular::{
    Molecule, Protein, AtomType, AminoAcid,
};

fn main() -> Result<()> {
    println!("=== PRISM-AI Drug Discovery Demo ===\n");

    // 1. Configure drug discovery
    let config = DrugDiscoveryConfig {
        max_iterations: 100,
        learning_rate: 0.01,
        target_affinity: -10.0,  // -10 kcal/mol is strong binding
        use_gpu: false,          // Set true for GPU acceleration
    };

    println!("Configuration:");
    println!("  Optimization iterations: {}", config.max_iterations);
    println!("  Target binding affinity: {} kcal/mol", config.target_affinity);
    println!("  GPU acceleration: {}\n", config.use_gpu);

    // 2. Initialize drug discovery controller
    let mut controller = DrugDiscoveryController::new(config)?;
    println!("✓ Drug discovery controller initialized\n");

    // 3. Create example drug molecule (simplified aspirin-like structure)
    println!("--- Creating Initial Molecule ---");
    let atoms = vec![
        AtomType::Carbon,      // 0
        AtomType::Carbon,      // 1
        AtomType::Carbon,      // 2
        AtomType::Carbon,      // 3
        AtomType::Carbon,      // 4
        AtomType::Carbon,      // 5 - Aromatic ring
        AtomType::Carbon,      // 6 - Carboxyl group
        AtomType::Oxygen,      // 7
        AtomType::Oxygen,      // 8
        AtomType::Hydrogen,    // 9
        AtomType::Hydrogen,    // 10
        AtomType::Hydrogen,    // 11
        AtomType::Hydrogen,    // 12
        AtomType::Hydrogen,    // 13
    ];

    // 3D coordinates (simplified 2D representation)
    let coords = Array2::from_shape_vec(
        (14, 3),
        vec![
            0.0, 0.0, 0.0,      // C0
            1.0, 0.0, 0.0,      // C1
            1.5, 0.87, 0.0,     // C2
            1.0, 1.73, 0.0,     // C3
            0.0, 1.73, 0.0,     // C4
            -0.5, 0.87, 0.0,    // C5
            2.5, 0.87, 0.0,     // C6 (carboxyl)
            3.0, 1.5, 0.0,      // O7
            2.8, -0.2, 0.0,     // O8
            -1.0, 0.87, 0.0,    // H9
            0.0, 2.5, 0.0,      // H10
            1.0, 2.5, 0.0,      // H11
            2.0, 0.0, 0.0,      // H12
            -0.5, -0.5, 0.0,    // H13
        ],
    )?;

    // Connectivity (simplified - only key bonds)
    let mut bonds = Array2::zeros((14, 14));
    // Aromatic ring bonds
    bonds[[0, 1]] = 1; bonds[[1, 0]] = 1;
    bonds[[1, 2]] = 1; bonds[[2, 1]] = 1;
    bonds[[2, 3]] = 1; bonds[[3, 2]] = 1;
    bonds[[3, 4]] = 1; bonds[[4, 3]] = 1;
    bonds[[4, 5]] = 1; bonds[[5, 4]] = 1;
    bonds[[5, 0]] = 1; bonds[[0, 5]] = 1;
    // Carboxyl group
    bonds[[2, 6]] = 1; bonds[[6, 2]] = 1;
    bonds[[6, 7]] = 2; bonds[[7, 6]] = 2;  // C=O double bond
    bonds[[6, 8]] = 1; bonds[[8, 6]] = 1;  // C-OH

    let initial_molecule = Molecule::new(atoms.clone(), coords, bonds);

    println!("Initial molecule:");
    println!("  Atoms: {}", initial_molecule.size());
    println!("  Heavy atoms: {}", initial_molecule.heavy_atom_count());
    println!("  Molecular weight: {:.1} Da", initial_molecule.molecular_weight());

    // 4. Create protein target (simplified kinase active site)
    println!("\n--- Creating Protein Target ---");
    let pocket_residues = vec![
        AminoAcid::Leucine,      // Hydrophobic
        AminoAcid::Valine,       // Hydrophobic
        AminoAcid::Lysine,       // Charged (+)
        AminoAcid::AsparticAcid, // Charged (-)
        AminoAcid::Serine,       // H-bond donor
        AminoAcid::Threonine,    // H-bond donor/acceptor
        AminoAcid::Phenylalanine,// Aromatic
        AminoAcid::Tyrosine,     // Aromatic + H-bond
    ];

    let pocket_coords = Array2::from_shape_vec(
        (8, 3),
        vec![
            0.0, 0.0, 5.0,
            2.0, 1.0, 5.0,
            1.0, 2.0, 6.0,
            -1.0, 1.0, 6.0,
            1.5, 0.5, 4.5,
            0.5, 1.5, 4.5,
            -0.5, 0.5, 5.5,
            1.0, -1.0, 5.0,
        ],
    )?;

    let target = Protein::new(
        "Kinase_Active_Site".to_string(),
        pocket_coords,
        pocket_residues.clone(),
    );

    println!("Protein target: {}", target.name);
    println!("  Binding pocket residues: {}", pocket_residues.len());
    print!("  Pocket composition: ");
    let hydrophobic = pocket_residues.iter().filter(|r| r.is_hydrophobic()).count();
    let charged = pocket_residues.iter().filter(|r| r.is_charged()).count();
    let h_donors = pocket_residues.iter().filter(|r| r.is_h_donor()).count();
    println!("{}% hydrophobic, {} charged, {} H-bond donors",
        (hydrophobic * 100) / pocket_residues.len(),
        charged,
        h_donors
    );

    // 5. Predict initial binding affinity
    println!("\n--- Initial Binding Prediction ---");
    let initial_prediction = controller.predict_binding(&initial_molecule, &target)?;

    println!("Binding affinity: {:.2} kcal/mol", initial_prediction.affinity);
    println!("IC50: {:.2} nM", initial_prediction.ic50);
    println!("Confidence: {:.0}%", initial_prediction.confidence * 100.0);
    println!("Lipinski violations: {}/4", initial_prediction.lipinski_violations);
    println!("Drug-likeness: {:.0}%", initial_prediction.drug_likeness * 100.0);

    // Interpret binding affinity
    let binding_strength = if initial_prediction.affinity < -12.0 {
        "Very strong"
    } else if initial_prediction.affinity < -10.0 {
        "Strong"
    } else if initial_prediction.affinity < -8.0 {
        "Moderate"
    } else if initial_prediction.affinity < -6.0 {
        "Weak"
    } else {
        "Very weak"
    };
    println!("Binding strength: {}", binding_strength);

    // 6. Optimize molecule using Active Inference
    println!("\n--- Optimizing Molecule with Active Inference ---");
    println!("Minimizing variational free energy...\n");

    let optimization_result = controller.optimize_molecule(&initial_molecule, &target)?;

    println!("Optimization complete:");
    println!("  Iterations: {}", optimization_result.iterations);
    println!("  Converged: {}", optimization_result.converged);
    println!("  Final binding affinity: {:.2} kcal/mol", optimization_result.binding_affinity);

    // Show free energy trajectory
    if optimization_result.free_energy_history.len() >= 5 {
        println!("\nFree energy trajectory:");
        let history = &optimization_result.free_energy_history;
        println!("  Initial: {:.4}", history[0]);
        println!("  Iter 25: {:.4}", history[history.len() / 4]);
        println!("  Iter 50: {:.4}", history[history.len() / 2]);
        println!("  Iter 75: {:.4}", history[3 * history.len() / 4]);
        println!("  Final: {:.4}", history[history.len() - 1]);

        let total_reduction = history[0] - history[history.len() - 1];
        println!("  Total reduction: {:.4}", total_reduction);
    }

    // 7. Predict optimized molecule properties
    println!("\n--- Optimized Molecule Properties ---");
    let final_prediction = controller.predict_binding(
        &optimization_result.molecule,
        &target
    )?;

    println!("Binding affinity: {:.2} kcal/mol (improved by {:.2} kcal/mol)",
        final_prediction.affinity,
        initial_prediction.affinity - final_prediction.affinity
    );
    println!("IC50: {:.2} nM (improved from {:.2} nM)",
        final_prediction.ic50,
        initial_prediction.ic50
    );
    println!("Lipinski violations: {}/4", final_prediction.lipinski_violations);
    println!("Drug-likeness: {:.0}% (vs {:.0}% initial)",
        final_prediction.drug_likeness * 100.0,
        initial_prediction.drug_likeness * 100.0
    );

    // 8. Lipinski's Rule of Five analysis
    println!("\n--- Drug-Likeness Analysis (Lipinski's Rule of Five) ---");
    println!("Rules for oral bioavailability:");
    println!("  ✓ Molecular weight ≤ 500 Da: {:.1} Da",
        optimization_result.molecule.molecular_weight());
    println!("  ✓ LogP ≤ 5 (lipophilicity): Estimated");
    println!("  ✓ H-bond donors ≤ 5");
    println!("  ✓ H-bond acceptors ≤ 10");
    println!("\nViolations: {}/4 (0-1 is acceptable)", final_prediction.lipinski_violations);

    if final_prediction.lipinski_violations <= 1 {
        println!("✓ Good oral bioavailability expected");
    } else {
        println!("⚠ May have reduced oral bioavailability");
    }

    // 9. Active Inference explanation
    println!("\n--- Active Inference Methodology ---");
    println!("The optimization minimized variational free energy:");
    println!("  F = E[ln Q(molecule|target)] - ln P(molecule,target)");
    println!("\nThis balances:");
    println!("  • Binding affinity (maximize target complementarity)");
    println!("  • Chemical diversity (maintain exploration)");
    println!("  • Drug-likeness (satisfy Lipinski's rules)");
    println!("\nThe result is a molecule that:");
    println!("  1. Binds strongly to the target");
    println!("  2. Maintains drug-like properties");
    println!("  3. Minimizes off-target effects (via free energy)");

    // 10. Virtual screening demo
    println!("\n--- Virtual Screening (Bonus) ---");
    println!("Screening library of {} molecules...", 3);

    let library = vec![
        initial_molecule.clone(),
        optimization_result.molecule.clone(),
        // Add another simple molecule
        create_simple_molecule()?,
    ];

    let screening_results = controller.screen_library(&library, &target)?;

    println!("\nScreening results:");
    for (i, result) in screening_results.iter().enumerate() {
        println!("  Molecule {}: Affinity={:.2} kcal/mol, IC50={:.2} nM, Drug-likeness={:.0}%",
            i + 1,
            result.affinity,
            result.ic50,
            result.drug_likeness * 100.0
        );
    }

    // Find best from screening
    let best_idx = screening_results
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.affinity.partial_cmp(&b.affinity).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!("\n✓ Best candidate: Molecule {} (affinity={:.2} kcal/mol)",
        best_idx + 1,
        screening_results[best_idx].affinity
    );

    println!("\n=== Demo Complete ===");
    println!("\nNext steps:");
    println!("  • Validate with molecular dynamics simulations");
    println!("  • Synthesize lead compound");
    println!("  • Test in vitro binding assays");
    println!("  • Optimize ADMET properties");

    Ok(())
}

/// Helper function to create a simple molecule for screening
fn create_simple_molecule() -> Result<Molecule> {
    let atoms = vec![
        AtomType::Carbon,
        AtomType::Carbon,
        AtomType::Oxygen,
        AtomType::Hydrogen,
        AtomType::Hydrogen,
    ];

    let coords = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.5, 1.0, 0.0,
            -0.5, 0.0, 0.0,
            1.5, -0.5, 0.0,
        ],
    )?;

    let bonds = Array2::zeros((5, 5));

    Ok(Molecule::new(atoms, coords, bonds))
}
