//! Drug Discovery Platform Demo
//!
//! Demonstrates the complete drug discovery workflow:
//! 1. Screen compound library against target protein
//! 2. Predict ADMET properties
//! 3. Optimize lead compound
//!
//! # Usage
//! ```bash
//! cargo run --example drug_discovery_demo --features cuda
//! ```

use prism_ai::applications::drug_discovery::*;
use ndarray::{Array2};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== PRISM-AI Drug Discovery Platform Demo ===\n");

    // 1. Initialize platform with GPU acceleration
    println!("1. Initializing drug discovery platform...");
    let config = PlatformConfig::default();
    let mut platform = DrugDiscoveryPlatform::new(config)?;
    println!("   ✅ Platform initialized (GPU: {})\n", if cfg!(feature = "cuda") { "enabled" } else { "disabled" });

    // 2. Define target protein
    println!("2. Loading target protein...");
    let target_protein = create_example_protein();
    println!("   Protein ID: {}", target_protein.id);
    println!("   Atoms: {}", target_protein.atom_types.len());
    println!("   Binding site residues: {:?}\n", target_protein.binding_site);

    // 3. Create compound library
    println!("3. Generating compound library...");
    let compound_library = create_compound_library(10);
    println!("   Library size: {} compounds\n", compound_library.len());

    // 4. Screen library
    println!("4. Screening compounds (GPU-accelerated docking + GNN prediction)...");
    let candidates = platform.screen_library(&target_protein, &compound_library)?;

    println!("   Found {} candidates passing affinity threshold\n", candidates.len());

    // 5. Display top candidates
    println!("5. Top 3 Drug Candidates:");
    println!("   ╔═══════════════════════════════════════════════════════════════╗");
    println!("   ║  Rank │ SMILES    │ Affinity  │ ADMET │ Overall │ Class   ║");
    println!("   ╠═══════════════════════════════════════════════════════════════╣");

    for (i, candidate) in candidates.iter().take(3).enumerate() {
        println!("   ║  #{:<4}│ {:<9} │ {:>7.2} │ {:>5.2} │ {:>7.2} │ {:7} ║",
            i + 1,
            &candidate.molecule.smiles[..9.min(candidate.molecule.smiles.len())],
            candidate.binding_affinity,
            candidate.admet_properties.overall_score(),
            candidate.overall_score,
            "Lead"
        );
    }
    println!("   ╚═══════════════════════════════════════════════════════════════╝\n");

    // 6. Optimize lead compound
    if let Some(lead) = candidates.first() {
        println!("6. Optimizing lead compound using Active Inference...");
        let optimization = platform.optimize_lead(
            &lead.molecule,
            &target_protein,
            10,  // max iterations
        )?;

        println!("   Affinity improvement: {:.2} kcal/mol", optimization.affinity_improvement);
        println!("   Optimization trajectory: {} steps", optimization.optimization_trajectory.len());
        println!("   Final molecule: {}\n", optimization.optimized_molecule.smiles);
    }

    // 7. Display ADMET profile
    if let Some(lead) = candidates.first() {
        println!("7. ADMET Profile (Lead Compound):");
        let admet = &lead.admet_properties;
        println!("   ┌─────────────────────────────────────────┐");
        println!("   │ Absorption (Caco-2):     {:>6.2}      │", admet.absorption);
        println!("   │ BBB Penetration:         {:>6.2}      │", admet.bbb_penetration);
        println!("   │ CYP450 Inhibition:       {:>6.2}      │", admet.cyp450_inhibition);
        println!("   │ hERG Inhibition:         {:>6.2}      │", admet.herg_inhibition);
        println!("   │ Solubility (logS):       {:>6.2}      │", admet.solubility_logs);
        println!("   │ Overall Score:           {:>6.2}      │", admet.overall_score());
        println!("   │ Confidence:              {:>6.2}      │", admet.confidence);
        println!("   └─────────────────────────────────────────┘\n");
    }

    println!("✅ Drug discovery demo complete!");
    println!("\n💡 Note: This demo uses placeholder structures.");
    println!("   In production, load actual PDB files and SMILES libraries.");

    Ok(())
}

/// Create example target protein
fn create_example_protein() -> Protein {
    // Simplified protein structure (in production: load from PDB)
    let n_atoms = 1000;
    let coords = Array2::from_shape_fn((n_atoms, 3), |(i, j)| {
        ((i * 3 + j) % 100) as f64 / 10.0
    });

    Protein {
        id: "1ABC".to_string(),
        coordinates: coords,
        atom_types: vec!["CA".to_string(); n_atoms],
        binding_site: vec![42, 43, 44, 87, 88, 89],  // Example residues
        secondary_structure: vec![
            SecondaryStructure::Helix,
            SecondaryStructure::Sheet,
            SecondaryStructure::Loop,
        ],
    }
}

/// Create example compound library
fn create_compound_library(n_compounds: usize) -> Vec<Molecule> {
    let mut library = Vec::with_capacity(n_compounds);

    for i in 0..n_compounds {
        let n_atoms = 20 + (i * 5);
        let coords = Array2::from_shape_fn((n_atoms, 3), |(atom, dim)| {
            ((atom * 7 + dim * 11) % 50) as f64 / 10.0
        });

        let mut bonds = Vec::new();
        for j in 0..(n_atoms - 1) {
            bonds.push((j, j + 1, BondType::Single));
        }

        library.push(Molecule {
            smiles: format!("C{}H{}O{}", n_atoms / 2, n_atoms / 3, n_atoms / 6),
            coordinates: coords,
            atom_types: vec!["C".to_string(); n_atoms],
            bonds,
            molecular_weight: 200.0 + (i as f64 * 50.0),
        });
    }

    library
}
