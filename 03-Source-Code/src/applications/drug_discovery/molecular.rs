//! Molecular Representation
//!
//! Simplified molecular structures for drug discovery.
//! Real implementation would use RDKit or similar chemistry library.

use ndarray::{Array1, Array2};

/// Atom types in simplified representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtomType {
    Carbon,
    Nitrogen,
    Oxygen,
    Sulfur,
    Hydrogen,
    Fluorine,
    Chlorine,
    Bromine,
}

impl AtomType {
    /// Get atomic number
    pub fn atomic_number(&self) -> u8 {
        match self {
            AtomType::Hydrogen => 1,
            AtomType::Carbon => 6,
            AtomType::Nitrogen => 7,
            AtomType::Oxygen => 8,
            AtomType::Fluorine => 9,
            AtomType::Sulfur => 16,
            AtomType::Chlorine => 17,
            AtomType::Bromine => 35,
        }
    }

    /// Get electronegativity (Pauling scale)
    pub fn electronegativity(&self) -> f64 {
        match self {
            AtomType::Hydrogen => 2.20,
            AtomType::Carbon => 2.55,
            AtomType::Nitrogen => 3.04,
            AtomType::Oxygen => 3.44,
            AtomType::Fluorine => 3.98,
            AtomType::Sulfur => 2.58,
            AtomType::Chlorine => 3.16,
            AtomType::Bromine => 2.96,
        }
    }
}

/// Simplified molecular representation
#[derive(Debug, Clone)]
pub struct Molecule {
    /// Atom types
    pub atoms: Vec<AtomType>,
    /// 3D coordinates [n_atoms, 3]
    pub coordinates: Array2<f64>,
    /// Bond connectivity matrix [n_atoms, n_atoms]
    pub bonds: Array2<u8>,
    /// Molecular descriptors (fingerprint for ML)
    pub descriptors: Array1<f64>,
}

impl Molecule {
    /// Create new molecule
    pub fn new(
        atoms: Vec<AtomType>,
        coordinates: Array2<f64>,
        bonds: Array2<u8>,
    ) -> Self {
        let descriptors = Self::compute_descriptors(&atoms, &coordinates, &bonds);

        Self {
            atoms,
            coordinates,
            bonds,
            descriptors,
        }
    }

    /// Compute molecular descriptors (simplified)
    fn compute_descriptors(
        atoms: &[AtomType],
        coordinates: &Array2<f64>,
        bonds: &Array2<u8>,
    ) -> Array1<f64> {
        let n_atoms = atoms.len();

        // Compute simple descriptors:
        // 1. Molecular weight
        let mol_weight: f64 = atoms
            .iter()
            .map(|a| a.atomic_number() as f64)
            .sum();

        // 2. Number of bonds
        let n_bonds: f64 = bonds.iter().filter(|&&b| b > 0).count() as f64 / 2.0;

        // 3. Average electronegativity
        let avg_electroneg: f64 = atoms
            .iter()
            .map(|a| a.electronegativity())
            .sum::<f64>() / n_atoms as f64;

        // 4. Radius of gyration (spatial extent)
        let centroid = coordinates.mean_axis(ndarray::Axis(0)).unwrap();
        let rog: f64 = coordinates
            .outer_iter()
            .map(|coord| {
                let dx = coord[0] - centroid[0];
                let dy = coord[1] - centroid[1];
                let dz = coord[2] - centroid[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum::<f64>()
            / n_atoms as f64;

        // 5-10: Atom type counts (normalized)
        let mut atom_counts = vec![0.0; 8];
        for atom in atoms {
            let idx = match atom {
                AtomType::Hydrogen => 0,
                AtomType::Carbon => 1,
                AtomType::Nitrogen => 2,
                AtomType::Oxygen => 3,
                AtomType::Fluorine => 4,
                AtomType::Sulfur => 5,
                AtomType::Chlorine => 6,
                AtomType::Bromine => 7,
            };
            atom_counts[idx] += 1.0 / n_atoms as f64;
        }

        let mut descriptors = vec![mol_weight, n_bonds, avg_electroneg, rog.sqrt()];
        descriptors.extend(atom_counts);

        Array1::from_vec(descriptors)
    }

    /// Number of atoms in molecule
    pub fn size(&self) -> usize {
        self.atoms.len()
    }

    /// Estimate molecular weight
    pub fn molecular_weight(&self) -> f64 {
        self.descriptors[0]
    }

    /// Number of heavy atoms (non-hydrogen)
    pub fn heavy_atom_count(&self) -> usize {
        self.atoms
            .iter()
            .filter(|&&a| a != AtomType::Hydrogen)
            .count()
    }
}

/// Protein target representation
#[derive(Debug, Clone)]
pub struct Protein {
    /// Protein name/ID
    pub name: String,
    /// Binding pocket coordinates [n_residues, 3]
    pub pocket_coords: Array2<f64>,
    /// Binding pocket residue types (simplified)
    pub pocket_residues: Vec<AminoAcid>,
    /// Pharmacophore features (simplified)
    pub pharmacophore: Array1<f64>,
}

impl Protein {
    /// Create new protein target
    pub fn new(
        name: String,
        pocket_coords: Array2<f64>,
        pocket_residues: Vec<AminoAcid>,
    ) -> Self {
        let pharmacophore = Self::compute_pharmacophore(&pocket_coords, &pocket_residues);

        Self {
            name,
            pocket_coords,
            pocket_residues,
            pharmacophore,
        }
    }

    /// Compute pharmacophore features (simplified)
    fn compute_pharmacophore(
        coords: &Array2<f64>,
        residues: &[AminoAcid],
    ) -> Array1<f64> {
        let n_residues = residues.len();

        // Simplified pharmacophore:
        // 1. Hydrophobic patches
        let hydrophobic = residues
            .iter()
            .filter(|r| r.is_hydrophobic())
            .count() as f64 / n_residues as f64;

        // 2. Charged residues
        let charged = residues
            .iter()
            .filter(|r| r.is_charged())
            .count() as f64 / n_residues as f64;

        // 3. H-bond donors
        let h_donors = residues
            .iter()
            .filter(|r| r.is_h_donor())
            .count() as f64 / n_residues as f64;

        // 4. H-bond acceptors
        let h_acceptors = residues
            .iter()
            .filter(|r| r.is_h_acceptor())
            .count() as f64 / n_residues as f64;

        // 5. Pocket volume (approximate from coordinates)
        let centroid = coords.mean_axis(ndarray::Axis(0)).unwrap();
        let volume: f64 = coords
            .outer_iter()
            .map(|coord| {
                let dx = coord[0] - centroid[0];
                let dy = coord[1] - centroid[1];
                let dz = coord[2] - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum::<f64>();

        Array1::from_vec(vec![hydrophobic, charged, h_donors, h_acceptors, volume])
    }
}

/// Simplified amino acid representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AminoAcid {
    Alanine,
    Arginine,
    Asparagine,
    AsparticAcid,
    Cysteine,
    Glutamine,
    GlutamicAcid,
    Glycine,
    Histidine,
    Isoleucine,
    Leucine,
    Lysine,
    Methionine,
    Phenylalanine,
    Proline,
    Serine,
    Threonine,
    Tryptophan,
    Tyrosine,
    Valine,
}

impl AminoAcid {
    /// Is this amino acid hydrophobic?
    pub fn is_hydrophobic(&self) -> bool {
        matches!(
            self,
            AminoAcid::Alanine
                | AminoAcid::Isoleucine
                | AminoAcid::Leucine
                | AminoAcid::Methionine
                | AminoAcid::Phenylalanine
                | AminoAcid::Tryptophan
                | AminoAcid::Tyrosine
                | AminoAcid::Valine
        )
    }

    /// Is this amino acid charged?
    pub fn is_charged(&self) -> bool {
        matches!(
            self,
            AminoAcid::Arginine
                | AminoAcid::Lysine
                | AminoAcid::AsparticAcid
                | AminoAcid::GlutamicAcid
        )
    }

    /// Can this amino acid donate H-bonds?
    pub fn is_h_donor(&self) -> bool {
        matches!(
            self,
            AminoAcid::Arginine
                | AminoAcid::Asparagine
                | AminoAcid::Glutamine
                | AminoAcid::Histidine
                | AminoAcid::Lysine
                | AminoAcid::Serine
                | AminoAcid::Threonine
                | AminoAcid::Tryptophan
                | AminoAcid::Tyrosine
        )
    }

    /// Can this amino acid accept H-bonds?
    pub fn is_h_acceptor(&self) -> bool {
        matches!(
            self,
            AminoAcid::Asparagine
                | AminoAcid::AsparticAcid
                | AminoAcid::Glutamine
                | AminoAcid::GlutamicAcid
                | AminoAcid::Histidine
                | AminoAcid::Serine
                | AminoAcid::Threonine
                | AminoAcid::Tyrosine
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_properties() {
        assert_eq!(AtomType::Carbon.atomic_number(), 6);
        assert_eq!(AtomType::Nitrogen.atomic_number(), 7);
        assert!((AtomType::Oxygen.electronegativity() - 3.44).abs() < 0.01);
    }

    #[test]
    fn test_molecule_creation() {
        // Simple H2O molecule
        let atoms = vec![AtomType::Oxygen, AtomType::Hydrogen, AtomType::Hydrogen];
        let coords = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 0.0, 0.0, // O
                0.96, 0.0, 0.0, // H
                -0.24, 0.93, 0.0, // H
            ],
        )
        .unwrap();
        let bonds = Array2::zeros((3, 3));

        let molecule = Molecule::new(atoms, coords, bonds);
        assert_eq!(molecule.size(), 3);
        assert_eq!(molecule.heavy_atom_count(), 1);
    }

    #[test]
    fn test_amino_acid_properties() {
        assert!(AminoAcid::Leucine.is_hydrophobic());
        assert!(AminoAcid::Lysine.is_charged());
        assert!(AminoAcid::Serine.is_h_donor());
        assert!(AminoAcid::AsparticAcid.is_h_acceptor());
    }
}
