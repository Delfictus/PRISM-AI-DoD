//! Information Hamiltonian for Quantum Consensus

use ndarray::{Array1, Array2};

pub struct InformationHamiltonian {
    n_llms: usize,
    temperature: f64,
}

impl InformationHamiltonian {
    pub fn new(n_llms: usize, temperature: f64) -> Self {
        Self { n_llms, temperature }
    }

    pub fn energy(&self, weights: &Array1<f64>, distances: &Array2<f64>) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.n_llms {
            for j in 0..self.n_llms {
                energy += weights[i] * weights[j] * distances[[i, j]];
            }
        }
        energy
    }

    pub fn gradient(&self, weights: &Array1<f64>, distances: &Array2<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(self.n_llms);
        for i in 0..self.n_llms {
            for j in 0..self.n_llms {
                grad[i] += 2.0 * weights[j] * distances[[i, j]];
            }
        }
        grad
    }
}
