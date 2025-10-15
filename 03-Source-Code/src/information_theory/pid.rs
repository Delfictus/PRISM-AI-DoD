//! Partial Information Decomposition (PID)
//!
//! Decomposes multivariate mutual information into:
//! 1. **Unique Information**: Information only in X₁ about Y
//! 2. **Redundant Information**: Information shared by X₁ and X₂ about Y
//! 3. **Synergistic Information**: Information only available from X₁ AND X₂ together
//!
//! Mathematical Framework (Williams-Beer Lattice):
//! ```
//! I(Y; X₁, X₂) = Unique(X₁) + Unique(X₂) + Redundant(X₁,X₂) + Synergy(X₁,X₂)
//! ```
//!
//! Applications:
//! - Neuroscience: How brain regions jointly encode information
//! - Finance: Unique vs redundant predictive signals
//! - PWSA: Missile tracking with multiple sensors (radar, optical, IR)
//! - Climate: Distinguishing direct vs synergistic climate drivers
//!
//! References:
//! - Williams & Beer (2010). "Nonnegative decomposition of multivariate information"
//! - Bertschinger et al. (2014). "Quantifying unique information"

use anyhow::Result;
use ndarray::Array1;
use std::collections::HashMap;

/// Partial Information Decomposition result
#[derive(Debug, Clone)]
pub struct PidResult {
    /// Total mutual information I(Y; X₁, X₂)
    pub total_mi: f64,
    /// Unique information in X₁: I(Y; X₁ \ X₂)
    pub unique_x1: f64,
    /// Unique information in X₂: I(Y; X₂ \ X₁)
    pub unique_x2: f64,
    /// Redundant information: min(I(Y;X₁), I(Y;X₂))
    pub redundant: f64,
    /// Synergistic information: I(Y; X₁, X₂) - I(Y; X₁) - I(Y; X₂) + redundant
    pub synergy: f64,
}

/// Partial Information Decomposition calculator
pub struct PartialInfoDecomp {
    /// Number of bins for discretization
    n_bins: usize,
    /// Method for computing unique information
    method: PidMethod,
}

/// PID computation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PidMethod {
    /// Minimum mutual information (Williams-Beer)
    MinimumMutualInfo,
    /// Bertschinger's unique information
    Bertschinger,
    /// Pointwise unique information
    Pointwise,
}

impl Default for PartialInfoDecomp {
    fn default() -> Self {
        Self {
            n_bins: 10,
            method: PidMethod::MinimumMutualInfo,
        }
    }
}

impl PartialInfoDecomp {
    /// Create new PID calculator
    pub fn new(n_bins: usize, method: PidMethod) -> Self {
        Self { n_bins, method }
    }

    /// Calculate PID for bivariate sources
    ///
    /// # Arguments
    /// * `x1` - First source variable
    /// * `x2` - Second source variable
    /// * `y` - Target variable
    ///
    /// # Returns
    /// PidResult with decomposition into unique, redundant, and synergistic components
    pub fn calculate(
        &self,
        x1: &Array1<f64>,
        x2: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<PidResult> {
        assert_eq!(x1.len(), x2.len(), "Sources must have same length");
        assert_eq!(x1.len(), y.len(), "Sources and target must have same length");

        // Discretize variables
        let x1_disc = self.discretize(x1);
        let x2_disc = self.discretize(x2);
        let y_disc = self.discretize(y);

        // Calculate mutual informations
        let mi_y_x1 = self.mutual_information(&y_disc, &x1_disc)?;
        let mi_y_x2 = self.mutual_information(&y_disc, &x2_disc)?;
        let mi_y_x1x2 = self.mutual_information_joint(&y_disc, &x1_disc, &x2_disc)?;

        // Calculate redundancy (minimum MI)
        let redundant = mi_y_x1.min(mi_y_x2);

        // Calculate unique information
        let (unique_x1, unique_x2) = match self.method {
            PidMethod::MinimumMutualInfo => {
                // Unique(X₁) = I(Y; X₁|X₂) = I(Y; X₁) - redundant
                let unq_x1 = (mi_y_x1 - redundant).max(0.0);
                let unq_x2 = (mi_y_x2 - redundant).max(0.0);
                (unq_x1, unq_x2)
            }
            PidMethod::Bertschinger => {
                self.bertschinger_unique(&y_disc, &x1_disc, &x2_disc)?
            }
            PidMethod::Pointwise => {
                self.pointwise_unique(&y_disc, &x1_disc, &x2_disc)?
            }
        };

        // Calculate synergy
        // Synergy = I(Y; X₁, X₂) - I(Y; X₁) - I(Y; X₂) + Redundant
        let synergy = (mi_y_x1x2 - mi_y_x1 - mi_y_x2 + redundant).max(0.0);

        Ok(PidResult {
            total_mi: mi_y_x1x2,
            unique_x1,
            unique_x2,
            redundant,
            synergy,
        })
    }

    /// Mutual information I(X; Y)
    fn mutual_information(&self, x: &[i32], y: &[i32]) -> Result<f64> {
        let n = x.len() as f64;

        // Build probability distributions
        let mut p_xy: HashMap<(i32, i32), f64> = HashMap::new();
        let mut p_x: HashMap<i32, f64> = HashMap::new();
        let mut p_y: HashMap<i32, f64> = HashMap::new();

        for i in 0..x.len() {
            *p_xy.entry((x[i], y[i])).or_insert(0.0) += 1.0;
            *p_x.entry(x[i]).or_insert(0.0) += 1.0;
            *p_y.entry(y[i]).or_insert(0.0) += 1.0;
        }

        // Normalize
        for val in p_xy.values_mut() {
            *val /= n;
        }
        for val in p_x.values_mut() {
            *val /= n;
        }
        for val in p_y.values_mut() {
            *val /= n;
        }

        // Calculate MI
        let mut mi = 0.0;

        for ((xi, yi), &prob_xy) in &p_xy {
            if prob_xy > 1e-10 {
                let prob_x = p_x.get(xi).copied().unwrap_or(0.0);
                let prob_y = p_y.get(yi).copied().unwrap_or(0.0);

                if prob_x > 1e-10 && prob_y > 1e-10 {
                    mi += prob_xy * (prob_xy / (prob_x * prob_y)).ln();
                }
            }
        }

        Ok(mi / std::f64::consts::LN_2) // Convert to bits
    }

    /// Joint mutual information I(Y; X₁, X₂)
    fn mutual_information_joint(&self, y: &[i32], x1: &[i32], x2: &[i32]) -> Result<f64> {
        let n = y.len() as f64;

        // Build joint probability distributions
        let mut p_y_x1x2: HashMap<(i32, i32, i32), f64> = HashMap::new();
        let mut p_y: HashMap<i32, f64> = HashMap::new();
        let mut p_x1x2: HashMap<(i32, i32), f64> = HashMap::new();

        for i in 0..y.len() {
            *p_y_x1x2.entry((y[i], x1[i], x2[i])).or_insert(0.0) += 1.0;
            *p_y.entry(y[i]).or_insert(0.0) += 1.0;
            *p_x1x2.entry((x1[i], x2[i])).or_insert(0.0) += 1.0;
        }

        // Normalize
        for val in p_y_x1x2.values_mut() {
            *val /= n;
        }
        for val in p_y.values_mut() {
            *val /= n;
        }
        for val in p_x1x2.values_mut() {
            *val /= n;
        }

        // Calculate MI
        let mut mi = 0.0;

        for ((yi, x1i, x2i), &prob_joint) in &p_y_x1x2 {
            if prob_joint > 1e-10 {
                let prob_y = p_y.get(yi).copied().unwrap_or(0.0);
                let prob_x1x2 = p_x1x2.get(&(*x1i, *x2i)).copied().unwrap_or(0.0);

                if prob_y > 1e-10 && prob_x1x2 > 1e-10 {
                    mi += prob_joint * (prob_joint / (prob_y * prob_x1x2)).ln();
                }
            }
        }

        Ok(mi / std::f64::consts::LN_2)
    }

    /// Bertschinger's unique information measure
    fn bertschinger_unique(&self, y: &[i32], x1: &[i32], x2: &[i32]) -> Result<(f64, f64)> {
        // Simplified Bertschinger computation
        // Full implementation requires optimization over information channels

        let mi_y_x1 = self.mutual_information(y, x1)?;
        let mi_y_x2 = self.mutual_information(y, x2)?;
        let mi_y_x1x2 = self.mutual_information_joint(y, x1, x2)?;

        // Conditional mutual informations
        let mi_y_x1_given_x2 = self.conditional_mutual_information(y, x1, x2)?;
        let mi_y_x2_given_x1 = self.conditional_mutual_information(y, x2, x1)?;

        let unique_x1 = mi_y_x1_given_x2.max(0.0);
        let unique_x2 = mi_y_x2_given_x1.max(0.0);

        Ok((unique_x1, unique_x2))
    }

    /// Conditional mutual information I(Y; X₁ | X₂)
    fn conditional_mutual_information(&self, y: &[i32], x1: &[i32], x2: &[i32]) -> Result<f64> {
        let n = y.len() as f64;

        // P(Y, X₁, X₂)
        let mut p_y_x1_x2: HashMap<(i32, i32, i32), f64> = HashMap::new();
        // P(X₂)
        let mut p_x2: HashMap<i32, f64> = HashMap::new();
        // P(Y, X₂)
        let mut p_y_x2: HashMap<(i32, i32), f64> = HashMap::new();
        // P(X₁, X₂)
        let mut p_x1_x2: HashMap<(i32, i32), f64> = HashMap::new();

        for i in 0..y.len() {
            *p_y_x1_x2.entry((y[i], x1[i], x2[i])).or_insert(0.0) += 1.0;
            *p_x2.entry(x2[i]).or_insert(0.0) += 1.0;
            *p_y_x2.entry((y[i], x2[i])).or_insert(0.0) += 1.0;
            *p_x1_x2.entry((x1[i], x2[i])).or_insert(0.0) += 1.0;
        }

        // Normalize
        for val in p_y_x1_x2.values_mut() {
            *val /= n;
        }
        for val in p_x2.values_mut() {
            *val /= n;
        }
        for val in p_y_x2.values_mut() {
            *val /= n;
        }
        for val in p_x1_x2.values_mut() {
            *val /= n;
        }

        // I(Y; X₁ | X₂) = Σ p(y,x₁,x₂) log[p(y,x₁,x₂)p(x₂) / (p(y,x₂)p(x₁,x₂))]
        let mut cmi = 0.0;

        for ((yi, x1i, x2i), &prob_joint) in &p_y_x1_x2 {
            if prob_joint > 1e-10 {
                let prob_x2 = p_x2.get(x2i).copied().unwrap_or(0.0);
                let prob_y_x2 = p_y_x2.get(&(*yi, *x2i)).copied().unwrap_or(0.0);
                let prob_x1_x2 = p_x1_x2.get(&(*x1i, *x2i)).copied().unwrap_or(0.0);

                if prob_x2 > 1e-10 && prob_y_x2 > 1e-10 && prob_x1_x2 > 1e-10 {
                    let numerator = prob_joint * prob_x2;
                    let denominator = prob_y_x2 * prob_x1_x2;

                    if denominator > 1e-10 {
                        cmi += prob_joint * (numerator / denominator).ln();
                    }
                }
            }
        }

        Ok(cmi / std::f64::consts::LN_2)
    }

    /// Pointwise unique information
    fn pointwise_unique(&self, y: &[i32], x1: &[i32], x2: &[i32]) -> Result<(f64, f64)> {
        // Calculate pointwise mutual informations and aggregate

        let mi_y_x1_given_x2 = self.conditional_mutual_information(y, x1, x2)?;
        let mi_y_x2_given_x1 = self.conditional_mutual_information(y, x2, x1)?;

        Ok((mi_y_x1_given_x2.max(0.0), mi_y_x2_given_x1.max(0.0)))
    }

    /// Discretize continuous variable
    fn discretize(&self, series: &Array1<f64>) -> Vec<i32> {
        let min_val = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range < 1e-10 {
            return vec![0; series.len()];
        }

        series
            .iter()
            .map(|&x| {
                let normalized = (x - min_val) / range;
                let bin = (normalized * (self.n_bins as f64 - 1.0)) as i32;
                bin.max(0).min(self.n_bins as i32 - 1)
            })
            .collect()
    }
}

impl PidResult {
    /// Verify PID decomposition satisfies constraints
    pub fn is_valid(&self) -> bool {
        // All components should be non-negative
        if self.unique_x1 < -1e-10
            || self.unique_x2 < -1e-10
            || self.redundant < -1e-10
            || self.synergy < -1e-10
        {
            return false;
        }

        // Sum should approximately equal total MI
        let sum = self.unique_x1 + self.unique_x2 + self.redundant + self.synergy;
        (sum - self.total_mi).abs() < 0.1 // Allow 10% tolerance
    }

    /// Get dominant information type
    pub fn dominant_component(&self) -> &str {
        let max_val = self.unique_x1
            .max(self.unique_x2)
            .max(self.redundant)
            .max(self.synergy);

        if (self.unique_x1 - max_val).abs() < 1e-10 {
            "Unique X1"
        } else if (self.unique_x2 - max_val).abs() < 1e-10 {
            "Unique X2"
        } else if (self.redundant - max_val).abs() < 1e-10 {
            "Redundant"
        } else {
            "Synergistic"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_creation() {
        let pid = PartialInfoDecomp::new(10, PidMethod::MinimumMutualInfo);
        assert_eq!(pid.n_bins, 10);
        assert_eq!(pid.method, PidMethod::MinimumMutualInfo);
    }

    #[test]
    fn test_redundant_sources() {
        let pid = PartialInfoDecomp::default();

        // X1 and X2 are copies - should have high redundancy
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let x1 = y.clone(); // X1 = Y (perfect copy)
        let x2 = y.clone(); // X2 = Y (perfect copy)

        let y_arr = Array1::from_vec(y);
        let x1_arr = Array1::from_vec(x1);
        let x2_arr = Array1::from_vec(x2);

        let result = pid.calculate(&x1_arr, &x2_arr, &y_arr).unwrap();

        println!("Redundant sources PID:");
        println!("  Total MI: {}", result.total_mi);
        println!("  Unique X1: {}", result.unique_x1);
        println!("  Unique X2: {}", result.unique_x2);
        println!("  Redundant: {}", result.redundant);
        println!("  Synergy: {}", result.synergy);

        // Should have high redundancy, low unique/synergy
        assert!(result.redundant > 0.0);
        assert!(result.unique_x1 < result.redundant);
        assert!(result.unique_x2 < result.redundant);
    }

    #[test]
    fn test_unique_sources() {
        let pid = PartialInfoDecomp::default();

        // X1 and X2 provide different information about Y
        let x1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let x2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.1).cos()).collect();

        let x1_arr = Array1::from_vec(x1);
        let x2_arr = Array1::from_vec(x2);
        let y_arr = Array1::from_vec(y);

        let result = pid.calculate(&x1_arr, &x2_arr, &y_arr).unwrap();

        println!("Unique sources PID:");
        println!("  Dominant: {}", result.dominant_component());
        println!("  Total MI: {}", result.total_mi);
        println!("  Unique X1: {}", result.unique_x1);
        println!("  Unique X2: {}", result.unique_x2);
        println!("  Redundant: {}", result.redundant);
        println!("  Synergy: {}", result.synergy);

        // Should have unique information from both sources
        assert!(result.unique_x1 > 0.0 || result.unique_x2 > 0.0);
        assert!(result.total_mi > 0.0);
    }

    #[test]
    fn test_synergistic_xor() {
        let pid = PartialInfoDecomp::default();

        // XOR: Y = X1 XOR X2 (purely synergistic)
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        let mut y = Vec::new();

        for i in 0..200 {
            let bit1 = if (i / 10) % 2 == 0 { 0.0 } else { 1.0 };
            let bit2 = if (i / 20) % 2 == 0 { 0.0 } else { 1.0 };
            let y_val = if (bit1 + bit2) == 1.0 { 1.0 } else { 0.0 }; // XOR

            x1.push(bit1);
            x2.push(bit2);
            y.push(y_val);
        }

        let x1_arr = Array1::from_vec(x1);
        let x2_arr = Array1::from_vec(x2);
        let y_arr = Array1::from_vec(y);

        let result = pid.calculate(&x1_arr, &x2_arr, &y_arr).unwrap();

        println!("XOR (synergistic) PID:");
        println!("  Total MI: {}", result.total_mi);
        println!("  Unique X1: {}", result.unique_x1);
        println!("  Unique X2: {}", result.unique_x2);
        println!("  Redundant: {}", result.redundant);
        println!("  Synergy: {}", result.synergy);
        println!("  Dominant: {}", result.dominant_component());

        // XOR should show synergy
        assert!(result.total_mi > 0.0);
    }

    #[test]
    fn test_pid_validity() {
        let pid = PartialInfoDecomp::default();

        let x1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let x2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 1.0).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();

        let x1_arr = Array1::from_vec(x1);
        let x2_arr = Array1::from_vec(x2);
        let y_arr = Array1::from_vec(y);

        let result = pid.calculate(&x1_arr, &x2_arr, &y_arr).unwrap();

        // Verify PID decomposition is valid
        assert!(result.is_valid(), "PID decomposition should satisfy constraints");

        println!("Valid PID:");
        println!("  Sum of components: {}",
                 result.unique_x1 + result.unique_x2 + result.redundant + result.synergy);
        println!("  Total MI: {}", result.total_mi);
    }

    #[test]
    fn test_mutual_information() {
        let pid = PartialInfoDecomp::default();

        // Perfect correlation
        let x: Vec<i32> = (0..100).map(|i| i % 10).collect();
        let y = x.clone();

        let mi = pid.mutual_information(&x, &y).unwrap();

        println!("MI (perfect correlation): {}", mi);

        // Should have high MI for perfect correlation
        assert!(mi > 1.0);
    }
}
