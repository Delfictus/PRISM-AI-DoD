//! Multiple Comparison Correction
//!
//! When testing transfer entropy across multiple time lags or variables,
//! we face the multiple comparisons problem: the chance of false discoveries
//! increases with the number of tests.
//!
//! This module implements:
//! 1. **Bonferroni Correction**: Conservative, controls Family-Wise Error Rate (FWER)
//! 2. **Benjamini-Hochberg FDR**: Controls False Discovery Rate, less conservative
//! 3. **Holm-Bonferroni**: Sequentially rejective Bonferroni
//!
//! Applications:
//! - Multiscale TE analysis (scanning multiple lags)
//! - Network analysis (testing many pairwise relationships)
//! - Feature selection (identifying relevant predictors)
//!
//! References:
//! - Bonferroni, C. (1936). "Teoria statistica delle classi"
//! - Benjamini & Hochberg (1995). "Controlling the false discovery rate"
//! - Holm, S. (1979). "A simple sequentially rejective multiple test procedure"

use anyhow::Result;

/// Multiple testing correction result
#[derive(Debug, Clone)]
pub struct CorrectedPValues {
    /// Original p-values
    pub original: Vec<f64>,
    /// Corrected p-values
    pub corrected: Vec<f64>,
    /// Rejection decisions at given alpha
    pub rejected: Vec<bool>,
    /// Correction method used
    pub method: CorrectionMethod,
    /// Significance level
    pub alpha: f64,
}

/// Correction methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrectionMethod {
    /// Bonferroni: α_corrected = α / m
    Bonferroni,
    /// Benjamini-Hochberg False Discovery Rate
    BenjaminiHochberg,
    /// Holm-Bonferroni (sequentially rejective)
    Holm,
    /// No correction (for reference)
    None,
}

/// Multiple testing corrector
pub struct MultipleTestingCorrection {
    /// Significance level
    alpha: f64,
    /// Correction method
    method: CorrectionMethod,
}

impl Default for MultipleTestingCorrection {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            method: CorrectionMethod::BenjaminiHochberg,
        }
    }
}

impl MultipleTestingCorrection {
    /// Create new multiple testing corrector
    pub fn new(alpha: f64, method: CorrectionMethod) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be in (0, 1)");

        Self { alpha, method }
    }

    /// Apply correction to p-values
    ///
    /// # Arguments
    /// * `p_values` - Original p-values from multiple tests
    ///
    /// # Returns
    /// CorrectedPValues with adjusted p-values and rejection decisions
    pub fn correct(&self, p_values: &[f64]) -> Result<CorrectedPValues> {
        // Validate p-values
        for &p in p_values {
            if p < 0.0 || p > 1.0 {
                anyhow::bail!("P-values must be in [0, 1]");
            }
        }

        let corrected = match self.method {
            CorrectionMethod::Bonferroni => self.bonferroni(p_values),
            CorrectionMethod::BenjaminiHochberg => self.benjamini_hochberg(p_values),
            CorrectionMethod::Holm => self.holm(p_values),
            CorrectionMethod::None => p_values.to_vec(),
        };

        // Determine rejections
        let rejected: Vec<bool> = corrected.iter().map(|&p| p < self.alpha).collect();

        Ok(CorrectedPValues {
            original: p_values.to_vec(),
            corrected,
            rejected,
            method: self.method,
            alpha: self.alpha,
        })
    }

    /// Bonferroni correction
    ///
    /// Adjusted p-value: p_corrected = min(m × p_original, 1.0)
    /// where m is the number of tests
    ///
    /// Controls Family-Wise Error Rate (FWER): P(at least one false positive) ≤ α
    fn bonferroni(&self, p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len() as f64;

        p_values
            .iter()
            .map(|&p| (p * m).min(1.0))
            .collect()
    }

    /// Benjamini-Hochberg FDR correction
    ///
    /// Controls False Discovery Rate: E[FP / (FP + TP)] ≤ α
    ///
    /// Procedure:
    /// 1. Sort p-values in ascending order
    /// 2. Find largest i such that p_(i) ≤ (i/m) × α
    /// 3. Reject all H_j for j ≤ i
    fn benjamini_hochberg(&self, p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len();

        if m == 0 {
            return Vec::new();
        }

        // Create indices for sorting
        let mut indexed: Vec<(usize, f64)> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by p-value
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Calculate BH adjusted p-values
        let mut adjusted = vec![1.0; m];

        // Backward pass to maintain monotonicity
        adjusted[indexed[m - 1].0] = indexed[m - 1].1;

        for i in (0..m-1).rev() {
            let raw_adjusted = indexed[i].1 * (m as f64) / ((i + 1) as f64);
            let monotonic_adjusted = raw_adjusted.min(adjusted[indexed[i + 1].0]);
            adjusted[indexed[i].0] = monotonic_adjusted.min(1.0);
        }

        adjusted
    }

    /// Holm-Bonferroni correction (step-down procedure)
    ///
    /// More powerful than Bonferroni while still controlling FWER
    ///
    /// Procedure:
    /// 1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
    /// 2. Reject H_(i) if p_(i) ≤ α/(m - i + 1) for all j ≤ i
    fn holm(&self, p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len();

        if m == 0 {
            return Vec::new();
        }

        // Create indexed p-values
        let mut indexed: Vec<(usize, f64)> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by p-value
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Calculate Holm-adjusted p-values
        let mut adjusted = vec![1.0; m];

        for (rank, &(original_idx, p_val)) in indexed.iter().enumerate() {
            let adjustment_factor = (m - rank) as f64;
            let adjusted_p = (p_val * adjustment_factor).min(1.0);

            // Enforce monotonicity
            if rank > 0 {
                adjusted[original_idx] = adjusted_p.max(adjusted[indexed[rank - 1].0]);
            } else {
                adjusted[original_idx] = adjusted_p;
            }
        }

        adjusted
    }

    /// Get number of discoveries at given FDR level
    ///
    /// For BH procedure, returns number of rejections
    pub fn n_discoveries(&self, p_values: &[f64]) -> Result<usize> {
        let corrected = self.correct(p_values)?;
        Ok(corrected.rejected.iter().filter(|&&r| r).count())
    }

    /// Estimate actual False Discovery Rate
    ///
    /// FDR ≈ (# false positives) / (# total positives)
    pub fn estimate_fdr(&self, p_values: &[f64], true_nulls: &[bool]) -> Result<f64> {
        assert_eq!(p_values.len(), true_nulls.len(), "Lengths must match");

        let corrected = self.correct(p_values)?;

        let mut false_positives = 0;
        let mut total_positives = 0;

        for i in 0..corrected.rejected.len() {
            if corrected.rejected[i] {
                total_positives += 1;

                if true_nulls[i] {
                    // Null is true but we rejected it
                    false_positives += 1;
                }
            }
        }

        if total_positives == 0 {
            return Ok(0.0);
        }

        Ok(false_positives as f64 / total_positives as f64)
    }
}

impl CorrectedPValues {
    /// Number of discoveries (rejections)
    pub fn n_discoveries(&self) -> usize {
        self.rejected.iter().filter(|&&r| r).count()
    }

    /// Get indices of discoveries
    pub fn discovery_indices(&self) -> Vec<usize> {
        self.rejected
            .iter()
            .enumerate()
            .filter(|(_, &r)| r)
            .map(|(i, _)| i)
            .collect()
    }

    /// Discovery rate
    pub fn discovery_rate(&self) -> f64 {
        self.n_discoveries() as f64 / self.original.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonferroni() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::Bonferroni);

        let p_values = vec![0.001, 0.01, 0.03, 0.05, 0.1];
        let result = corrector.correct(&p_values).unwrap();

        println!("Bonferroni correction:");
        for i in 0..p_values.len() {
            println!("  p={:.4} -> p_adj={:.4} (reject={})",
                     result.original[i], result.corrected[i], result.rejected[i]);
        }

        // With 5 tests and α=0.05, threshold is 0.05/5 = 0.01
        // So only p=0.001 should be significant
        assert!(result.rejected[0]); // 0.001 * 5 = 0.005 < 0.05
        assert!(!result.rejected[1]); // 0.01 * 5 = 0.05 = 0.05 (boundary)
        assert!(!result.rejected[2]); // 0.03 * 5 = 0.15 > 0.05
    }

    #[test]
    fn test_benjamini_hochberg() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::BenjaminiHochberg);

        let p_values = vec![0.001, 0.008, 0.015, 0.04, 0.1, 0.5];
        let result = corrector.correct(&p_values).unwrap();

        println!("Benjamini-Hochberg FDR correction:");
        for i in 0..p_values.len() {
            println!("  p={:.4} -> p_adj={:.4} (reject={})",
                     result.original[i], result.corrected[i], result.rejected[i]);
        }

        // BH is less conservative than Bonferroni
        // Should reject more nulls
        assert!(result.n_discoveries() >= 2);
        assert!(result.rejected[0]); // Smallest p-value should be rejected
    }

    #[test]
    fn test_holm() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::Holm);

        let p_values = vec![0.005, 0.012, 0.025, 0.08, 0.15];
        let result = corrector.correct(&p_values).unwrap();

        println!("Holm-Bonferroni correction:");
        for i in 0..p_values.len() {
            println!("  p={:.4} -> p_adj={:.4} (reject={})",
                     result.original[i], result.corrected[i], result.rejected[i]);
        }

        // Holm is more powerful than Bonferroni
        // Should reject at least as many as Bonferroni
        assert!(result.rejected[0]); // Smallest p-value
    }

    #[test]
    fn test_no_correction() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::None);

        let p_values = vec![0.01, 0.03, 0.06, 0.1];
        let result = corrector.correct(&p_values).unwrap();

        // No correction: just compare to alpha
        assert_eq!(result.corrected, result.original);
        assert!(result.rejected[0]); // 0.01 < 0.05
        assert!(result.rejected[1]); // 0.03 < 0.05
        assert!(!result.rejected[2]); // 0.06 > 0.05
    }

    #[test]
    fn test_all_null() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::BenjaminiHochberg);

        // All p-values high (all null hypotheses true)
        let p_values = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let result = corrector.correct(&p_values).unwrap();

        // Should reject none
        assert_eq!(result.n_discoveries(), 0);
    }

    #[test]
    fn test_all_alternative() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::BenjaminiHochberg);

        // All p-values very low (all alternatives true)
        let p_values = vec![0.001, 0.002, 0.003, 0.004, 0.005];
        let result = corrector.correct(&p_values).unwrap();

        println!("All alternatives - discoveries: {}", result.n_discoveries());

        // Should reject all or most
        assert!(result.n_discoveries() >= 4);
    }

    #[test]
    fn test_comparison_conservative() {
        // Compare methods on same data

        let p_values = vec![0.005, 0.01, 0.02, 0.05, 0.1, 0.2];

        let bonf = MultipleTestingCorrection::new(0.05, CorrectionMethod::Bonferroni);
        let bh = MultipleTestingCorrection::new(0.05, CorrectionMethod::BenjaminiHochberg);
        let holm = MultipleTestingCorrection::new(0.05, CorrectionMethod::Holm);

        let bonf_result = bonf.correct(&p_values).unwrap();
        let bh_result = bh.correct(&p_values).unwrap();
        let holm_result = holm.correct(&p_values).unwrap();

        println!("Method comparison:");
        println!("  Bonferroni discoveries: {}", bonf_result.n_discoveries());
        println!("  BH FDR discoveries: {}", bh_result.n_discoveries());
        println!("  Holm discoveries: {}", holm_result.n_discoveries());

        // Expected ordering: Bonferroni <= Holm <= BH (in terms of rejections)
        assert!(bonf_result.n_discoveries() <= holm_result.n_discoveries());
        assert!(holm_result.n_discoveries() <= bh_result.n_discoveries());
    }

    #[test]
    fn test_fdr_estimation() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::BenjaminiHochberg);

        // Simulate: first 5 are alternatives, last 5 are nulls
        let p_values = vec![
            0.001, 0.005, 0.01, 0.015, 0.02,  // Alternatives
            0.3, 0.4, 0.5, 0.6, 0.7,           // Nulls
        ];

        let true_nulls = vec![
            false, false, false, false, false,
            true, true, true, true, true,
        ];

        let fdr = corrector.estimate_fdr(&p_values, &true_nulls).unwrap();

        println!("Estimated FDR: {:.3}", fdr);

        // FDR should be low since we have clear signal
        assert!(fdr < 0.2);
    }

    #[test]
    fn test_discovery_indices() {
        let corrector = MultipleTestingCorrection::new(0.05, CorrectionMethod::BenjaminiHochberg);

        let p_values = vec![0.001, 0.1, 0.005, 0.5, 0.01];
        let result = corrector.correct(&p_values).unwrap();

        let indices = result.discovery_indices();

        println!("Discovery indices: {:?}", indices);

        // Should include indices of small p-values
        assert!(indices.contains(&0)); // p=0.001
        assert!(indices.contains(&2)); // p=0.005
    }
}
