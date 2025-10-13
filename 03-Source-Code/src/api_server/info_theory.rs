//! Information-theoretic metrics for API responses
//!
//! Provides mutual information, transfer entropy, entropy rates, and other
//! information-theoretic measures for quantifying uncertainty and information flow.

use std::collections::HashMap;

/// Information-theoretic metrics for a data stream or signal
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InfoTheoryMetrics {
    /// Shannon entropy H(X) in bits
    pub entropy: f64,

    /// Entropy rate H'(X) - entropy per time unit
    pub entropy_rate: f64,

    /// Mutual information I(X;Y) in bits
    pub mutual_information: Option<f64>,

    /// Transfer entropy TE(X→Y) in bits
    pub transfer_entropy: Option<f64>,

    /// Channel capacity in bits/second
    pub channel_capacity: Option<f64>,

    /// Fisher information - lower bound on estimation variance
    pub fisher_information: Option<f64>,

    /// Kullback-Leibler divergence from baseline
    pub kl_divergence: Option<f64>,
}

impl Default for InfoTheoryMetrics {
    fn default() -> Self {
        Self {
            entropy: 0.0,
            entropy_rate: 0.0,
            mutual_information: None,
            transfer_entropy: None,
            channel_capacity: None,
            fisher_information: None,
            kl_divergence: None,
        }
    }
}

/// Calculate Shannon entropy H(X) = -Σ p(x) log₂ p(x)
pub fn shannon_entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Calculate joint entropy H(X,Y) = -Σ p(x,y) log₂ p(x,y)
pub fn joint_entropy(joint_probs: &[(f64, f64, f64)]) -> f64 {
    joint_probs
        .iter()
        .filter(|&&(_, _, p)| p > 0.0)
        .map(|&(_, _, p)| -p * p.log2())
        .sum()
}

/// Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
///
/// Measures how much knowing Y reduces uncertainty about X
pub fn mutual_information(
    x_probs: &[f64],
    y_probs: &[f64],
    joint_probs: &[(f64, f64, f64)],
) -> f64 {
    let h_x = shannon_entropy(x_probs);
    let h_y = shannon_entropy(y_probs);
    let h_xy = joint_entropy(joint_probs);

    h_x + h_y - h_xy
}

/// Calculate transfer entropy TE(X→Y) = I(Y_future; X_past | Y_past)
///
/// Measures directional information flow from X to Y
pub fn transfer_entropy(
    source_past: &[f64],
    target_past: &[f64],
    target_future: &[f64],
) -> f64 {
    // Compute conditional mutual information
    // TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

    let n = source_past.len().min(target_past.len()).min(target_future.len());
    if n < 2 {
        return 0.0;
    }

    // Build empirical probability distributions
    let mut counts_y_given_y_past: HashMap<(i32, i32), u32> = HashMap::new();
    let mut counts_y_given_y_x_past: HashMap<(i32, i32, i32), u32> = HashMap::new();

    for i in 0..n {
        let y_past = discretize(target_past[i]);
        let x_past = discretize(source_past[i]);
        let y_future = discretize(target_future[i]);

        *counts_y_given_y_past.entry((y_past, y_future)).or_insert(0) += 1;
        *counts_y_given_y_x_past.entry((y_past, x_past, y_future)).or_insert(0) += 1;
    }

    // Calculate conditional entropies
    let h_y_given_y_past = conditional_entropy_from_counts(&counts_y_given_y_past, n as f64);
    let h_y_given_y_x_past = conditional_entropy_3d(&counts_y_given_y_x_past, n as f64);

    // Transfer entropy is the reduction in uncertainty
    (h_y_given_y_past - h_y_given_y_x_past).max(0.0)
}

/// Calculate entropy rate H'(X) for a time series
///
/// Measures entropy per time step, indicating predictability
pub fn entropy_rate(time_series: &[f64], window_size: usize) -> f64 {
    if time_series.len() < window_size + 1 {
        return 0.0;
    }

    let mut block_entropies = Vec::new();

    for k in 1..=window_size {
        let blocks = extract_blocks(time_series, k);
        let probs = estimate_probabilities(&blocks);
        let h_k = shannon_entropy(&probs);
        block_entropies.push(h_k);
    }

    // Entropy rate is the limit: H'(X) = lim_{n→∞} [H(X_n) - H(X_{n-1})]
    if block_entropies.len() >= 2 {
        block_entropies.last().unwrap() - block_entropies[block_entropies.len() - 2]
    } else {
        block_entropies.last().copied().unwrap_or(0.0)
    }
}

/// Calculate channel capacity C = max I(X;Y)
///
/// Maximum information that can be reliably transmitted
pub fn channel_capacity(snr_db: f64) -> f64 {
    // Shannon-Hartley theorem: C = log₂(1 + SNR)
    let snr_linear = 10_f64.powf(snr_db / 10.0);
    (1.0 + snr_linear).log2()
}

/// Calculate Fisher information for parameter estimation
///
/// Measures how much information data carries about parameter θ
pub fn fisher_information(observations: &[f64], parameter_estimate: f64) -> f64 {
    // For Gaussian: I(θ) = n/σ²
    // This is a simplified version - real implementation depends on likelihood model

    let n = observations.len() as f64;
    let variance = compute_variance(observations, parameter_estimate);

    if variance > 1e-10 {
        n / variance
    } else {
        f64::INFINITY
    }
}

/// Calculate Kullback-Leibler divergence D_KL(P || Q)
///
/// Measures how much distribution P diverges from Q
pub fn kl_divergence(p_probs: &[f64], q_probs: &[f64]) -> f64 {
    assert_eq!(p_probs.len(), q_probs.len(), "Probability distributions must have same length");

    p_probs
        .iter()
        .zip(q_probs.iter())
        .filter(|&(&p, &q)| p > 0.0 && q > 0.0)
        .map(|(&p, &q)| p * (p / q).log2())
        .sum()
}

/// Estimate information metrics from sensor data
pub fn estimate_sensor_info_metrics(
    sensor_data: &[f64],
    reference_signal: Option<&[f64]>,
    noise_level: f64,
) -> InfoTheoryMetrics {
    // Discretize continuous data for entropy estimation
    let discretized = sensor_data.iter().map(|&x| discretize(x)).collect::<Vec<_>>();
    let probs = estimate_discrete_probabilities(&discretized);

    let entropy = shannon_entropy(&probs);
    let entropy_rate = entropy_rate(sensor_data, 5);

    // Calculate SNR and channel capacity
    let signal_power = compute_variance(sensor_data, 0.0);
    let snr_db = 10.0 * (signal_power / (noise_level * noise_level)).log10();
    let capacity = channel_capacity(snr_db);

    let mut metrics = InfoTheoryMetrics {
        entropy,
        entropy_rate,
        channel_capacity: Some(capacity),
        ..Default::default()
    };

    // If reference signal provided, calculate mutual information and transfer entropy
    if let Some(reference) = reference_signal {
        if reference.len() >= sensor_data.len() {
            let ref_discretized = reference[..sensor_data.len()]
                .iter()
                .map(|&x| discretize(x))
                .collect::<Vec<_>>();
            let ref_probs = estimate_discrete_probabilities(&ref_discretized);

            // Build joint probability distribution
            let joint_probs = build_joint_distribution(&discretized, &ref_discretized);

            metrics.mutual_information = Some(mutual_information(&probs, &ref_probs, &joint_probs));

            // Calculate transfer entropy (reference → sensor)
            if sensor_data.len() > 2 && reference.len() > sensor_data.len() {
                let te = transfer_entropy(
                    &reference[..sensor_data.len() - 1],
                    &sensor_data[..sensor_data.len() - 1],
                    &sensor_data[1..],
                );
                metrics.transfer_entropy = Some(te);
            }
        }
    }

    // Fisher information
    let mean = sensor_data.iter().sum::<f64>() / sensor_data.len() as f64;
    metrics.fisher_information = Some(fisher_information(sensor_data, mean));

    metrics
}

// Helper functions

fn discretize(value: f64) -> i32 {
    // Discretize to 100 bins
    (value * 10.0).round() as i32
}

fn estimate_probabilities(blocks: &[Vec<i32>]) -> Vec<f64> {
    let mut counts: HashMap<Vec<i32>, u32> = HashMap::new();
    for block in blocks {
        *counts.entry(block.clone()).or_insert(0) += 1;
    }

    let total = blocks.len() as f64;
    counts.values().map(|&count| count as f64 / total).collect()
}

fn estimate_discrete_probabilities(values: &[i32]) -> Vec<f64> {
    let mut counts: HashMap<i32, u32> = HashMap::new();
    for &val in values {
        *counts.entry(val).or_insert(0) += 1;
    }

    let total = values.len() as f64;
    counts.values().map(|&count| count as f64 / total).collect()
}

fn extract_blocks(series: &[f64], block_size: usize) -> Vec<Vec<i32>> {
    series
        .windows(block_size)
        .map(|window| window.iter().map(|&x| discretize(x)).collect())
        .collect()
}

fn build_joint_distribution(x: &[i32], y: &[i32]) -> Vec<(f64, f64, f64)> {
    let mut counts: HashMap<(i32, i32), u32> = HashMap::new();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        *counts.entry((xi, yi)).or_insert(0) += 1;
    }

    let total = x.len() as f64;
    counts
        .into_iter()
        .map(|((xi, yi), count)| (xi as f64, yi as f64, count as f64 / total))
        .collect()
}

fn conditional_entropy_from_counts(counts: &HashMap<(i32, i32), u32>, total: f64) -> f64 {
    let mut h = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total;
            h -= p * p.log2();
        }
    }
    h
}

fn conditional_entropy_3d(counts: &HashMap<(i32, i32, i32), u32>, total: f64) -> f64 {
    let mut h = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total;
            h -= p * p.log2();
        }
    }
    h
}

fn compute_variance(data: &[f64], mean: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&uniform);
        assert!((h - 2.0).abs() < 1e-10); // log₂(4) = 2

        // Deterministic has zero entropy
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&deterministic);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_channel_capacity() {
        // 10 dB SNR
        let c = channel_capacity(10.0);
        assert!(c > 3.0 && c < 4.0); // Should be ~3.46 bits
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let d = kl_divergence(&p, &q);
        assert!(d.abs() < 1e-10); // Same distribution → 0 divergence
    }

    #[test]
    fn test_estimate_sensor_metrics() {
        let sensor_data = vec![1.0, 1.5, 2.0, 1.8, 2.2, 1.9, 2.1];
        let metrics = estimate_sensor_info_metrics(&sensor_data, None, 0.1);

        assert!(metrics.entropy > 0.0);
        assert!(metrics.channel_capacity.is_some());
        assert!(metrics.fisher_information.is_some());
    }
}
