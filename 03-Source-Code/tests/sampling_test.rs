//! Comprehensive tests for sampling strategies

use prism_ai::orchestration::local_llm::{TokenSampler, SamplingConfig};
use anyhow::Result;

#[test]
fn test_sampler_creation() {
    let sampler = TokenSampler::default();
    assert_eq!(sampler.config().temperature, 1.0);
}

#[test]
fn test_greedy_sampling_deterministic() -> Result<()> {
    let config = SamplingConfig::greedy();
    let sampler = TokenSampler::new(config);

    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];

    // Should always return same token
    for _ in 0..10 {
        let token = sampler.sample(&logits, &[])?;
        assert_eq!(token, 3); // Token with highest logit
    }

    Ok(())
}

#[test]
fn test_temperature_sampling() -> Result<()> {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Low temperature should favor highest logit
    let low_temp = TokenSampler::new(SamplingConfig {
        temperature: 0.1,
        ..Default::default()
    });

    let mut high_logit_count = 0;
    for _ in 0..100 {
        let token = low_temp.sample(&logits, &[])?;
        if token == 4 { // Highest logit index
            high_logit_count += 1;
        }
    }

    // Should heavily favor highest logit
    assert!(high_logit_count > 90);

    Ok(())
}

#[test]
fn test_top_k_filtering() -> Result<()> {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 2,
        ..Default::default()
    };
    let sampler = TokenSampler::new(config);

    let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];

    // Sample multiple times
    let mut tokens = std::collections::HashSet::new();
    for _ in 0..50 {
        let token = sampler.sample(&logits, &[])?;
        tokens.insert(token);
    }

    // Should only sample from top-2 tokens (indices 1 and 3)
    assert!(tokens.len() <= 2);
    assert!(tokens.contains(&1) || tokens.contains(&3));

    Ok(())
}

#[test]
fn test_config_presets() {
    let greedy = SamplingConfig::greedy();
    assert_eq!(greedy.temperature, 0.0);
    assert_eq!(greedy.top_k, 1);

    let standard = SamplingConfig::standard();
    assert_eq!(standard.temperature, 0.7);
    assert_eq!(standard.top_k, 50);
    assert_eq!(standard.top_p, 0.9);

    let creative = SamplingConfig::creative();
    assert_eq!(creative.temperature, 0.9);
    assert_eq!(creative.top_k, 100);
    assert_eq!(creative.top_p, 0.95);

    let precise = SamplingConfig::precise();
    assert_eq!(precise.temperature, 0.3);
    assert_eq!(precise.top_k, 10);

    let min_p = SamplingConfig::min_p_recommended();
    assert_eq!(min_p.min_p, 0.05);
}

#[test]
fn test_repetition_penalty() -> Result<()> {
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 1,
        repetition_penalty: 2.0,
        ..Default::default()
    };
    let sampler = TokenSampler::new(config);

    let logits = vec![1.0, 2.0, 3.0, 4.0];

    // Without previous tokens, should pick token 3
    let token1 = sampler.sample(&logits, &[])?;
    assert_eq!(token1, 3);

    // With token 3 in history, penalty should make it less likely
    let token2 = sampler.sample(&logits, &[3])?;
    // Should pick different token due to penalty
    assert_ne!(token2, 3);

    Ok(())
}

#[test]
fn test_sampling_with_uniform_logits() -> Result<()> {
    let sampler = TokenSampler::default();

    // All equal logits
    let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    // Should be able to sample any token
    let mut tokens = std::collections::HashSet::new();
    for _ in 0..100 {
        let token = sampler.sample(&logits, &[])?;
        tokens.insert(token);
    }

    // Should sample from all tokens eventually
    assert!(tokens.len() >= 3);

    Ok(())
}

#[test]
fn test_extreme_logits() -> Result<()> {
    let sampler = TokenSampler::default();

    // One very high logit
    let logits = vec![0.0, 0.0, 100.0, 0.0, 0.0];

    for _ in 0..10 {
        let token = sampler.sample(&logits, &[])?;
        assert_eq!(token, 2); // Should always pick the extreme value
    }

    Ok(())
}

#[test]
fn test_default_config() {
    let config = SamplingConfig::default();
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.top_k, 0);
    assert_eq!(config.top_p, 1.0);
    assert_eq!(config.min_p, 0.0);
    assert_eq!(config.repetition_penalty, 1.0);
}

#[test]
fn test_config_update() {
    let mut sampler = TokenSampler::default();
    assert_eq!(sampler.config().temperature, 1.0);

    let new_config = SamplingConfig::greedy();
    sampler.set_config(new_config);
    assert_eq!(sampler.config().temperature, 0.0);
}
