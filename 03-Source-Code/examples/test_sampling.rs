//! Sampling Strategies Demonstration

use anyhow::Result;
use prism_ai::orchestration::local_llm::{TokenSampler, SamplingConfig};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════╗");
    println!("║  SAMPLING STRATEGIES DEMONSTRATION       ║");
    println!("╚══════════════════════════════════════════╝\n");

    // Create sample logits (vocabulary of 10 tokens)
    let logits = vec![0.1, 0.8, 0.3, 1.5, 0.2, 0.9, 0.4, 1.2, 0.6, 0.5];
    println!("Sample logits: {:?}\n", logits);

    test_greedy_sampling(&logits)?;
    test_temperature_sampling(&logits)?;
    test_top_k_sampling(&logits)?;
    test_top_p_sampling(&logits)?;
    test_min_p_sampling(&logits)?;
    test_preset_configs(&logits)?;

    println!("\n╔══════════════════════════════════════════╗");
    println!("║  ALL DEMONSTRATIONS COMPLETED            ║");
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

fn test_greedy_sampling(logits: &[f32]) -> Result<()> {
    println!("═══ TEST 1: Greedy Sampling (Deterministic) ═══\n");

    let config = SamplingConfig::greedy();
    let sampler = TokenSampler::new(config);

    println!("Config:");
    println!("  Temperature: {}", sampler.config().temperature);
    println!("  Top-k: {}", sampler.config().top_k);

    println!("\nSampling 10 times:");
    for i in 0..10 {
        let token = sampler.sample(logits, &[])?;
        print!("{} ", token);
        if (i + 1) % 10 == 0 {
            println!();
        }
    }

    println!("\n✓ Greedy sampling always picks highest logit (token 3)\n");

    Ok(())
}

fn test_temperature_sampling(logits: &[f32]) -> Result<()> {
    println!("═══ TEST 2: Temperature Sampling ═══\n");

    let temps = vec![0.1, 0.5, 1.0, 1.5, 2.0];

    for temp in temps {
        let config = SamplingConfig {
            temperature: temp,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);

        print!("Temperature {:.1}: ", temp);
        for _ in 0..20 {
            let token = sampler.sample(logits, &[])?;
            print!("{}", token);
        }
        println!();
    }

    println!("\n✓ Lower temperature = more deterministic (favors high logits)");
    println!("✓ Higher temperature = more random (explores more tokens)\n");

    Ok(())
}

fn test_top_k_sampling(logits: &[f32]) -> Result<()> {
    println!("═══ TEST 3: Top-k Sampling ═══\n");

    let top_ks = vec![1, 2, 5];

    for k in top_ks {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: k,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);

        print!("Top-k={}: ", k);
        let mut tokens = std::collections::HashSet::new();
        for _ in 0..50 {
            let token = sampler.sample(logits, &[])?;
            tokens.insert(token);
        }
        println!("Unique tokens sampled: {:?}", tokens);
    }

    println!("\n✓ Top-k limits sampling to k most likely tokens\n");

    Ok(())
}

fn test_top_p_sampling(logits: &[f32]) -> Result<()> {
    println!("═══ TEST 4: Top-p (Nucleus) Sampling ═══\n");

    let top_ps = vec![0.5, 0.9, 0.95];

    for p in top_ps {
        let config = SamplingConfig {
            temperature: 1.0,
            top_p: p,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);

        print!("Top-p={:.2}: ", p);
        let mut tokens = std::collections::HashSet::new();
        for _ in 0..50 {
            let token = sampler.sample(logits, &[])?;
            tokens.insert(token);
        }
        println!("Unique tokens sampled: {:?}", tokens);
    }

    println!("\n✓ Top-p samples from smallest set with cumulative probability >= p\n");

    Ok(())
}

fn test_min_p_sampling(logits: &[f32]) -> Result<()> {
    println!("═══ TEST 5: Min-p Sampling (2025 State-of-the-Art) ═══\n");

    let config = SamplingConfig::min_p_recommended();
    let sampler = TokenSampler::new(config);

    println!("Config (Recommended by major providers):");
    println!("  Min-p: {}", sampler.config().min_p);
    println!("  Temperature: {}", sampler.config().temperature);

    print!("\nSampled tokens: ");
    for _ in 0..20 {
        let token = sampler.sample(logits, &[])?;
        print!("{}", token);
    }
    println!();

    println!("\n✓ Min-p dynamically filters based on top token probability");
    println!("✓ More adaptive than top-p across different distributions\n");

    Ok(())
}

fn test_preset_configs(logits: &[f32]) -> Result<()> {
    println!("═══ TEST 6: Preset Configurations ═══\n");

    let presets = vec![
        ("Standard", SamplingConfig::standard()),
        ("Creative", SamplingConfig::creative()),
        ("Precise", SamplingConfig::precise()),
    ];

    for (name, config) in presets {
        let sampler = TokenSampler::new(config);

        println!("{} Config:", name);
        println!("  Temperature: {}", sampler.config().temperature);
        println!("  Top-k: {}", sampler.config().top_k);
        println!("  Top-p: {}", sampler.config().top_p);
        println!("  Repetition penalty: {}", sampler.config().repetition_penalty);

        print!("  Sample: ");
        for _ in 0..20 {
            let token = sampler.sample(logits, &[])?;
            print!("{}", token);
        }
        println!("\n");
    }

    println!("✓ Different presets for different use cases");
    println!("  - Standard: Balanced quality and diversity");
    println!("  - Creative: More exploration and variety");
    println!("  - Precise: Conservative and focused\n");

    Ok(())
}
