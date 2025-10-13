//! BPE Tokenizer Demonstration
//!
//! Shows BPE tokenization and training
//!
//! Usage:
//!   cargo run --example test_bpe_tokenizer --features cuda

use anyhow::Result;
use prism_ai::orchestration::local_llm::{BPETokenizer, SpecialTokens};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════╗");
    println!("║  BPE TOKENIZER DEMONSTRATION             ║");
    println!("╚══════════════════════════════════════════╝\n");

    // Test 1: Byte-level tokenization
    println!("═══ TEST 1: Byte-Level Tokenization ═══\n");
    test_byte_level()?;

    // Test 2: Special tokens
    println!("\n═══ TEST 2: Special Tokens ═══\n");
    test_special_tokens()?;

    // Test 3: Train tokenizer
    println!("\n═══ TEST 3: Training Tokenizer ═══\n");
    test_training()?;

    // Test 4: Unicode handling
    println!("\n═══ TEST 4: Unicode Support ═══\n");
    test_unicode()?;

    // Test 5: Compression demonstration
    println!("\n═══ TEST 5: Compression via BPE ═══\n");
    test_compression()?;

    println!("\n╔══════════════════════════════════════════╗");
    println!("║  ALL TESTS COMPLETED                     ║");
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

fn test_byte_level() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = "Hello, World!";
    println!("Original text: \"{}\"", text);

    let tokens = tokenizer.encode(text, false)?;
    println!("Token IDs: {:?}", tokens);
    println!("Token count: {}", tokens.len());

    let decoded = tokenizer.decode(&tokens, false)?;
    println!("Decoded text: \"{}\"", decoded);

    assert_eq!(text, decoded);
    println!("✓ Byte-level encoding/decoding successful");

    Ok(())
}

fn test_special_tokens() -> Result<()> {
    let mut tokenizer = BPETokenizer::new();

    // Register special tokens
    tokenizer.register_special_tokens(SpecialTokens {
        bos_token: Some(1),
        eos_token: Some(2),
        pad_token: Some(0),
        unk_token: Some(3),
    });

    println!("Registered special tokens:");
    println!("  BOS (Beginning of Sequence): 1");
    println!("  EOS (End of Sequence): 2");
    println!("  PAD (Padding): 0");
    println!("  UNK (Unknown): 3");

    let text = "Hello";
    println!("\nText: \"{}\"", text);

    // Encode without special tokens
    let tokens_plain = tokenizer.encode(text, false)?;
    println!("Without special tokens: {:?}", tokens_plain);

    // Encode with special tokens
    let tokens_special = tokenizer.encode(text, true)?;
    println!("With special tokens: {:?}", tokens_special);

    assert_eq!(tokens_special[0], 1); // BOS
    assert_eq!(tokens_special[tokens_special.len() - 1], 2); // EOS

    // Decode skipping special tokens
    let decoded = tokenizer.decode(&tokens_special, true)?;
    assert_eq!(decoded, text);

    println!("✓ Special tokens working correctly");

    Ok(())
}

fn test_training() -> Result<()> {
    // Training corpus with repeated patterns
    let corpus = "\
        The quick brown fox jumps over the lazy dog.\
        The dog was lazy, but the fox was quick.\
        Quick foxes and lazy dogs are common.\
        The brown fox and the lazy dog met.\
    ";

    println!("Training corpus: {} bytes", corpus.len());
    println!("Training vocab size: 300 tokens\n");

    // Train tokenizer
    let tokenizer = BPETokenizer::train(corpus, 300)?;

    println!("Trained tokenizer:");
    println!("  Vocabulary size: {}", tokenizer.vocab_size());
    println!("  Number of merges: {}", tokenizer.merges().len());

    // Show some learned merges
    println!("\nFirst 5 learned merges:");
    for (i, ((t1, t2), new_id)) in tokenizer.merges().iter().take(5).enumerate() {
        let token1 = tokenizer.id_to_token(*t1).unwrap_or_else(|| format!("#{}", t1));
        let token2 = tokenizer.id_to_token(*t2).unwrap_or_else(|| format!("#{}", t2));
        println!("  {}. ({}, {}) -> {}", i + 1, token1, token2, new_id);
    }

    // Test encoding with trained tokenizer
    let test_text = "The quick fox";
    let tokens = tokenizer.encode(test_text, false)?;
    println!("\nTest encoding \"{}\":", test_text);
    println!("  Tokens: {:?}", tokens);
    println!("  Token count: {} (vs {} bytes)", tokens.len(), test_text.len());

    let decoded = tokenizer.decode(&tokens, false)?;
    assert_eq!(decoded, test_text);

    println!("✓ Training successful");

    Ok(())
}

fn test_unicode() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let test_cases = vec![
        ("English", "Hello World"),
        ("Chinese", "你好世界"),
        ("Japanese", "こんにちは"),
        ("Arabic", "مرحبا"),
        ("Russian", "Привет"),
        ("Emoji", "🚀🌟⭐"),
        ("Mixed", "Hello 世界 🌍"),
    ];

    println!("Testing Unicode support:\n");

    for (lang, text) in test_cases {
        let tokens = tokenizer.encode(text, false)?;
        let decoded = tokenizer.decode(&tokens, false)?;

        println!("{:10} \"{}\"", lang, text);
        println!("           {} bytes -> {} tokens", text.len(), tokens.len());

        assert_eq!(decoded, text);
    }

    println!("\n✓ All Unicode tests passed");

    Ok(())
}

fn test_compression() -> Result<()> {
    // Demonstrate compression effect of BPE

    let text = "hello hello hello world world world test test";
    println!("Original text: \"{}\"", text);
    println!("Text length: {} bytes\n", text.len());

    // Byte-level tokenizer (no merges)
    let tokenizer_base = BPETokenizer::new();
    let tokens_base = tokenizer_base.encode(text, false)?;
    println!("Byte-level tokenization:");
    println!("  Token count: {} (1 token per byte)", tokens_base.len());

    // Train tokenizer with merges
    let training_text = text.repeat(10); // More training data for better merges
    let tokenizer_trained = BPETokenizer::train(&training_text, 300)?;
    let tokens_trained = tokenizer_trained.encode(text, false)?;
    println!("\nBPE tokenization (with {} merges):", tokenizer_trained.merges().len());
    println!("  Token count: {}", tokens_trained.len());

    let compression_ratio = tokens_base.len() as f64 / tokens_trained.len() as f64;
    println!("\nCompression ratio: {:.2}x", compression_ratio);
    println!("Token reduction: {:.1}%",
        (1.0 - tokens_trained.len() as f64 / tokens_base.len() as f64) * 100.0
    );

    // Verify correctness
    let decoded = tokenizer_trained.decode(&tokens_trained, false)?;
    assert_eq!(decoded, text);

    println!("✓ Compression demonstration complete");

    Ok(())
}
