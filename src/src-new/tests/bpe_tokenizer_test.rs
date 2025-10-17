//! Comprehensive tests for BPE tokenizer

use prism_ai::orchestration::local_llm::{BPETokenizer, SpecialTokens};
use anyhow::Result;

#[test]
fn test_tokenizer_initialization() {
    let tokenizer = BPETokenizer::new();
    assert_eq!(tokenizer.vocab_size(), 256);
    assert_eq!(tokenizer.merges().len(), 0);
}

#[test]
fn test_byte_level_encode_simple() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let tokens = tokenizer.encode("A", false)?;
    assert_eq!(tokens, vec![65]); // ASCII 'A'

    let tokens = tokenizer.encode("Hello", false)?;
    assert_eq!(tokens, vec![72, 101, 108, 108, 111]); // H e l l o

    Ok(())
}

#[test]
fn test_byte_level_decode_simple() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = tokenizer.decode(&[65], false)?;
    assert_eq!(text, "A");

    let text = tokenizer.decode(&[72, 101, 108, 108, 111], false)?;
    assert_eq!(text, "Hello");

    Ok(())
}

#[test]
fn test_encode_decode_roundtrip() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let original = "The quick brown fox jumps over the lazy dog";
    let tokens = tokenizer.encode(original, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, original);

    Ok(())
}

#[test]
fn test_unicode_encoding() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    // Test various Unicode scripts
    let test_cases = vec![
        "Hello 世界",           // Chinese
        "Привет мир",          // Russian
        "مرحبا",               // Arabic
        "こんにちは",           // Japanese
        "🚀🌟",                 // Emojis
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text, false)?;
        let decoded = tokenizer.decode(&tokens, false)?;
        assert_eq!(decoded, text, "Failed for: {}", text);
    }

    Ok(())
}

#[test]
fn test_special_tokens_bos_eos() -> Result<()> {
    let mut tokenizer = BPETokenizer::new();

    tokenizer.register_special_tokens(SpecialTokens {
        bos_token: Some(1),
        eos_token: Some(2),
        pad_token: Some(0),
        unk_token: Some(3),
    });

    // Encode with special tokens
    let tokens = tokenizer.encode("Hello", true)?;

    assert_eq!(tokens[0], 1); // BOS
    assert_eq!(tokens[tokens.len() - 1], 2); // EOS
    assert!(tokens.len() > 2); // Should have content in middle

    Ok(())
}

#[test]
fn test_special_tokens_skip_decoding() -> Result<()> {
    let mut tokenizer = BPETokenizer::new();

    tokenizer.register_special_tokens(SpecialTokens {
        bos_token: Some(1),
        eos_token: Some(2),
        pad_token: Some(0),
        unk_token: None,
    });

    let tokens = vec![1, 72, 105, 2]; // BOS H i EOS

    // Decode with special tokens
    let text_with = tokenizer.decode(&tokens, false)?;
    assert!(text_with.len() > 2);

    // Decode skipping special tokens
    let text_without = tokenizer.decode(&tokens, true)?;
    assert_eq!(text_without, "Hi");

    Ok(())
}

#[test]
fn test_train_basic() -> Result<()> {
    let text = "aaabdaaabac";
    let tokenizer = BPETokenizer::train(text, 260)?;

    assert_eq!(tokenizer.vocab_size(), 260);
    assert_eq!(tokenizer.merges().len(), 4); // 260 - 256 = 4 merges

    // Verify can encode and decode
    let tokens = tokenizer.encode("aaa", false)?;
    let decoded = tokenizer.decode(&tokens, false)?;
    assert_eq!(decoded, "aaa");

    Ok(())
}

#[test]
fn test_train_compression() -> Result<()> {
    // Train on repeated patterns
    let text = "hello hello hello world world";
    let tokenizer = BPETokenizer::train(text, 300)?;

    // Encode the same text
    let tokens = tokenizer.encode("hello hello", false)?;

    // Should be fewer tokens than byte-level due to merges
    assert!(tokens.len() < "hello hello".len());

    Ok(())
}

#[test]
fn test_vocab_lookups() {
    let tokenizer = BPETokenizer::new();

    // Test ASCII lookups
    for i in 0..128 {
        let ch = i as u8 as char;
        let token_str = ch.to_string();

        if let Some(token_id) = tokenizer.token_to_id(&token_str) {
            assert_eq!(token_id, i);

            let recovered = tokenizer.id_to_token(i);
            assert_eq!(recovered, Some(token_str));
        }
    }
}

#[test]
fn test_empty_input() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let tokens = tokenizer.encode("", false)?;
    assert_eq!(tokens.len(), 0);

    let decoded = tokenizer.decode(&[], false)?;
    assert_eq!(decoded, "");

    Ok(())
}

#[test]
fn test_single_character() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let tokens = tokenizer.encode("a", false)?;
    assert_eq!(tokens, vec![97]); // ASCII 'a'

    let decoded = tokenizer.decode(&[97], false)?;
    assert_eq!(decoded, "a");

    Ok(())
}

#[test]
fn test_whitespace() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = "Hello World";
    let tokens = tokenizer.encode(text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);
    assert!(tokens.contains(&32)); // Space character

    Ok(())
}

#[test]
fn test_newlines_tabs() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = "Line1\nLine2\tTab";
    let tokens = tokenizer.encode(text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);

    Ok(())
}

#[test]
fn test_long_text() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    // Generate long text
    let text = "The quick brown fox ".repeat(100);

    let tokens = tokenizer.encode(&text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);
    assert_eq!(tokens.len(), text.len()); // Byte-level

    Ok(())
}

#[test]
fn test_numbers() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = "12345 67890";
    let tokens = tokenizer.encode(text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);

    Ok(())
}

#[test]
fn test_special_characters() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
    let tokens = tokenizer.encode(text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);

    Ok(())
}

#[test]
fn test_train_merge_order() -> Result<()> {
    // Train on text where we can predict merge order
    let text = "aaabbbaaabbbaaa";
    let tokenizer = BPETokenizer::train(text, 258)?; // 256 + 2 merges

    assert_eq!(tokenizer.merges().len(), 2);

    // First merge should be most frequent pair
    let (first_pair, _) = tokenizer.merges()[0];

    // Should merge either (a,a) or (b,b) as they're most frequent
    let a_id = 97; // ASCII 'a'
    let b_id = 98; // ASCII 'b'

    assert!(first_pair == (a_id, a_id) || first_pair == (b_id, b_id));

    Ok(())
}

#[test]
fn test_default_special_tokens() {
    let tokenizer = BPETokenizer::new();
    let special = tokenizer.special_tokens();

    assert_eq!(special.bos_token, None);
    assert_eq!(special.eos_token, None);
    assert_eq!(special.pad_token, None);
    assert_eq!(special.unk_token, None);
}

#[test]
fn test_tokenizer_default_trait() {
    let tokenizer = BPETokenizer::default();
    assert_eq!(tokenizer.vocab_size(), 256);
}

#[test]
fn test_case_sensitivity() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let upper = tokenizer.encode("HELLO", false)?;
    let lower = tokenizer.encode("hello", false)?;

    assert_ne!(upper, lower); // Case should matter

    Ok(())
}

#[test]
fn test_consecutive_spaces() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    let text = "Hello    World"; // Multiple spaces
    let tokens = tokenizer.encode(text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);

    Ok(())
}

#[test]
fn test_utf8_boundaries() -> Result<()> {
    let tokenizer = BPETokenizer::new();

    // Multi-byte UTF-8 characters
    let text = "café naïve résumé";
    let tokens = tokenizer.encode(text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    assert_eq!(decoded, text);

    Ok(())
}
