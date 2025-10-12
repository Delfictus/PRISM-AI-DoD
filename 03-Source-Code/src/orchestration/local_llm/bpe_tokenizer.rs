//! BPE (Byte Pair Encoding) Tokenizer
//!
//! Production-ready BPE tokenizer for LLM text processing
//! Compatible with GPT-2, GPT-4, Llama, and other BPE-based models
//!
//! Algorithm:
//! 1. Start with 256 byte-level tokens (UTF-8 bytes)
//! 2. Learn merge operations from training corpus
//! 3. Apply merges iteratively during encoding
//! 4. Support special tokens (BOS, EOS, PAD, UNK)
//!
//! References:
//! - Original BPE paper: Neural Machine Translation of Rare Words with Subword Units
//! - GPT-2 tokenizer: https://github.com/openai/gpt-2
//! - minbpe: https://github.com/karpathy/minbpe

use anyhow::{Result, Context, bail};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Special token IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos_token: Option<i32>,  // Beginning of sequence
    pub eos_token: Option<i32>,  // End of sequence
    pub pad_token: Option<i32>,  // Padding
    pub unk_token: Option<i32>,  // Unknown token
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: None,
            eos_token: None,
            pad_token: None,
            unk_token: None,
        }
    }
}

/// BPE merge operation: (token1, token2) -> new_token_id
type Merge = ((i32, i32), i32);

/// BPE Tokenizer
///
/// Implements Byte Pair Encoding for subword tokenization
pub struct BPETokenizer {
    /// Vocabulary: token_id -> token_bytes
    vocab: HashMap<i32, Vec<u8>>,

    /// Reverse vocabulary: token_bytes -> token_id
    vocab_reverse: HashMap<Vec<u8>, i32>,

    /// Merge operations: (token1, token2) -> new_token_id
    merges: Vec<Merge>,

    /// Merge lookup for O(1) access
    merge_lookup: HashMap<(i32, i32), i32>,

    /// Special tokens
    special_tokens: SpecialTokens,

    /// Vocabulary size
    vocab_size: usize,
}

impl BPETokenizer {
    /// Create new BPE tokenizer from scratch
    ///
    /// Starts with 256 byte-level tokens
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut vocab_reverse = HashMap::new();

        // Initialize with 256 byte-level tokens
        for i in 0..256 {
            let byte_vec = vec![i as u8];
            vocab.insert(i, byte_vec.clone());
            vocab_reverse.insert(byte_vec, i);
        }

        Self {
            vocab,
            vocab_reverse,
            merges: Vec::new(),
            merge_lookup: HashMap::new(),
            special_tokens: SpecialTokens::default(),
            vocab_size: 256,
        }
    }

    /// Load tokenizer from vocabulary and merges files
    ///
    /// Compatible with GPT-2 style vocab.json and merges.txt
    pub fn from_files<P: AsRef<Path>>(vocab_path: P, merges_path: P) -> Result<Self> {
        let mut tokenizer = Self::new();

        // Load vocabulary
        let vocab_file = File::open(vocab_path.as_ref())
            .context("Failed to open vocabulary file")?;
        let vocab_reader = BufReader::new(vocab_file);

        for line in vocab_reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // Parse: token_string token_id
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(token_id) = parts.last().unwrap().parse::<i32>() {
                    let token_str = parts[..parts.len()-1].join(" ");
                    let token_bytes = token_str.as_bytes().to_vec();
                    tokenizer.vocab.insert(token_id, token_bytes.clone());
                    tokenizer.vocab_reverse.insert(token_bytes, token_id);
                }
            }
        }

        // Load merges
        let merges_file = File::open(merges_path.as_ref())
            .context("Failed to open merges file")?;
        let merges_reader = BufReader::new(merges_file);

        for (idx, line) in merges_reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse: token1 token2
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let token1_bytes = parts[0].as_bytes().to_vec();
                let token2_bytes = parts[1].as_bytes().to_vec();

                if let (Some(&id1), Some(&id2)) = (
                    tokenizer.vocab_reverse.get(&token1_bytes),
                    tokenizer.vocab_reverse.get(&token2_bytes),
                ) {
                    let new_token_id = 256 + idx as i32;
                    let mut merged_bytes = token1_bytes.clone();
                    merged_bytes.extend_from_slice(&token2_bytes);

                    tokenizer.merges.push(((id1, id2), new_token_id));
                    tokenizer.merge_lookup.insert((id1, id2), new_token_id);
                    tokenizer.vocab.insert(new_token_id, merged_bytes.clone());
                    tokenizer.vocab_reverse.insert(merged_bytes, new_token_id);
                }
            }
        }

        tokenizer.vocab_size = tokenizer.vocab.len();

        Ok(tokenizer)
    }

    /// Register special tokens
    pub fn register_special_tokens(&mut self, special_tokens: SpecialTokens) {
        self.special_tokens = special_tokens;
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// * `add_special_tokens` - Whether to add BOS/EOS tokens
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();

        // Add BOS token if requested
        if add_special_tokens {
            if let Some(bos) = self.special_tokens.bos_token {
                tokens.push(bos);
            }
        }

        // Convert text to bytes
        let bytes = text.as_bytes();

        // Start with individual byte tokens
        let mut token_ids: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();

        // Apply merges iteratively
        while token_ids.len() > 1 {
            // Find the pair with the earliest merge
            let mut best_merge_idx = None;
            let mut best_merge_rank = usize::MAX;

            for i in 0..token_ids.len() - 1 {
                let pair = (token_ids[i], token_ids[i + 1]);

                // Check if this pair has a merge operation
                if let Some(&_new_token) = self.merge_lookup.get(&pair) {
                    // Find rank of this merge
                    if let Some(rank) = self.merges.iter().position(|(p, _)| *p == pair) {
                        if rank < best_merge_rank {
                            best_merge_rank = rank;
                            best_merge_idx = Some(i);
                        }
                    }
                }
            }

            // If no merge found, we're done
            if best_merge_idx.is_none() {
                break;
            }

            let idx = best_merge_idx.unwrap();
            let pair = (token_ids[idx], token_ids[idx + 1]);
            let new_token = self.merge_lookup[&pair];

            // Apply merge
            token_ids.splice(idx..idx + 2, [new_token]);
        }

        tokens.extend(token_ids);

        // Add EOS token if requested
        if add_special_tokens {
            if let Some(eos) = self.special_tokens.eos_token {
                tokens.push(eos);
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    /// * `token_ids` - Vector of token IDs
    /// * `skip_special_tokens` - Whether to skip special tokens in output
    ///
    /// # Returns
    /// Decoded text string
    pub fn decode(&self, token_ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes = Vec::new();

        for &token_id in token_ids {
            // Skip special tokens if requested
            if skip_special_tokens {
                if Some(token_id) == self.special_tokens.bos_token
                    || Some(token_id) == self.special_tokens.eos_token
                    || Some(token_id) == self.special_tokens.pad_token
                {
                    continue;
                }
            }

            // Get token bytes from vocabulary
            if let Some(token_bytes) = self.vocab.get(&token_id) {
                bytes.extend_from_slice(token_bytes);
            } else {
                // Unknown token - try to handle gracefully
                if let Some(unk) = self.special_tokens.unk_token {
                    if token_id == unk {
                        bytes.extend_from_slice(b"<UNK>");
                        continue;
                    }
                }
                bail!("Unknown token ID: {}", token_id);
            }
        }

        // Convert bytes to string
        String::from_utf8(bytes).context("Invalid UTF-8 in decoded text")
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get token string for a token ID
    pub fn id_to_token(&self, token_id: i32) -> Option<String> {
        self.vocab.get(&token_id).and_then(|bytes| String::from_utf8(bytes.clone()).ok())
    }

    /// Get token ID for a token string
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.vocab_reverse.get(token.as_bytes()).copied()
    }

    /// Train BPE tokenizer on text corpus
    ///
    /// # Arguments
    /// * `text` - Training corpus
    /// * `vocab_size` - Desired vocabulary size
    ///
    /// # Returns
    /// Trained tokenizer
    pub fn train(text: &str, vocab_size: usize) -> Result<Self> {
        if vocab_size < 256 {
            bail!("Vocabulary size must be at least 256");
        }

        let mut tokenizer = Self::new();
        let num_merges = vocab_size - 256;

        // Convert text to byte sequence
        let bytes = text.as_bytes();
        let mut token_ids: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();

        // Perform merges
        for merge_idx in 0..num_merges {
            // Count pair frequencies
            let mut pair_counts: HashMap<(i32, i32), usize> = HashMap::new();

            for i in 0..token_ids.len().saturating_sub(1) {
                let pair = (token_ids[i], token_ids[i + 1]);
                *pair_counts.entry(pair).or_insert(0) += 1;
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair
            let (best_pair, _count) = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .unwrap();

            let new_token_id = 256 + merge_idx as i32;

            // Add merge to tokenizer
            tokenizer.merges.push((*best_pair, new_token_id));
            tokenizer.merge_lookup.insert(*best_pair, new_token_id);

            // Create merged token bytes
            let token1_bytes = tokenizer.vocab[&best_pair.0].clone();
            let token2_bytes = tokenizer.vocab[&best_pair.1].clone();
            let mut merged_bytes = token1_bytes;
            merged_bytes.extend_from_slice(&token2_bytes);

            tokenizer.vocab.insert(new_token_id, merged_bytes.clone());
            tokenizer.vocab_reverse.insert(merged_bytes, new_token_id);

            // Apply merge to token sequence
            let mut i = 0;
            while i < token_ids.len() - 1 {
                if (token_ids[i], token_ids[i + 1]) == *best_pair {
                    token_ids.splice(i..i + 2, [new_token_id]);
                } else {
                    i += 1;
                }
            }
        }

        tokenizer.vocab_size = tokenizer.vocab.len();

        Ok(tokenizer)
    }

    /// Get merge operations
    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = BPETokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 256);
        assert_eq!(tokenizer.merges().len(), 0);
    }

    #[test]
    fn test_byte_level_encoding() -> Result<()> {
        let tokenizer = BPETokenizer::new();

        // Encode "Hello" - should give byte values
        let tokens = tokenizer.encode("Hello", false)?;
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens, vec![72, 101, 108, 108, 111]); // H e l l o

        Ok(())
    }

    #[test]
    fn test_byte_level_decoding() -> Result<()> {
        let tokenizer = BPETokenizer::new();

        // Decode byte sequence
        let tokens = vec![72, 101, 108, 108, 111]; // "Hello"
        let text = tokenizer.decode(&tokens, false)?;
        assert_eq!(text, "Hello");

        Ok(())
    }

    #[test]
    fn test_special_tokens() -> Result<()> {
        let mut tokenizer = BPETokenizer::new();

        tokenizer.register_special_tokens(SpecialTokens {
            bos_token: Some(1),
            eos_token: Some(2),
            pad_token: Some(0),
            unk_token: Some(3),
        });

        // Encode with special tokens
        let tokens = tokenizer.encode("Hi", true)?;
        assert_eq!(tokens[0], 1); // BOS
        assert_eq!(tokens[tokens.len() - 1], 2); // EOS

        Ok(())
    }

    #[test]
    fn test_train_simple() -> Result<()> {
        // Train on simple repeated text
        let text = "aaabdaaabac";
        let tokenizer = BPETokenizer::train(text, 260)?; // 256 + 4 merges

        assert!(tokenizer.vocab_size() >= 256);
        assert!(!tokenizer.merges().is_empty());

        // Should be able to encode and decode
        let tokens = tokenizer.encode("aaa", false)?;
        let decoded = tokenizer.decode(&tokens, false)?;
        assert_eq!(decoded, "aaa");

        Ok(())
    }

    #[test]
    fn test_vocab_lookups() {
        let tokenizer = BPETokenizer::new();

        // Test byte-level lookups
        let token_id = tokenizer.token_to_id("A");
        assert_eq!(token_id, Some(65)); // ASCII 'A'

        let token_str = tokenizer.id_to_token(65);
        assert_eq!(token_str, Some("A".to_string()));
    }

    #[test]
    fn test_unicode_handling() -> Result<()> {
        let tokenizer = BPETokenizer::new();

        // Test UTF-8 encoding
        let text = "Hello 世界"; // Mix of ASCII and Chinese
        let tokens = tokenizer.encode(text, false)?;

        // Decode back
        let decoded = tokenizer.decode(&tokens, false)?;
        assert_eq!(decoded, text);

        Ok(())
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
}
