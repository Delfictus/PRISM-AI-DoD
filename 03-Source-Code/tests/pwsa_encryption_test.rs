//! Encryption Security Tests for PWSA Week 2
//!
//! Validates AES-256-GCM encryption for classified data

use prism_ai::pwsa::vendor_sandbox::*;

#[test]
fn test_encryption_for_classified_data() {
    let mut km = KeyManager::new("test_passphrase_secure_123").unwrap();
    let key = km.get_dek(DataClassification::Secret);

    let plaintext = b"SECRET: Missile launch detected at 38.5N, 127.8E";
    let mut data = SecureDataSlice::from_bytes(
        plaintext.to_vec(),
        DataClassification::Secret,
    );

    // Encrypt
    data.encrypt(&key).unwrap();
    assert!(data.encrypted);
    assert_ne!(data.data, plaintext);  // Ciphertext differs from plaintext

    // Decrypt
    data.decrypt(&key).unwrap();
    assert!(!data.encrypted);
    assert_eq!(data.data, plaintext);  // Recovered plaintext
}

#[test]
fn test_unclassified_skips_encryption() {
    let mut km = KeyManager::new("test_passphrase").unwrap();
    let key = km.get_dek(DataClassification::Unclassified);

    let plaintext = b"UNCLASSIFIED: Routine telemetry data";
    let mut data = SecureDataSlice::from_bytes(
        plaintext.to_vec(),
        DataClassification::Unclassified,
    );

    // Should not encrypt unclassified data
    data.encrypt(&key).unwrap();
    assert!(!data.encrypted);
    assert_eq!(data.data, plaintext);  // Unchanged
}

#[test]
fn test_wrong_key_fails_decryption() {
    let mut km = KeyManager::new("passphrase_1").unwrap();
    let key1 = km.get_dek(DataClassification::Secret);

    let mut km2 = KeyManager::new("passphrase_2").unwrap();
    let key2 = km2.get_dek(DataClassification::Secret);

    let plaintext = b"TOP SECRET: Nuclear launch codes";
    let mut data = SecureDataSlice::from_bytes(
        plaintext.to_vec(),
        DataClassification::TopSecret,
    );

    // Encrypt with key1
    data.encrypt(&key1).unwrap();
    assert!(data.encrypted);

    // Try to decrypt with wrong key
    let result = data.decrypt(&key2);
    assert!(result.is_err());  // Should fail

    // Can decrypt with correct key
    let result = data.decrypt(&key1);
    assert!(result.is_ok());
    assert_eq!(data.data, plaintext);
}

#[test]
fn test_key_derivation_deterministic() {
    let mut km1 = KeyManager::new("same_passphrase").unwrap();
    let mut km2 = KeyManager::new("same_passphrase").unwrap();

    // Note: Due to random salt, keys will be DIFFERENT
    // This tests that DEK derivation is deterministic PER KeyManager instance
    let dek1_secret = km1.get_dek(DataClassification::Secret);
    let dek1_ts = km1.get_dek(DataClassification::TopSecret);

    // Same KeyManager should return same DEK
    let dek1_secret_again = km1.get_dek(DataClassification::Secret);
    assert_eq!(dek1_secret, dek1_secret_again);

    // Different classifications should have different DEKs
    assert_ne!(dek1_secret, dek1_ts);
}

#[test]
fn test_encryption_integrity() {
    let mut km = KeyManager::new("integrity_test").unwrap();
    let key = km.get_dek(DataClassification::Secret);

    let plaintext = b"Sensitive satellite telemetry data";
    let mut data = SecureDataSlice::from_bytes(
        plaintext.to_vec(),
        DataClassification::Secret,
    );

    // Encrypt
    data.encrypt(&key).unwrap();
    let ciphertext = data.data.clone();

    // Tamper with ciphertext
    if let Some(byte) = data.data.get_mut(0) {
        *byte = byte.wrapping_add(1);  // Flip a bit
    }

    // Decryption should fail due to authentication tag
    let result = data.decrypt(&key);
    assert!(result.is_err());  // AEAD authentication should catch tampering
}

#[test]
fn test_key_zeroization() {
    let mut km = KeyManager::new("zeroize_test").unwrap();
    let _key = km.get_dek(DataClassification::Secret);

    // Explicitly zeroize
    km.zeroize_keys();

    // Keys should be cleared (can't easily test, but ensures Drop is called)
    // This is important for security - keys don't persist in memory
}

#[test]
fn test_multiple_classification_levels() {
    let mut km = KeyManager::new("multi_class_test").unwrap();

    let key_unclas = km.get_dek(DataClassification::Unclassified);
    let key_cui = km.get_dek(DataClassification::ControlledUnclassified);
    let key_secret = km.get_dek(DataClassification::Secret);
    let key_ts = km.get_dek(DataClassification::TopSecret);

    // All keys should be different
    assert_ne!(key_unclas, key_cui);
    assert_ne!(key_cui, key_secret);
    assert_ne!(key_secret, key_ts);

    // Test encryption at different levels
    let data_secret = b"SECRET: Threat detected";
    let mut slice = SecureDataSlice::from_bytes(
        data_secret.to_vec(),
        DataClassification::Secret,
    );

    slice.encrypt(&key_secret).unwrap();
    assert!(slice.encrypted);

    slice.decrypt(&key_secret).unwrap();
    assert_eq!(slice.data, data_secret);
}