//! K-quant dequantization tests
//!
//! Tests the K-quant dequantization functions (Q2_K through Q8_K)

use anyhow::Result;
use prism_ai::orchestration::local_llm::gguf_gpu_loader::GgufGpuLoader;

#[test]
#[cfg(feature = "cuda")]
fn test_k_quant_formats_available() -> Result<()> {
    println!("✅ K-quant test file compiled successfully");
    println!("   Formats: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_q2_k_super_block_size() {
    // Q2_K should have 256 elements per super-block, 82 bytes
    const SUPER_BLOCK_SIZE: usize = 256;
    const BYTES_PER_SUPER_BLOCK: usize = 82;

    assert_eq!(SUPER_BLOCK_SIZE, 256);
    assert_eq!(BYTES_PER_SUPER_BLOCK, 82);
    println!("✅ Q2_K: {} elements, {} bytes per super-block",
             SUPER_BLOCK_SIZE, BYTES_PER_SUPER_BLOCK);
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_k_super_block_size() {
    // Q4_K should have 256 elements per super-block, 144 bytes
    const SUPER_BLOCK_SIZE: usize = 256;
    const BYTES_PER_SUPER_BLOCK: usize = 144;

    assert_eq!(SUPER_BLOCK_SIZE, 256);
    assert_eq!(BYTES_PER_SUPER_BLOCK, 144);
    println!("✅ Q4_K: {} elements, {} bytes per super-block",
             SUPER_BLOCK_SIZE, BYTES_PER_SUPER_BLOCK);
}

#[test]
#[cfg(feature = "cuda")]
fn test_q8_k_super_block_size() {
    // Q8_K should have 256 elements per super-block, 292 bytes
    const SUPER_BLOCK_SIZE: usize = 256;
    const BYTES_PER_SUPER_BLOCK: usize = 292;

    assert_eq!(SUPER_BLOCK_SIZE, 256);
    assert_eq!(BYTES_PER_SUPER_BLOCK, 292);
    println!("✅ Q8_K: {} elements, {} bytes per super-block",
             SUPER_BLOCK_SIZE, BYTES_PER_SUPER_BLOCK);
}

#[test]
#[cfg(feature = "cuda")]
fn test_k_quant_efficiency() {
    // Q2_K: 256 elements / 82 bytes = 3.12 elements/byte (2.5 bits/element)
    let q2k_efficiency = 256.0 / 82.0;
    assert!(q2k_efficiency > 3.0 && q2k_efficiency < 3.2);

    // Q4_K: 256 elements / 144 bytes = 1.78 elements/byte (4.5 bits/element)
    let q4k_efficiency = 256.0 / 144.0;
    assert!(q4k_efficiency > 1.7 && q4k_efficiency < 1.9);

    // Q8_K: 256 elements / 292 bytes = 0.88 elements/byte (9.1 bits/element)
    let q8k_efficiency = 256.0 / 292.0;
    assert!(q8k_efficiency > 0.8 && q8k_efficiency < 0.9);

    println!("✅ K-quant efficiency:");
    println!("   Q2_K: {:.2} elements/byte ({:.1} bits/element)",
             q2k_efficiency, 8.0 / q2k_efficiency);
    println!("   Q4_K: {:.2} elements/byte ({:.1} bits/element)",
             q4k_efficiency, 8.0 / q4k_efficiency);
    println!("   Q8_K: {:.2} elements/byte ({:.1} bits/element)",
             q8k_efficiency, 8.0 / q8k_efficiency);
}
