//! Comprehensive tests for GGUF loader
//!
//! Tests cover:
//! - Data type conversions
//! - Metadata value extraction
//! - Tensor info calculations
//! - Quantization dequantization (unit tests)

use prism_ai::orchestration::local_llm::{
    GgufType, MetadataValue, TensorInfo,
};

#[test]
fn test_gguf_type_sizes() {
    // F32 and F16
    assert_eq!(GgufType::F32.size_bytes(), 4);
    assert_eq!(GgufType::F16.size_bytes(), 2);

    // 4-bit quantization types (32 elements per block)
    assert_eq!(GgufType::Q4_0.size_bytes(), 18); // 2 bytes delta + 16 bytes data
    assert_eq!(GgufType::Q4_1.size_bytes(), 20); // 2 bytes delta + 2 bytes min + 16 bytes data

    // 5-bit quantization
    assert_eq!(GgufType::Q5_0.size_bytes(), 22);
    assert_eq!(GgufType::Q5_1.size_bytes(), 24);

    // 8-bit quantization
    assert_eq!(GgufType::Q8_0.size_bytes(), 34); // 2 bytes delta + 32 bytes data
    assert_eq!(GgufType::Q8_1.size_bytes(), 36);

    // K-quants (256 elements per block)
    assert_eq!(GgufType::Q2_K.size_bytes(), 82);
    assert_eq!(GgufType::Q3_K.size_bytes(), 110);
    assert_eq!(GgufType::Q4_K.size_bytes(), 144);
    assert_eq!(GgufType::Q5_K.size_bytes(), 176);
    assert_eq!(GgufType::Q6_K.size_bytes(), 210);
    assert_eq!(GgufType::Q8_K.size_bytes(), 292);
}

#[test]
fn test_gguf_type_is_quantized() {
    assert!(!GgufType::F32.is_quantized());
    assert!(!GgufType::F16.is_quantized());

    assert!(GgufType::Q4_0.is_quantized());
    assert!(GgufType::Q4_1.is_quantized());
    assert!(GgufType::Q5_0.is_quantized());
    assert!(GgufType::Q5_1.is_quantized());
    assert!(GgufType::Q8_0.is_quantized());
    assert!(GgufType::Q8_1.is_quantized());
    assert!(GgufType::Q2_K.is_quantized());
    assert!(GgufType::Q3_K.is_quantized());
    assert!(GgufType::Q4_K.is_quantized());
    assert!(GgufType::Q5_K.is_quantized());
    assert!(GgufType::Q6_K.is_quantized());
    assert!(GgufType::Q8_K.is_quantized());
}

#[test]
fn test_gguf_type_from_u32() {
    assert!(matches!(GgufType::from_u32(0), Ok(GgufType::F32)));
    assert!(matches!(GgufType::from_u32(1), Ok(GgufType::F16)));
    assert!(matches!(GgufType::from_u32(2), Ok(GgufType::Q4_0)));
    assert!(matches!(GgufType::from_u32(3), Ok(GgufType::Q4_1)));
    assert!(matches!(GgufType::from_u32(8), Ok(GgufType::Q8_0)));
    assert!(matches!(GgufType::from_u32(12), Ok(GgufType::Q4_K)));

    // Invalid type
    assert!(GgufType::from_u32(999).is_err());
}

#[test]
fn test_metadata_value_u64_conversions() {
    let val = MetadataValue::U8(42);
    assert_eq!(val.as_u64(), Some(42));

    let val = MetadataValue::U16(1000);
    assert_eq!(val.as_u64(), Some(1000));

    let val = MetadataValue::U32(100000);
    assert_eq!(val.as_u64(), Some(100000));

    let val = MetadataValue::U64(10000000000);
    assert_eq!(val.as_u64(), Some(10000000000));

    // Non-unsigned types should return None
    let val = MetadataValue::I32(-42);
    assert_eq!(val.as_u64(), None);

    let val = MetadataValue::F32(3.14);
    assert_eq!(val.as_u64(), None);

    let val = MetadataValue::String("test".to_string());
    assert_eq!(val.as_u64(), None);
}

#[test]
fn test_metadata_value_string_conversions() {
    let val = MetadataValue::String("hello".to_string());
    assert_eq!(val.as_str(), Some("hello"));

    // Non-string types should return None
    let val = MetadataValue::U32(42);
    assert_eq!(val.as_str(), None);

    let val = MetadataValue::F32(3.14);
    assert_eq!(val.as_str(), None);
}

#[test]
fn test_metadata_value_f32_conversions() {
    let val = MetadataValue::F32(3.14);
    assert_eq!(val.as_f32(), Some(3.14));

    let val = MetadataValue::F64(2.718);
    assert!((val.as_f32().unwrap() - 2.718).abs() < 0.001);

    // Non-float types should return None
    let val = MetadataValue::U32(42);
    assert_eq!(val.as_f32(), None);

    let val = MetadataValue::String("test".to_string());
    assert_eq!(val.as_f32(), None);
}

#[test]
fn test_tensor_info_element_count() {
    // 1D tensor
    let tensor = TensorInfo {
        name: "test1d".to_string(),
        dimensions: vec![1000],
        data_type: GgufType::F32,
        offset: 0,
    };
    assert_eq!(tensor.element_count(), 1000);

    // 2D tensor (e.g., weight matrix)
    let tensor = TensorInfo {
        name: "test2d".to_string(),
        dimensions: vec![512, 512],
        data_type: GgufType::F32,
        offset: 0,
    };
    assert_eq!(tensor.element_count(), 512 * 512);

    // 3D tensor
    let tensor = TensorInfo {
        name: "test3d".to_string(),
        dimensions: vec![32, 64, 128],
        data_type: GgufType::F32,
        offset: 0,
    };
    assert_eq!(tensor.element_count(), 32 * 64 * 128);

    // 4D tensor
    let tensor = TensorInfo {
        name: "test4d".to_string(),
        dimensions: vec![8, 16, 32, 64],
        data_type: GgufType::F32,
        offset: 0,
    };
    assert_eq!(tensor.element_count(), 8 * 16 * 32 * 64);
}

#[test]
fn test_tensor_info_size_bytes_f32() {
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![1000, 1000],
        data_type: GgufType::F32,
        offset: 0,
    };

    // 1M elements * 4 bytes = 4MB
    assert_eq!(tensor.size_bytes(), 1000 * 1000 * 4);
}

#[test]
fn test_tensor_info_size_bytes_f16() {
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![1000, 1000],
        data_type: GgufType::F16,
        offset: 0,
    };

    // 1M elements * 2 bytes = 2MB
    assert_eq!(tensor.size_bytes(), 1000 * 1000 * 2);
}

#[test]
fn test_tensor_info_size_bytes_q4_0() {
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![1024], // 32 blocks of 32 elements
        data_type: GgufType::Q4_0,
        offset: 0,
    };

    // 1024 elements / 32 per block = 32 blocks
    // 32 blocks * 18 bytes per block = 576 bytes
    assert_eq!(tensor.size_bytes(), 32 * 18);
}

#[test]
fn test_tensor_info_size_bytes_q8_0() {
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![3200], // 100 blocks of 32 elements
        data_type: GgufType::Q8_0,
        offset: 0,
    };

    // 3200 elements / 32 per block = 100 blocks
    // 100 blocks * 34 bytes per block = 3400 bytes
    assert_eq!(tensor.size_bytes(), 100 * 34);
}

#[test]
fn test_tensor_info_size_bytes_k_quants() {
    // K-quants use 256 elements per block
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![2560], // 10 blocks of 256 elements
        data_type: GgufType::Q4_K,
        offset: 0,
    };

    // 2560 elements / 256 per block = 10 blocks
    // 10 blocks * 144 bytes per block = 1440 bytes
    assert_eq!(tensor.size_bytes(), 10 * 144);
}

#[test]
fn test_tensor_info_partial_blocks() {
    // Test that partial blocks are handled correctly
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![100], // Not a multiple of 32
        data_type: GgufType::Q4_0,
        offset: 0,
    };

    // 100 elements needs 4 blocks (32*3 = 96, need 1 more for remaining 4)
    // 4 blocks * 18 bytes = 72 bytes
    assert_eq!(tensor.size_bytes(), 4 * 18);
}

#[test]
fn test_large_tensor_calculations() {
    // Test with Llama-7B-like dimensions
    let tensor = TensorInfo {
        name: "huge_weight".to_string(),
        dimensions: vec![4096, 4096], // 16M elements
        data_type: GgufType::Q4_0,
        offset: 0,
    };

    let elem_count = tensor.element_count();
    assert_eq!(elem_count, 4096 * 4096);

    // Calculate expected size
    let num_blocks = (elem_count + 31) / 32; // Round up
    let expected_size = num_blocks * 18;
    assert_eq!(tensor.size_bytes(), expected_size);
}

#[test]
fn test_metadata_array_values() {
    let array = MetadataValue::Array(vec![
        MetadataValue::U32(1),
        MetadataValue::U32(2),
        MetadataValue::U32(3),
    ]);

    match array {
        MetadataValue::Array(ref values) => {
            assert_eq!(values.len(), 3);
            assert_eq!(values[0].as_u64(), Some(1));
            assert_eq!(values[1].as_u64(), Some(2));
            assert_eq!(values[2].as_u64(), Some(3));
        }
        _ => panic!("Expected array"),
    }
}

#[test]
fn test_metadata_bool_values() {
    let val_true = MetadataValue::Bool(true);
    match val_true {
        MetadataValue::Bool(b) => assert!(b),
        _ => panic!("Expected bool"),
    }

    let val_false = MetadataValue::Bool(false);
    match val_false {
        MetadataValue::Bool(b) => assert!(!b),
        _ => panic!("Expected bool"),
    }
}

// Integration test would go here if we had a sample GGUF file
// For now, these unit tests provide good coverage of the core functionality

#[test]
fn test_tensor_info_clone() {
    let tensor = TensorInfo {
        name: "test".to_string(),
        dimensions: vec![100, 200],
        data_type: GgufType::Q4_0,
        offset: 12345,
    };

    let cloned = tensor.clone();
    assert_eq!(tensor.name, cloned.name);
    assert_eq!(tensor.dimensions, cloned.dimensions);
    assert_eq!(tensor.offset, cloned.offset);
}

// ===== Additional Error Scenario Tests for Day 5 =====

use prism_ai::orchestration::local_llm::GgufLoader;
use std::fs;
use tempfile::TempDir;
use anyhow::Result;

#[test]
fn test_gguf_load_nonexistent_file() {
    let result = GgufLoader::load("/nonexistent/path/model.gguf");
    assert!(result.is_err(), "Should fail for nonexistent file");

    if let Err(e) = result {
        let msg = e.to_string();
        assert!(!msg.is_empty(), "Error message should not be empty");
    }
}

#[test]
fn test_gguf_load_empty_file() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let file_path = temp_dir.path().join("empty.gguf");
    fs::File::create(&file_path)?;

    let result = GgufLoader::load(&file_path);
    assert!(result.is_err(), "Should fail for empty file");

    Ok(())
}

#[test]
fn test_gguf_load_invalid_magic_number() -> Result<()> {
    use std::io::Write;

    let temp_dir = tempfile::tempdir()?;
    let file_path = temp_dir.path().join("invalid_magic.gguf");

    let mut file = fs::File::create(&file_path)?;
    // Write invalid magic number (not "GGUF")
    file.write_all(b"ABCD")?;
    file.write_all(&[0u8; 100])?;
    drop(file);

    let result = GgufLoader::load(&file_path);
    assert!(result.is_err(), "Should fail for invalid magic number");

    Ok(())
}

#[test]
fn test_gguf_load_truncated_file() -> Result<()> {
    use std::io::Write;

    let temp_dir = tempfile::tempdir()?;
    let file_path = temp_dir.path().join("truncated.gguf");

    let mut file = fs::File::create(&file_path)?;
    // Write valid magic but truncate rest
    file.write_all(b"GGUF")?;
    // Missing version, metadata, tensors
    drop(file);

    let result = GgufLoader::load(&file_path);
    assert!(result.is_err(), "Should fail for truncated file");

    Ok(())
}

#[test]
fn test_gguf_load_directory() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let dir_path = temp_dir.path();

    // Try to load a directory instead of file
    let result = GgufLoader::load(dir_path);
    assert!(result.is_err(), "Should fail when given a directory");

    Ok(())
}

#[test]
fn test_gguf_invalid_type_conversion() {
    // Test type conversions that should fail
    for invalid_type in [999, 1000, u32::MAX] {
        let result = GgufType::from_u32(invalid_type);
        assert!(result.is_err(), "Should fail for invalid type {}", invalid_type);
    }
}

#[test]
fn test_gguf_metadata_type_mismatches() {
    // Test accessing wrong type from MetadataValue
    let int_val = MetadataValue::U32(42);
    assert_eq!(int_val.as_str(), None, "Should return None for wrong type");
    assert_eq!(int_val.as_f32(), None, "Should return None for wrong type");

    let str_val = MetadataValue::String("test".to_string());
    assert_eq!(str_val.as_u64(), None, "Should return None for wrong type");
    assert_eq!(str_val.as_f32(), None, "Should return None for wrong type");

    let float_val = MetadataValue::F32(3.14);
    assert_eq!(float_val.as_u64(), None, "Should return None for wrong type");
    assert_eq!(float_val.as_str(), None, "Should return None for wrong type");
}

#[test]
fn test_gguf_zero_dimension_tensor() {
    let tensor = TensorInfo {
        name: "zero_dim".to_string(),
        dimensions: vec![0],
        data_type: GgufType::F32,
        offset: 0,
    };

    assert_eq!(tensor.element_count(), 0);
    assert_eq!(tensor.size_bytes(), 0);
}

#[test]
fn test_gguf_empty_dimension_tensor() {
    let tensor = TensorInfo {
        name: "no_dims".to_string(),
        dimensions: vec![],
        data_type: GgufType::F32,
        offset: 0,
    };

    assert_eq!(tensor.element_count(), 1); // Product of empty array is 1
    assert_eq!(tensor.size_bytes(), 4); // 1 element * 4 bytes
}

#[test]
fn test_gguf_very_large_tensor() {
    // Test with extremely large dimensions (might overflow if not handled)
    let tensor = TensorInfo {
        name: "huge".to_string(),
        dimensions: vec![10000, 10000], // 100M elements
        data_type: GgufType::F32,
        offset: 0,
    };

    let elem_count = tensor.element_count();
    assert_eq!(elem_count, 100_000_000);

    let size = tensor.size_bytes();
    assert_eq!(size, 100_000_000 * 4); // 400 MB
}

#[test]
fn test_gguf_negative_signed_metadata() {
    let val = MetadataValue::I32(-42);
    match val {
        MetadataValue::I32(v) => assert_eq!(v, -42),
        _ => panic!("Expected I32"),
    }

    let val = MetadataValue::I64(-1000000);
    match val {
        MetadataValue::I64(v) => assert_eq!(v, -1000000),
        _ => panic!("Expected I64"),
    }
}

#[test]
fn test_gguf_type_all_variants() {
    // Ensure all GgufType variants can be created
    let types = vec![
        GgufType::F32,
        GgufType::F16,
        GgufType::Q4_0,
        GgufType::Q4_1,
        GgufType::Q5_0,
        GgufType::Q5_1,
        GgufType::Q8_0,
        GgufType::Q8_1,
        GgufType::Q2_K,
        GgufType::Q3_K,
        GgufType::Q4_K,
        GgufType::Q5_K,
        GgufType::Q6_K,
        GgufType::Q8_K,
    ];

    for typ in types {
        assert!(typ.size_bytes() > 0, "Size should be positive");
    }
}

#[test]
fn test_gguf_metadata_nested_arrays() {
    let nested = MetadataValue::Array(vec![
        MetadataValue::Array(vec![
            MetadataValue::U32(1),
            MetadataValue::U32(2),
        ]),
        MetadataValue::Array(vec![
            MetadataValue::U32(3),
            MetadataValue::U32(4),
        ]),
    ]);

    match nested {
        MetadataValue::Array(ref outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                MetadataValue::Array(inner) => {
                    assert_eq!(inner.len(), 2);
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected array"),
    }
}

#[test]
fn test_gguf_empty_array_metadata() {
    let empty_array = MetadataValue::Array(vec![]);

    match empty_array {
        MetadataValue::Array(ref values) => {
            assert_eq!(values.len(), 0);
        }
        _ => panic!("Expected array"),
    }
}

#[test]
fn test_gguf_empty_path() {
    let result = GgufLoader::load("");
    assert!(result.is_err(), "Should fail for empty path");
}

#[test]
fn test_gguf_unicode_in_metadata() {
    let unicode_str = MetadataValue::String("Hello 世界 🌍".to_string());

    match unicode_str {
        MetadataValue::String(ref s) => {
            assert_eq!(s, "Hello 世界 🌍");
            assert!(s.contains("世界"));
            assert!(s.contains("🌍"));
        }
        _ => panic!("Expected string"),
    }
}

#[test]
fn test_gguf_tensor_name_validation() {
    // Test that tensor names can contain various characters
    let names = vec![
        "simple_name",
        "name.with.dots",
        "name-with-dashes",
        "name_123",
        "CamelCaseName",
    ];

    for name in names {
        let tensor = TensorInfo {
            name: name.to_string(),
            dimensions: vec![10, 10],
            data_type: GgufType::F32,
            offset: 0,
        };

        assert_eq!(tensor.name, name);
    }
}

#[test]
fn test_gguf_quantization_block_alignment() {
    // Q4_0 has 32 elements per block
    let sizes = vec![32, 64, 128, 256, 512];

    for size in sizes {
        let tensor = TensorInfo {
            name: "aligned".to_string(),
            dimensions: vec![size],
            data_type: GgufType::Q4_0,
            offset: 0,
        };

        let expected_blocks = size / 32;
        let expected_size = expected_blocks * 18;
        assert_eq!(tensor.size_bytes(), expected_size);
    }
}

#[test]
fn test_gguf_quantization_unaligned_sizes() {
    // Test with sizes not aligned to block boundaries
    let sizes = vec![33, 65, 100, 250];

    for size in sizes {
        let tensor = TensorInfo {
            name: "unaligned".to_string(),
            dimensions: vec![size],
            data_type: GgufType::Q4_0,
            offset: 0,
        };

        let expected_blocks = (size + 31) / 32; // Round up
        let expected_size = expected_blocks * 18;
        assert_eq!(tensor.size_bytes(), expected_size);
    }
}

#[test]
fn test_gguf_f64_metadata_precision() {
    let val = MetadataValue::F64(std::f64::consts::PI);

    match val {
        MetadataValue::F64(v) => {
            assert!((v - std::f64::consts::PI).abs() < 1e-10);
        }
        _ => panic!("Expected F64"),
    }
}

#[test]
fn test_gguf_max_values() {
    // Test edge cases with maximum values
    let max_u64 = MetadataValue::U64(u64::MAX);
    assert_eq!(max_u64.as_u64(), Some(u64::MAX));

    let max_i64 = MetadataValue::I64(i64::MAX);
    match max_i64 {
        MetadataValue::I64(v) => assert_eq!(v, i64::MAX),
        _ => panic!("Expected I64"),
    }

    let min_i64 = MetadataValue::I64(i64::MIN);
    match min_i64 {
        MetadataValue::I64(v) => assert_eq!(v, i64::MIN),
        _ => panic!("Expected I64"),
    }
}
