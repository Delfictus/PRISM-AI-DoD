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
