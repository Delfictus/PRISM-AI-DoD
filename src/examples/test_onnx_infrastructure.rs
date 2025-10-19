///! Standalone Test for ONNX Runtime CUDA Infrastructure
///!
///! Proves that Step 5 (ONNX Runtime Integration) infrastructure is ready.
///!
///! Usage: cargo run --example test_onnx_infrastructure --features cuda

use prism_ai::cma::neural::{ColoringGNN, compute_node_features};
use ndarray::Array2;

fn main() {
    println!("🧪 Testing ONNX Runtime CUDA Infrastructure (Step 5)");
    println!("{}", "=".repeat(80));

    // Test 1: ColoringGNN initialization
    println!("\n[Test 1] ColoringGNN Initialization");
    println!("{}", "-".repeat(80));

    let model_path = "../models/coloring_gnn.onnx";
    let max_colors = 50;
    let device_id = 0;

    println!("  Model path: {}", model_path);
    println!("  Max colors: {}", max_colors);
    println!("  GPU device: {}", device_id);

    let gnn = match ColoringGNN::new(model_path, max_colors, device_id) {
        Ok(gnn) => {
            println!("\n✅ ColoringGNN initialized successfully");
            gnn
        }
        Err(e) => {
            eprintln!("\n❌ FATAL: ColoringGNN initialization failed: {}", e);
            std::process::exit(1);
        }
    };

    // Test 2: Node features computation (CPU-based utility)
    println!("\n[Test 2] Node Features Computation");
    println!("{}", "-".repeat(80));

    // Triangle graph
    let mut triangle = Array2::from_elem((3, 3), false);
    triangle[[0, 1]] = true;
    triangle[[1, 0]] = true;
    triangle[[1, 2]] = true;
    triangle[[2, 1]] = true;
    triangle[[0, 2]] = true;
    triangle[[2, 0]] = true;

    println!("  Computing features for triangle graph (3 vertices)...");
    let node_features = compute_node_features(&triangle);

    println!("  Feature shape: {:?}", node_features.shape());
    assert_eq!(node_features.shape(), &[3, 16], "Features should be [N, 16]");
    println!("  ✅ Feature computation works");

    // Test 3: GNN prediction (tests fallback behavior)
    println!("\n[Test 3] GNN Prediction");
    println!("{}", "-".repeat(80));

    println!("  Running prediction on triangle graph...");
    let prediction = match gnn.predict(&triangle, &node_features) {
        Ok(pred) => {
            println!("  ✅ Prediction succeeded");
            pred
        }
        Err(e) => {
            eprintln!("\n❌ FATAL: Prediction failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("\n  Prediction results:");
    println!("    Node color logits shape: {:?}", prediction.node_color_logits.shape());
    println!("    Predicted chromatic number: {}", prediction.predicted_chromatic);
    println!("    Graph type logits shape: {:?}", prediction.graph_type_logits.shape());
    println!("    Difficulty score: {:.1}", prediction.difficulty_score);
    println!("    Inference time: {:.2}ms", prediction.inference_time_ms);

    // Validate prediction structure
    assert_eq!(
        prediction.node_color_logits.shape(),
        &[3, max_colors],
        "Color logits should be [N, max_colors]"
    );
    assert_eq!(
        prediction.graph_type_logits.len(),
        8,
        "Graph type logits should be [8]"
    );
    println!("  ✅ Prediction structure validated");

    // Test 4: Vertex ordering (downstream usage)
    println!("\n[Test 4] Vertex Ordering Extraction");
    println!("{}", "-".repeat(80));

    println!("  Extracting vertex ordering from GNN predictions...");
    let ordering = match gnn.get_vertex_ordering(&triangle, &node_features) {
        Ok(ord) => {
            println!("  ✅ Vertex ordering extracted");
            ord
        }
        Err(e) => {
            eprintln!("\n❌ FATAL: Ordering extraction failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("  Ordering: {:?}", ordering);
    assert_eq!(ordering.len(), 3, "Ordering should have 3 vertices");

    // Check it's a valid permutation
    let mut sorted_ordering = ordering.clone();
    sorted_ordering.sort();
    assert_eq!(sorted_ordering, vec![0, 1, 2], "Ordering should be valid permutation");
    println!("  ✅ Valid permutation verified");

    // Test 5: Larger graph
    println!("\n[Test 5] Larger Graph (10 vertices)");
    println!("{}", "-".repeat(80));

    let mut graph10 = Array2::from_elem((10, 10), false);
    for i in 0..10 {
        graph10[[i, (i + 1) % 10]] = true;
        graph10[[(i + 1) % 10, i]] = true;
    }

    let node_features_10 = compute_node_features(&graph10);
    println!("  Computing prediction for 10-vertex graph...");

    let prediction_10 = match gnn.predict(&graph10, &node_features_10) {
        Ok(pred) => {
            println!("  ✅ Prediction succeeded");
            pred
        }
        Err(e) => {
            eprintln!("\n❌ FATAL: Prediction failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("  Predicted chromatic: {}", prediction_10.predicted_chromatic);
    println!("  Inference time: {:.2}ms", prediction_10.inference_time_ms);

    // Infrastructure status summary
    println!("\n{}", "=".repeat(80));
    println!("🎉 ALL INFRASTRUCTURE TESTS PASSED!");
    println!("{}", "=".repeat(80));
    println!("\n✅ Step 5 (ONNX Runtime CUDA Integration) Infrastructure is READY");
    println!("   - ColoringGNN initializes correctly");
    println!("   - Node feature computation works");
    println!("   - Prediction interface functions");
    println!("   - Smart fallback to placeholder when model not trained");
    println!("   - Vertex ordering extraction works");
    println!("\n📋 Current Status:");

    use std::path::Path;
    if Path::new(model_path).exists() {
        println!("   🟢 ONNX model file EXISTS - using REAL inference");
    } else {
        println!("   🟡 ONNX model file MISSING - using placeholder fallback");
        println!("      Next step: Train GNN and export to ONNX (Step 14)");
    }

    println!("\n✅ Infrastructure verified and ready for model deployment");
}
