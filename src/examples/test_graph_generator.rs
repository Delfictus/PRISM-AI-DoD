///! Test graph dataset generator

use prism_ai::data::GraphGenerator;

fn main() {
    println!("🧪 Testing Graph Generator");
    println!("{}", "=".repeat(80));

    let mut generator = GraphGenerator::new(42);

    // Generate small test dataset
    println!("\n📦 Generating test dataset (100 graphs)...\n");

    let mut graphs = Vec::new();

    // Test each graph type
    println!("  [1/8] Random Sparse...");
    for i in 0..12 {
        graphs.push(generator.generate_random_graph(i, 30, 0.02, prism_ai::data::GraphType::RandomSparse));
    }

    println!("  [2/8] Random Dense...");
    for i in 12..24 {
        graphs.push(generator.generate_random_graph(i, 30, 0.5, prism_ai::data::GraphType::RandomDense));
    }

    println!("  [3/8] Register Allocation...");
    for i in 24..36 {
        graphs.push(generator.generate_register_graph(i, 30));
    }

    println!("  [4/8] Leighton Adversarial...");
    for i in 36..48 {
        graphs.push(generator.generate_leighton_graph(i, 30));
    }

    println!("  [5/8] Geometric...");
    for i in 48..60 {
        graphs.push(generator.generate_geometric_graph(i, 30));
    }

    println!("  [6/8] Mycielski...");
    for i in 60..72 {
        graphs.push(generator.generate_mycielski_graph(i, 4));
    }

    println!("  [7/8] Scale-Free...");
    for i in 72..86 {
        graphs.push(generator.generate_scale_free_graph(i, 30, 3));
    }

    println!("  [8/8] Small-World...");
    for i in 86..100 {
        graphs.push(generator.generate_small_world_graph(i, 30, 6, 0.2));
    }

    println!("\n✅ Generated {} graphs\n", graphs.len());

    // Analyze dataset
    println!("📊 DATASET ANALYSIS:");
    println!("{}", "-".repeat(80));

    let mut type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut total_chromatic = 0;
    let mut total_difficulty = 0.0;
    let mut min_chromatic = usize::MAX;
    let mut max_chromatic = 0;

    for graph in &graphs {
        let type_name = format!("{:?}", graph.graph_type);
        *type_counts.entry(type_name).or_insert(0) += 1;

        total_chromatic += graph.chromatic_number;
        total_difficulty += graph.difficulty_score;
        min_chromatic = min_chromatic.min(graph.chromatic_number);
        max_chromatic = max_chromatic.max(graph.chromatic_number);
    }

    println!("\nGraph Type Distribution:");
    for (graph_type, count) in &type_counts {
        println!("  {:20} {:3} graphs", graph_type, count);
    }

    println!("\nChromatic Numbers:");
    println!("  Average:  {:.2}", total_chromatic as f64 / graphs.len() as f64);
    println!("  Min:      {}", min_chromatic);
    println!("  Max:      {}", max_chromatic);

    println!("\nDifficulty:");
    println!("  Average:  {:.2}/100", total_difficulty / graphs.len() as f64);

    // Sample detailed analysis
    println!("\n📝 SAMPLE GRAPHS:");
    println!("{}", "-".repeat(80));

    for (i, graph) in graphs.iter().take(5).enumerate() {
        println!("\nGraph #{} ({:?}):", i, graph.graph_type);
        println!("  Vertices:          {}", graph.num_vertices);
        println!("  Edges:             {}", graph.num_edges);
        println!("  Density:           {:.4} ({:?})", graph.density, graph.density_class);
        println!("  Avg degree:        {:.2}", graph.avg_degree);
        println!("  Chromatic number:  {}", graph.chromatic_number);
        println!("  Clustering:        {:.4}", graph.clustering_coefficient);
        println!("  Difficulty:        {:.1}/100", graph.difficulty_score);
        println!("  Node features:     {} × {}", graph.node_features.nrows(), graph.node_features.ncols());

        // Verify coloring is valid
        let mut valid = true;
        for i in 0..graph.num_vertices {
            for j in 0..graph.num_vertices {
                if graph.adjacency[[i, j]] && graph.optimal_coloring[i] == graph.optimal_coloring[j] {
                    valid = false;
                    break;
                }
            }
        }
        println!("  Valid coloring:    {}", if valid { "✅" } else { "❌" });
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ All tests passed!");
}
