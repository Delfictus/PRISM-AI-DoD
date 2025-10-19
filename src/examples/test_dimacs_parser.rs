///! Test DIMACS parser on real benchmarks and display characteristics

use prism_ai::data::DimacsGraph;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_.col_file>", args[0]);
        eprintln!("Example: {} benchmarks/dimacs/DSJC125.5.col", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Parsing DIMACS file: {}", path);
    println!("{}", "=".repeat(80));

    match DimacsGraph::from_file(path) {
        Ok(graph) => {
            println!("\n📊 GRAPH: {}", graph.name);
            println!("  Vertices: {}", graph.num_vertices);
            println!("  Edges: {}", graph.num_edges);
            if let Some(chi) = graph.known_chromatic {
                println!("  Known χ(G): {}", chi);
            }

            let c = &graph.characteristics;
            println!("\n🔍 DENSITY:");
            println!("  Edge density: {:.4}", c.edge_density);
            println!("  Class: {:?}", c.density_class);

            println!("\n📈 DEGREE DISTRIBUTION:");
            println!("  Average degree: {:.2}", c.avg_degree);
            println!("  Maximum degree: {}", c.max_degree);
            println!("  Variance: {:.2}", c.degree_variance);

            println!("\n🌐 STRUCTURE:");
            println!("  Clustering coefficient: {:.4}", c.clustering_coefficient);
            println!("  Diameter estimate: {}", c.diameter_estimate);
            println!("  Transitivity: {:.4}", c.transitivity);

            println!("\n🎨 CHROMATIC BOUNDS:");
            println!("  Clique lower bound: {}", c.clique_lower_bound);
            println!("  Greedy upper bound: {}", c.greedy_upper_bound);
            println!("  Degeneracy: {}", c.degeneracy);
            println!("  Gap: {}", c.greedy_upper_bound - c.clique_lower_bound);

            println!("\n🏷️  CLASSIFICATION:");
            println!("  Graph type: {:?}", c.graph_type);
            println!("  Difficulty score: {:.1}/100", c.difficulty_score);

            let s = &c.recommended_strategy;
            println!("\n🎯 RECOMMENDED STRATEGY:");
            println!("  Use TDA: {} (weight: {:.2})", s.use_tda, s.tda_weight);
            println!("  Use GNN: {} (confidence: {:.2})", s.use_gnn, s.gnn_confidence);
            println!("  Explore/Exploit: {:.2}", s.exploration_vs_exploitation);
            println!("  Temperature: {:.2}", s.temperature_scaling);
            println!("  Parallel factor: {:.2}x", s.parallel_attempts_factor);
        }
        Err(e) => {
            eprintln!("❌ Error parsing file: {}", e);
            std::process::exit(1);
        }
    }
}
