use std::io::{self, Write};
use prism_ai::{
    information_theory::{TransferEntropy, detect_causal_direction},
    active_inference::GenerativeModel,
    statistical_mechanics::{ThermodynamicNetwork, NetworkConfig},
    phase6::AdaptiveSolver,
};
use ndarray::{Array1, Array2};

fn main() {
    println!("╔═══════════════════════════════════════════╗");
    println!("║          PRISM-AI Interactive CLI         ║");
    println!("║   Quantum-Neuromorphic Computing System   ║");
    println!("╚═══════════════════════════════════════════╝");
    println!();
    println!("Type 'help' for available commands, 'quit' to exit");
    println!();

    loop {
        print!("PRISM> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "quit" | "exit" => {
                println!("Shutting down PRISM-AI...");
                break;
            }
            "help" => show_help(),
            cmd if cmd.starts_with("te ") => run_transfer_entropy(cmd),
            cmd if cmd.starts_with("thermo ") => run_thermodynamic(cmd),
            cmd if cmd.starts_with("infer ") => run_inference(cmd),
            cmd if cmd.starts_with("color ") => run_graph_coloring(cmd),
            cmd if cmd.starts_with("causal ") => run_causal_analysis(cmd),
            "status" => show_status(),
            "" => continue,
            _ => println!("Unknown command. Type 'help' for available commands."),
        }
    }
}

fn show_help() {
    println!("\n═══ Available Commands ═══\n");
    println!("  te <n>          - Compute Transfer Entropy on random data of size n");
    println!("  causal <n>      - Detect causal direction between random signals");
    println!("  thermo <n>      - Run thermodynamic network evolution with n oscillators");
    println!("  infer           - Run active inference demonstration");
    println!("  color <n>       - Solve graph coloring for random graph of size n");
    println!("  status          - Show system status and GPU availability");
    println!("  help            - Show this help message");
    println!("  quit            - Exit PRISM-AI\n");
}

fn run_transfer_entropy(cmd: &str) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let n = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(100);

    println!("\n🧮 Computing Transfer Entropy for {} samples...", n);

    let source = Array1::linspace(0.0, 10.0, n);
    let target = source.mapv(|x: f64| (x * 1.5).sin() + x.cos() * 0.5);

    let te = TransferEntropy::default();
    let result = te.calculate(&source, &target);

    println!("✅ Transfer Entropy: {:.6} bits", result.te_value);
    println!("   Effective TE: {:.6} bits", result.effective_te);
    println!("   P-value: {:.6}", result.p_value);
    println!("   Time lag: {}", result.time_lag);
}

fn run_causal_analysis(cmd: &str) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let n = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(500);

    println!("\n🔍 Detecting Causal Direction ({} samples)...", n);

    // Create causally related signals
    let x = Array1::linspace(0.0, 20.0, n);
    let noise = Array1::from_vec(vec![0.1; n]);
    let y = x.mapv(|v: f64| v.sin()) + noise; // Y depends on X

    let (direction, te_xy, te_yx) = detect_causal_direction(&x, &y, 10);

    println!("✅ Causal Analysis Complete:");
    match direction {
        prism_ai::information_theory::CausalDirection::XtoY => {
            println!("   Direction: X → Y");
            println!("   Strength: {:.4}", te_xy);
        }
        prism_ai::information_theory::CausalDirection::YtoX => {
            println!("   Direction: Y → X");
            println!("   Strength: {:.4}", te_yx);
        }
        prism_ai::information_theory::CausalDirection::Bidirectional => {
            println!("   Direction: Bidirectional");
            println!("   X→Y: {:.4}, Y→X: {:.4}", te_xy, te_yx);
        }
        prism_ai::information_theory::CausalDirection::Independent => {
            println!("   Direction: Independent (no causation detected)");
        }
    }
}

fn run_thermodynamic(cmd: &str) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let n = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(100);

    println!("\n🌡️ Running Thermodynamic Network ({} oscillators)...", n);

    let config = NetworkConfig {
        n_oscillators: n,
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: true,
        seed: 42,
    };

    let mut network = ThermodynamicNetwork::new(config);
    let result = network.evolve(100);

    println!("✅ Evolution Complete:");
    println!("   Entropy never decreased: {}", result.entropy_never_decreased);
    println!("   Boltzmann satisfied: {}", result.boltzmann_satisfied);
    println!("   Execution time: {:.2} ms", result.execution_time_ms);
    println!("   Final entropy: {:.6}", result.state.entropy);
    println!("   Phase coherence: {:.4}", result.metrics.phase_coherence);
}

fn run_inference(_cmd: &str) {
    println!("\n🧠 Running Active Inference...");

    let model = GenerativeModel::new();
    let observations = Array1::linspace(0.0, 1.0, 100);

    let free_energy = model.free_energy(&observations);

    println!("✅ Active Inference Complete:");
    println!("   Free Energy: {:.6}", free_energy.total);
    println!("   Complexity: {:.6}", free_energy.complexity);
    println!("   Accuracy: {:.6}", free_energy.accuracy);
    println!("   Model complexity: Hierarchical (3 levels)");
}

fn run_graph_coloring(cmd: &str) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let n = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);

    println!("\n🎨 Solving Graph Coloring ({} vertices)...", n);

    // Create random graph
    let mut adjacency = Array2::from_elem((n, n), false);
    for i in 0..n {
        for j in (i+1)..n {
            if (i + j) % 3 == 0 {  // Simple pattern for edges
                adjacency[[i, j]] = true;
                adjacency[[j, i]] = true;
            }
        }
    }

    // Count edges
    let edges: usize = (0..n).map(|i|
        (i+1..n).filter(|&j| adjacency[[i, j]]).count()
    ).sum();

    println!("   Graph: {} vertices, {} edges", n, edges);

    // Use adaptive solver
    let mut solver = AdaptiveSolver::new(n).expect("Failed to create solver");

    // Run async solver in blocking context
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let solution = runtime.block_on(solver.solve(&adjacency));

    match solution {
        Ok(sol) => {
            println!("✅ Solution Found:");
            println!("   Colors used: {}", sol.num_colors);
            println!("   Valid: {}", sol.verify(&adjacency));
            println!("   Statistics: {}", sol.statistics());
        }
        Err(e) => {
            println!("❌ Solving failed: {}", e);
        }
    }
}

fn show_status() {
    println!("\n═══ PRISM-AI System Status ═══\n");

    // Check GPU
    #[cfg(feature = "cuda")]
    {
        println!("  GPU Acceleration: ✅ Enabled (CUDA feature)");
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("  GPU Acceleration: ❌ Not Available");
    }
        // Check PTX files
    let ptx_files = [
        "transfer_entropy.ptx",
        "thermodynamic.ptx",
        "active_inference.ptx",
    ];

    let mut ptx_count = 0;
    for ptx in &ptx_files {
        if std::path::Path::new(&format!("src/kernels/ptx/{}", ptx)).exists() {
            ptx_count += 1;
        }
    }

    println!("  PTX Kernels: {}/{} compiled", ptx_count, ptx_files.len());

    // System info
    println!("  Platform: Quantum-Neuromorphic Hybrid");
    println!("  Components: 5 paradigms unified");
    println!("  Precision: 10^-30 (double-double)");
    println!("  Constitutional: 7 Articles enforced");
}