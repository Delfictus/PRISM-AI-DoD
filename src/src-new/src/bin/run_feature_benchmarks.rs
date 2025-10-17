//! Run Feature Optimization Benchmarks
//!
//! REVOLUTIONARY GPU-accelerated feature processing benchmarks

use prism_ai::gpu::feature_optimization_benchmark::FeatureOptimizationBenchmark;
use anyhow::Result;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      PRISM-AI Feature Optimization Benchmarks 🚀        ║");
    println!("║         ONLY ADVANCE - NO COMPROMISES!                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // Initialize benchmark suite
    let mut benchmark = FeatureOptimizationBenchmark::new()?;

    // Run all benchmarks
    println!("\n⚡ Starting comprehensive GPU feature optimization benchmarks...");
    let results = benchmark.run_all_benchmarks()?;

    // Print detailed report
    benchmark.print_report();

    // Save results to file
    save_results(&results)?;

    println!("\n✅ Benchmarks completed successfully!");
    println!("🎯 Results saved to: feature_optimization_results.txt");

    Ok(())
}

/// Save benchmark results to file
fn save_results(results: &prism_ai::gpu::feature_optimization_benchmark::BenchmarkResults) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("feature_optimization_results.txt")?;

    writeln!(file, "PRISM-AI Feature Optimization Benchmark Results")?;
    writeln!(file, "{}", "=".repeat(60))?;
    writeln!(file)?;
    writeln!(file, "Performance Timings:")?;
    writeln!(file, "  Multi-scale fusion: {:.3}s", results.multi_scale_time)?;
    writeln!(file, "  Attention selection: {:.3}s", results.attention_time)?;
    writeln!(file, "  Cross-modal fusion: {:.3}s", results.cross_modal_time)?;
    writeln!(file, "  Feature engineering: {:.3}s", results.engineering_time)?;
    writeln!(file, "  Information optimization: {:.3}s", results.information_opt_time)?;
    writeln!(file)?;
    writeln!(file, "Overall Metrics:")?;
    writeln!(file, "  Total features processed: {}", results.total_features_processed)?;
    writeln!(file, "  Throughput: {:.2} GB/s", results.throughput_gbps)?;
    writeln!(file)?;
    writeln!(file, "GPU Utilization:")?;
    writeln!(file, "  Compute efficiency: {:.2} GFLOPS", results.fusion_metrics.compute_throughput)?;
    writeln!(file, "  Memory efficiency: {:.2}%", results.fusion_metrics.memory_efficiency * 100.0)?;

    Ok(())
}