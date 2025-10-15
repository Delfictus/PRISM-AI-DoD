//! GPU Memory Pool Demo
//!
//! Demonstrates memory allocation tracking and pooling recommendations.
//! Run with: cargo run --example memory_pool_demo --features cuda

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use prism_ai::gpu::memory_pool::{GpuMemoryPool, MemoryPoolConfig};

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU Memory Pool Demo - Worker 2 Infrastructure          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create memory pool tracker
    let config = MemoryPoolConfig {
        max_pool_size_bytes: 256 * 1024 * 1024, // 256 MB
        max_buffers_per_size: 16,
        min_pool_size_bytes: 4096,
        enable_tracking: true,
    };

    let pool = GpuMemoryPool::new(config);

    println!("🚀 Simulating GPU memory allocation patterns...\n");

    // Scenario 1: Matrix multiplications (same size - high reuse potential)
    println!("1️⃣  Matrix Multiplication Pattern (1024x1024)");
    for _ in 0..20 {
        let size_bytes = 1024 * 1024 * 4; // 4 MB per matrix
        pool.record_allocation(size_bytes);
        pool.record_deallocation(size_bytes);
    }
    println!("   ✅ Recorded 20 allocations/deallocations\n");

    // Scenario 2: Batch processing (repeated sizes)
    println!("2️⃣  Batch Processing Pattern (various batch sizes)");
    for _ in 0..10 {
        // Batch size 32
        pool.record_allocation(32 * 1024 * 4);
        pool.record_deallocation(32 * 1024 * 4);
    }
    for _ in 0..8 {
        // Batch size 64
        pool.record_allocation(64 * 1024 * 4);
        pool.record_deallocation(64 * 1024 * 4);
    }
    println!("   ✅ Recorded batch processing patterns\n");

    // Scenario 3: Dynamic allocation (high fragmentation)
    println!("3️⃣  Dynamic Allocation Pattern (varying sizes)");
    for i in 1..=15 {
        let size_bytes = i * 512 * 1024; // Varying sizes
        pool.record_allocation(size_bytes);
        if i % 3 == 0 {
            pool.record_deallocation(size_bytes);
        }
    }
    println!("   ✅ Recorded dynamic allocations\n");

    // Scenario 4: Small allocations (below min pool size)
    println!("4️⃣  Small Allocation Pattern (<4KB - not tracked)");
    for _ in 0..50 {
        pool.record_allocation(1024); // 1 KB (below minimum)
    }
    println!("   ✅ Small allocations (filtered out)\n");

    // Display statistics
    println!("═══════════════════════════════════════════════════════════\n");
    println!("{}", pool.get_report());
    println!("\n═══════════════════════════════════════════════════════════\n");

    // Export JSON
    println!("📄 Exporting to JSON format...\n");
    let json = pool.export_json()?;
    println!("{}\n", json);

    // Analysis
    let stats = pool.get_stats();
    println!("═══════════════════════════════════════════════════════════");
    println!("📊 ANALYSIS");
    println!("═══════════════════════════════════════════════════════════\n");

    if stats.reuse_potential() > 50.0 {
        println!("✅ HIGH REUSE DETECTED ({:.1}%)", stats.reuse_potential());
        println!("   Recommendation: Implement memory pooling for:");
        for (size, count) in stats.top_allocation_sizes(3) {
            println!("   • {} bytes ({} allocations)", size, count);
        }
        println!("\n   Expected benefit: 30-50% reduction in allocation overhead");
    } else {
        println!("ℹ️  LOW REUSE ({:.1}%)", stats.reuse_potential());
        println!("   Recommendation: Memory pooling may not provide significant benefits");
    }

    println!();

    if stats.fragmentation_estimate() > 30.0 {
        println!("⚠️  HIGH FRAGMENTATION ({:.1}%)", stats.fragmentation_estimate());
        println!("   Recommendation: Consider fixed-size buffer strategy");
        println!("   • Pre-allocate common sizes at startup");
        println!("   • Use power-of-2 sizing to reduce fragmentation");
    } else {
        println!("✅ LOW FRAGMENTATION ({:.1}%)", stats.fragmentation_estimate());
        println!("   Memory allocation patterns are well-behaved");
    }

    println!("\n═══════════════════════════════════════════════════════════\n");

    println!("💡 Integration Example:\n");
    println!("```rust");
    println!("// In kernel_executor.rs:");
    println!("use crate::gpu::memory_pool::{{GpuMemoryPool, MemoryPoolConfig}};");
    println!();
    println!("let pool = GpuMemoryPool::with_default_config();");
    println!();
    println!("// Track allocations");
    println!("pool.record_allocation(size_bytes);");
    println!("let buffer = device.alloc_zeros::<f32>(size)?;");
    println!("// ... use buffer ...");
    println!("pool.record_deallocation(size_bytes);");
    println!();
    println!("// Analyze patterns");
    println!("let stats = pool.get_stats();");
    println!("println!(\"Reuse potential: {{:.1}}%\", stats.reuse_potential());");
    println!("```\n");

    println!("✅ Memory Pool Demo Complete!");
    println!("\n💡 Use this data to:");
    println!("   • Identify frequently-allocated sizes");
    println!("   • Estimate memory pooling benefits");
    println!("   • Optimize allocation strategies");
    println!("   • Reduce GPU memory fragmentation");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("⚠️  CUDA feature required");
    eprintln!("   Run: cargo run --example memory_pool_demo --features cuda");
    std::process::exit(1);
}
