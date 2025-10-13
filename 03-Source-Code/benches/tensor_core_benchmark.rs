//! Tensor Core Performance Benchmark
//!
//! Comprehensive benchmarking suite comparing Tensor Core performance
//! against FP32 baseline for various matrix sizes.
//!
//! Run with: cargo bench --features cuda --bench tensor_core_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(feature = "cuda")]
use prism_ai::gpu::kernel_executor::get_global_executor;

#[cfg(feature = "cuda")]
fn benchmark_matmul_fp32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_fp32");

    let executor = get_global_executor().unwrap();
    let executor = executor.lock().unwrap();

    let sizes = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    for (m, k, n) in sizes {
        group.bench_with_input(
            BenchmarkId::new("matrix_multiply", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                let a = vec![0.1f32; m * k];
                let b_mat = vec![0.1f32; k * n];

                b.iter(|| {
                    let _ = executor.matrix_multiply(
                        black_box(&a),
                        black_box(&b_mat),
                        black_box(m),
                        black_box(k),
                        black_box(n)
                    );
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_tensor_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_core");

    let executor = get_global_executor().unwrap();
    let executor = executor.lock().unwrap();

    let sizes = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    for (m, k, n) in sizes {
        group.bench_with_input(
            BenchmarkId::new("tensor_core_matmul", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                let a = vec![0.1f32; m * k];
                let b_mat = vec![0.1f32; k * n];

                b.iter(|| {
                    let _ = executor.tensor_core_matmul(
                        black_box(&a),
                        black_box(&b_mat),
                        black_box(m),
                        black_box(k),
                        black_box(n)
                    );
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_tensor_core_wmma(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_core_wmma");

    let executor = get_global_executor().unwrap();
    let executor = executor.lock().unwrap();

    let sizes = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    for (m, k, n) in sizes {
        group.bench_with_input(
            BenchmarkId::new("wmma", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                let a = vec![0.1f32; m * k];
                let b_mat = vec![0.1f32; k * n];

                b.iter(|| {
                    let _ = executor.tensor_core_matmul_wmma(
                        black_box(&a),
                        black_box(&b_mat),
                        black_box(m),
                        black_box(k),
                        black_box(n)
                    );
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_speedup_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_comparison");
    group.significance_level(0.1).sample_size(100);

    let executor = get_global_executor().unwrap();
    let executor = executor.lock().unwrap();

    let m = 256;
    let k = 256;
    let n = 256;

    let a = vec![0.1f32; m * k];
    let b_mat = vec![0.1f32; k * n];

    group.bench_function("fp32_baseline", |b| {
        b.iter(|| {
            let _ = executor.matrix_multiply(
                black_box(&a),
                black_box(&b_mat),
                black_box(m),
                black_box(k),
                black_box(n)
            );
        });
    });

    group.bench_function("tensor_core_fp16", |b| {
        b.iter(|| {
            let _ = executor.tensor_core_matmul(
                black_box(&a),
                black_box(&b_mat),
                black_box(m),
                black_box(k),
                black_box(n)
            );
        });
    });

    group.bench_function("tensor_core_wmma", |b| {
        b.iter(|| {
            let _ = executor.tensor_core_matmul_wmma(
                black_box(&a),
                black_box(&b_mat),
                black_box(m),
                black_box(k),
                black_box(n)
            );
        });
    });

    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    benchmark_matmul_fp32,
    benchmark_tensor_core,
    benchmark_tensor_core_wmma,
    benchmark_speedup_comparison
);

#[cfg(feature = "cuda")]
criterion_main!(benches);

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("⚠️  CUDA feature required for benchmarks");
    eprintln!("   Run: cargo bench --features cuda --bench tensor_core_benchmark");
}
