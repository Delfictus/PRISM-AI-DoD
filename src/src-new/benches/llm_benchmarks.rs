//! Performance benchmarks for Local LLM components
//!
//! Run with: cargo bench --bench llm_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_ai::orchestration::local_llm::{
    GgufType, BPETokenizer, TokenSampler, SamplingConfig,
};

fn bench_gguf_type_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gguf_type_size");

    let types = vec![
        ("F32", GgufType::F32),
        ("F16", GgufType::F16),
        ("Q4_0", GgufType::Q4_0),
        ("Q4_1", GgufType::Q4_1),
        ("Q8_0", GgufType::Q8_0),
    ];

    for (name, gguf_type) in types {
        group.bench_with_input(BenchmarkId::from_parameter(name), &gguf_type, |b, gt| {
            b.iter(|| {
                black_box(gt.type_size());
                black_box(gt.block_size());
            });
        });
    }

    group.finish();
}

fn bench_tokenizer_encode(c: &mut Criterion) {
    let tokenizer = BPETokenizer::new(32000);
    let texts = vec![
        ("short", "Hello"),
        ("medium", "The quick brown fox jumps over the lazy dog"),
        ("long", "The quick brown fox jumps over the lazy dog. ".repeat(10)),
        ("unicode", "Hello 你好 мир 🚀"),
    ];

    let mut group = c.benchmark_group("tokenizer_encode");

    for (name, text) in texts {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, t| {
            b.iter(|| {
                tokenizer.encode(black_box(t), false).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_tokenizer_decode(c: &mut Criterion) {
    let tokenizer = BPETokenizer::new(32000);

    let token_sequences = vec![
        ("short", vec![100, 200, 300]),
        ("medium", (0..50).collect::<Vec<i32>>()),
        ("long", (0..200).collect::<Vec<i32>>()),
    ];

    let mut group = c.benchmark_group("tokenizer_decode");

    for (name, tokens) in token_sequences {
        group.bench_with_input(BenchmarkId::from_parameter(name), &tokens, |b, t| {
            b.iter(|| {
                tokenizer.decode(black_box(t)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sampling_greedy(c: &mut Criterion) {
    let sampler = TokenSampler::new(SamplingConfig::greedy());
    let vocab_sizes = vec![1000, 10000, 50000];

    let mut group = c.benchmark_group("sampling_greedy");

    for vocab_size in vocab_sizes {
        let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.001).collect();

        group.bench_with_input(BenchmarkId::from_parameter(vocab_size), &logits, |b, l| {
            b.iter(|| {
                sampler.sample(black_box(l), black_box(&[])).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sampling_temperature(c: &mut Criterion) {
    let temperatures = vec![0.1, 0.7, 1.0, 1.5];
    let vocab_size = 32000;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.001).sin()).collect();

    let mut group = c.benchmark_group("sampling_temperature");

    for temp in temperatures {
        let config = SamplingConfig {
            temperature: temp,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(temp), &sampler, |b, s| {
            b.iter(|| {
                s.sample(black_box(&logits), black_box(&[])).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sampling_top_k(c: &mut Criterion) {
    let top_k_values = vec![1, 10, 50, 100];
    let vocab_size = 32000;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.001).sin()).collect();

    let mut group = c.benchmark_group("sampling_top_k");

    for k in top_k_values {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: k,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(k), &sampler, |b, s| {
            b.iter(|| {
                s.sample(black_box(&logits), black_box(&[])).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sampling_top_p(c: &mut Criterion) {
    let top_p_values = vec![0.5, 0.9, 0.95, 0.99];
    let vocab_size = 32000;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.001).sin()).collect();

    let mut group = c.benchmark_group("sampling_top_p");

    for p in top_p_values {
        let config = SamplingConfig {
            temperature: 1.0,
            top_p: p,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(p), &sampler, |b, s| {
            b.iter(|| {
                s.sample(black_box(&logits), black_box(&[])).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sampling_with_repetition_penalty(c: &mut Criterion) {
    let vocab_size = 32000;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.001).sin()).collect();

    let context_lengths = vec![0, 10, 50, 100];

    let mut group = c.benchmark_group("sampling_repetition_penalty");

    for ctx_len in context_lengths {
        let config = SamplingConfig {
            temperature: 1.0,
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let sampler = TokenSampler::new(config);
        let context: Vec<i32> = (0..ctx_len).collect();

        group.bench_with_input(BenchmarkId::from_parameter(ctx_len), &context, |b, ctx| {
            b.iter(|| {
                sampler.sample(black_box(&logits), black_box(ctx)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_tokenizer_roundtrip(c: &mut Criterion) {
    let tokenizer = BPETokenizer::new(32000);
    let texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
    ];

    let mut group = c.benchmark_group("tokenizer_roundtrip");

    for (i, text) in texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::from_parameter(i), text, |b, t| {
            b.iter(|| {
                let tokens = tokenizer.encode(black_box(t), false).unwrap();
                tokenizer.decode(black_box(&tokens)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sampling_strategies_comparison(c: &mut Criterion) {
    let vocab_size = 32000;
    let logits: Vec<f32> = (0..vocab_size).map(|i| {
        // Create a realistic distribution with a peak
        let x = (i as f32 - 1000.0) / 100.0;
        (-x * x).exp() + (i as f32 * 0.0001).sin() * 0.1
    }).collect();

    let strategies = vec![
        ("greedy", SamplingConfig::greedy()),
        ("standard", SamplingConfig::standard()),
        ("creative", SamplingConfig::creative()),
        ("precise", SamplingConfig::precise()),
        ("min_p", SamplingConfig::min_p_recommended()),
    ];

    let mut group = c.benchmark_group("sampling_strategies");

    for (name, config) in strategies {
        let sampler = TokenSampler::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(name), &sampler, |b, s| {
            b.iter(|| {
                s.sample(black_box(&logits), black_box(&[])).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_batch_tokenization(c: &mut Criterion) {
    let tokenizer = BPETokenizer::new(32000);
    let batch_sizes = vec![1, 10, 50, 100];
    let text = "The quick brown fox jumps over the lazy dog";

    let mut group = c.benchmark_group("batch_tokenization");

    for batch_size in batch_sizes {
        let texts: Vec<&str> = vec![text; batch_size];

        group.bench_with_input(BenchmarkId::from_parameter(batch_size), &texts, |b, t| {
            b.iter(|| {
                for text in t {
                    tokenizer.encode(black_box(text), false).unwrap();
                }
            });
        });
    }

    group.finish();
}

fn bench_sampling_sequence_generation(c: &mut Criterion) {
    let vocab_size = 32000;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let sampler = TokenSampler::new(SamplingConfig::standard());

    let sequence_lengths = vec![10, 50, 100, 200];

    let mut group = c.benchmark_group("sequence_generation");

    for seq_len in sequence_lengths {
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, &len| {
            b.iter(|| {
                let mut context = vec![];
                for _ in 0..len {
                    let token = sampler.sample(black_box(&logits), black_box(&context)).unwrap();
                    context.push(token);
                }
                black_box(context)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gguf_type_calculations,
    bench_tokenizer_encode,
    bench_tokenizer_decode,
    bench_sampling_greedy,
    bench_sampling_temperature,
    bench_sampling_top_k,
    bench_sampling_top_p,
    bench_sampling_with_repetition_penalty,
    bench_tokenizer_roundtrip,
    bench_sampling_strategies_comparison,
    bench_batch_tokenization,
    bench_sampling_sequence_generation,
);

criterion_main!(benches);
