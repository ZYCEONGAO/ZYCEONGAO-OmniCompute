//! # benches::jit_latency
//!
//! Microbenchmarks measuring JIT translation latencies and caching hot-paths
//! using the Criterion framework.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omni_core::{HardwareProfile, TargetBackend, JitEngine, OmniRuntime, VirtualSlidingPager};
use std::sync::Arc;
use std::time::Duration;

/// Creates a mock AMD hardware profile for benchmarking JIT translation paths.
fn create_mock_amdgpu_profile() -> HardwareProfile {
    HardwareProfile {
        target_backend: TargetBackend::AmdGpu,
        vram_bytes: 16 * 1024 * 1024 * 1024,
        l2_cache_bytes: 4 * 1024 * 1024,
        unified_memory: false,
        device_name: "AMD Radeon RX 7900 XTX".to_string(),
    }
}

/// Benchmarks JIT compilation from PTX source to AMDGCN HIP kernels.
fn bench_jit_compilation(c: &mut Criterion) {
    let hw = create_mock_amdgpu_profile();
    let engine = JitEngine::new(&hw).unwrap();

    let fake_ptx = b".version 8.0\n.target sm_90\n mma.sync.aligned.m16n8k16 ";

    let mut group = c.benchmark_group("JIT Compilation");
    
    // Warm up the engine
    let _ = engine.compile_ptx(fake_ptx, 100).unwrap();

    group.bench_function("Cold JIT Compilation Path", |b| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;
            for i in 0..iters {
                // Clear cache to ensure we benchmark the cold compile latency path
                engine.clear_cache();
                let start = std::time::Instant::now();
                let _res = engine.compile_ptx(black_box(fake_ptx), black_box(i));
                total_duration += start.elapsed();
            }
            total_duration
        });
    });

    group.bench_function("Hot Cache JIT Path", |b| {
        // Prepare cached compiled block
        let _ = engine.compile_ptx(fake_ptx, 999).unwrap();
        b.iter(|| {
            let _res = engine.compile_ptx(black_box(fake_ptx), black_box(999)).unwrap();
        });
    });

    group.finish();
}

/// Benchmarks the virtual sliding memory page manager.
fn bench_memory_paging(c: &mut Criterion) {
    let runtime = OmniRuntime::init().unwrap();
    let pager = &runtime.pager;
    
    // Pre-allocate virtual layer address spaces
    let layer1 = runtime.allocator.alloc(16 * 1024 * 1024, false).unwrap();
    let layer2 = runtime.allocator.alloc(16 * 1024 * 1024, false).unwrap();
    let layer3 = runtime.allocator.alloc(16 * 1024 * 1024, false).unwrap();

    c.bench_function("Pager Window Slide Duration", |b| {
        b.iter(|| {
            pager.slide_window(
                black_box(&[layer1]),
                black_box(&[layer2]),
                black_box(&[layer3]),
                black_box(35.5)
            ).unwrap();
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_jit_compilation, bench_memory_paging
}

criterion_main!(benches);
