//! # benches::jit_latency
//!
//! Microbenchmarks measuring JIT translation latencies and caching hot-paths
//! using the Criterion framework.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omni_core::{
    hardware::{HardwareProfile, OperatingSystem, TargetBackend},
    jit::JitEngine,
    memory::{allocator::DuvasAllocator, pager::VirtualSlidingPager},
};
use std::time::Duration;

/// Creates a mock AMD hardware profile for benchmarking JIT translation paths.
fn create_mock_amdgpu_profile() -> HardwareProfile {
    HardwareProfile {
        target_backend: TargetBackend::AmdRocm {
            gfx_arch: "gfx1100".to_string(),
            wavefront_size: 32,
            compute_units: 96,
        },
        vram_bytes: 16 * 1024 * 1024 * 1024,
        l2_cache_bytes: 4 * 1024 * 1024,
        fp16_tflops: 123.0,
        memory_bandwidth_gbps: 960.0,
        atomic_support: true,
        tensor_core_support: true,
        os: OperatingSystem::Linux,
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
                // Re-instantiate engine to benchmark the cold compile latency path
                let local_engine = JitEngine::new(&hw).unwrap();
                let start = std::time::Instant::now();
                let _res = local_engine.compile_ptx(black_box(fake_ptx), black_box(i));
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
    let mut allocator = omni_core::memory::allocator::DuvasAllocator::new(24 * 1024 * 1024 * 1024);
    
    // Pre-allocate virtual layer address spaces
    let layer1 = allocator.alloc(16 * 1024 * 1024, false).unwrap();
    let layer2 = allocator.alloc(16 * 1024 * 1024, false).unwrap();
    let layer3 = allocator.alloc(16 * 1024 * 1024, false).unwrap();

    let pager = VirtualSlidingPager::new(4096, 6 * 1024 * 1024, std::sync::Arc::new(allocator));

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
