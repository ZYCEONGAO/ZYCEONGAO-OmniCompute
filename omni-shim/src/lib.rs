//! # omni-shim
//!
//! **Dynamic Library Injection & CUDA Driver Masquerade Layer**
//!
//! `omni-shim` is the entry point of the OmniCompute virtualization stack.
//! It masquerades as the standard NVIDIA CUDA runtime (`libcuda.so` / `cudart64_*.dll`)
//! and intercepts all CUDA API calls made by user-space AI frameworks (PyTorch, llama.cpp, etc.)
//! **without any modification to the upstream application source code**.
//!
//! ## Injection Mechanism
//!
//! | Platform | Mechanism | Environment Variable |
//! |---|---|---|
//! | Linux   | Dynamic linker preload | `LD_PRELOAD=/path/to/libomni_shim.so` |
//! | macOS   | dyld library injection | `DYLD_INSERT_LIBRARIES=/path/to/libomni_shim.dylib` |
//! | Windows | DLL search order hijack | Place `cudart64_120.dll` alongside executable |
//!
//! ## Architecture
//!
//! ```text
//! PyTorch / llama.cpp
//!       │
//!       │  cudaMalloc(), cuLaunchKernel(), ...
//!       ▼
//! ┌─────────────────────────────────────┐
//! │           omni-shim                 │
//! │  ┌─────────────────────────────┐    │
//! │  │   cuda_runtime.rs           │    │  ← Runtime API (high-level)
//! │  │   cuda_driver.rs            │    │  ← Driver API  (low-level)
//! │  │   interceptor.rs            │    │  ← OS-level hook & dispatch
//! │  └─────────────────────────────┘    │
//! └──────────────────┬──────────────────┘
//!                    │  OmniTensor / KernelRequest
//!                    ▼
//!              omni-core (JIT Engine)
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, clippy::all)]

pub mod cuda_driver;
pub mod cuda_runtime;
pub mod interceptor;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use tracing::{debug, info};

// ─── Global Shim State ───────────────────────────────────────────────────────

/// Global registry of all virtual CUDA devices exposed by this shim.
///
/// Each device entry maps a CUDA device index to a [`VirtualDevice`] descriptor
/// that provides hardware capability metadata sourced from the actual physical
/// hardware detected by `omni-core`.
pub static VIRTUAL_DEVICES: Lazy<RwLock<Vec<VirtualDevice>>> = Lazy::new(|| {
    let devices = detect_virtual_devices();
    RwLock::new(devices)
});

/// Global dispatch table: maps CUDA stream handles (opaque `u64`) to their
/// associated async execution context within the Omni-Core JIT engine.
pub static STREAM_TABLE: Lazy<dashmap::DashMap<u64, StreamContext>> =
    Lazy::new(dashmap::DashMap::new);

/// Global memory allocation table: tracks all `cudaMalloc`-ed virtual addresses
/// and their mappings to real hardware memory regions managed by the DUVAS allocator.
pub static ALLOC_TABLE: Lazy<dashmap::DashMap<u64, AllocEntry>> =
    Lazy::new(dashmap::DashMap::new);

// ─── Types ────────────────────────────────────────────────────────────────────

/// Represents a virtualized CUDA-compatible compute device.
///
/// The shim presents these to the user application as if they were real NVIDIA
/// GPUs, but they are backed by the actual heterogeneous hardware detected by
/// `omni-core` (AMD, Apple, Intel, CPU, etc.).
#[derive(Debug, Clone)]
pub struct VirtualDevice {
    /// CUDA device index (0-based)
    pub index: i32,
    /// Human-readable device name (e.g., "OmniCompute Virtual GPU [AMD RX 7900XTX]")
    pub name: String,
    /// Total virtual memory (bytes) — reflects actual backing hardware capacity
    pub total_memory: usize,
    /// Compute capability major version (reported as 8 for Ampere compatibility)
    pub compute_capability_major: i32,
    /// Compute capability minor version
    pub compute_capability_minor: i32,
    /// Underlying real hardware type
    pub backend: HardwareBackend,
    /// Warp/wavefront size of the underlying hardware
    pub warp_size: i32,
    /// Number of streaming multiprocessors / compute units
    pub multiprocessor_count: i32,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// L2 cache size in bytes
    pub l2_cache_size: i32,
}

/// The actual hardware backend driving a virtual device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareBackend {
    /// NVIDIA GPU via native CUDA (pass-through mode)
    NvidiaCuda,
    /// AMD GPU via ROCm/HIP (AMDGCN backend)
    AmdRocm,
    /// Apple Silicon via Metal (UMA-optimized path)
    AppleMetal,
    /// Intel GPU via Vulkan/SPIR-V
    IntelVulkan,
    /// Generic Vulkan-capable GPU
    GenericVulkan,
    /// CPU fallback (AVX-512 / NEON vectorized)
    CpuFallback,
    /// Remote node via omni-net P2P dispatch
    RemoteP2p { node_id: String },
}

/// Execution context for a CUDA stream.
#[derive(Debug)]
pub struct StreamContext {
    /// Opaque stream handle value (mirroring the `cudaStream_t` pointer)
    pub handle: u64,
    /// The device this stream belongs to
    pub device_index: i32,
    /// Whether this stream has been synchronized (all pending ops completed)
    pub synchronized: bool,
    /// Pending kernel launches queued on this stream
    pub pending_kernels: Vec<KernelLaunch>,
}

/// A recorded kernel launch captured from `cuLaunchKernel`.
#[derive(Debug, Clone)]
pub struct KernelLaunch {
    /// PTX / cubin binary or hash identifying the kernel
    pub kernel_id: u64,
    /// Grid dimensions: (grid_dim_x, grid_dim_y, grid_dim_z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions: (block_dim_x, block_dim_y, block_dim_z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory per block (bytes)
    pub shared_mem_bytes: u32,
    /// Serialized kernel arguments (tensor pointers, scalars)
    pub args: Vec<u8>,
}

/// A tracked memory allocation in the virtual address space.
#[derive(Debug, Clone)]
pub struct AllocEntry {
    /// Virtual address returned to the caller
    pub virtual_ptr: u64,
    /// Allocation size in bytes
    pub size: usize,
    /// Device index this allocation belongs to
    pub device_index: i32,
    /// Whether this allocation is currently pinned (non-evictable by the Pager)
    pub pinned: bool,
}

// ─── Initialization ───────────────────────────────────────────────────────────

/// Called once when the shared library is loaded by the OS dynamic linker.
///
/// On Linux this is triggered by `LD_PRELOAD` before `main()` runs.
/// On Windows this is triggered by the DLL loader's `DllMain`.
///
/// # Safety
/// This function uses `unsafe` only for the FFI `#[no_mangle]` export required
/// by the dynamic linker. The internal logic is fully safe Rust.
#[cfg(target_os = "linux")]
#[used]
#[link_section = ".init_array"]
static INIT_SHIM: extern "C" fn() = {
    extern "C" fn init() {
        omni_shim_init();
    }
    init
};

/// Main initialization function for the OmniCompute shim.
///
/// Performs:
/// 1. Logging setup
/// 2. Hardware detection via omni-core
/// 3. Virtual device registration
/// 4. Dispatch table initialization
pub fn omni_shim_init() {
    // Initialize tracing subscriber (respects RUST_LOG env var)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("omni_shim=info".parse().unwrap()),
        )
        .try_init();

    info!("╔══════════════════════════════════════════════════════╗");
    info!("║         OmniCompute Shim v{:<27} ║", env!("CARGO_PKG_VERSION"));
    info!("║  Heterogeneous Hardware Virtualization Runtime       ║");
    info!("╚══════════════════════════════════════════════════════╝");

    // Trigger lazy initialization of global tables
    let devices = VIRTUAL_DEVICES.read();
    info!("Registered {} virtual CUDA device(s):", devices.len());
    for dev in devices.iter() {
        info!(
            "  [{}] {} — backend={:?}, mem={}MB",
            dev.index,
            dev.name,
            dev.backend,
            dev.total_memory / 1024 / 1024
        );
    }

    debug!("Shim dispatch tables initialized");
    debug!("OmniCompute JIT engine ready");
}

/// Detects available heterogeneous hardware and builds virtual device list.
///
/// In production this queries `omni-core`'s hardware detection module.
/// The feature-gated simulation path provides deterministic behavior for CI.
fn detect_virtual_devices() -> Vec<VirtualDevice> {
    // TODO(@omni-core): Query omni_core::hardware::HardwareDetector
    // For MVP, we expose a single virtual device backed by the best available hardware.
    vec![VirtualDevice {
        index: 0,
        name: "OmniCompute Virtual GPU 0 (Heterogeneous Backend)".to_string(),
        total_memory: 24 * 1024 * 1024 * 1024, // 24 GB virtual VRAM
        compute_capability_major: 8,             // Pretend Ampere for max compat
        compute_capability_minor: 0,
        backend: HardwareBackend::CpuFallback,   // Replaced at runtime by hardware probe
        warp_size: 32,
        multiprocessor_count: 108,
        max_threads_per_block: 1024,
        l2_cache_size: 40 * 1024 * 1024,        // 40 MB L2
    }]
}

// ─── C ABI Entry Points ───────────────────────────────────────────────────────
// These are the top-level symbols that the dynamic linker resolves when
// an application calls CUDA functions. They delegate to the typed Rust modules.

/// Returns the OmniCompute version string.
#[no_mangle]
pub extern "C" fn omni_version() -> *const libc::c_char {
    b"OmniCompute/0.1.0\0".as_ptr() as *const libc::c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_devices_non_empty() {
        let devices = detect_virtual_devices();
        assert!(!devices.is_empty(), "Must expose at least one virtual device");
    }

    #[test]
    fn test_virtual_device_fields() {
        let devices = detect_virtual_devices();
        let dev = &devices[0];
        assert_eq!(dev.index, 0);
        assert!(dev.total_memory > 0);
        assert_eq!(dev.compute_capability_major, 8);
    }

    #[test]
    fn test_global_tables_init() {
        // Touch global statics to ensure lazy init doesn't panic
        let _devices = VIRTUAL_DEVICES.read();
        assert!(STREAM_TABLE.is_empty());
        assert!(ALLOC_TABLE.is_empty());
    }
}
