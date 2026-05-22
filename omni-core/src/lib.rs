//! # omni-core
//!
//! The central brain of OmniCompute — MLIR-based JIT compilation engine,
//! heterogeneous code generation backends, and virtual memory management.
//!
//! ## Subsystems
//!
//! - [`mlir`] — `omni.tensor` dialect definition and optimization passes
//! - [`codegen`] — Hardware-specific machine code emission backends
//! - [`memory`] — Virtual Sliding Pager and DUVAS allocator
//! - [`jit`] — Top-level JIT engine coordinating the full compilation pipeline
//! - [`hardware`] — Runtime hardware detection and capability probing
//!
//! ## JIT Pipeline
//!
//! ```text
//! cuLaunchKernel (intercepted by omni-shim)
//!       |
//!       v
//! [1] IR Lifting:    PTX / Triton IR  ->  omni.tensor dialect
//!       |
//!       v
//! [2] Optimization:  Operator Fusion, Affine Transform, Loop Tiling
//!       |
//!       v
//! [3] Codegen:       omni.tensor  ->  AMDGCN | Metal MSL | SPIR-V
//!       |
//!       v
//! [4] Execution:     Load compiled binary into hardware runtime
//! ```

#![warn(missing_docs, clippy::all)]
#![allow(dead_code)] // Allow during active development

pub mod codegen;
pub mod hardware;
pub mod jit;
pub mod memory;
pub mod mlir;

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

// ─── Public Re-exports ────────────────────────────────────────────────────────

pub use hardware::{HardwareDetector, HardwareProfile, TargetBackend};
pub use jit::JitEngine;
pub use memory::allocator::DuvasAllocator;
pub use memory::pager::VirtualSlidingPager;

// ─── Core Engine ─────────────────────────────────────────────────────────────

/// The top-level OmniCompute runtime instance.
///
/// Holds all subsystem handles. Created once per process by the shim on startup
/// and stored in a global `Arc<OmniRuntime>`.
pub struct OmniRuntime {
    /// JIT compilation engine
    pub jit: Arc<JitEngine>,
    /// Virtual memory pager
    pub pager: Arc<VirtualSlidingPager>,
    /// Unified virtual address space allocator
    pub allocator: Arc<DuvasAllocator>,
    /// Detected hardware profile
    pub hardware: HardwareProfile,
}

impl OmniRuntime {
    /// Creates and initializes the OmniCompute runtime.
    ///
    /// This function:
    /// 1. Detects the physical hardware (via `HardwareDetector`)
    /// 2. Initializes the JIT engine with the detected backend
    /// 3. Starts the virtual memory pager
    /// 4. Returns a fully-initialized runtime ready to accept kernel dispatches
    pub fn init() -> Result<Arc<Self>> {
        info!("OmniRuntime: initializing...");

        let hardware = HardwareDetector::probe()?;
        info!("OmniRuntime: detected backend = {:?}", hardware.target_backend);
        info!(
            "OmniRuntime: physical VRAM = {} MB, L2 cache = {} KB",
            hardware.vram_bytes / 1024 / 1024,
            hardware.l2_cache_bytes / 1024,
        );

        let jit       = Arc::new(JitEngine::new(&hardware)?);
        let allocator = Arc::new(DuvasAllocator::new(hardware.vram_bytes));
        let pager     = Arc::new(VirtualSlidingPager::new(
            hardware.vram_bytes,
            hardware.l2_cache_bytes,
            Arc::clone(&allocator),
        ));

        info!("OmniRuntime: all subsystems ready");
        Ok(Arc::new(Self { jit, pager, allocator, hardware }))
    }
}

// ─── Version ──────────────────────────────────────────────────────────────────

/// Returns the omni-core version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_not_empty() {
        assert!(!version().is_empty());
    }
}
