//! # codegen
//!
//! Subsystem for emitting target-specific compute kernel machine code or shading languages.
//!
//! Exposes a unified `CodeGenerator` interface that transforms hardware-agnostic
//! `omni.tensor` MLIR modules into optimized executable binaries or source buffers.

pub mod amdgpu;
pub mod metal;
pub mod spirv;

use crate::mlir::dialect::OmniModule;
pub use amdgpu::AmdGpuCodegen;
pub use metal::MetalCodegen;
pub use spirv::SpirvCodegen;

/// The output of a code generation pass.
#[derive(Debug, Clone)]
pub struct GeneratedKernel {
    /// Name of the kernel / entry point function
    pub name: String,
    /// The generated source code (e.g., Metal MSL) or compiled binary byte payload (e.g., AMDGCN ELF, SPIR-V)
    pub payload: Vec<u8>,
    /// Thread block dimensions hint [x, y, z] to guide invocation
    pub block_size: [u32; 3],
}

/// Common interface for all target code generation backends.
pub trait CodeGenerator {
    /// Performs lowering from high-level `omni.tensor` IR into optimized backend assembly/binary.
    fn generate(&self, module: &OmniModule) -> anyhow::Result<GeneratedKernel>;
}
