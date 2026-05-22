//! # jit
//!
//! Top-level即时编译 (JIT) 编译引擎。
//! Coordinates the JIT compilation pipeline by linking the IR Lifter,
//! the Pass Manager, and target-specific Code Generators.

use crate::hardware::{HardwareProfile, TargetBackend};
use crate::mlir::{IrLifter, PassManager};
use crate::codegen::{AmdGpuCodegen, MetalCodegen, SpirvCodegen, CodeGenerator, GeneratedKernel};
use anyhow::{bail, Result};
use dashmap::DashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Top-level JIT engine that compiles CUDA PTX / Triton IR to target hardware formats.
pub struct JitEngine {
    /// Target hardware profile
    hardware: HardwareProfile,
    /// Thread-safe Cache of compiled kernels: MD5/hash -> GeneratedKernel
    kernel_cache: DashMap<u64, GeneratedKernel>,
}

impl JitEngine {
    /// Creates a new JIT compiler engine optimized for the target hardware.
    pub fn new(hardware: &HardwareProfile) -> Result<Self> {
        info!("JitEngine: initializing compiler for {:?}", hardware.target_backend);
        Ok(Self {
            hardware: hardware.clone(),
            kernel_cache: DashMap::new(),
        })
    }

    /// Compiles raw CUDA PTX bytes into a target hardware kernel.
    ///
    /// ## Compilation Workflow
    ///
    /// 1. **Cache Lookup**: Skips compilation if the kernel hash is already present in `kernel_cache`.
    /// 2. **IR Lifting**: Feeds PTX bytes into the `IrLifter` to reverse-lift instructions into
    ///    hardware-agnostic `omni.tensor` dialect representation.
    /// 3. **MLIR Optimizations**: Runs the `PassManager` to perform Operator Fusion, Loop Tiling, and DCE.
    /// 4. **Codegen**: Emits target machine code or shader byte payload using the appropriate generator.
    pub fn compile_ptx(&self, ptx_bytes: &[u8], kernel_id: u64) -> Result<GeneratedKernel> {
        // Compute unique kernel hash to query compilation cache
        let kernel_hash = Self::hash_ptx(ptx_bytes, kernel_id);

        if let Some(cached) = self.kernel_cache.get(&kernel_hash) {
            debug!("JitEngine: cache HIT for kernel 0x{:x}", kernel_id);
            return Ok(cached.clone());
        }

        debug!("JitEngine: cache MISS. Starting JIT pipeline for kernel 0x{:x}", kernel_id);

        // 1. Lift PTX into high-level MLIR dialect
        let mut lifter = IrLifter::new();
        let mut module = lifter.lift_ptx(ptx_bytes, kernel_id)?;

        // 2. Run optimization passes
        let pass_manager = PassManager::new(self.hardware.clone());
        pass_manager.run(&mut module)?;

        // 3. Lower and emit code for target platform
        let kernel = match self.hardware.target_backend {
            TargetBackend::AmdRocm { .. } => {
                let codegen = AmdGpuCodegen::new(self.hardware.clone());
                codegen.generate(&module)?
            }
            TargetBackend::AppleMetal { .. } => {
                let codegen = MetalCodegen::new(self.hardware.clone());
                codegen.generate(&module)?
            }
            TargetBackend::VulkanGeneric { .. } => {
                let codegen = SpirvCodegen::new(self.hardware.clone());
                codegen.generate(&module)?
            }
            TargetBackend::CpuVectorized { .. } | TargetBackend::RemoteP2p { .. } => {
                // For CPU fallback, leverage SPIR-V/GLSL pipeline or emit simplified reference code.
                let codegen = SpirvCodegen::new(self.hardware.clone());
                codegen.generate(&module)?
            }
        };

        debug!(
            "JitEngine: JIT compilation successful. Generated kernel '{}' ({} bytes payload)",
            kernel.name,
            kernel.payload.len()
        );

        // Cache the compilation output
        self.kernel_cache.insert(kernel_hash, kernel.clone());

        Ok(kernel)
    }

    /// Computes a lightweight fast hash for PTX bytes and kernel identifiers.
    fn hash_ptx(ptx_bytes: &[u8], kernel_id: u64) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        hash = hash.wrapping_mul(0x00000100000001b3) ^ kernel_id;
        for &b in ptx_bytes {
            hash = hash.wrapping_mul(0x00000100000001b3) ^ (b as u64);
        }
        hash
    }

    /// Clears the compilation cache.
    pub fn clear_cache(&self) {
        self.kernel_cache.clear();
        info!("JitEngine: JIT kernel cache flushed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HardwareProfile;

    fn create_mock_hardware() -> HardwareProfile {
        HardwareProfile {
            target_backend: TargetBackend::VulkanGeneric {
                device_name: "Mock Intel Arc GPU".to_string(),
                max_invocations: 1024,
            },
            vram_bytes: 4 * 1024 * 1024 * 1024,
            l2_cache_bytes: 1024 * 1024,
            fp16_tflops: 10.0,
            memory_bandwidth_gbps: 256.0,
            atomic_support: true,
            tensor_core_support: false,
            os: crate::hardware::OperatingSystem::Windows,
        }
    }

    #[test]
    fn test_jit_compilation_cache() {
        let hw = create_mock_hardware();
        let engine = JitEngine::new(&hw).unwrap();

        let fake_ptx = b".version 8.0\n.target sm_90\n mma.sync.aligned.m16n8k16 ";
        
        // Compile first time (cache miss)
        let kernel1 = engine.compile_ptx(fake_ptx, 42).unwrap();
        assert!(kernel1.name.contains("matmul"));

        // Compile second time (cache hit)
        let kernel2 = engine.compile_ptx(fake_ptx, 42).unwrap();
        assert_eq!(kernel1.name, kernel2.name);
        assert_eq!(kernel1.payload, kernel2.payload);
    }
}
