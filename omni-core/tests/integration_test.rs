use anyhow::Result;
use omni_core::{
    memory::allocator::DuvasAllocator,
    hardware::{HardwareDetector, TargetBackend},
    jit::JitEngine,
};

#[test]
fn test_end_to_end_jit_pipeline() -> Result<()> {
    // 1. Detect hardware
    let profile = HardwareDetector::probe()?;
    
    // We expect some backend (AMDGPU, Metal, or SPIR-V)
    assert!(
        matches!(profile.target_backend, TargetBackend::AmdRocm { .. } | TargetBackend::AppleMetal { .. } | TargetBackend::VulkanGeneric { .. } | TargetBackend::CpuVectorized { .. }),
        "Expected valid hardware backend"
    );

    // 2. Initialize memory allocator
    let mut allocator = DuvasAllocator::new(profile.vram_bytes);
    let ptr = allocator.alloc(1024, false)?;
    assert!(ptr > 0);

    let engine = JitEngine::new(&profile)?;
    let ptx_bytes = b"dummy ptx payload";
    let binary = engine.compile_ptx(ptx_bytes, 1)?;
    
    // Binary should contain codegen artifacts
    assert!(!binary.payload.is_empty(), "JIT compilation produced empty payload");

    // 5. Cleanup
    allocator.free(ptr)?;

    Ok(())
}
