//! # hardware
//!
//! Runtime hardware detection and capability profiling.
//!
//! Probes the host machine at startup to determine:
//! - Which compute backend to use (AMD / Apple / Intel / CPU)
//! - Physical memory capacity and topology
//! - Microarchitecture features (wavefront size, SRAM layout, UMA vs. discrete)
//!
//! The resulting [`HardwareProfile`] drives all downstream decisions in the
//! JIT engine and memory pager.

use anyhow::{bail, Result};
use tracing::{debug, info, warn};

// ─── Target Backend ───────────────────────────────────────────────────────────

/// The hardware backend selected for code generation and execution.
///
/// Determined at runtime by [`HardwareDetector::probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetBackend {
    /// AMD GPU — emits AMDGCN assembly via LLVM ROCm backend.
    /// Supports RDNA2 (Wave32/64), RDNA3, and older GCN architectures.
    AmdRocm {
        /// GFX architecture string (e.g., "gfx1100" for RDNA3)
        gfx_arch: String,
        /// Wavefront execution width (32 for RDNA2+, 64 for older GCN)
        wavefront_size: u32,
        /// Number of compute units
        compute_units: u32,
    },

    /// Apple Silicon — emits Metal Shading Language via metal-cpp.
    /// Benefits from Unified Memory Architecture (UMA): zero-copy H2D transfers.
    AppleMetal {
        /// Chip generation (e.g., "M3 Max")
        chip_name: String,
        /// Total unified memory in bytes (shared CPU + GPU)
        unified_memory_bytes: usize,
        /// GPU core count
        gpu_cores: u32,
    },

    /// Intel Arc / other Vulkan-capable GPUs — emits SPIR-V.
    VulkanGeneric {
        /// Vulkan device name
        device_name: String,
        /// Maximum compute shader invocations per workgroup
        max_invocations: u32,
    },

    /// CPU fallback — emits AVX-512 / NEON vectorized code.
    /// Used when no GPU is available or for scalar fallback.
    CpuVectorized {
        /// Number of physical CPU cores
        core_count: u32,
        /// SIMD width in bits (256 for AVX2, 512 for AVX-512, 128 for NEON)
        simd_width: u32,
    },

    /// Remote P2P node via omni-net — dispatches to a networked heterogeneous device.
    RemoteP2p {
        /// Node peer ID
        peer_id: String,
        /// Estimated round-trip latency in microseconds
        latency_us: u64,
    },
}

impl TargetBackend {
    /// Returns a user-friendly name of the active compute device.
    pub fn device_name(&self) -> String {
        match self {
            Self::AmdRocm { gfx_arch, .. } => format!("AMD Radeon GPU ({})", gfx_arch),
            Self::AppleMetal { chip_name, .. } => chip_name.clone(),
            Self::VulkanGeneric { device_name, .. } => device_name.clone(),
            Self::CpuVectorized { core_count, simd_width } => {
                format!("Host CPU ({} Cores, {}bit SIMD)", core_count, simd_width)
            }
            Self::RemoteP2p { peer_id, .. } => format!("Remote Peer Node ({})", peer_id),
        }
    }
}

// ─── Hardware Profile ─────────────────────────────────────────────────────────

/// Complete hardware capability profile for the local machine.
///
/// Produced by [`HardwareDetector::probe`] and consumed by:
/// - [`JitEngine`]: selects the correct codegen backend
/// - [`VirtualSlidingPager`]: calibrates page size and prefetch windows
/// - [`DuvasAllocator`]: sets the virtual address space bounds
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Selected compute backend
    pub target_backend: TargetBackend,

    /// Total device VRAM / unified memory (bytes)
    pub vram_bytes: usize,

    /// L2 on-chip cache size (bytes) — used to calibrate virtual page size
    pub l2_cache_bytes: usize,

    /// Peak theoretical FP16 TFLOPS (for scheduler priority scoring)
    pub fp16_tflops: f64,

    /// Peak memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,

    /// Whether the device supports hardware-accelerated atomic operations
    pub atomic_support: bool,

    /// Whether Tensor Core / Matrix Core equivalents are available
    pub tensor_core_support: bool,

    /// Operating system
    pub os: OperatingSystem,
}

/// Host operating system — affects injection mechanism selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatingSystem {
    Linux,
    MacOs,
    Windows,
}

impl OperatingSystem {
    /// Returns the current OS at compile time.
    pub fn current() -> Self {
        #[cfg(target_os = "linux")]   { OperatingSystem::Linux }
        #[cfg(target_os = "macos")]   { OperatingSystem::MacOs }
        #[cfg(target_os = "windows")] { OperatingSystem::Windows }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows"
        )))]
        { OperatingSystem::Linux } // default
    }

    /// Returns the appropriate LD_PRELOAD / DYLD_INSERT env var for this OS.
    pub fn preload_env_var(&self) -> &'static str {
        match self {
            OperatingSystem::Linux   => "LD_PRELOAD",
            OperatingSystem::MacOs   => "DYLD_INSERT_LIBRARIES",
            OperatingSystem::Windows => "N/A (DLL search order hijack)",
        }
    }
}

// ─── Hardware Detector ────────────────────────────────────────────────────────

/// Probes the host hardware at runtime to build a [`HardwareProfile`].
pub struct HardwareDetector;

impl HardwareDetector {
    /// Probes all available hardware and returns the best available profile.
    ///
    /// Detection priority:
    /// 1. AMD GPU (ROCm)
    /// 2. Apple Silicon (Metal / UMA)
    /// 3. Vulkan-capable GPU
    /// 4. CPU vectorized fallback
    ///
    /// In the future this will use platform-native APIs:
    /// - Linux: `/sys/class/drm/`, `rocm-smi`, `vulkaninfo`
    /// - macOS: `Metal::MTLCreateSystemDefaultDevice()`
    /// - Windows: `IDXGIFactory::EnumAdapters()`
    pub fn probe() -> Result<HardwareProfile> {
        debug!("HardwareDetector: starting hardware probe...");

        // Try detection in priority order
        if let Some(profile) = Self::try_detect_amd() {
            info!("HardwareDetector: AMD GPU detected");
            return Ok(profile);
        }

        if let Some(profile) = Self::try_detect_apple_metal() {
            info!("HardwareDetector: Apple Silicon detected");
            return Ok(profile);
        }

        if let Some(profile) = Self::try_detect_vulkan() {
            info!("HardwareDetector: Vulkan GPU detected");
            return Ok(profile);
        }

        // Always succeeds — CPU fallback
        Ok(Self::cpu_fallback())
    }

    /// Attempts to detect an AMD GPU via ROCm.
    fn try_detect_amd() -> Option<HardwareProfile> {
        // Production: query rocm-smi or /sys/class/drm/card*/device/vendor (0x1002 = AMD)
        // For simulation, check environment variable override
        if std::env::var("OMNI_FORCE_BACKEND").as_deref() == Ok("amd") {
            return Some(HardwareProfile {
                target_backend: TargetBackend::AmdRocm {
                    gfx_arch: "gfx1100".to_string(), // RDNA3 (RX 7900 XTX)
                    wavefront_size: 32,
                    compute_units: 96,
                },
                vram_bytes: 24 * 1024 * 1024 * 1024,   // 24 GB GDDR6
                l2_cache_bytes: 6 * 1024 * 1024,        // 6 MB L2
                fp16_tflops: 123.0,                      // RX 7900 XTX peak
                memory_bandwidth_gbps: 960.0,
                atomic_support: true,
                tensor_core_support: true,               // WMMA on RDNA3
                os: OperatingSystem::current(),
            });
        }

        // On Linux, check for AMD GPU in sysfs
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/dev/kfd").exists() {
                debug!("HardwareDetector: /dev/kfd found — AMD ROCm likely available");
                return Some(HardwareProfile {
                    target_backend: TargetBackend::AmdRocm {
                        gfx_arch: "gfx906".to_string(), // conservative default
                        wavefront_size: 64,
                        compute_units: 60,
                    },
                    vram_bytes: 16 * 1024 * 1024 * 1024,
                    l2_cache_bytes: 4 * 1024 * 1024,
                    fp16_tflops: 26.0,
                    memory_bandwidth_gbps: 512.0,
                    atomic_support: true,
                    tensor_core_support: false,
                    os: OperatingSystem::Linux,
                });
            }
        }

        None
    }

    /// Attempts to detect Apple Silicon via Metal availability.
    fn try_detect_apple_metal() -> Option<HardwareProfile> {
        #[cfg(target_os = "macos")]
        {
            // Production: call metal::Device::system_default()
            // Check if we are on Apple Silicon (arm64 macOS)
            #[cfg(target_arch = "aarch64")]
            {
                debug!("HardwareDetector: aarch64 macOS detected — Apple Silicon");
                return Some(HardwareProfile {
                    target_backend: TargetBackend::AppleMetal {
                        chip_name: "Apple Silicon".to_string(),
                        unified_memory_bytes: 24 * 1024 * 1024 * 1024,
                        gpu_cores: 30,
                    },
                    vram_bytes: 24 * 1024 * 1024 * 1024,  // UMA — total system RAM
                    l2_cache_bytes: 24 * 1024 * 1024,      // M-series L2 is large
                    fp16_tflops: 27.0,
                    memory_bandwidth_gbps: 300.0,
                    atomic_support: true,
                    tensor_core_support: true,
                    os: OperatingSystem::MacOs,
                });
            }
        }

        if std::env::var("OMNI_FORCE_BACKEND").as_deref() == Ok("metal") {
            return Some(HardwareProfile {
                target_backend: TargetBackend::AppleMetal {
                    chip_name: "Apple M3 Max (simulated)".to_string(),
                    unified_memory_bytes: 96 * 1024 * 1024 * 1024,
                    gpu_cores: 40,
                },
                vram_bytes: 96 * 1024 * 1024 * 1024,
                l2_cache_bytes: 48 * 1024 * 1024,
                fp16_tflops: 49.0,
                memory_bandwidth_gbps: 400.0,
                atomic_support: true,
                tensor_core_support: true,
                os: OperatingSystem::current(),
            });
        }

        None
    }

    /// Attempts to detect a Vulkan-capable GPU.
    fn try_detect_vulkan() -> Option<HardwareProfile> {
        // Production: enumerate VkPhysicalDevice via ash / vulkano
        if std::env::var("OMNI_FORCE_BACKEND").as_deref() == Ok("vulkan") {
            return Some(HardwareProfile {
                target_backend: TargetBackend::VulkanGeneric {
                    device_name: "Vulkan Device (simulated)".to_string(),
                    max_invocations: 1024,
                },
                vram_bytes: 8 * 1024 * 1024 * 1024,
                l2_cache_bytes: 2 * 1024 * 1024,
                fp16_tflops: 10.0,
                memory_bandwidth_gbps: 256.0,
                atomic_support: true,
                tensor_core_support: false,
                os: OperatingSystem::current(),
            });
        }
        None
    }

    /// CPU-only fallback — always available.
    fn cpu_fallback() -> HardwareProfile {
        let core_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);

        // Detect SIMD width from target features
        #[cfg(target_feature = "avx512f")]
        let simd_width = 512u32;
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        let simd_width = 256u32;
        #[cfg(all(target_arch = "aarch64", not(target_feature = "avx2")))]
        let simd_width = 128u32;
        #[cfg(not(any(
            target_feature = "avx512f",
            target_feature = "avx2",
            target_arch = "aarch64"
        )))]
        let simd_width = 128u32;

        warn!("HardwareDetector: no GPU detected, using CPU vectorized fallback");
        HardwareProfile {
            target_backend: TargetBackend::CpuVectorized { core_count, simd_width },
            vram_bytes: 8 * 1024 * 1024 * 1024,  // 8 GB system RAM estimate
            l2_cache_bytes: 1024 * 1024,           // 1 MB L2 per core typical
            fp16_tflops: (core_count as f64) * 0.001, // rough estimate
            memory_bandwidth_gbps: 50.0,
            atomic_support: true,
            tensor_core_support: false,
            os: OperatingSystem::current(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_always_succeeds() {
        let profile = HardwareDetector::probe().unwrap();
        assert!(profile.vram_bytes > 0);
        assert!(profile.l2_cache_bytes > 0);
    }

    #[test]
    fn test_os_detection() {
        let os = OperatingSystem::current();
        // Should be one of the known variants
        assert!(matches!(
            os,
            OperatingSystem::Linux | OperatingSystem::MacOs | OperatingSystem::Windows
        ));
    }

    #[test]
    fn test_cpu_fallback_core_count() {
        let profile = HardwareDetector::probe().unwrap();
        if let TargetBackend::CpuVectorized { core_count, .. } = profile.target_backend {
            assert!(core_count >= 1);
        }
    }
}
