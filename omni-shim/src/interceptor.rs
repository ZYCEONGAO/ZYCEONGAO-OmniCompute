//! # interceptor
//!
//! OS-level dynamic link interception and dispatch logic.
//!
//! This module provides the platform-specific mechanics for:
//! 1. **Hook registration** — installing the OmniCompute shim into the dynamic
//!    linker's resolution path
//! 2. **Dispatch table** — a fast lookup table for forwarding any unimplemented
//!    symbol to its real CUDA counterpart (when available)
//! 3. **Thread-local state** — per-thread current device tracking
//!
//! ## Platform Injection Guides
//!
//! ### Linux
//! ```bash
//! export LD_PRELOAD=/path/to/libomni_shim.so
//! python inference.py
//! ```
//!
//! ### macOS
//! ```bash
//! export DYLD_INSERT_LIBRARIES=/path/to/libomni_shim.dylib
//! export DYLD_FORCE_FLAT_NAMESPACE=1
//! python inference.py
//! ```
//!
//! ### Windows
//! Place `cudart64_120.dll` (built from this crate) in the same directory as
//! the Python interpreter or the executable, ahead of the real CUDA DLL in
//! the DLL search order.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use tracing::{debug, info, warn};

// ─── Thread-Local Device State ────────────────────────────────────────────────

std::thread_local! {
    /// The currently active virtual device index for this thread.
    /// Mirrors CUDA's per-thread `cudaSetDevice()` semantics.
    static CURRENT_DEVICE: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };

    /// Stack of CUDA contexts pushed via `cuCtxPushCurrent`.
    /// Most frameworks use context stacks for multi-device management.
    static CONTEXT_STACK: std::cell::RefCell<Vec<u64>> =
        std::cell::RefCell::new(Vec::new());
}

/// Returns the current thread's active device index.
pub fn current_device() -> i32 {
    CURRENT_DEVICE.with(|d| d.get())
}

/// Sets the current thread's active device index.
pub fn set_current_device(index: i32) {
    CURRENT_DEVICE.with(|d| d.set(index));
}

// ─── Symbol Dispatch Table ────────────────────────────────────────────────────

/// A registered hook entry: maps a CUDA symbol name to OmniCompute's handler.
pub struct HookEntry {
    /// The CUDA API symbol name (e.g., `"cudaMalloc"`)
    pub symbol: &'static str,
    /// The OmniCompute replacement function pointer
    pub handler: *const libc::c_void,
    /// Whether this hook has been validated (symbol exists in the real CUDA lib)
    pub validated: bool,
}

// SAFETY: HookEntry contains raw function pointers which are Send+Sync
// because they are static references to compiled Rust functions.
unsafe impl Send for HookEntry {}
unsafe impl Sync for HookEntry {}

/// Global dispatch table for CUDA symbol overrides.
///
/// This is queried by the dynamic linker stub on platforms that support
/// runtime symbol interposition (e.g., via `dlsym` / `GetProcAddress` wrapping).
pub static DISPATCH_TABLE: Lazy<RwLock<HashMap<&'static str, HookEntry>>> =
    Lazy::new(|| {
        let mut table = HashMap::new();

        macro_rules! register_hook {
            ($sym:ident) => {
                table.insert(
                    stringify!($sym),
                    HookEntry {
                        symbol: stringify!($sym),
                        handler: $sym as *const libc::c_void,
                        validated: false,
                    },
                );
            };
        }

        // ── CUDA Runtime API ──────────────────────────────────────────────
        register_hook!(cudaGetDeviceCount);
        register_hook!(cudaGetDeviceProperties);
        register_hook!(cudaSetDevice);
        register_hook!(cudaGetDevice);
        register_hook!(cudaMalloc);
        register_hook!(cudaFree);
        register_hook!(cudaMemcpy);
        register_hook!(cudaMemcpyAsync);
        register_hook!(cudaMemset);
        register_hook!(cudaStreamCreate);
        register_hook!(cudaStreamDestroy);
        register_hook!(cudaStreamSynchronize);
        register_hook!(cudaLaunchKernel);
        register_hook!(cudaGetLastError);
        register_hook!(cudaGetErrorString);
        register_hook!(cudaGetErrorName);
        register_hook!(cudaDeviceSynchronize);
        register_hook!(cudaRuntimeGetVersion);

        // ── CUDA Driver API ───────────────────────────────────────────────
        register_hook!(cuInit);
        register_hook!(cuDriverGetVersion);
        register_hook!(cuDeviceGetCount);
        register_hook!(cuDeviceGet);
        register_hook!(cuDeviceGetAttribute);
        register_hook!(cuCtxCreate);
        register_hook!(cuCtxGetCurrent);
        register_hook!(cuCtxSynchronize);
        register_hook!(cuModuleLoad);
        register_hook!(cuModuleLoadData);
        register_hook!(cuModuleGetFunction);
        register_hook!(cuModuleUnload);
        register_hook!(cuLaunchKernel);
        register_hook!(cuMemAlloc);
        register_hook!(cuMemFree);
        register_hook!(cuMemcpyHtoD);
        register_hook!(cuMemcpyDtoH);

        RwLock::new(table)
    });

// Forward declarations (symbols defined in sibling modules)
extern "C" {
    fn cudaGetDeviceCount(count: *mut libc::c_int) -> u32;
    fn cudaGetDeviceProperties(prop: *mut libc::c_void, device: libc::c_int) -> u32;
    fn cudaSetDevice(device: libc::c_int) -> u32;
    fn cudaGetDevice(device: *mut libc::c_int) -> u32;
    fn cudaMalloc(dev_ptr: *mut *mut libc::c_void, size: libc::size_t) -> u32;
    fn cudaFree(dev_ptr: *mut libc::c_void) -> u32;
    fn cudaMemcpy(dst: *mut libc::c_void, src: *const libc::c_void, count: libc::size_t, kind: u32) -> u32;
    fn cudaMemcpyAsync(dst: *mut libc::c_void, src: *const libc::c_void, count: libc::size_t, kind: u32, stream: *mut libc::c_void) -> u32;
    fn cudaMemset(dev_ptr: *mut libc::c_void, value: libc::c_int, count: libc::size_t) -> u32;
    fn cudaStreamCreate(p_stream: *mut *mut libc::c_void) -> u32;
    fn cudaStreamDestroy(stream: *mut libc::c_void) -> u32;
    fn cudaStreamSynchronize(stream: *mut libc::c_void) -> u32;
    fn cudaLaunchKernel(func: *const libc::c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, args: *mut *mut libc::c_void, shared_mem: libc::size_t, stream: *mut libc::c_void) -> u32;
    fn cudaGetLastError() -> u32;
    fn cudaGetErrorString(error: u32) -> *const libc::c_char;
    fn cudaGetErrorName(error: u32) -> *const libc::c_char;
    fn cudaDeviceSynchronize() -> u32;
    fn cudaRuntimeGetVersion(version: *mut libc::c_int) -> u32;
    fn cuInit(flags: libc::c_uint) -> u32;
    fn cuDriverGetVersion(version: *mut libc::c_int) -> u32;
    fn cuDeviceGetCount(count: *mut libc::c_int) -> u32;
    fn cuDeviceGet(device: *mut libc::c_int, ordinal: libc::c_int) -> u32;
    fn cuDeviceGetAttribute(pi: *mut libc::c_int, attrib: libc::c_int, dev: libc::c_int) -> u32;
    fn cuCtxCreate(pctx: *mut *mut libc::c_void, flags: libc::c_uint, dev: libc::c_int) -> u32;
    fn cuCtxGetCurrent(pctx: *mut *mut libc::c_void) -> u32;
    fn cuCtxSynchronize() -> u32;
    fn cuModuleLoad(module: *mut *mut libc::c_void, fname: *const libc::c_char) -> u32;
    fn cuModuleLoadData(module: *mut *mut libc::c_void, image: *const libc::c_void) -> u32;
    fn cuModuleGetFunction(hfunc: *mut *mut libc::c_void, hmod: *mut libc::c_void, name: *const libc::c_char) -> u32;
    fn cuModuleUnload(hmod: *mut libc::c_void) -> u32;
    fn cuLaunchKernel(func: *mut libc::c_void, gx: libc::c_uint, gy: libc::c_uint, gz: libc::c_uint, bx: libc::c_uint, by: libc::c_uint, bz: libc::c_uint, shmem: libc::c_uint, stream: *mut libc::c_void, params: *mut *mut libc::c_void, extra: *mut *mut libc::c_void) -> u32;
    fn cuMemAlloc(dptr: *mut u64, bytesize: libc::size_t) -> u32;
    fn cuMemFree(dptr: u64) -> u32;
    fn cuMemcpyHtoD(dst: u64, src: *const libc::c_void, count: libc::size_t) -> u32;
    fn cuMemcpyDtoH(dst: *mut libc::c_void, src: u64, count: libc::size_t) -> u32;
}

// ─── Hook Statistics ──────────────────────────────────────────────────────────

/// Runtime statistics for the shim's interception activity.
#[derive(Debug, Default)]
pub struct InterceptStats {
    /// Total number of CUDA API calls intercepted
    pub total_calls: u64,
    /// Number of `cudaMalloc` calls (memory pressure indicator)
    pub malloc_calls: u64,
    /// Number of `cuLaunchKernel` / `cudaLaunchKernel` calls (compute intensity)
    pub launch_calls: u64,
    /// Number of `cudaMemcpy` calls (bandwidth indicator)
    pub memcpy_calls: u64,
    /// Total bytes allocated via virtual cudaMalloc
    pub bytes_allocated: u64,
    /// Total bytes transferred via cudaMemcpy
    pub bytes_transferred: u64,
}

/// Global interception statistics — query via `omnicompute status`.
pub static STATS: Lazy<RwLock<InterceptStats>> =
    Lazy::new(|| RwLock::new(InterceptStats::default()));

/// Records a call event into the global statistics tracker.
pub fn record_call(call_type: CallType, bytes: u64) {
    let mut stats = STATS.write();
    stats.total_calls += 1;
    match call_type {
        CallType::Malloc  => { stats.malloc_calls += 1; stats.bytes_allocated += bytes; }
        CallType::Launch  => { stats.launch_calls += 1; }
        CallType::Memcpy  => { stats.memcpy_calls += 1; stats.bytes_transferred += bytes; }
        CallType::Other   => {}
    }
}

/// Categories of CUDA API calls for statistics tracking.
#[derive(Debug, Clone, Copy)]
pub enum CallType {
    Malloc,
    Launch,
    Memcpy,
    Other,
}

// ─── Windows DLL Entry ────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "system" fn DllMain(
    _hinstdll: *mut libc::c_void,
    fdw_reason: u32,
    _lp_reserved: *mut libc::c_void,
) -> i32 {
    const DLL_PROCESS_ATTACH: u32 = 1;
    if fdw_reason == DLL_PROCESS_ATTACH {
        crate::omni_shim_init();
    }
    1 // TRUE
}

// ─── Status Report ────────────────────────────────────────────────────────────

/// Prints a formatted status report of the shim's current activity.
pub fn print_status() {
    let stats = STATS.read();
    let table = DISPATCH_TABLE.read();
    info!("═══════════════════════════════════════════");
    info!("  OmniCompute Shim — Interception Status");
    info!("═══════════════════════════════════════════");
    info!("  Registered hooks : {}", table.len());
    info!("  Total API calls  : {}", stats.total_calls);
    info!("  Kernel launches  : {}", stats.launch_calls);
    info!("  cudaMalloc calls : {}", stats.malloc_calls);
    info!("  Bytes allocated  : {} MB", stats.bytes_allocated / 1024 / 1024);
    info!("  Bytes transferred: {} MB", stats.bytes_transferred / 1024 / 1024);
    info!("═══════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_table_populated() {
        let table = DISPATCH_TABLE.read();
        assert!(table.contains_key("cudaMalloc"));
        assert!(table.contains_key("cuLaunchKernel"));
        assert!(table.contains_key("cuModuleLoad"));
        assert!(table.len() >= 30, "Expected at least 30 hooks");
    }

    #[test]
    fn test_thread_local_device() {
        set_current_device(0);
        assert_eq!(current_device(), 0);
    }

    #[test]
    fn test_record_call_stats() {
        {
            let mut stats = STATS.write();
            *stats = InterceptStats::default();
        }
        record_call(CallType::Launch, 0);
        record_call(CallType::Malloc, 4096);
        let stats = STATS.read();
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.launch_calls, 1);
        assert_eq!(stats.malloc_calls, 1);
        assert_eq!(stats.bytes_allocated, 4096);
    }
}
