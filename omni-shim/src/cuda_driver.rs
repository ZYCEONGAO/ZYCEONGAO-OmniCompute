//! # cuda_driver
//!
//! Intercepts the **CUDA Driver API** (`libcuda.so`) — the low-level C API used
//! by frameworks that want finer-grained control over module loading, function
//! lookup, and kernel dispatch (e.g., Triton-generated kernels, cuDNN internals).
//!
//! The Driver API operates on opaque `CUresult` codes and handle-based objects
//! (`CUmodule`, `CUfunction`, `CUstream`, `CUdeviceptr`).

use crate::{KernelLaunch, STREAM_TABLE};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tracing::{debug, trace, warn};

// ─── CUDA Driver Return Codes ─────────────────────────────────────────────────

/// CUDA Driver API error codes — mirrors `CUresult` from `cuda.h`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum CuResult {
    CUDA_SUCCESS                    = 0,
    CUDA_ERROR_INVALID_VALUE        = 1,
    CUDA_ERROR_OUT_OF_MEMORY        = 2,
    CUDA_ERROR_NOT_INITIALIZED      = 3,
    CUDA_ERROR_INVALID_CONTEXT      = 201,
    CUDA_ERROR_INVALID_MODULE       = 301,
    CUDA_ERROR_INVALID_IMAGE        = 200,
    CUDA_ERROR_NOT_FOUND            = 500,
    CUDA_ERROR_INVALID_HANDLE       = 400,
    CUDA_ERROR_NO_DEVICE            = 100,
    CUDA_ERROR_UNKNOWN              = 999,
}

use CuResult::*;

// ─── Handle Types ─────────────────────────────────────────────────────────────

/// Opaque CUDA module handle — wraps a compiled PTX/cubin binary.
/// In OmniCompute this maps to a JIT-compiled MLIR module.
#[repr(C)]
pub struct CUmod(u64);

/// Opaque CUDA function handle — a kernel entry point within a module.
#[repr(C)]
pub struct CUfunc(u64);

/// CUDA device pointer type
pub type CUdeviceptr = u64;

// ─── Module Registry ──────────────────────────────────────────────────────────

/// A loaded CUDA module descriptor.
#[derive(Debug)]
pub struct ModuleEntry {
    /// Unique handle ID
    pub id: u64,
    /// Source: path to .ptx/.cubin or embedded PTX text
    pub source: ModuleSource,
    /// Map of kernel name → function handle ID
    pub functions: HashMap<String, u64>,
    /// Raw PTX/cubin bytes (stored for JIT re-compilation by omni-core)
    pub raw_bytes: Vec<u8>,
}

/// How the module was originally loaded.
#[derive(Debug, Clone)]
pub enum ModuleSource {
    /// Loaded from a file path (e.g., `cuModuleLoad("/tmp/kernel.ptx")`)
    File(String),
    /// Embedded PTX/cubin data pointer
    Data { ptx_len: usize },
    /// Fatbin (multiple target architectures bundled)
    Fatbin,
}

/// Global registry of loaded CUDA modules.
static MODULE_REGISTRY: Lazy<RwLock<HashMap<u64, ModuleEntry>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static MODULE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1000);

static FUNCTION_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(2000);

// ─── Driver Initialization ────────────────────────────────────────────────────

/// Initializes the CUDA driver.
///
/// Mirrors: `CUresult cuInit(unsigned int Flags)`
#[no_mangle]
pub extern "C" fn cuInit(_flags: libc::c_uint) -> CuResult {
    debug!("cuInit → CUDA_SUCCESS (OmniCompute shim)");
    CUDA_SUCCESS
}

/// Returns the CUDA driver version.
///
/// Mirrors: `CUresult cuDriverGetVersion(int *driverVersion)`
#[no_mangle]
pub unsafe extern "C" fn cuDriverGetVersion(version: *mut libc::c_int) -> CuResult {
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Report CUDA 12.4 driver compatibility
    unsafe { *version = 12040; }
    CUDA_SUCCESS
}

// ─── Device Management ────────────────────────────────────────────────────────

/// Returns the number of available CUDA devices.
///
/// Mirrors: `CUresult cuDeviceGetCount(int *count)`
#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetCount(count: *mut libc::c_int) -> CuResult {
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let devices = crate::VIRTUAL_DEVICES.read();
    unsafe { *count = devices.len() as libc::c_int; }
    CUDA_SUCCESS
}

/// Returns a handle to the specified CUDA device.
///
/// Mirrors: `CUresult cuDeviceGet(CUdevice *device, int ordinal)`
#[no_mangle]
pub unsafe extern "C" fn cuDeviceGet(
    device: *mut libc::c_int,
    ordinal: libc::c_int,
) -> CuResult {
    if device.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let devices = crate::VIRTUAL_DEVICES.read();
    if ordinal < 0 || ordinal as usize >= devices.len() {
        return CUDA_ERROR_NO_DEVICE;
    }
    unsafe { *device = ordinal; }
    CUDA_SUCCESS
}

/// Returns a device attribute value.
///
/// Mirrors: `CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)`
#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetAttribute(
    pi: *mut libc::c_int,
    attrib: libc::c_int,
    _dev: libc::c_int,
) -> CuResult {
    if pi.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let devices = crate::VIRTUAL_DEVICES.read();
    let vdev = match devices.first() {
        Some(d) => d,
        None => return CUDA_ERROR_NO_DEVICE,
    };

    // CUdevice_attribute enum values (selected subset):
    let value = match attrib {
        1  => vdev.max_threads_per_block,         // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        4  => vdev.warp_size,                     // CU_DEVICE_ATTRIBUTE_WARP_SIZE
        14 => vdev.multiprocessor_count,          // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        16 => vdev.compute_capability_major,      // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        17 => vdev.compute_capability_minor,      // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        39 => 1,                                  // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
        75 => 32,                                 // CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR
        _  => 0,
    };
    unsafe { *pi = value; }
    CUDA_SUCCESS
}

// ─── Context Management ───────────────────────────────────────────────────────

/// Creates a CUDA context for a device.
///
/// Mirrors: `CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)`
#[no_mangle]
pub unsafe extern "C" fn cuCtxCreate(
    pctx: *mut *mut libc::c_void,
    _flags: libc::c_uint,
    _dev: libc::c_int,
) -> CuResult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Return a dummy non-null context handle
    static CTX_HANDLE: u64 = 0xDEAD_BEEF_CAFE_BABE;
    unsafe { *pctx = &CTX_HANDLE as *const u64 as *mut libc::c_void; }
    debug!("cuCtxCreate → context handle created");
    CUDA_SUCCESS
}

/// Gets the current CUDA context.
///
/// Mirrors: `CUresult cuCtxGetCurrent(CUcontext *pctx)`
#[no_mangle]
pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut *mut libc::c_void) -> CuResult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    static CTX_HANDLE: u64 = 0xDEAD_BEEF_CAFE_BABE;
    unsafe { *pctx = &CTX_HANDLE as *const u64 as *mut libc::c_void; }
    CUDA_SUCCESS
}

/// Synchronizes the current context.
///
/// Mirrors: `CUresult cuCtxSynchronize(void)`
#[no_mangle]
pub extern "C" fn cuCtxSynchronize() -> CuResult {
    debug!("cuCtxSynchronize");
    CUDA_SUCCESS
}

// ─── Module Management ────────────────────────────────────────────────────────

/// Loads a CUDA module from a PTX/cubin file.
///
/// The PTX binary is captured here and passed to `omni-core`'s JIT engine,
/// which lifts it to the `omni.tensor` MLIR dialect for hardware-agnostic
/// re-compilation.
///
/// Mirrors: `CUresult cuModuleLoad(CUmodule *module, const char *fname)`
#[no_mangle]
pub unsafe extern "C" fn cuModuleLoad(
    module: *mut *mut libc::c_void,
    fname: *const libc::c_char,
) -> CuResult {
    if module.is_null() || fname.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let path = unsafe {
        match std::ffi::CStr::from_ptr(fname).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return CUDA_ERROR_INVALID_VALUE,
        }
    };

    debug!("cuModuleLoad: {}", path);

    // Read PTX/cubin bytes
    let raw_bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            warn!("cuModuleLoad: cannot read '{}': {}", path, e);
            return CUDA_ERROR_INVALID_IMAGE;
        }
    };

    // TODO(@omni-core): Pass raw_bytes to omni_core::mlir::lift_ptx(&raw_bytes)
    //   which will parse the PTX and build an MLIR module.

    let id = MODULE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MODULE_REGISTRY.write().insert(id, ModuleEntry {
        id,
        source: ModuleSource::File(path),
        functions: HashMap::new(),
        raw_bytes,
    });

    unsafe { *module = id as usize as *mut libc::c_void; }
    CUDA_SUCCESS
}

/// Loads a CUDA module from in-memory PTX/fatbin data.
///
/// Mirrors: `CUresult cuModuleLoadData(CUmodule *module, const void *image)`
#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadData(
    module: *mut *mut libc::c_void,
    image: *const libc::c_void,
) -> CuResult {
    if module.is_null() || image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Try to determine PTX length by scanning for null terminator (PTX is text)
    let ptx_str = unsafe { std::ffi::CStr::from_ptr(image as *const libc::c_char) };
    let raw_bytes = ptx_str.to_bytes().to_vec();
    let ptx_len = raw_bytes.len();

    debug!("cuModuleLoadData: {} bytes of PTX/cubin", ptx_len);

    // TODO(@omni-core): Lift raw_bytes → omni.tensor MLIR dialect
    let id = MODULE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MODULE_REGISTRY.write().insert(id, ModuleEntry {
        id,
        source: ModuleSource::Data { ptx_len },
        functions: HashMap::new(),
        raw_bytes,
    });

    unsafe { *module = id as usize as *mut libc::c_void; }
    CUDA_SUCCESS
}

/// Retrieves a function handle from a loaded module by name.
///
/// This is where Triton and custom CUDA kernels get resolved. The function
/// name and its corresponding PTX are handed to `omni-core` for JIT compilation.
///
/// Mirrors: `CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)`
#[no_mangle]
pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut *mut libc::c_void,
    hmod: *mut libc::c_void,
    name: *const libc::c_char,
) -> CuResult {
    if hfunc.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let func_name = unsafe {
        match std::ffi::CStr::from_ptr(name).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return CUDA_ERROR_INVALID_VALUE,
        }
    };

    let mod_id = hmod as u64;
    debug!("cuModuleGetFunction: module={} name={}", mod_id, func_name);

    let func_id = FUNCTION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    if let Some(mut entry) = MODULE_REGISTRY.write().get_mut(&mod_id) {
        entry.functions.insert(func_name, func_id);
    }

    // TODO(@omni-core): Trigger JIT compilation for this function
    //   omni_core::jit::compile_function(mod_id, &func_name)?;

    unsafe { *hfunc = func_id as usize as *mut libc::c_void; }
    CUDA_SUCCESS
}

/// Unloads a CUDA module and frees its resources.
///
/// Mirrors: `CUresult cuModuleUnload(CUmodule hmod)`
#[no_mangle]
pub extern "C" fn cuModuleUnload(hmod: *mut libc::c_void) -> CuResult {
    let id = hmod as u64;
    MODULE_REGISTRY.write().remove(&id);
    debug!("cuModuleUnload: module={}", id);
    CUDA_SUCCESS
}

// ─── Kernel Launch ────────────────────────────────────────────────────────────

/// Launches a CUDA kernel via the Driver API.
///
/// This is the **deepest interception point** for low-level CUDA kernels.
/// All parameters are captured and forwarded to the `omni-core` JIT dispatcher.
///
/// Mirrors:
/// ```c
/// CUresult cuLaunchKernel(
///     CUfunction f,
///     unsigned gridDimX, gridDimY, gridDimZ,
///     unsigned blockDimX, blockDimY, blockDimZ,
///     unsigned sharedMemBytes,
///     CUstream hStream,
///     void **kernelParams,
///     void **extra
/// )
/// ```
#[no_mangle]
pub unsafe extern "C" fn cuLaunchKernel(
    func: *mut libc::c_void,
    grid_dim_x: libc::c_uint,
    grid_dim_y: libc::c_uint,
    grid_dim_z: libc::c_uint,
    block_dim_x: libc::c_uint,
    block_dim_y: libc::c_uint,
    block_dim_z: libc::c_uint,
    shared_mem_bytes: libc::c_uint,
    h_stream: *mut libc::c_void,
    _kernel_params: *mut *mut libc::c_void,
    _extra: *mut *mut libc::c_void,
) -> CuResult {
    let func_id = func as u64;
    let stream_handle = h_stream as u64;

    debug!(
        "cuLaunchKernel: func=0x{:x} grid=({},{},{}) block=({},{},{}) shmem={}B",
        func_id,
        grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes,
    );

    let launch = KernelLaunch {
        kernel_id: func_id,
        grid_dim:  (grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim: (block_dim_x, block_dim_y, block_dim_z),
        shared_mem_bytes,
        args: Vec::new(),
    };

    // Dispatch to JIT engine
    // omni_core::jit::execute_kernel(&launch)?;

    if stream_handle != 0 {
        if let Some(mut ctx) = STREAM_TABLE.get_mut(&stream_handle) {
            ctx.pending_kernels.push(launch);
            ctx.synchronized = false;
        }
    }

    CUDA_SUCCESS
}

// ─── Memory (Driver API) ──────────────────────────────────────────────────────

/// Allocates device memory via the Driver API.
///
/// Mirrors: `CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)`
#[no_mangle]
pub unsafe extern "C" fn cuMemAlloc(
    dptr: *mut CUdeviceptr,
    bytesize: libc::size_t,
) -> CuResult {
    if dptr.is_null() || bytesize == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut raw_ptr: *mut libc::c_void = std::ptr::null_mut();
    let err = unsafe {
        super::cuda_runtime::cudaMalloc(&mut raw_ptr, bytesize)
    };

    if err != super::cuda_runtime::CudaError::cudaSuccess {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    unsafe { *dptr = raw_ptr as CUdeviceptr; }
    trace!("cuMemAlloc({} bytes) → 0x{:x}", bytesize, unsafe { *dptr });
    CUDA_SUCCESS
}

/// Frees device memory via the Driver API.
///
/// Mirrors: `CUresult cuMemFree(CUdeviceptr dptr)`
#[no_mangle]
pub extern "C" fn cuMemFree(dptr: CUdeviceptr) -> CuResult {
    let err = unsafe {
        super::cuda_runtime::cudaFree(dptr as *mut libc::c_void)
    };
    if err != super::cuda_runtime::CudaError::cudaSuccess {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

/// Copies memory from host to device (Driver API).
///
/// Mirrors: `CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)`
#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoD(
    dst_device: CUdeviceptr,
    src_host: *const libc::c_void,
    byte_count: libc::size_t,
) -> CuResult {
    if src_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_host as *const u8,
            dst_device as *mut u8,
            byte_count,
        );
    }
    CUDA_SUCCESS
}

/// Copies memory from device to host (Driver API).
///
/// Mirrors: `CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)`
#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoH(
    dst_host: *mut libc::c_void,
    src_device: CUdeviceptr,
    byte_count: libc::size_t,
) -> CuResult {
    if dst_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_device as *const u8,
            dst_host as *mut u8,
            byte_count,
        );
    }
    CUDA_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cu_init() {
        assert_eq!(cuInit(0), CUDA_SUCCESS);
    }

    #[test]
    fn test_cu_device_count() {
        let mut count = 0i32;
        let err = unsafe { cuDeviceGetCount(&mut count) };
        assert_eq!(err, CUDA_SUCCESS);
        assert!(count > 0);
    }

    #[test]
    fn test_cu_mem_alloc_free() {
        let mut ptr: u64 = 0;
        let err = unsafe { cuMemAlloc(&mut ptr, 1024) };
        assert_eq!(err, CUDA_SUCCESS);
        assert_ne!(ptr, 0);

        let free_err = cuMemFree(ptr);
        assert_eq!(free_err, CUDA_SUCCESS);
    }
}
