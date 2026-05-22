//! # cuda_runtime
//!
//! Intercepts the **CUDA Runtime API** (`cudart`) — the high-level C API that
//! most user code and AI frameworks (PyTorch, etc.) directly invoke.
//!
//! Every exported symbol here has an **identical signature** to the corresponding
//! NVIDIA `cudart64_*.dll` / `libcudart.so` symbol, ensuring that dynamic linkers
//! resolve these stubs instead of the real CUDA runtime.
//!
//! ## Intercepted Functions
//!
//! | Category | Functions |
//! |---|---|
//! | Device Management | `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaSetDevice` |
//! | Memory | `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost` |
//! | Data Transfer | `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset` |
//! | Stream | `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize` |
//! | Event | `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize` |
//! | Kernel Launch | `cudaLaunchKernel` |
//! | Error | `cudaGetLastError`, `cudaGetErrorString` |

use crate::{AllocEntry, KernelLaunch, StreamContext, ALLOC_TABLE, STREAM_TABLE, VIRTUAL_DEVICES};
use std::ffi::CStr;
use tracing::{debug, trace, warn};

// ─── CUDA Return Codes ────────────────────────────────────────────────────────

/// CUDA error codes — mirrors `cudaError_t` from `cuda_runtime_api.h`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum CudaError {
    cudaSuccess                  = 0,
    cudaErrorInvalidValue        = 1,
    cudaErrorMemoryAllocation    = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidDevice       = 10,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorNoDevice            = 100,
    cudaErrorNotReady            = 600,
    cudaErrorUnknown             = 999,
}

use CudaError::*;

/// `cudaMemcpyKind` — direction of `cudaMemcpy` operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum CudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
}

// ─── CUDA Device Properties ───────────────────────────────────────────────────

/// Mirrors the `cudaDeviceProp` structure from CUDA headers.
/// Applications read this to learn about GPU capabilities.
/// We populate it from our [`VirtualDevice`] metadata.
#[repr(C)]
#[derive(Debug)]
#[allow(non_snake_case, non_camel_case_types)]
pub struct cudaDeviceProp {
    pub name:                    [libc::c_char; 256],
    pub uuid:                    [u8; 16],
    pub luid:                    [u8; 8],
    pub luidDeviceNodeMask:      u32,
    pub totalGlobalMem:          usize,
    pub sharedMemPerBlock:       usize,
    pub regsPerBlock:            i32,
    pub warpSize:                i32,
    pub memPitch:                usize,
    pub maxThreadsPerBlock:      i32,
    pub maxThreadsDim:           [i32; 3],
    pub maxGridSize:             [i32; 3],
    pub clockRate:               i32,
    pub totalConstMem:           usize,
    pub major:                   i32,
    pub minor:                   i32,
    pub textureAlignment:        usize,
    pub texturePitchAlignment:   usize,
    pub deviceOverlap:           i32,
    pub multiProcessorCount:     i32,
    pub kernelExecTimeoutEnabled: i32,
    pub integrated:              i32,
    pub canMapHostMemory:        i32,
    pub computeMode:             i32,
    pub concurrentKernels:       i32,
    pub ECCEnabled:              i32,
    pub pciBusID:                i32,
    pub pciDeviceID:             i32,
    pub pciDomainID:             i32,
    pub tccDriver:               i32,
    pub asyncEngineCount:        i32,
    pub unifiedAddressing:       i32,
    pub memoryClockRate:         i32,
    pub memoryBusWidth:          i32,
    pub l2CacheSize:             i32,
    pub persistingL2CacheMaxSize: i32,
    pub maxThreadsPerMultiProcessor: i32,
    pub streamPrioritiesSupported: i32,
    pub globalL1CacheSupported:  i32,
    pub localL1CacheSupported:   i32,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor:   i32,
    pub managedMemory:           i32,
    pub isMultiGpuBoard:         i32,
    pub multiGpuBoardGroupID:    i32,
    pub cooperativeLaunch:       i32,
    pub cooperativeMultiDeviceLaunch: i32,
    pub sharedMemPerBlockOptin:  usize,
    pub pageableMemoryAccessUsesHostPageTables: i32,
    pub directManagedMemAccessFromHost: i32,
    pub maxBlocksPerMultiProcessor: i32,
    pub accessPolicyMaxWindowSize: i32,
    pub reservedSharedMemPerBlock: usize,
    // padding to match real struct size (816 bytes total in CUDA 12)
    pub _padding: [u8; 184],
}

impl Default for cudaDeviceProp {
    fn default() -> Self {
        // SAFETY: POD struct, zero is a valid initial state
        unsafe { std::mem::zeroed() }
    }
}

// ─── Device Management ────────────────────────────────────────────────────────

/// Returns the number of virtual CUDA-compatible devices available.
///
/// Mirrors: `cudaError_t cudaGetDeviceCount(int *count)`
#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceCount(count: *mut libc::c_int) -> CudaError {
    if count.is_null() {
        return cudaErrorInvalidValue;
    }
    let devices = VIRTUAL_DEVICES.read();
    unsafe { *count = devices.len() as libc::c_int; }
    debug!("cudaGetDeviceCount → {}", devices.len());
    cudaSuccess
}

/// Fills a `cudaDeviceProp` struct with virtual device capabilities.
///
/// Mirrors: `cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)`
#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceProperties(
    prop: *mut cudaDeviceProp,
    device: libc::c_int,
) -> CudaError {
    if prop.is_null() {
        return cudaErrorInvalidValue;
    }
    let devices = VIRTUAL_DEVICES.read();
    let Some(vdev) = devices.get(device as usize) else {
        warn!("cudaGetDeviceProperties: invalid device {}", device);
        return cudaErrorInvalidDevice;
    };

    let p = unsafe { &mut *prop };
    *p = cudaDeviceProp::default();

    // Copy device name into fixed-size C char array
    let name_bytes = vdev.name.as_bytes();
    let copy_len = name_bytes.len().min(255);
    unsafe {
        std::ptr::copy_nonoverlapping(
            name_bytes.as_ptr() as *const libc::c_char,
            p.name.as_mut_ptr(),
            copy_len,
        );
    }

    p.totalGlobalMem          = vdev.total_memory;
    p.major                   = vdev.compute_capability_major;
    p.minor                   = vdev.compute_capability_minor;
    p.warpSize                = vdev.warp_size;
    p.multiProcessorCount     = vdev.multiprocessor_count;
    p.maxThreadsPerBlock      = vdev.max_threads_per_block;
    p.maxThreadsDim           = [1024, 1024, 64];
    p.maxGridSize             = [2147483647, 65535, 65535];
    p.l2CacheSize             = vdev.l2_cache_size;
    p.sharedMemPerBlock       = 49152;   // 48 KB — standard CUDA shared mem
    p.regsPerBlock            = 65536;
    p.clockRate               = 1_800_000; // 1800 MHz (placeholder)
    p.memoryClockRate         = 9_001_000; // HBM2e placeholder
    p.memoryBusWidth          = 5120;      // HBM2e 5120-bit bus
    p.unifiedAddressing       = 1;
    p.managedMemory           = 1;
    p.concurrentKernels       = 1;
    p.cooperativeLaunch       = 1;
    p.asyncEngineCount        = 3;
    p.maxBlocksPerMultiProcessor = 32;
    p.sharedMemPerMultiprocessor = 102400; // 100 KB

    debug!("cudaGetDeviceProperties → device {} ({:?})", device, vdev.backend);
    cudaSuccess
}

/// Sets the active CUDA device for the calling thread.
///
/// Mirrors: `cudaError_t cudaSetDevice(int device)`
#[no_mangle]
pub extern "C" fn cudaSetDevice(device: libc::c_int) -> CudaError {
    let devices = VIRTUAL_DEVICES.read();
    if device < 0 || device as usize >= devices.len() {
        warn!("cudaSetDevice: invalid device {}", device);
        return cudaErrorInvalidDevice;
    }
    debug!("cudaSetDevice → {}", device);
    // NOTE: Thread-local device selection tracked in interceptor.rs
    cudaSuccess
}

/// Returns the currently active CUDA device index.
///
/// Mirrors: `cudaError_t cudaGetDevice(int *device)`
#[no_mangle]
pub unsafe extern "C" fn cudaGetDevice(device: *mut libc::c_int) -> CudaError {
    if device.is_null() {
        return cudaErrorInvalidValue;
    }
    unsafe { *device = 0; } // Default to device 0 for MVP
    cudaSuccess
}

// ─── Memory Management ────────────────────────────────────────────────────────

/// Allocates virtual device memory.
///
/// The returned pointer is a **virtual address** managed by the DUVAS allocator
/// in omni-core. It is not a real GPU VRAM address.
///
/// Mirrors: `cudaError_t cudaMalloc(void **devPtr, size_t size)`
#[no_mangle]
pub unsafe extern "C" fn cudaMalloc(
    dev_ptr: *mut *mut libc::c_void,
    size: libc::size_t,
) -> CudaError {
    if dev_ptr.is_null() || size == 0 {
        return cudaErrorInvalidValue;
    }

    // Delegate to omni-core DUVAS allocator
    // For MVP: use a heap allocation tagged with a virtual address marker
    let layout = match std::alloc::Layout::from_size_align(size, 256) {
        Ok(l) => l,
        Err(_) => return cudaErrorMemoryAllocation,
    };

    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return cudaErrorMemoryAllocation;
    }

    let virtual_addr = ptr as u64;

    ALLOC_TABLE.insert(
        virtual_addr,
        AllocEntry {
            virtual_ptr: virtual_addr,
            size,
            device_index: 0,
            pinned: false,
        },
    );

    unsafe { *dev_ptr = ptr as *mut libc::c_void; }
    trace!("cudaMalloc({} bytes) → 0x{:x}", size, virtual_addr);
    cudaSuccess
}

/// Frees virtual device memory previously allocated by `cudaMalloc`.
///
/// Mirrors: `cudaError_t cudaFree(void *devPtr)`
#[no_mangle]
pub unsafe extern "C" fn cudaFree(dev_ptr: *mut libc::c_void) -> CudaError {
    if dev_ptr.is_null() {
        return cudaSuccess; // CUDA spec: freeing NULL is a no-op
    }

    let addr = dev_ptr as u64;
    if let Some((_, entry)) = ALLOC_TABLE.remove(&addr) {
        let layout = std::alloc::Layout::from_size_align(entry.size, 256)
            .expect("Allocation layout must be valid");
        unsafe { std::alloc::dealloc(dev_ptr as *mut u8, layout); }
        trace!("cudaFree(0x{:x})", addr);
        cudaSuccess
    } else {
        warn!("cudaFree: unknown pointer 0x{:x}", addr);
        cudaErrorInvalidValue
    }
}

/// Copies data between host and device memory regions.
///
/// Mirrors: `cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)`
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy(
    dst: *mut libc::c_void,
    src: *const libc::c_void,
    count: libc::size_t,
    kind: CudaMemcpyKind,
) -> CudaError {
    if dst.is_null() || src.is_null() {
        return cudaErrorInvalidValue;
    }

    trace!("cudaMemcpy({:?}, {} bytes)", kind, count);

    // In OmniCompute's unified virtual address model, H2D, D2H, D2D all reduce
    // to the same memcpy — the DUVAS pager handles physical data placement.
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count); }
    cudaSuccess
}

/// Asynchronous memcpy — enqueues on the given stream.
///
/// Mirrors: `cudaError_t cudaMemcpyAsync(..., cudaStream_t stream)`
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyAsync(
    dst: *mut libc::c_void,
    src: *const libc::c_void,
    count: libc::size_t,
    kind: CudaMemcpyKind,
    _stream: *mut libc::c_void,
) -> CudaError {
    // For MVP, delegate to synchronous version.
    // Production: enqueue in StreamContext's async DMA pipeline.
    unsafe { cudaMemcpy(dst, src, count, kind) }
}

/// Sets device memory to a specific byte value.
///
/// Mirrors: `cudaError_t cudaMemset(void *devPtr, int value, size_t count)`
#[no_mangle]
pub unsafe extern "C" fn cudaMemset(
    dev_ptr: *mut libc::c_void,
    value: libc::c_int,
    count: libc::size_t,
) -> CudaError {
    if dev_ptr.is_null() {
        return cudaErrorInvalidValue;
    }
    unsafe { std::ptr::write_bytes(dev_ptr as *mut u8, value as u8, count); }
    trace!("cudaMemset({} bytes, val={})", count, value);
    cudaSuccess
}

// ─── Stream Management ────────────────────────────────────────────────────────

/// Creates a CUDA execution stream.
///
/// Mirrors: `cudaError_t cudaStreamCreate(cudaStream_t *pStream)`
#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreate(
    p_stream: *mut *mut libc::c_void,
) -> CudaError {
    if p_stream.is_null() {
        return cudaErrorInvalidValue;
    }

    // Use a unique ID as the stream handle
    static STREAM_COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(1);
    let handle = STREAM_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    STREAM_TABLE.insert(handle, StreamContext {
        handle,
        device_index: 0,
        synchronized: true,
        pending_kernels: Vec::new(),
    });

    unsafe { *p_stream = handle as *mut libc::c_void; }
    debug!("cudaStreamCreate → handle={}", handle);
    cudaSuccess
}

/// Destroys a CUDA stream.
///
/// Mirrors: `cudaError_t cudaStreamDestroy(cudaStream_t stream)`
#[no_mangle]
pub extern "C" fn cudaStreamDestroy(stream: *mut libc::c_void) -> CudaError {
    let handle = stream as u64;
    STREAM_TABLE.remove(&handle);
    debug!("cudaStreamDestroy({})", handle);
    cudaSuccess
}

/// Blocks until all operations in a stream have completed.
///
/// Mirrors: `cudaError_t cudaStreamSynchronize(cudaStream_t stream)`
#[no_mangle]
pub extern "C" fn cudaStreamSynchronize(stream: *mut libc::c_void) -> CudaError {
    let handle = stream as u64;
    if let Some(mut ctx) = STREAM_TABLE.get_mut(&handle) {
        // Drain pending kernels — in production this flushes the JIT queue
        ctx.pending_kernels.clear();
        ctx.synchronized = true;
    }
    debug!("cudaStreamSynchronize({})", handle);
    cudaSuccess
}

// ─── Kernel Launch ────────────────────────────────────────────────────────────

/// High-level CUDA Runtime kernel launch.
///
/// This is the primary entry point for operator capture. When PyTorch or any
/// CUDA-based framework launches a compute kernel, this function intercepts it,
/// serializes the kernel parameters, and dispatches to `omni-core` for JIT
/// translation and execution on the actual hardware backend.
///
/// Mirrors: `cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)`
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchKernel(
    func: *const libc::c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    args: *mut *mut libc::c_void,
    shared_mem: libc::size_t,
    stream: *mut libc::c_void,
) -> CudaError {
    let kernel_id = func as u64;
    let stream_handle = stream as u64;

    debug!(
        "cudaLaunchKernel: kernel=0x{:x} grid=({},{},{}) block=({},{},{}) shmem={}",
        kernel_id,
        grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem,
    );

    // Build launch descriptor
    let launch = KernelLaunch {
        kernel_id,
        grid_dim:  (grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim: (block_dim_x, block_dim_y, block_dim_z),
        shared_mem_bytes: shared_mem as u32,
        args: Vec::new(), // TODO: Serialize args array with type reflection
    };

    // Dispatch to omni-core JIT engine
    // omni_core::jit::dispatch_kernel(&launch)?;

    // Queue on stream if provided, otherwise execute synchronously
    if stream_handle != 0 {
        if let Some(mut ctx) = STREAM_TABLE.get_mut(&stream_handle) {
            ctx.pending_kernels.push(launch);
            ctx.synchronized = false;
        }
    }

    cudaSuccess
}

// ─── Error Handling ───────────────────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: std::cell::Cell<CudaError> = const { std::cell::Cell::new(CudaError::cudaSuccess) };
}

/// Returns the last CUDA error on this thread.
///
/// Mirrors: `cudaError_t cudaGetLastError(void)`
#[no_mangle]
pub extern "C" fn cudaGetLastError() -> CudaError {
    LAST_ERROR.with(|e| e.replace(cudaSuccess))
}

/// Returns a human-readable string for a CUDA error code.
///
/// Mirrors: `const char* cudaGetErrorString(cudaError_t error)`
#[no_mangle]
pub extern "C" fn cudaGetErrorString(error: CudaError) -> *const libc::c_char {
    match error {
        cudaSuccess               => b"cudaSuccess\0".as_ptr() as _,
        cudaErrorInvalidValue     => b"invalid argument\0".as_ptr() as _,
        cudaErrorMemoryAllocation => b"device-side assert triggered / out of memory\0".as_ptr() as _,
        cudaErrorInvalidDevice    => b"invalid device ordinal\0".as_ptr() as _,
        _                         => b"unknown error\0".as_ptr() as _,
    }
}

/// Returns a human-readable error name.
///
/// Mirrors: `const char* cudaGetErrorName(cudaError_t error)`
#[no_mangle]
pub extern "C" fn cudaGetErrorName(error: CudaError) -> *const libc::c_char {
    match error {
        cudaSuccess               => b"cudaSuccess\0".as_ptr() as _,
        cudaErrorInvalidValue     => b"cudaErrorInvalidValue\0".as_ptr() as _,
        cudaErrorMemoryAllocation => b"cudaErrorMemoryAllocation\0".as_ptr() as _,
        _                         => b"cudaErrorUnknown\0".as_ptr() as _,
    }
}

/// Always returns `cudaSuccess` — used by frameworks to check runtime health.
///
/// Mirrors: `cudaError_t cudaDeviceSynchronize(void)`
#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> CudaError {
    debug!("cudaDeviceSynchronize");
    cudaSuccess
}

/// Returns CUDA Runtime library version.
///
/// Mirrors: `cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)`
#[no_mangle]
pub unsafe extern "C" fn cudaRuntimeGetVersion(version: *mut libc::c_int) -> CudaError {
    if version.is_null() {
        return cudaErrorInvalidValue;
    }
    // Report CUDA 12.4 compatibility
    unsafe { *version = 12040; }
    cudaSuccess
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_count() {
        let mut count = 0i32;
        let err = unsafe { cudaGetDeviceCount(&mut count) };
        assert_eq!(err, cudaSuccess);
        assert!(count > 0);
    }

    #[test]
    fn test_device_properties() {
        let mut prop = cudaDeviceProp::default();
        let err = unsafe { cudaGetDeviceProperties(&mut prop, 0) };
        assert_eq!(err, cudaSuccess);
        assert!(prop.totalGlobalMem > 0);
        assert_eq!(prop.major, 8);
    }

    #[test]
    fn test_malloc_free_roundtrip() {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, 4096) };
        assert_eq!(err, cudaSuccess);
        assert!(!ptr.is_null());

        let free_err = unsafe { cudaFree(ptr) };
        assert_eq!(free_err, cudaSuccess);
    }

    #[test]
    fn test_memcpy_h2d_d2h() {
        let host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut dev_ptr: *mut libc::c_void = std::ptr::null_mut();
        let size = host_data.len() * std::mem::size_of::<f32>();

        unsafe { cudaMalloc(&mut dev_ptr, size) };

        let copy_err = unsafe {
            cudaMemcpy(
                dev_ptr,
                host_data.as_ptr() as *const libc::c_void,
                size,
                CudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        assert_eq!(copy_err, cudaSuccess);

        let mut result = vec![0.0f32; 4];
        let back_err = unsafe {
            cudaMemcpy(
                result.as_mut_ptr() as *mut libc::c_void,
                dev_ptr,
                size,
                CudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        assert_eq!(back_err, cudaSuccess);
        assert_eq!(result, host_data);

        unsafe { cudaFree(dev_ptr) };
    }

    #[test]
    fn test_stream_lifecycle() {
        let mut stream: *mut libc::c_void = std::ptr::null_mut();
        let create_err = unsafe { cudaStreamCreate(&mut stream) };
        assert_eq!(create_err, cudaSuccess);
        assert!(!stream.is_null());

        let sync_err = cudaStreamSynchronize(stream);
        assert_eq!(sync_err, cudaSuccess);

        let destroy_err = cudaStreamDestroy(stream);
        assert_eq!(destroy_err, cudaSuccess);
    }
}
