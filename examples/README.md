# OmniCompute Examples

This directory contains example workloads demonstrating OmniCompute's zero-invasion virtualization.

## Vector Addition (PyTorch)

`vector_add.py` is a standard PyTorch script that explicitly allocates tensors on the `cuda` device.

Normally, this script will crash on AMD/Apple hardware with a "CUDA not available" error.
With OmniCompute, the script executes seamlessly without modifying a single line of Python code.

### Running the Example

Make sure the `omnicompute` CLI is built and in your `PATH`:

```bash
cargo build --release
export PATH=$PATH:$(pwd)/../target/release
```

Run the script wrapped by OmniCompute:

```bash
omnicompute run python vector_add.py
```

### What happens under the hood?

1. **`omni-shim`** intercepts PyTorch's `cudaGetDeviceCount` and `cudaMalloc` calls, returning virtual hardware profiles and virtual memory pointers.
2. When PyTorch dispatches the `a + b` kernel, `omni-shim` captures the PTX/SASS payload.
3. **`omni-core`**'s JIT compiler translates the kernel to native MSL (Apple), AMDGCN (AMD), or SPIR-V (Vulkan) in milliseconds.
4. The memory pager transparently handles data migration between the host and the actual hardware backend.
