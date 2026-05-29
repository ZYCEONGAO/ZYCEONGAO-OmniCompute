# Contributing to OmniCompute

Thank you for your interest in contributing to OmniCompute! This document provides
guidelines and information for contributors.

## Getting Started

### Prerequisites

- **Rust 1.82+** with `cargo`
- **CMake 3.20+** (for LLVM/MLIR integration)
- **Git** for version control

### Building from Source

```bash
git clone https://github.com/ZYCEONGAO/ZYCEONGAO-OmniCompute.git
cd ZYCEONGAO-OmniCompute

# Build all workspace crates
cargo build --workspace

# Run the full test suite
cargo test --workspace

# Check formatting
cargo fmt --all -- --check

# Run linter
cargo clippy --workspace --all-targets
```

## Development Workflow

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally.
3. Create a **feature branch** from `master`:
   ```bash
   git checkout -b feat/my-feature
   ```
4. Make your changes, ensuring:
   - All tests pass (`cargo test --workspace`)
   - Code is formatted (`cargo fmt --all`)
   - No clippy warnings (`cargo clippy --workspace`)
5. **Commit** with a descriptive message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add Vulkan compute pipeline support
   fix: resolve memory leak in DUVAS allocator
   docs: update Metal codegen documentation
   ```
6. **Push** to your fork and open a **Pull Request**.

## Project Structure

| Crate | Description |
|---|---|
| `omni-shim` | CUDA Runtime/Driver API interception layer |
| `omni-core` | MLIR JIT compiler, codegen backends, memory management |
| `omni-net` | P2P network, task scheduling, zero-trust crypto |
| `omni-cli` | Developer-facing CLI toolchain |

## Areas of Contribution

We especially welcome contributions in the following areas:

### Compiler & IR
- New MLIR optimization passes (operator fusion, loop tiling)
- Additional codegen backends (Intel oneAPI, Qualcomm Hexagon)
- CUDA-to-MLIR lifting improvements

### Runtime & Memory
- Virtual Sliding Pager optimizations
- DUVAS allocator performance tuning
- Async DMA pipeline improvements

### Network & Security
- libp2p transport layer optimizations
- NAT traversal improvements
- Zero-knowledge proof integration for verifiable compute

### Testing & Benchmarks
- Integration test coverage
- Real-world workload benchmarks (PyTorch, llama.cpp)
- Fuzz testing for serialization layers

## Code Style

- Follow standard Rust conventions (`rustfmt` defaults)
- All public items must have documentation comments (`///`)
- Use `tracing` for logging (not `println!` or `eprintln!`)
- Error handling: prefer `anyhow::Result` for application code, `thiserror` for library errors
- Unsafe code must include `// SAFETY:` comments explaining invariants

## Reporting Issues

When reporting bugs, please include:

- Rust version (`rustc --version`)
- Operating system and hardware
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output

## License

By contributing to OmniCompute, you agree that your contributions will be licensed
under the [Apache License 2.0](LICENSE).

---

Thank you for helping democratize compute!
