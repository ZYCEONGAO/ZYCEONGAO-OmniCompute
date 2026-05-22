//! # mlir::passes
//!
//! Optimization passes for the `omni.tensor` MLIR dialect.
//!
//! This module implements key compiler optimization passes that operate on
//! the high-level intermediate representation (`OmniModule`) before lowering
//! to concrete target machine/shading code.
//!
//! ## Optimizations
//!
//! 1. **Operator Fusion** — Fuses memory-bound elementwise operations (like bias add,
//!    activation functions) into preceding compute-bound operations (like MatMul/GEMM)
//!    to eliminate round-trips to global device memory (GDDR/HBM).
//! 2. **Loop Tiling & Vectorization** — Restructures loop dimensions into cache-aligned tiles
//!    (e.g., $16 \times 16$ or $64 \times 64$ sub-blocks) matching the target hardware's
//!    SRAM or Apple Silicon UMA cache properties.
//! 3. **Constant Folding & Dead Code Elimination (DCE)** — Simplifies expressions with compile-time
//!    constants and prunes unused SSA operations to reduce runtime memory and execution overhead.

use crate::mlir::dialect::{OmniModule, OmniOp, OpKind, TensorType, AttrValue};
use crate::hardware::HardwareProfile;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

// ─── Pass Manager ────────────────────────────────────────────────────────────

/// Coordinates the optimization pipeline on `omni.tensor` modules.
pub struct PassManager {
    /// Target hardware profile to guide optimization heuristics
    hardware: HardwareProfile,
}

impl PassManager {
    /// Creates a new optimization pass manager for a given hardware profile.
    pub fn new(hardware: HardwareProfile) -> Self {
        Self { hardware }
    }

    /// Runs all optimization passes on a module in sequence.
    ///
    /// The optimization pipeline order:
    /// 1. Constant Folding
    /// 2. Operator Fusion (critical for memory bandwidth saving)
    /// 3. Loop Tiling & Vectorization (crucial for SRAM utilization)
    /// 4. Dead Code Elimination (DCE)
    pub fn run(&self, module: &mut OmniModule) -> anyhow::Result<()> {
        let op_count_before = module.op_count();
        debug!("PassManager: starting optimizations on {} ops", op_count_before);

        self.run_constant_folding(module)?;
        self.run_operator_fusion(module)?;
        self.run_loop_tiling(module)?;
        self.run_dce(module)?;

        info!(
            "PassManager: optimization completed. Ops: {} -> {} (fused/eliminated {})",
            op_count_before,
            module.op_count(),
            op_count_before as isize - module.op_count() as isize
        );

        Ok(())
    }

    // ─── [Pass 1] Constant Folding ───────────────────────────────────────────

    /// Folds operations whose inputs are fully constant.
    ///
    /// E.g., `Add(const(2.0), const(3.0))` becomes `const(5.0)`.
    fn run_constant_folding(&self, module: &mut OmniModule) -> anyhow::Result<()> {
        debug!("PassManager: running Constant Folding...");
        let mut constants: HashMap<String, f64> = HashMap::new();
        let mut folded_ops = Vec::new();

        // 1. Identify existing constants or foldable patterns
        for op in &module.ops {
            match &op.kind {
                OpKind::Fill { value } => {
                    constants.insert(op.output.clone(), *value);
                    folded_ops.push(op.clone());
                }
                OpKind::Add if op.inputs.iter().all(|i| constants.contains_key(i)) => {
                    let val_a = constants.get(&op.inputs[0]).unwrap();
                    let val_b = constants.get(&op.inputs[1]).unwrap();
                    let sum = val_a + val_b;
                    constants.insert(op.output.clone(), sum);
                    let mut folded = op.clone();
                    folded.kind = OpKind::Fill { value: sum };
                    folded_ops.push(folded);
                }
                OpKind::Mul if op.inputs.iter().all(|i| constants.contains_key(i)) => {
                    let val_a = constants.get(&op.inputs[0]).unwrap();
                    let val_b = constants.get(&op.inputs[1]).unwrap();
                    let prod = val_a * val_b;
                    constants.insert(op.output.clone(), prod);
                    let mut folded = op.clone();
                    folded.kind = OpKind::Fill { value: prod };
                    folded_ops.push(folded);
                }
                _ => {
                    folded_ops.push(op.clone());
                }
            }
        }

        module.ops = folded_ops;
        Ok(())
    }

    // ─── [Pass 2] Operator Fusion ─────────────────────────────────────────────

    /// Fuses elementwise ops into preceding linear algebra ops (e.g., MatMul + BiasAdd -> Gemm).
    ///
    /// ## High-level Concept
    ///
    /// Memory bandwidth is the primary bottleneck in modern deep learning execution.
    /// Fusing operations allows intermediate results to stay in high-speed register files or
    /// local shared memory (L1/SRAM) rather than writing back to VRAM (GDDR/HBM) and loading again.
    fn run_operator_fusion(&self, module: &mut OmniModule) -> anyhow::Result<()> {
        debug!("PassManager: running Operator Fusion...");
        if module.ops.is_empty() {
            return Ok(());
        }

        let mut fused_ops: Vec<OmniOp> = Vec::new();
        let mut skipped_indices = HashSet::new();

        for i in 0..module.ops.len() {
            if skipped_indices.contains(&i) {
                continue;
            }

            let current_op = &module.ops[i];

            // Target pattern: MatMul / BatchedMatMul followed by an elementwise activation/add
            if (current_op.kind == OpKind::MatMul || current_op.kind == OpKind::BatchedMatMul)
                && i + 1 < module.ops.len()
            {
                let next_op = &module.ops[i + 1];

                // If next operation consumes the output of the current MatMul and is a fuse-eligible activation
                if next_op.inputs.contains(&current_op.output)
                    && (next_op.kind == OpKind::Relu
                        || next_op.kind == OpKind::Gelu
                        || next_op.kind == OpKind::Silu
                        || matches!(next_op.kind, OpKind::Add))
                {
                    debug!(
                        "PassManager: fusing {:?} with activation {:?}",
                        current_op.kind, next_op.kind
                    );

                    // Create a fused operation representation
                    let mut fused_op = current_op.clone();
                    
                    // Store the fusion attribute
                    fused_op.attrs.insert(
                        "fused_activation".to_string(),
                        AttrValue::String(format!("{:?}", next_op.kind)),
                    );

                    // If it was an Add (e.g., residual bias add), add the other input to the fused op
                    if matches!(next_op.kind, OpKind::Add) {
                        for input in &next_op.inputs {
                            if input != &current_op.output {
                                fused_op.inputs.push(input.clone());
                            }
                        }
                    }

                    // The output becomes the activation output
                    fused_op.output = next_op.output.clone();
                    fused_op.result_type = next_op.result_type.clone();

                    fused_ops.push(fused_op);
                    skipped_indices.insert(i + 1); // skip activation op since it's fused
                    continue;
                }
            }

            fused_ops.push(current_op.clone());
        }

        module.ops = fused_ops;
        Ok(())
    }

    // ─── [Pass 3] Loop Tiling & Vectorization ─────────────────────────────────

    /// Rewrites kernel loop layouts into cache-aligned tiles matching host hardware.
    ///
    /// Guides the codegen layer on grid/block sizes or wavefront vectorization lanes.
    fn run_loop_tiling(&self, module: &mut OmniModule) -> anyhow::Result<()> {
        debug!(
            "PassManager: optimizing tiling parameters for backend {:?}",
            self.hardware.target_backend
        );

        // Adjust default tile parameters based on hardware registers / caches
        let tile_size = match self.hardware.target_backend {
            crate::hardware::TargetBackend::AmdRocm { .. } => 64, // Optimal for Wave64 AMD architectures
            crate::hardware::TargetBackend::AppleMetal { .. } => 32,  // Optimal for Apple M-series threadgroup sizes
            crate::hardware::TargetBackend::VulkanGeneric { .. } => 16,  // Standard safe subgroup size
            crate::hardware::TargetBackend::CpuVectorized { .. } | crate::hardware::TargetBackend::RemoteP2p { .. } => 8,      // Fits vector registers (AVX2/AVX-512)
        };

        for op in &mut module.ops {
            if op.kind == OpKind::MatMul || op.kind == OpKind::BatchedMatMul {
                // Annotate the operation with loop tiling configuration
                op.attrs.insert("tile_m".into(), AttrValue::Int(tile_size));
                op.attrs.insert("tile_n".into(), AttrValue::Int(tile_size));
                op.attrs.insert("tile_k".into(), AttrValue::Int(16)); // typically smaller tile dimension on inner loop
                
                // Vectorization factor (e.g. read 4 elements / 128 bits at once for memory efficiency)
                op.attrs.insert("vector_width".into(), AttrValue::Int(4));
                debug!(
                    "PassManager: tiled MatMul {} with tile size {}",
                    op.output, tile_size
                );
            }
        }

        Ok(())
    }

    // ─── [Pass 4] Dead Code Elimination (DCE) ─────────────────────────────────

    /// Removes operations whose outputs are never read in the module.
    fn run_dce(&self, module: &mut OmniModule) -> anyhow::Result<()> {
        debug!("PassManager: running Dead Code Elimination...");
        
        let mut active_ops = Vec::new();
        let mut referenced_values = HashSet::new();

        // 1. Mark: Sweep from bottom to find all used variables
        for op in module.ops.iter().rev() {
            // Let's assume the final output is always considered active.
            // In a real JIT compilation flow, the last output node or any output designated as
            // a function return value is the root of the dependency chain.
            let is_root = op.id == (module.op_count() - 1) as u64;

            if is_root || referenced_values.contains(&op.output) {
                // If this operation's output is required, all of its input values are now required
                for input in &op.inputs {
                    referenced_values.insert(input.clone());
                }
                active_ops.push(op.clone());
            } else {
                debug!("PassManager: pruning dead operation %{} = {:?}", op.output, op.kind);
            }
        }

        // Reverse back to maintain topological order
        active_ops.reverse();
        module.ops = active_ops;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::dialect::{ElementType, OmniOp, TensorType};
    use crate::hardware::TargetBackend;

    fn create_mock_hardware() -> HardwareProfile {
        HardwareProfile {
            target_backend: TargetBackend::AmdRocm {
                gfx_arch: "gfx1100".to_string(),
                wavefront_size: 32,
                compute_units: 96,
            },
            vram_bytes: 8 * 1024 * 1024 * 1024,
            l2_cache_bytes: 4 * 1024 * 1024,
            fp16_tflops: 123.0,
            memory_bandwidth_gbps: 960.0,
            atomic_support: true,
            tensor_core_support: true,
            os: crate::hardware::OperatingSystem::Windows,
        }
    }

    #[test]
    fn test_constant_folding() {
        let mut module = OmniModule::new();
        
        let const_op = OmniOp {
            id: 0,
            kind: OpKind::Fill { value: 2.5 },
            inputs: vec![],
            output: "v0".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![1], ElementType::F32),
        };
        module.push_op(const_op);

        let add_op = OmniOp {
            id: 0,
            kind: OpKind::Add,
            inputs: vec!["v0".into(), "v0".into()],
            output: "v1".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![1], ElementType::F32),
        };
        module.push_op(add_op);

        let pm = PassManager::new(create_mock_hardware());
        pm.run_constant_folding(&mut module).unwrap();

        // The second op should be folded into a Fill of 5.0
        assert_eq!(module.ops.len(), 2);
        assert_eq!(module.ops[1].kind, OpKind::Fill { value: 5.0 });
    }

    #[test]
    fn test_operator_fusion() {
        let mut module = OmniModule::new();

        let matmul = OmniOp {
            id: 0,
            kind: OpKind::MatMul,
            inputs: vec!["a".into(), "b".into()],
            output: "c".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![128, 128], ElementType::F32),
        };
        module.push_op(matmul);

        let relu = OmniOp {
            id: 0,
            kind: OpKind::Relu,
            inputs: vec!["c".into()],
            output: "d".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![128, 128], ElementType::F32),
        };
        module.push_op(relu);

        let pm = PassManager::new(create_mock_hardware());
        pm.run_operator_fusion(&mut module).unwrap();

        // The activation op should be fused, reducing op count to 1
        assert_eq!(module.ops.len(), 1);
        assert_eq!(module.ops[0].output, "d");
        assert!(module.ops[0].attrs.contains_key("fused_activation"));
    }
}
