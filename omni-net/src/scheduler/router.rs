//! # scheduler::router
//!
//! Smart task router and neural network blind splitter.
//!
//! Partitions computational graphs into high-sensitivity local sub-graphs
//! and anonymous, obfuscated mathematical operations dispatchable to global long-tail nodes.
//! Implements elastic load balancing to optimize globally-distributed job scheduling.

use omni_core::mlir::dialect::{OmniModule, OmniOp, OpKind, TensorType};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Metadata profiles of active long-tail worker nodes in the P2P network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    /// Unique peer identity
    pub peer_id: String,
    /// Measured FP16 throughput in TFLOPS
    pub fp16_tflops: f32,
    /// Network latency Round-Trip Time (RTT) in milliseconds
    pub rtt_ms: f32,
    /// Physical memory capacity available on the worker's device
    pub vram_capacity_bytes: u64,
    /// Current CPU / accelerator computational load (percentage)
    pub current_load: f32,
}

/// A dispatched task unit containing code payload and target worker assignments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDispatch {
    /// Target worker node peer ID
    pub target_peer_id: String,
    /// Operation ID to track results
    pub op_id: u64,
    /// The mathematical operation payload
    pub op: OmniOp,
    /// Calculated priority index
    pub priority: f32,
}

/// Task Router governing task splitting and heterogeneous scheduling.
pub struct TaskRouter {
    /// Local machine capacity limits
    _local_rtt_penalty: f32,
}

impl TaskRouter {
    /// Creates a new Task Router.
    pub fn new() -> Self {
        Self { _local_rtt_penalty: 1.2 }
    }

    /// Neural Network Blind Splitting Algorithm.
    ///
    /// Partitions an `OmniModule` graph into:
    /// 1. **Local Set** ($\mathcal{M}_{\text{local}}$): High-sensitivity operations (e.g. Embedding,
    ///    Softmax, Token generation heads) that run strictly on the local secure machine.
    /// 2. **Global Set** ($\mathcal{O}_{\text{global}}$): Standard computation-heavy, privacy-flat math
    ///    layers (e.g. intermediate multi-head attention weights, standard MatMuls) eligible for public distribution.
    pub fn blind_split(&self, module: &OmniModule) -> Result<(Vec<OmniOp>, Vec<OmniOp>)> {
        debug!("TaskRouter: performing blind splitting on {} operations", module.op_count());

        let mut local_ops = Vec::new();
        let mut global_ops = Vec::new();

        for op in &module.ops {
            match &op.kind {
                // High-sensitivity layers holding direct user token data
                OpKind::Gather { .. } | OpKind::Scatter { .. } | OpKind::Softmax { .. } => {
                    debug!("TaskRouter: pinning high-sensitivity op %{} ({:?}) to LOCAL set", op.output, op.kind);
                    local_ops.push(op.clone());
                }
                // Compute-heavy math-only transformations containing no direct token indices
                OpKind::MatMul | OpKind::BatchedMatMul | OpKind::Add | OpKind::Mul | OpKind::UnaryFunc { .. } => {
                    global_ops.push(op.clone());
                }
                // Fallback to local
                _ => {
                    local_ops.push(op.clone());
                }
            }
        }

        info!(
            "TaskRouter: split completed. LocalOps={} (secure), GlobalOps={} (offloadable)",
            local_ops.len(),
            global_ops.len()
        );

        Ok((local_ops, global_ops))
    }

    /// Elastic Load Balancing Algorithm.
    ///
    /// Schedules a set of dispatchable operations to the available global long-tail nodes.
    /// Uses a heuristic scoring function targeting minimum execution latency and maximum throughput:
    ///
    /// $$\text{Score} = \frac{\text{Tflops}}{\text{Load}} \times \frac{1}{\text{Rtt\_ms} + 1.0}$$
    pub fn schedule_global_ops(
        &self,
        global_ops: &[OmniOp],
        workers: &[WorkerNode],
    ) -> Result<Vec<TaskDispatch>> {
        if workers.is_empty() {
            bail!("TaskRouter: no active P2P worker nodes available for scheduling");
        }

        debug!("TaskRouter: scheduling {} offloaded operations", global_ops.len());

        let mut dispatches = Vec::new();

        for (i, op) in global_ops.iter().enumerate() {
            // Find the best worker using the scoring heuristic
            let mut best_worker = &workers[0];
            let mut best_score = -1.0f32;

            for worker in workers {
                // Avoid overloaded workers
                let load_factor = if worker.current_load >= 99.0 { 100.0 } else { 1.0 + worker.current_load / 100.0 };
                
                // Heuristic score: higher Tflops, lower latency, lower load is better
                let score = (worker.fp16_tflops / load_factor) * (1000.0 / (worker.rtt_ms + 1.0));

                if score > best_score {
                    best_score = score;
                    best_worker = worker;
                }
            }

            // Assign priority to track critical-path operations
            let priority = best_score * (1.0 / (i + 1) as f32);

            dispatches.push(TaskDispatch {
                target_peer_id: best_worker.peer_id.clone(),
                op_id: op.id,
                op: op.clone(),
                priority,
            });

            debug!(
                "TaskRouter: scheduled op %{} -> worker '{}' (score={:.2})",
                op.output, best_worker.peer_id, best_score
            );
        }

        Ok(dispatches)
    }
}

impl Default for TaskRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::mlir::dialect::{ElementType, OmniOp, TensorType};
    use std::collections::HashMap;

    #[test]
    fn test_blind_splitting() {
        let mut module = OmniModule::new();
        
        // 1. Gather (Sensitive embedding lookup)
        module.push_op(OmniOp {
            id: 0,
            kind: OpKind::Gather { axis: 0 },
            inputs: vec!["tokens".into(), "weights".into()],
            output: "v0".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![1, 4096], ElementType::F32),
        });

        // 2. MatMul (Privacy-flat layer weight multiplication)
        module.push_op(OmniOp {
            id: 0,
            kind: OpKind::MatMul,
            inputs: vec!["v0".into(), "layer1".into()],
            output: "v1".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![1, 4096], ElementType::F32),
        });

        let router = TaskRouter::new();
        let (local, global) = router.blind_split(&module).unwrap();

        assert_eq!(local.len(), 1);
        assert_eq!(local[0].kind, OpKind::Gather { axis: 0 });

        assert_eq!(global.len(), 1);
        assert_eq!(global[0].kind, OpKind::MatMul);
    }

    #[test]
    fn test_elastic_scheduling() {
        let op = OmniOp {
            id: 12,
            kind: OpKind::MatMul,
            inputs: vec!["a".into(), "b".into()],
            output: "c".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![64, 64], ElementType::F32),
        };

        let worker_a = WorkerNode {
            peer_id: "worker-A-tflops-100".to_string(),
            fp16_tflops: 100.0,
            rtt_ms: 50.0,
            vram_capacity_bytes: 16 * 1024 * 1024 * 1024,
            current_load: 10.0,
        };

        let worker_b = WorkerNode {
            peer_id: "worker-B-tflops-20".to_string(),
            fp16_tflops: 20.0,
            rtt_ms: 10.0,
            vram_capacity_bytes: 8 * 1024 * 1024 * 1024,
            current_load: 5.0,
        };

        let router = TaskRouter::new();
        let dispatches = router.schedule_global_ops(&[op], &[worker_a, worker_b]).unwrap();

        assert_eq!(dispatches.len(), 1);
        // Worker A has way higher Tflops (100 vs 20) which outweighs the RTT difference in score
        assert_eq!(dispatches[0].target_peer_id, "worker-A-tflops-100");
    }
}
