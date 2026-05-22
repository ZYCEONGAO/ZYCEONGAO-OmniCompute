//! # p2p::protocol
//!
//! OmniCompute P2P network message exchange protocol.
//!
//! Defines standard message types for peer-to-peer discovery, status heartbeats,
//! latency probing (Ping/Pong), task dispatch requests, and calculation response payloads.
//! Uses binary serialization (`bincode` / `bytes`) to guarantee minimal serialization overhead.

use crate::scheduler::router::{WorkerNode, TaskDispatch};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Standard binary message payload exchanged between OmniCompute nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    /// Latency Probe Ping containing current Unix timestamp in nanoseconds
    Ping {
        /// Sent timestamp (nanoseconds)
        sent_at_ns: u64,
    },
    /// Latency Probe Pong reflecting the sent timestamp
    Pong {
        /// Original Ping sent timestamp (nanoseconds)
        ping_sent_ns: u64,
        /// Worker receipt timestamp (nanoseconds)
        received_at_ns: u64,
    },
    /// Heartbeat dispatched periodically by workers to report their hardware capacities
    Heartbeat {
        /// Active worker profile
        profile: WorkerNode,
        /// Current uptime in seconds
        uptime_secs: u64,
    },
    /// Dispatch request offloading a mathematical computation task
    TaskRequest {
        /// The task details
        task: TaskDispatch,
        /// Encrypted / obfuscated input parameters buffer
        input_data: Vec<u8>,
    },
    /// Compute result returned back to the coordinating local host node
    TaskResponse {
        /// Matching operation ID
        op_id: u64,
        /// Flag indicating successful remote JIT execution
        success: bool,
        /// Encrypted or raw result tensor data buffer
        output_data: Vec<u8>,
        /// Optional error details if compilation or execution failed
        error_message: Option<String>,
    },
}

impl NetworkMessage {
    /// Encodes a network message into a compact binary byte array.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let bytes = bincode::serialize(self)?;
        Ok(bytes)
    }

    /// Decodes a compact binary byte array back into a `NetworkMessage`.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let message = bincode::deserialize(bytes)?;
        Ok(message)
    }

    /// Helper to get current Unix Epoch timestamp in nanoseconds.
    pub fn current_time_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Calculates RTT latency (Round-Trip Time) in milliseconds given a ping timestamp.
    pub fn calculate_rtt(ping_sent_ns: u64) -> f32 {
        let now = Self::current_time_ns();
        if now <= ping_sent_ns {
            0.0f32
        } else {
            let elapsed_ns = now - ping_sent_ns;
            (elapsed_ns as f64 / 1_000_000.0) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::mlir::dialect::{ElementType, OmniOp, OpKind, TensorType};
    use std::collections::HashMap;

    #[test]
    fn test_protocol_serialization() {
        let op = OmniOp {
            id: 7,
            kind: OpKind::Add,
            inputs: vec!["a".into(), "b".into()],
            output: "c".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![32], ElementType::F32),
        };

        let worker = WorkerNode {
            peer_id: "test-peer".to_string(),
            fp16_tflops: 42.0,
            rtt_ms: 5.0,
            vram_capacity_bytes: 4 * 1024 * 1024 * 1024,
            current_load: 0.0,
        };

        let msg = NetworkMessage::TaskRequest {
            task: TaskDispatch {
                target_peer_id: "test-peer".to_string(),
                op_id: 7,
                op,
                priority: 10.0,
            },
            input_data: vec![1, 2, 3, 4],
        };

        let encoded = msg.encode().unwrap();
        assert!(!encoded.is_empty());

        let decoded = NetworkMessage::decode(&encoded).unwrap();
        if let NetworkMessage::TaskRequest { task, input_data } = decoded {
            assert_eq!(task.op_id, 7);
            assert_eq!(input_data, vec![1, 2, 3, 4]);
        } else {
            panic!("Decoded message kind mismatched!");
        }
    }

    #[test]
    fn test_rtt_calculation() {
        let sent = NetworkMessage::current_time_ns() - 10_000_000; // 10 ms ago
        let rtt = NetworkMessage::calculate_rtt(sent);
        assert!(rtt >= 9.5 && rtt <= 15.0); // should be around 10ms
    }
}
