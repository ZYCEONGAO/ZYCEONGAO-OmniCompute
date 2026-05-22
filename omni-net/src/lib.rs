//! # omni-net
//!
//! Decentralized zero-trust P2P compute routing and scheduling network for OmniCompute.
//!
//! Exposes:
//! - **P2P transport swarm** (`p2p`) — NAT-traversed, Kademlia DHT node networking.
//! - **Task Router** (`scheduler`) — Heuristic load balancing and blind graph splitting.
//! - **Zero-Trust Security** (`crypto`) — Homomorphic tensor obfuscation and TEE enclave sandboxes.

pub mod crypto;
pub mod p2p;
pub mod scheduler;

pub use crypto::obfuscator::{ObfuscationMatrix, ControlFlowObfuscator};
pub use crypto::tee::{TeeSandbox, TeeType};
pub use p2p::node::OmniNode;
pub use p2p::protocol::NetworkMessage;
pub use scheduler::router::{TaskRouter, TaskDispatch, WorkerNode};
