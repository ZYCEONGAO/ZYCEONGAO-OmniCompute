//! # mlir
//!
//! High-level intermediate representation system based on MLIR concepts.
//! Includes the `omni.tensor` dialect definition and optimization passes.

pub mod dialect;
pub mod passes;

pub use dialect::{ElementType, IrLifter, OmniModule, OmniOp, OpKind, TensorType};
pub use passes::PassManager;
