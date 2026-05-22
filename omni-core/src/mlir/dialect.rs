//! # mlir::dialect
//!
//! Definition of the `omni.tensor` MLIR dialect — OmniCompute's central
//! hardware-agnostic intermediate representation.
//!
//! ## Design Rationale
//!
//! Traditional binary-level translation (QEMU, ZLUDA) operates on machine code,
//! incurring 30-50% overhead from decode-re-encode cycles and losing high-level
//! semantic information needed for optimization.
//!
//! OmniCompute instead intercepts at the **IR layer**: CUDA PTX / Triton IR is
//! reverse-lifted into `omni.tensor` — a high-level algebraic dialect that
//! expresses compute in terms of tensor shapes, element types, and mathematical
//! operations, independent of any specific hardware instruction set.
//!
//! ## Dialect Hierarchy
//!
//! ```text
//! [High Level]
//!   omni.tensor   -- hardware-agnostic tensor algebra (this module)
//!       |
//!       v  (lowering passes in passes.rs)
//!   omni.vector   -- explicit SIMD / wavefront vectorization
//!       |
//!       v
//!   omni.hardware -- hardware-specific hints (wavefront size, SRAM layout)
//!       |
//!       v
//!   Target IR     -- AMDGCN / Metal MSL / SPIR-V / AVX-512 intrinsics
//! [Low Level]
//! ```

use anyhow::{bail, Result};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;
use tracing::debug;

// ─── Type System ─────────────────────────────────────────────────────────────

/// Supported element types in the `omni.tensor` dialect.
///
/// Mirrors the type system of modern ML frameworks, with explicit width
/// encoding to enable optimal hardware mapping (e.g., BF16 on TPU, FP16
/// on GPU tensor cores).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    /// 8-bit unsigned integer (quantized activations)
    U8,
    /// 8-bit integer (quantized weights, INT8 inference)
    I8,
    /// 16-bit integer
    I16,
    /// 32-bit integer (attention masks, indices)
    I32,
    /// 64-bit integer
    I64,
    /// 16-bit IEEE 754 float (half precision)
    F16,
    /// 16-bit Brain Float (Google, preferred for training)
    BF16,
    /// 32-bit IEEE 754 float (full precision)
    F32,
    /// 64-bit IEEE 754 float (double precision)
    F64,
    /// 8-bit float (FP8 for H100/MI300X)
    F8E4M3,
    /// Boolean (masks, attention patterns)
    Bool,
    /// Opaque / unknown — used during lifting before type inference
    Opaque,
}

impl ElementType {
    /// Returns the size of one element in bytes.
    pub fn byte_width(&self) -> usize {
        match self {
            ElementType::U8 | ElementType::I8 | ElementType::Bool | ElementType::F8E4M3 => 1,
            ElementType::I16 | ElementType::F16 | ElementType::BF16 => 2,
            ElementType::I32 | ElementType::F32 => 4,
            ElementType::I64 | ElementType::F64 => 8,
            ElementType::Opaque => 0,
        }
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            ElementType::F8E4M3
            | ElementType::F16
            | ElementType::BF16
            | ElementType::F32
            | ElementType::F64
        )
    }

    /// Returns the MLIR type string for debug output.
    pub fn mlir_name(&self) -> &'static str {
        match self {
            ElementType::U8     => "ui8",
            ElementType::I8     => "i8",
            ElementType::I16    => "i16",
            ElementType::I32    => "i32",
            ElementType::I64    => "i64",
            ElementType::F16    => "f16",
            ElementType::BF16   => "bf16",
            ElementType::F32    => "f32",
            ElementType::F64    => "f64",
            ElementType::F8E4M3 => "f8E4M3FN",
            ElementType::Bool   => "i1",
            ElementType::Opaque => "opaque",
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mlir_name())
    }
}

/// A ranked tensor type: an n-dimensional array with known element type.
///
/// Dimensions may be static (known at compile time) or dynamic
/// (represented as `None`, inferred at JIT compilation time).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    /// Tensor dimensions. `None` means dynamic/unknown at this position.
    pub shape: Vec<Option<i64>>,
    /// Element scalar type
    pub element_type: ElementType,
    /// Memory layout encoding
    pub encoding: MemoryEncoding,
}

impl TensorType {
    /// Creates a new static tensor type (all dimensions known).
    pub fn static_tensor(shape: Vec<i64>, element_type: ElementType) -> Self {
        Self {
            shape: shape.into_iter().map(Some).collect(),
            element_type,
            encoding: MemoryEncoding::RowMajor,
        }
    }

    /// Creates a new dynamic tensor type (all dimensions unknown).
    pub fn dynamic(rank: usize, element_type: ElementType) -> Self {
        Self {
            shape: vec![None; rank],
            element_type,
            encoding: MemoryEncoding::RowMajor,
        }
    }

    /// Returns the tensor rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements if all dimensions are static.
    pub fn num_elements(&self) -> Option<i64> {
        self.shape.iter().try_fold(1i64, |acc, &dim| {
            dim.map(|d| acc * d)
        })
    }

    /// Returns the total byte size if all dimensions are static.
    pub fn byte_size(&self) -> Option<usize> {
        self.num_elements()
            .map(|n| n as usize * self.element_type.byte_width())
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor<")?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i > 0 { write!(f, "x")?; }
            match dim {
                Some(d) => write!(f, "{}", d)?,
                None    => write!(f, "?")?,
            }
        }
        write!(f, "x{}>", self.element_type)
    }
}

/// Memory layout encoding for tensor storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryEncoding {
    /// Row-major (C-order) — default for PyTorch, CUDA
    RowMajor,
    /// Column-major (Fortran-order) — used by some BLAS libraries
    ColumnMajor,
    /// Custom tiled layout (for hardware-specific SRAM mapping)
    Tiled { tile_shape: Vec<i64> },
    /// Compressed sparse (CSR/CSC for sparse attention)
    Sparse,
}

// ─── Operations ───────────────────────────────────────────────────────────────

/// An `omni.tensor` operation — the fundamental unit of computation.
///
/// Each operation represents a mathematical transformation on tensors.
/// The JIT engine processes sequences of operations to build a compute graph
/// that is then optimized and lowered to hardware machine code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniOp {
    /// Unique operation ID within the current module
    pub id: u64,
    /// Operation kind
    pub kind: OpKind,
    /// Input tensor SSA value names
    pub inputs: Vec<String>,
    /// Output tensor SSA value name
    pub output: String,
    /// Static attributes (e.g., kernel size, stride, padding)
    pub attrs: HashMap<String, AttrValue>,
    /// Inferred output tensor type
    pub result_type: TensorType,
}

/// All supported `omni.tensor` operation kinds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpKind {
    // ── Linear Algebra ─────────────────────────────────────────────────────
    /// General matrix multiplication: C = alpha * A @ B + beta * C
    /// Maps to: cuBLAS sgemm / rocBLAS / Metal MPSMatrixMultiplication
    MatMul,
    /// Batched matrix multiplication
    BatchedMatMul,
    /// Matrix-vector product
    Gemv,

    // ── Convolution ────────────────────────────────────────────────────────
    /// N-D convolution (spatial, temporal, volumetric)
    Conv { num_spatial_dims: usize },
    /// Transposed / deconvolution
    ConvTranspose { num_spatial_dims: usize },
    /// Depthwise convolution (for MobileNet-style models)
    DepthwiseConv,

    // ── Attention ──────────────────────────────────────────────────────────
    /// Scaled dot-product attention (FlashAttention-compatible)
    /// Q, K, V -> softmax(Q @ K^T / sqrt(d_k)) @ V
    ScaledDotProductAttention { causal: bool },
    /// Multi-head attention
    MultiHeadAttention { num_heads: u32 },

    // ── Elementwise ────────────────────────────────────────────────────────
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Element-wise subtraction
    Sub,
    /// Generic element-wise unary function
    UnaryFunc { func: UnaryFunc },
    /// Generic element-wise binary function
    BinaryFunc { func: BinaryFunc },

    // ── Activation Functions ───────────────────────────────────────────────
    /// Rectified Linear Unit
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Gaussian Error Linear Unit
    Gelu,
    /// SiLU / Swish activation (used in LLaMA)
    Silu,
    /// Softmax over a specified axis
    Softmax { axis: i32 },

    // ── Normalization ──────────────────────────────────────────────────────
    /// Layer Normalization
    LayerNorm { eps: f64 },
    /// Group Normalization
    GroupNorm { num_groups: u32, eps: f64 },
    /// RMS Normalization (used in LLaMA)
    RmsNorm { eps: f64 },

    // ── Reduction ──────────────────────────────────────────────────────────
    /// Sum reduction over specified axes
    ReduceSum { axes: Vec<i32>, keep_dims: bool },
    /// Mean reduction
    ReduceMean { axes: Vec<i32>, keep_dims: bool },
    /// Max reduction
    ReduceMax { axes: Vec<i32>, keep_dims: bool },

    // ── Shape Manipulation ─────────────────────────────────────────────────
    /// Reshape without copying data
    Reshape,
    /// Transpose axes
    Transpose { perm: Vec<u32> },
    /// Concatenate tensors along an axis
    Concat { axis: i32 },
    /// Split tensor along an axis
    Split { axis: i32, split_sizes: Vec<i64> },
    /// Broadcast to a target shape
    Broadcast,
    /// Gather / embedding lookup
    Gather { axis: i32 },
    /// Scatter / embedding update
    Scatter { axis: i32 },

    // ── Memory ────────────────────────────────────────────────────────────
    /// Explicit memory copy (may trigger pager)
    MemCopy,
    /// Fill tensor with a constant
    Fill { value: f64 },

    // ── Custom / Opaque ───────────────────────────────────────────────────
    /// Unknown / unlifted operation — requires PTX-level fallback
    Opaque { ptx_hash: u64 },
}

/// Unary mathematical functions for `UnaryFunc` ops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryFunc {
    Exp, Log, Sqrt, Rsqrt, Abs, Neg, Ceil, Floor, Round,
    Cos, Sin, Tan, Erf,
}

/// Binary mathematical functions for `BinaryFunc` ops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryFunc {
    Pow, Max, Min, Mod, Atan2,
}

/// Attribute value types for operation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    IntList(Vec<i64>),
}

// ─── Module ───────────────────────────────────────────────────────────────────

/// An `omni.tensor` module — a sequence of operations forming a compute graph.
///
/// Produced by the IR lifter from captured PTX/Triton IR, and consumed by
/// the optimization passes and codegen backends.
#[derive(Debug, Default)]
pub struct OmniModule {
    /// Ordered list of operations (topologically sorted)
    pub ops: Vec<OmniOp>,
    /// SSA value type map: value_name -> TensorType
    pub value_types: HashMap<String, TensorType>,
    /// Module-level attributes (e.g., target_backend hint)
    pub attrs: HashMap<String, AttrValue>,
    /// Operation ID counter
    next_id: u64,
}

impl OmniModule {
    /// Creates an empty `omni.tensor` module.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an operation to the module.
    pub fn push_op(&mut self, mut op: OmniOp) -> u64 {
        let id = self.next_id;
        op.id = id;
        self.next_id += 1;

        // Register result type in SSA table
        self.value_types.insert(op.output.clone(), op.result_type.clone());
        self.ops.push(op);
        id
    }

    /// Returns the number of operations in this module.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Looks up the tensor type of an SSA value by name.
    pub fn value_type(&self, name: &str) -> Option<&TensorType> {
        self.value_types.get(name)
    }

    /// Prints a human-readable representation of the module for debugging.
    pub fn dump(&self) -> String {
        let mut out = String::from("module omni.tensor {\n");
        for op in &self.ops {
            out.push_str(&format!(
                "  %{} = omni.{:?}({}) : {}\n",
                op.output,
                op.kind,
                op.inputs.join(", "),
                op.result_type,
            ));
        }
        out.push('}');
        out
    }
}

// ─── IR Lifter ────────────────────────────────────────────────────────────────

/// Lifts a captured CUDA kernel into an `omni.tensor` module.
///
/// This is the first stage of the JIT pipeline. The lifter analyzes the
/// binary structure of intercepted PTX / Triton IR and reconstructs
/// the high-level tensor operation semantics.
pub struct IrLifter {
    /// Counter for generating fresh SSA value names
    value_counter: u64,
}

impl IrLifter {
    /// Creates a new IR lifter.
    pub fn new() -> Self {
        Self { value_counter: 0 }
    }

    /// Generates a fresh SSA value name.
    fn fresh_value(&mut self) -> String {
        let name = format!("v{}", self.value_counter);
        self.value_counter += 1;
        name
    }

    /// Lifts raw PTX bytes into an `omni.tensor` module.
    ///
    /// # Algorithm
    ///
    /// 1. Parse PTX text to extract kernel signature (parameter types, shapes)
    /// 2. Identify operation patterns via signature matching:
    ///    - Tensor shapes + GEMM-pattern → `OpKind::MatMul`
    ///    - Softmax pattern → `OpKind::Softmax`
    ///    - etc.
    /// 3. Build `OmniOp` nodes and link them via SSA values
    ///
    /// For PTX patterns that cannot be matched, creates `OpKind::Opaque`
    /// nodes with the original PTX hash for direct binary fallback.
    pub fn lift_ptx(&mut self, ptx_bytes: &[u8], kernel_id: u64) -> Result<OmniModule> {
        let mut module = OmniModule::new();

        // Try to parse as PTX text
        let ptx_str = std::str::from_utf8(ptx_bytes)
            .unwrap_or("<binary cubin>");

        debug!(
            "IrLifter: lifting kernel 0x{:x}, {} bytes of PTX",
            kernel_id,
            ptx_bytes.len()
        );

        // Production: full PTX parser with pattern matching
        // MVP: Detect operation class from kernel name / metadata heuristics
        let op_kind = if ptx_str.contains("mma.sync") || ptx_str.contains("wmma") {
            OpKind::MatMul
        } else if ptx_str.contains("ex2.approx") && ptx_str.contains("div.approx") {
            OpKind::Softmax { axis: -1 }
        } else if ptx_str.contains("tanh.approx") || ptx_str.contains("ex2.approx") {
            OpKind::UnaryFunc { func: UnaryFunc::Exp }
        } else {
            // Unknown pattern — create opaque node for PTX-level fallback
            OpKind::Opaque {
                ptx_hash: Self::hash_bytes(ptx_bytes),
            }
        };

        let input_a = self.fresh_value();
        let input_b = self.fresh_value();
        let output  = self.fresh_value();

        let op = OmniOp {
            id: 0, // assigned by push_op
            kind: op_kind,
            inputs: vec![input_a.clone(), input_b.clone()],
            output: output.clone(),
            attrs: HashMap::new(),
            result_type: TensorType::dynamic(2, ElementType::F16),
        };

        module.value_types.insert(input_a, TensorType::dynamic(2, ElementType::F16));
        module.value_types.insert(input_b, TensorType::dynamic(2, ElementType::F16));
        module.push_op(op);

        debug!("IrLifter: lifted {} ops from kernel 0x{:x}", module.op_count(), kernel_id);
        Ok(module)
    }

    /// Simple polynomial hash for PTX bytes (used to identify opaque kernels).
    fn hash_bytes(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325u64, |acc, &b| {
            acc.wrapping_mul(0x00000100000001b3).wrapping_add(b as u64)
        })
    }
}

impl Default for IrLifter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_type_widths() {
        assert_eq!(ElementType::F32.byte_width(), 4);
        assert_eq!(ElementType::F16.byte_width(), 2);
        assert_eq!(ElementType::BF16.byte_width(), 2);
        assert_eq!(ElementType::I8.byte_width(), 1);
        assert_eq!(ElementType::F64.byte_width(), 8);
    }

    #[test]
    fn test_tensor_type_display() {
        let t = TensorType::static_tensor(vec![128, 256], ElementType::F16);
        assert_eq!(format!("{}", t), "tensor<128x256xf16>");
    }

    #[test]
    fn test_dynamic_tensor_display() {
        let t = TensorType::dynamic(3, ElementType::BF16);
        assert_eq!(format!("{}", t), "tensor<?x?x?xbf16>");
    }

    #[test]
    fn test_tensor_byte_size() {
        let t = TensorType::static_tensor(vec![4, 4], ElementType::F32);
        assert_eq!(t.byte_size(), Some(4 * 4 * 4));
    }

    #[test]
    fn test_module_push_and_count() {
        let mut module = OmniModule::new();
        assert_eq!(module.op_count(), 0);

        let op = OmniOp {
            id: 0,
            kind: OpKind::MatMul,
            inputs: vec!["a".into(), "b".into()],
            output: "c".into(),
            attrs: HashMap::new(),
            result_type: TensorType::static_tensor(vec![64, 64], ElementType::F32),
        };
        module.push_op(op);
        assert_eq!(module.op_count(), 1);
        assert!(module.value_type("c").is_some());
    }

    #[test]
    fn test_ir_lifter_creates_module() {
        let mut lifter = IrLifter::new();
        let fake_ptx = b".version 8.0\n.target sm_90\n";
        let module = lifter.lift_ptx(fake_ptx, 0xDEAD).unwrap();
        assert!(module.op_count() >= 1);
    }
}
