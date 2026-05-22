//! # codegen::spirv
//!
//! SPIR-V / Vulkan GLSL code generation backend.
//! Lowers `omni.tensor` IR into Vulkan-compliant GLSL compute shaders
//! designed for compilation to SPIR-V.
//!
//! Compatible with Intel Arc, AMD (via Vulkan driver path), and WebGPU targets.

use crate::codegen::{CodeGenerator, GeneratedKernel};
use crate::mlir::dialect::{OmniModule, OpKind, ElementType};
use crate::hardware::HardwareProfile;
use anyhow::{bail, Result};
use tracing::debug;

/// SPIR-V / Vulkan Compute Shader Generator.
pub struct SpirvCodegen {
    /// Hardware profile of the target device
    _hardware: HardwareProfile,
}

impl SpirvCodegen {
    /// Creates a new SPIR-V code generator.
    pub fn new(hardware: HardwareProfile) -> Self {
        Self { _hardware: hardware }
    }

    /// Maps `ElementType` to GLSL datatype strings.
    fn map_element_type(t: &ElementType) -> &'static str {
        match t {
            ElementType::U8 => "uint",
            ElementType::I8 => "int",
            ElementType::I16 => "int",
            ElementType::I32 => "int",
            ElementType::I64 => "int64_t",
            ElementType::F16 => "float16_t",
            ElementType::BF16 => "float16_t",
            ElementType::F32 => "float",
            ElementType::F64 => "double",
            _ => "float",
        }
    }
}

impl CodeGenerator for SpirvCodegen {
    fn generate(&self, module: &OmniModule) -> Result<GeneratedKernel> {
        debug!("SpirvCodegen: starting SPIR-V GLSL code generation...");

        let mut glsl_code = String::new();
        glsl_code.push_str("#version 450 core\n");
        glsl_code.push_str("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n");
        glsl_code.push_str("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n\n");

        let mut kernel_name = "omni_kernel".to_string();
        let mut block_size = [16, 16, 1]; // Vulkan standard local size 16x16 default

        for op in &module.ops {
            match &op.kind {
                OpKind::MatMul | OpKind::BatchedMatMul => {
                    kernel_name = format!("omni_matmul_{}", op.id);
                    
                    // SPIR-V Tiling parameters
                    let tile_m = op.attrs.get("tile_m")
                        .and_then(|a| if let crate::mlir::dialect::AttrValue::Int(v) = a { Some(*v) } else { None })
                        .unwrap_or(16);
                    let tile_n = op.attrs.get("tile_n")
                        .and_then(|a| if let crate::mlir::dialect::AttrValue::Int(v) = a { Some(*v) } else { None })
                        .unwrap_or(16);
                    let tile_k = op.attrs.get("tile_k")
                        .and_then(|a| if let crate::mlir::dialect::AttrValue::Int(v) = a { Some(*v) } else { None })
                        .unwrap_or(16);

                    let dtype = Self::map_element_type(&op.result_type.element_type);
                    block_size = [16, 16, 1];

                    glsl_code.push_str(&format!(
                        "layout(local_size_x = {local_x}, local_size_y = {local_y}, local_size_z = 1) in;\n\n\
                        layout(std430, binding = 0) readonly buffer BufferA {{ {datatype} A[]; }};\n\
                        layout(std430, binding = 1) readonly buffer BufferB {{ {datatype} B[]; }};\n\
                        layout(std430, binding = 2) writeonly buffer BufferC {{ {datatype} C[]; }};\n\n\
                        layout(push_constant) uniform Params {{\n\
                        \tint M; int N; int K;\n\
                        }} params;\n\n\
                        shared {datatype} sA[{tile_m}][{tile_k}];\n\
                        shared {datatype} sB[{tile_k}][{tile_n}];\n\n\
                        void main() {{\n\
                        \tint tx = int(gl_LocalInvocationID.x);\n\
                        \tint ty = int(gl_LocalInvocationID.y);\n\
                        \tint row = int(gl_WorkGroupID.y) * {tile_m} + ty;\n\
                        \tint col = int(gl_WorkGroupID.x) * {tile_n} + tx;\n\
                        \n\
                        \t{datatype} sum = {datatype}(0.0);\n\
                        \n\
                        \tfor (int ph = 0; ph < (params.K + {tile_k} - 1) / {tile_k}; ++ph) {{\n\
                        \t\tif (row < params.M && ph * {tile_k} + tx < params.K) {{\n\
                        \t\t\tsA[ty][tx] = A[row * params.K + ph * {tile_k} + tx];\n\
                        \t\t}} else {{\n\
                        \t\t\tsA[ty][tx] = {datatype}(0.0);\n\
                        \t\t}}\n\
                        \n\
                        \t\tif (ph * {tile_k} + ty < params.K && col < params.N) {{\n\
                        \t\t\tsB[ty][tx] = B[(ph * {tile_k} + ty) * params.N + col];\n\
                        \t\t}} else {{\n\
                        \t\t\tsB[ty][tx] = {datatype}(0.0);\n\
                        \t\t}}\n\
                        \n\
                        \t\tbarrier();\n\
                        \n\
                        \t\tfor (int k = 0; k < {tile_k}; ++k) {{\n\
                        \t\t\tsum += sA[ty][k] * sB[k][tx];\n\
                        \t\t}}\n\
                        \n\
                        \t\tbarrier();\n\
                        \t}}\n\
                        \n\
                        \tif (row < params.M && col < params.N) {{\n\
                        ",
                        local_x = block_size[0],
                        local_y = block_size[1],
                        datatype = dtype,
                        tile_m = tile_m,
                        tile_n = tile_n,
                        tile_k = tile_k,
                    ));

                    // Fuse post-activation
                    if let Some(crate::mlir::dialect::AttrValue::String(act)) = op.attrs.get("fused_activation") {
                        match act.as_str() {
                            "Relu" => glsl_code.push_str("\t\tsum = max(sum, float16_t(0.0));\n"),
                            "Gelu" => glsl_code.push_str("\t\tsum = sum * float16_t(0.5) * (float16_t(1.0) + erf(sum * float16_t(0.70710678)));\n"),
                            "Silu" => glsl_code.push_str("\t\tsum = sum / (float16_t(1.0) + exp(-sum));\n"),
                            _ => {}
                        }
                    }

                    glsl_code.push_str("\t\tC[row * params.N + col] = sum;\n\t}\n}\n");
                }
                OpKind::Relu => {
                    kernel_name = format!("omni_relu_{}", op.id);
                    let dtype = Self::map_element_type(&op.result_type.element_type);
                    glsl_code.push_str(&format!(
                        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
                        layout(std430, binding = 0) readonly buffer BufferX {{ {datatype} x[]; }};\n\
                        layout(std430, binding = 1) writeonly buffer BufferY {{ {datatype} y[]; }};\n\n\
                        layout(push_constant) uniform Params {{ int n; }} params;\n\n\
                        void main() {{\n\
                        \tint idx = int(gl_GlobalInvocationID.x);\n\
                        \tif (idx < params.n) {{\n\
                        \t\ty[idx] = max(x[idx], {datatype}(0.0));\n\
                        \t}}\n\
                        }}\n",
                        datatype = dtype
                    ));
                    block_size = [256, 1, 1];
                }
                OpKind::Add => {
                    kernel_name = format!("omni_add_{}", op.id);
                    let dtype = Self::map_element_type(&op.result_type.element_type);
                    glsl_code.push_str(&format!(
                        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n\
                        layout(std430, binding = 0) readonly buffer BufferA {{ {datatype} a[]; }};\n\
                        layout(std430, binding = 1) readonly buffer BufferB {{ {datatype} b[]; }};\n\
                        layout(std430, binding = 2) writeonly buffer BufferC {{ {datatype} c[]; }};\n\n\
                        layout(push_constant) uniform Params {{ int n; }} params;\n\n\
                        void main() {{\n\
                        \tint idx = int(gl_GlobalInvocationID.x);\n\
                        \tif (idx < params.n) {{\n\
                        \t\tc[idx] = a[idx] + b[idx];\n\
                        \t}}\n\
                        }}\n",
                        datatype = dtype
                    ));
                    block_size = [256, 1, 1];
                }
                OpKind::Opaque { ptx_hash } => {
                    kernel_name = format!("omni_fallback_{}", op.id);
                    glsl_code.push_str(&format!(
                        "// Fallback opaque block for PTX hash: 0x{:x}\n\
                        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\
                        layout(std430, binding = 0) readonly buffer BufferIn {{ float A[]; }};\n\
                        layout(std430, binding = 1) writeonly buffer BufferOut {{ float B[]; }};\n\
                        layout(push_constant) uniform Params {{ int n; }} params;\n\n\
                        void main() {{\n\
                        \tint idx = int(gl_GlobalInvocationID.x);\n\
                        \tif (idx < params.n) B[idx] = A[idx];\n\
                        }}\n",
                        ptx_hash
                    ));
                    block_size = [256, 1, 1];
                }
                _ => {
                    bail!("SpirvCodegen: operation kind {:?} not yet implemented", op.kind);
                }
            }
        }

        Ok(GeneratedKernel {
            name: kernel_name,
            payload: glsl_code.into_bytes(),
            block_size,
        })
    }
}
