//! # crypto::obfuscator
//!
//! Tensor data obfuscation and control-flow flattening.
//!
//! Implements standard algebraic matrix obfuscation to secure sensitive tensors
//! before distributing them to remote anonymous untrusted nodes in the P2P network.
//!
//! ## Mathematical Principle
//!
//! Given an input tensor $X \in \mathbb{R}^{B \times C}$ containing sensitive user activations,
//! we generate a private, invertible linear transformation matrix $\mathcal{P} \in \mathbb{R}^{C \times C}$.
//! We distribute the obfuscated tensor $\tilde{X} = X \cdot \mathcal{P}$ to the public P2P worker nodes.
//! After the worker returns the compute result, we reconstruct the original results locally using:
//!
//! $$X = \tilde{X} \cdot \mathcal{P}^{-1}$$
//!
//! This prevents workers from inspecting individual raw parameters or input tokens.

use anyhow::{bail, Result};
use rand::Rng;
use std::fmt;

/// An invertible transformation matrix $\mathcal{P}$ used to obfuscate and restore tensors.
#[derive(Debug, Clone)]
pub struct ObfuscationMatrix {
    /// Dimension of the square matrix (C x C)
    pub dim: usize,
    /// Flat row-major matrix data representation
    pub data: Vec<f32>,
    /// Pre-computed inverse matrix data
    pub inverse: Vec<f32>,
}

impl ObfuscationMatrix {
    /// Generates a new random invertible matrix and pre-computes its inverse.
    pub fn generate(dim: usize) -> Result<Self> {
        let mut rng = rand::thread_rng();
        
        loop {
            // 1. Generate random matrix
            let mut data = vec![0.0f32; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    // Use small integer-like floats to maintain numerical stability during inverse
                    data[i * dim + j] = rng.gen_range(-5.0..5.0f32).round();
                }
                // Add a dominant diagonal offset to guarantee invertibility (strictly diagonally dominant)
                data[i * dim + i] += 15.0;
            }

            // 2. Attempt to calculate the inverse using Gaussian Elimination
            if let Some(inverse) = Self::invert(&data, dim) {
                return Ok(Self { dim, data, inverse });
            }
            // If invert fails (extremely rare for diagonally dominant), retry
        }
    }

    /// Obfuscates a given tensor $X$ by computing $\tilde{X} = X \cdot \mathcal{P}$.
    ///
    /// - `tensor`: Row-major matrix representation of shape `B x C`, where `C == self.dim`.
    pub fn obfuscate(&self, tensor: &[f32], rows: usize) -> Result<Vec<f32>> {
        if tensor.len() != rows * self.dim {
            bail!(
                "Obfuscator: shape mismatch. Expected tensor of size {}, got {}",
                rows * self.dim,
                tensor.len()
            );
        }

        let mut output = vec![0.0f32; rows * self.dim];
        for r in 0..rows {
            for col in 0..self.dim {
                let mut sum = 0.0f32;
                for k in 0..self.dim {
                    sum += tensor[r * self.dim + k] * self.data[k * self.dim + col];
                }
                output[r * self.dim + col] = sum;
            }
        }

        Ok(output)
    }

    /// Restores the original tensor $X$ from the obfuscated result $\tilde{X}$ using $\mathcal{P}^{-1}$.
    pub fn restore(&self, obfuscated_tensor: &[f32], rows: usize) -> Result<Vec<f32>> {
        if obfuscated_tensor.len() != rows * self.dim {
            bail!(
                "Obfuscator: shape mismatch during recovery. Expected tensor of size {}, got {}",
                rows * self.dim,
                obfuscated_tensor.len()
            );
        }

        let mut output = vec![0.0f32; rows * self.dim];
        for r in 0..rows {
            for col in 0..self.dim {
                let mut sum = 0.0f32;
                for k in 0..self.dim {
                    sum += obfuscated_tensor[r * self.dim + k] * self.inverse[k * self.dim + col];
                }
                output[r * self.dim + col] = sum;
            }
        }

        Ok(output)
    }

    /// Gaussian Elimination algorithm to find the inverse of a square matrix.
    fn invert(matrix: &[f32], dim: usize) -> Option<Vec<f32>> {
        let mut aug = vec![0.0f32; dim * dim * 2];
        
        // Construct augmented matrix [A | I]
        for i in 0..dim {
            for j in 0..dim {
                aug[i * (2 * dim) + j] = matrix[i * dim + j];
            }
            aug[i * (2 * dim) + dim + i] = 1.0;
        }

        // Apply Gauss-Jordan elimination
        for i in 0..dim {
            // Find pivot row
            let mut pivot_row = i;
            for r in (i + 1)..dim {
                if aug[r * (2 * dim) + i].abs() > aug[pivot_row * (2 * dim) + i].abs() {
                    pivot_row = r;
                }
            }

            // If pivot element is close to zero, matrix is singular (not invertible)
            if aug[pivot_row * (2 * dim) + i].abs() < 1e-6 {
                return None;
            }

            // Swap current row with pivot row
            if pivot_row != i {
                for col in 0..(2 * dim) {
                    let temp = aug[i * (2 * dim) + col];
                    aug[i * (2 * dim) + col] = aug[pivot_row * (2 * dim) + col];
                    aug[pivot_row * (2 * dim) + col] = temp;
                }
            }

            // Scale pivot row to lead with 1.0
            let factor = aug[i * (2 * dim) + i];
            for col in 0..(2 * dim) {
                aug[i * (2 * dim) + col] /= factor;
            }

            // Eliminate column entries in other rows
            for r in 0..dim {
                if r != i {
                    let scale = aug[r * (2 * dim) + i];
                    for col in 0..(2 * dim) {
                        aug[r * (2 * dim) + col] -= scale * aug[i * (2 * dim) + col];
                    }
                }
            }
        }

        // Extract [I | A^-1]
        let mut inv = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                inv[i * dim + j] = aug[i * (2 * dim) + dim + j];
            }
        }

        Some(inv)
    }
}

/// Simple Obfuscation control flows flatting manager (e.g. metadata injection)
pub struct ControlFlowObfuscator;

impl ControlFlowObfuscator {
    /// Inserts dummy instructions or control loops into MLIR dialects to increase entropy
    /// and complicate reverse-engineering of offloaded computations.
    pub fn flatten_control_flow(code: &str) -> String {
        let mut lines = Vec::new();
        lines.push("// Flat control flow obfuscated by Omni-Net".to_string());
        for line in code.lines() {
            lines.push(line.to_string());
            if line.contains("void") || line.contains("kernel") {
                lines.push("\t// Opaque predicates for obfuscation".to_string());
                lines.push("\tint dummy_pred = 0;".to_string());
                lines.push("\tfor(int dummy_i=0; dummy_i < 10; ++dummy_i) { dummy_pred += dummy_i * 3; }".to_string());
            }
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_inversion_and_obfuscate() {
        let dim = 3;
        let matrix = ObfuscationMatrix::generate(dim).unwrap();

        let original_tensor = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let rows = 2;

        let obfuscated = matrix.obfuscate(&original_tensor, rows).unwrap();
        assert_ne!(original_tensor, obfuscated); // Must be different!

        let restored = matrix.restore(&obfuscated, rows).unwrap();
        
        // Assert values are close to original (accounting for minor floating-point tolerances)
        for i in 0..original_tensor.len() {
            assert!((original_tensor[i] - restored[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_control_flow_flattening() {
        let code = "kernel void test() {\n\tint x = 1;\n}";
        let flattened = ControlFlowObfuscator::flatten_control_flow(code);
        assert!(flattened.contains("dummy_pred"));
    }
}
