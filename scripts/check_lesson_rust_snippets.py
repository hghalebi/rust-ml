#!/usr/bin/env python3
"""Compile-check Rust snippets embedded in authored lesson Markdown files."""

from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def common_f64() -> str:
    return textwrap.dedent(
        """
        #![allow(dead_code, unused_variables, unused_mut, unused_assignments, unused_imports)]
        type Vector = Vec<f64>;

        fn dot(a: &[f64], b: &[f64]) -> f64 {
            let mut sum = 0.0;
            for i in 0..a.len() {
                sum += a[i] * b[i];
            }
            sum
        }

        fn mat_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
            let rows = matrix.len();
            let mut result = vec![0.0; rows];
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..vector.len() {
                    sum += matrix[r][c] * vector[c];
                }
                result[r] = sum;
            }
            result
        }

        fn relu_vec(x: &[f64]) -> Vec<f64> {
            x.iter().map(|v| v.max(0.0)).collect()
        }

        fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
            a.iter().zip(b).map(|(x, y)| x + y).collect()
        }

        struct SelfAttention;
        impl SelfAttention {
            fn forward(&self, x: &[Vector]) -> Vec<Vector> {
                x.to_vec()
            }
        }

        struct FeedForward {
            w1: Vec<Vec<f64>>,
            w2: Vec<Vec<f64>>,
        }
        impl FeedForward {
            fn forward(&self, x: &[Vector]) -> Vec<Vector> {
                x.to_vec()
            }
        }
        """
    )


def prelude_vector() -> str:
    return textwrap.dedent(
        """
        #![allow(dead_code, unused_variables, unused_mut, unused_assignments, unused_imports)]

        #[derive(Debug, Clone)]
        pub struct Vector {
            data: Vec<f32>,
        }

        impl Vector {
            pub fn new(data: Vec<f32>) -> Self {
                Self { data }
            }

            pub fn len(&self) -> usize {
                self.data.len()
            }

            pub fn dot(&self, other: &Vector) -> f32 {
                assert_eq!(self.len(), other.len(), "dot: dimension mismatch");
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            }

            pub fn add(&self, other: &Vector) -> Vector {
                assert_eq!(self.len(), other.len(), "add: dimension mismatch");
                Vector::new(
                    self.data
                        .iter()
                        .zip(other.data.iter())
                        .map(|(a, b)| a + b)
                        .collect(),
                )
            }

            pub fn map<F>(&self, f: F) -> Vector
            where
                F: Fn(f32) -> f32,
            {
                Vector::new(self.data.iter().copied().map(f).collect())
            }

            pub fn as_slice(&self) -> &[f32] {
                &self.data
            }
        }
        """
    )


def prelude_vector_matrix() -> str:
    return prelude_vector() + textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        pub struct Matrix {
            rows: usize,
            cols: usize,
            data: Vec<f32>,
        }

        impl Matrix {
            pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
                assert_eq!(rows * cols, data.len(), "matrix: invalid data length");
                Self { rows, cols, data }
            }

            pub fn zeros(rows: usize, cols: usize) -> Self {
                Self {
                    rows,
                    cols,
                    data: vec![0.0; rows * cols],
                }
            }

            pub fn get(&self, r: usize, c: usize) -> f32 {
                self.data[r * self.cols + c]
            }

            pub fn set(&mut self, r: usize, c: usize, value: f32) {
                self.data[r * self.cols + c] = value;
            }

            pub fn mul_vec(&self, x: &Vector) -> Vector {
                assert_eq!(self.cols, x.len(), "mul_vec: dimension mismatch");
                let mut out = vec![0.0; self.rows];
                for (r, slot) in out.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for c in 0..self.cols {
                        sum += self.get(r, c) * x.as_slice()[c];
                    }
                    *slot = sum;
                }
                Vector::new(out)
            }
        }
        """
    )


def prelude_linear() -> str:
    return prelude_vector_matrix() + textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        pub struct Linear {
            weight: Matrix,
            bias: Vector,
        }

        impl Linear {
            pub fn new(weight: Matrix, bias: Vector) -> Self {
                Self { weight, bias }
            }

            pub fn forward(&self, x: &Vector) -> Vector {
                self.weight.mul_vec(x).add(&self.bias)
            }
        }
        """
    )


def prelude_sequence() -> str:
    return prelude_vector() + textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        pub struct Sequence {
            tokens: Vec<Vector>,
        }

        impl Sequence {
            pub fn new(tokens: Vec<Vector>) -> Self {
                assert!(!tokens.is_empty(), "sequence cannot be empty");
                let d = tokens[0].len();
                assert!(tokens.iter().all(|t| t.len() == d));
                Self { tokens }
            }

            pub fn len(&self) -> usize {
                self.tokens.len()
            }

            pub fn tokens(&self) -> &[Vector] {
                &self.tokens
            }
        }
        """
    )


def prelude_attention() -> str:
    return prelude_linear() + textwrap.dedent(
        """
        pub fn softmax(xs: &[f32]) -> Vec<f32> {
            let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.into_iter().map(|e| e / sum).collect()
        }

        #[derive(Debug, Clone)]
        pub struct Sequence {
            tokens: Vec<Vector>,
        }

        impl Sequence {
            pub fn new(tokens: Vec<Vector>) -> Self {
                assert!(!tokens.is_empty(), "sequence cannot be empty");
                let d = tokens[0].len();
                assert!(tokens.iter().all(|t| t.len() == d));
                Self { tokens }
            }

            pub fn len(&self) -> usize {
                self.tokens.len()
            }

            pub fn tokens(&self) -> &[Vector] {
                &self.tokens
            }
        }
        """
    )


def prelude_linear_attention() -> str:
    return prelude_attention() + textwrap.dedent(
        """
        pub fn phi(v: &Vector) -> Vector {
            let eps = 1e-6;
            v.map(|x| x.max(0.0) + eps)
        }
        """
    )


def prelude_transformer() -> str:
    return prelude_linear_attention() + textwrap.dedent(
        """
        pub fn relu(v: &Vector) -> Vector {
            v.map(|x| x.max(0.0))
        }

        pub fn layer_norm(x: &Vector) -> Vector {
            let n = x.len() as f32;
            let mean: f32 = x.as_slice().iter().sum::<f32>() / n;
            let var: f32 = x
                .as_slice()
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / n;

            let eps = 1e-5;
            Vector::new(
                x.as_slice()
                    .iter()
                    .map(|v| (v - mean) / (var + eps).sqrt())
                    .collect(),
            )
        }

        #[derive(Debug, Clone)]
        pub struct FeedForward {
            l1: Linear,
            l2: Linear,
        }

        impl FeedForward {
            pub fn new(l1: Linear, l2: Linear) -> Self {
                Self { l1, l2 }
            }

            pub fn forward(&self, x: &Vector) -> Vector {
                let h = relu(&self.l1.forward(x));
                self.l2.forward(&h)
            }
        }

        #[derive(Debug, Clone)]
        pub struct LinearAttention {
            w_q: Linear,
            w_k: Linear,
            w_v: Linear,
        }

        impl LinearAttention {
            pub fn new(w_q: Linear, w_k: Linear, w_v: Linear) -> Self {
                Self { w_q, w_k, w_v }
            }

            pub fn forward(&self, seq: &Sequence) -> Sequence {
                seq.clone()
            }
        }
        """
    )


def chunk_parts() -> dict[str, str]:
    allow = "#![allow(dead_code, unused_variables, unused_mut, unused_imports)]\n"
    parts: dict[str, str] = {"allow": allow}
    parts["vector"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Vector {
            data: Vec<f32>,
        }

        impl Vector {
            fn new(data: Vec<f32>) -> Self {
                Self { data }
            }

            fn len(&self) -> usize {
                self.data.len()
            }

            fn dot(&self, other: &Vector) -> f32 {
                assert_eq!(self.len(), other.len());
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            }
        }
        """
    )
    parts["vector_no_dot"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Vector {
            data: Vec<f32>,
        }

        impl Vector {
            fn new(data: Vec<f32>) -> Self {
                Self { data }
            }

            fn len(&self) -> usize {
                self.data.len()
            }
        }
        """
    )
    parts["matrix"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Matrix {
            rows: usize,
            cols: usize,
            data: Vec<f32>,
        }

        impl Matrix {
            fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
                assert_eq!(rows * cols, data.len());
                Self { rows, cols, data }
            }

            fn get(&self, r: usize, c: usize) -> f32 {
                self.data[r * self.cols + c]
            }

            fn mul_vec(&self, x: &Vector) -> Vector {
                assert_eq!(self.cols, x.len());
                let mut out = vec![0.0; self.rows];
                for (r, slot) in out.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for c in 0..self.cols {
                        sum += self.get(r, c) * x.data[c];
                    }
                    *slot = sum;
                }
                Vector::new(out)
            }
        }
        """
    )
    parts["matrix_no_mul"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Matrix {
            rows: usize,
            cols: usize,
            data: Vec<f32>,
        }

        impl Matrix {
            fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
                assert_eq!(rows * cols, data.len());
                Self { rows, cols, data }
            }

            fn get(&self, r: usize, c: usize) -> f32 {
                self.data[r * self.cols + c]
            }
        }
        """
    )
    parts["linear"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Linear {
            weight: Matrix,
            bias: Vector,
        }

        impl Linear {
            fn new(weight: Matrix, bias: Vector) -> Self {
                assert_eq!(weight.rows, bias.len());
                Self { weight, bias }
            }

            fn forward(&self, x: &Vector) -> Vector {
                let wx = self.weight.mul_vec(x);
                let data = wx
                    .data
                    .iter()
                    .zip(self.bias.data.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                Vector::new(data)
            }
        }
        """
    )
    parts["relu"] = textwrap.dedent(
        """
        fn relu(x: &Vector) -> Vector {
            Vector::new(x.data.iter().map(|v| v.max(0.0)).collect())
        }
        """
    )
    parts["sequence"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Sequence {
            tokens: Vec<Vector>,
        }

        impl Sequence {
            fn new(tokens: Vec<Vector>) -> Self {
                assert!(!tokens.is_empty());
                let dim = tokens[0].len();
                for token in &tokens {
                    assert_eq!(token.len(), dim);
                }
                Self { tokens }
            }

            fn len(&self) -> usize {
                self.tokens.len()
            }

            fn add(&self, other: &Sequence) -> Sequence {
                assert_eq!(self.len(), other.len());
                Sequence::new(
                    self.tokens
                        .iter()
                        .zip(other.tokens.iter())
                        .map(|(left, right)| Vector::new(
                            left.data
                                .iter()
                                .zip(right.data.iter())
                                .map(|(a, b)| a + b)
                                .collect(),
                        ))
                        .collect(),
                )
            }
        }
        """
    )
    parts["attention_score"] = "fn attention_score(q: &Vector, k: &Vector) -> f32 { q.dot(k) }\n"
    parts["scaled_attention_score"] = textwrap.dedent(
        """
        fn scaled_attention_score(q: &Vector, k: &Vector) -> f32 {
            let dk = q.len() as f32;
            q.dot(k) / dk.sqrt()
        }
        """
    )
    parts["softmax"] = textwrap.dedent(
        """
        fn softmax(xs: &[f32]) -> Vec<f32> {
            let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.into_iter().map(|e| e / sum).collect()
        }
        """
    )
    parts["weighted_sum"] = textwrap.dedent(
        """
        fn weighted_sum(weights: &[f32], values: &[Vector]) -> Vector {
            assert!(!values.is_empty());
            assert_eq!(weights.len(), values.len());
            let dim = values[0].len();
            let mut out = vec![0.0; dim];
            for (weight, value) in weights.iter().zip(values.iter()) {
                for (slot, component) in out.iter_mut().zip(value.data.iter()) {
                    *slot += weight * component;
                }
            }
            Vector::new(out)
        }
        """
    )
    parts["attention_head"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct AttentionHead {
            w_q: Linear,
            w_k: Linear,
            w_v: Linear,
        }

        impl AttentionHead {
            fn forward(&self, seq: &Sequence) -> Sequence {
                let queries: Vec<Vector> = seq.tokens.iter().map(|x| self.w_q.forward(x)).collect();
                let keys: Vec<Vector> = seq.tokens.iter().map(|x| self.w_k.forward(x)).collect();
                let values: Vec<Vector> = seq.tokens.iter().map(|x| self.w_v.forward(x)).collect();
                let mut outputs = Vec::with_capacity(seq.len());
                for q in &queries {
                    let scores: Vec<f32> =
                        keys.iter().map(|k| scaled_attention_score(q, k)).collect();
                    let weights = softmax(&scores);
                    let out = weighted_sum(&weights, &values);
                    outputs.push(out);
                }
                Sequence::new(outputs)
            }
        }
        """
    )
    parts["concat"] = textwrap.dedent(
        """
        fn concat_vectors(vectors: &[Vector]) -> Vector {
            let mut out = Vec::new();
            for v in vectors {
                out.extend_from_slice(&v.data);
            }
            Vector::new(out)
        }
        """
    )
    parts["mha_struct"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct MultiHeadAttention {
            heads: Vec<AttentionHead>,
            w_o: Linear,
        }
        """
    )
    parts["mha_impl"] = textwrap.dedent(
        """
        impl MultiHeadAttention {
            fn forward(&self, seq: &Sequence) -> Sequence {
                let head_outputs: Vec<Sequence> =
                    self.heads.iter().map(|head| head.forward(seq)).collect();
                let mut final_tokens = Vec::with_capacity(seq.len());
                for token_index in 0..seq.len() {
                    let token_parts: Vec<Vector> = head_outputs
                        .iter()
                        .map(|head_seq| head_seq.tokens[token_index].clone())
                        .collect();
                    let joined = concat_vectors(&token_parts);
                    let projected = self.w_o.forward(&joined);
                    final_tokens.push(projected);
                }
                Sequence::new(final_tokens)
            }
        }
        """
    )
    parts["pos_struct"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct PositionalEncoding {
            d_model: usize,
        }

        impl PositionalEncoding {
            fn new(d_model: usize) -> Self {
                Self { d_model }
            }
        }
        """
    )
    parts["pos_encode"] = textwrap.dedent(
        """
        impl PositionalEncoding {
            fn encode_position(&self, pos: usize) -> Vector {
                let mut values = vec![0.0; self.d_model];
                for (i, slot) in values.iter_mut().enumerate() {
                    let exponent = (2 * (i / 2)) as f32 / self.d_model as f32;
                    let angle = pos as f32 / 10000_f32.powf(exponent);
                    *slot = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                }
                Vector::new(values)
            }
        }
        """
    )
    parts["pos_add"] = textwrap.dedent(
        """
        impl PositionalEncoding {
            fn add_to(&self, seq: &Sequence) -> Sequence {
                let mut out = Vec::with_capacity(seq.len());
                for pos in 0..seq.len() {
                    let pe = self.encode_position(pos);
                    let token_with_pos = Vector::new(
                        seq.tokens[pos]
                            .data
                            .iter()
                            .zip(pe.data.iter())
                            .map(|(a, b)| a + b)
                            .collect(),
                    );
                    out.push(token_with_pos);
                }
                Sequence::new(out)
            }
        }
        """
    )
    parts["residual"] = "fn residual(x: &Sequence, fx: &Sequence) -> Sequence { x.add(fx) }\n"
    parts["layernorm"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct LayerNorm {
            eps: f32,
        }

        impl LayerNorm {
            fn new() -> Self {
                Self { eps: 1e-5 }
            }

            fn forward(&self, x: &Vector) -> Vector {
                let mean = x.data.iter().sum::<f32>() / x.len() as f32;
                let var = x
                    .data
                    .iter()
                    .map(|value| {
                        let delta = value - mean;
                        delta * delta
                    })
                    .sum::<f32>()
                    / x.len() as f32;
                let denom = (var + self.eps).sqrt();
                Vector::new(x.data.iter().map(|v| (v - mean) / denom).collect())
            }
        }
        """
    )
    parts["layernorm_seq"] = textwrap.dedent(
        """
        impl LayerNorm {
            fn forward_sequence(&self, seq: &Sequence) -> Sequence {
                Sequence::new(seq.tokens.iter().map(|token| self.forward(token)).collect())
            }
        }
        """
    )
    parts["feedforward_struct"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct FeedForward {
            linear1: Linear,
            linear2: Linear,
        }
        """
    )
    parts["feedforward_token"] = textwrap.dedent(
        """
        impl FeedForward {
            fn forward_token(&self, x: &Vector) -> Vector {
                let hidden = relu(&self.linear1.forward(x));
                self.linear2.forward(&hidden)
            }
        }
        """
    )
    parts["feedforward_seq"] = textwrap.dedent(
        """
        impl FeedForward {
            fn forward_sequence(&self, seq: &Sequence) -> Sequence {
                Sequence::new(
                    seq.tokens
                        .iter()
                        .map(|token| self.forward_token(token))
                        .collect(),
                )
            }
        }
        """
    )
    parts["block_struct"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct TransformerEncoderBlock {
            attention: MultiHeadAttention,
            norm1: LayerNorm,
            feed_forward: FeedForward,
            norm2: LayerNorm,
        }
        """
    )
    parts["block_impl"] = textwrap.dedent(
        """
        impl TransformerEncoderBlock {
            fn forward(&self, x: &Sequence) -> Sequence {
                let attention_out = self.attention.forward(x);
                let after_residual = residual(x, &attention_out);
                let after_norm = self.norm1.forward_sequence(&after_residual);
                let ff_out = self.feed_forward.forward_sequence(&after_norm);
                let after_ff_residual = residual(&after_norm, &ff_out);
                self.norm2.forward_sequence(&after_ff_residual)
            }
        }
        """
    )
    parts["encoder"] = textwrap.dedent(
        """
        #[derive(Debug, Clone)]
        struct Encoder {
            blocks: Vec<TransformerEncoderBlock>,
        }

        impl Encoder {
            fn forward(&self, x: &Sequence) -> Sequence {
                let mut current = x.clone();
                for block in &self.blocks {
                    current = block.forward(&current);
                }
                current
            }
        }
        """
    )
    return parts


def combine(parts: dict[str, str], *names: str) -> str:
    seen: set[str] = set()
    out = [parts["allow"]]
    for name in names:
        if name not in seen:
            out.append(parts[name])
            seen.add(name)
    return "".join(out)


def extract_blocks(markdown_file: Path) -> list[str]:
    text = markdown_file.read_text(encoding="utf-8")
    return [block.strip() for block in re.findall(r"```rust\n(.*?)```", text, re.S)]


def wrap_known_block(path: str, idx: int, block: str) -> str:
    key = f"{path}:{idx}"
    if key == "lessons/01-foundations/01-core-idea.md:1":
        return (
            common_f64()
            + f"fn main() {{ let (w1, x1, w2, x2, b) = (1.0_f64, 2.0, 3.0, 4.0, 5.0); {block} let _ = z; }}\n"
        )
    if key == "lessons/01-foundations/01-core-idea.md:2":
        return common_f64() + f"fn main() {{ {block} let _ = z; }}\n"
    if key == "lessons/01-foundations/02-reading-algebra-like-a-programmer.md:1":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let _ = (same, x1, x2, squared, first_example, sum, model(1.0, 2.0), y, y_hat, dot(&values, &values), matrix, d_loss_d_w);\n}\n"
        )
    if key == "lessons/01-foundations/03-rust-syntax-for-ml.md:1":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let mut n = Neuron { w1: 1.0, w2: 2.0, b: 3.0 };\n    let _ = (x, y, add(1.0, 2.0), pair, values, n.forward(1.0, 2.0));\n    n.update_bias(0.5);\n}\n"
        )
    if key == "lessons/01-foundations/solutions.md:1":
        return (
            common_f64()
            + f"fn main() {{ let (w1, x1, w2, x2, b) = (1.0_f64, 2.0, 3.0, 4.0, 5.0); {block} let _ = z; }}\n"
        )
    if key == "lessons/01-foundations/solutions.md:2":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let _ = (x1, x3);\n}\n"
        )
    if key == "lessons/01-foundations/solutions.md:3":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let _ = dot(&[1.0, 2.0], &[3.0, 4.0]);\n}\n"
        )
    if key == "lessons/01-foundations/solutions.md:4":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let n = Neuron { w1: 1.0, w2: 2.0, b: 3.0 };\n    let _ = n.forward(1.0, 2.0);\n}\n"
        )
    if key == "lessons/02-vectors/01-scalars-vectors-matrices.md:1":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let _ = (x, v, w);\n}\n"
        )
    if key == "lessons/02-vectors/01-scalars-vectors-matrices.md:2":
        return common_f64() + "fn main() {\n" + textwrap.indent(block, "    ") + "\n    let _ = w;\n}\n"
    if key == "lessons/02-vectors/02-sum-dot-product-and-mat-vec.md:1":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];\n    let vector = vec![5.0, 6.0];\n    let _ = (dot(&vector, &vector), mat_vec_mul(&matrix, &vector));\n}\n"
        )
    if key == "lessons/02-vectors/03-sigmoid-loss-and-gradient-descent.md:1":
        return (
            common_f64()
            + "fn main() { let y_hat = 0.8_f64; let y = 1.0_f64; let mut w = 2.0_f64; let learning_rate = 0.1_f64; let d_loss_d_w = 0.3_f64;\n"
            + textwrap.indent(block, "    ")
            + "\n    let _ = (sigmoid(0.0), loss, w);\n}\n"
        )
    if key == "lessons/02-vectors/exercises.md:1":
        return common_f64() + f"fn main() {{ let mut w = 2.0_f64; let learning_rate = 0.1_f64; let d_loss_d_w = 0.3_f64; {block} let _ = w; }}\n"
    if key == "lessons/02-vectors/solutions.md:1":
        return (
            common_f64()
            + "fn main() {\n"
            + textwrap.indent(block, "    ")
            + "\n    let _ = dot(&[1.0, 2.0], &[3.0, 4.0]);\n}\n"
        )
    if key == "lessons/07-transformer/01-tiny-transformer-from-first-principles.md:1":
        return common_f64() + (
            'fn main() { let queries = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; let keys = vec![vec![5.0, 6.0], vec![7.0, 8.0]]; let mut scores = vec![0.0; 2]; let i = 0usize; let j = 1usize; let scale = 2.0_f64; '
            + block
            + " let _ = scores; }\n"
        )
    if key == "lessons/07-transformer/01-tiny-transformer-from-first-principles.md:2":
        return (
            common_f64()
            + "struct Demo { w1: Vec<Vec<f64>>, w2: Vec<Vec<f64>> }\nimpl Demo { fn run(&self, x: &[f64]) -> Vec<f64> {\n"
            + textwrap.indent(block, "        ")
            + "\n} }\nfn main() { let demo = Demo { w1: vec![vec![1.0, 0.0], vec![0.0, 1.0]], w2: vec![vec![1.0, 0.0], vec![0.0, 1.0]] }; let _ = demo.run(&[1.0, -2.0]); }\n"
        )
    if key == "lessons/07-transformer/01-tiny-transformer-from-first-principles.md:3":
        return common_f64() + f"fn main() {{ let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; let attention_out = vec![vec![0.5, 0.5], vec![0.5, 0.5]]; let i = 0usize; let _y = {block}; }}\n"
    if key == "lessons/07-transformer/01-tiny-transformer-from-first-principles.md:4":
        trimmed = block.replace("type Vector = Vec<f64>;\n\n", "", 1)
        return (
            common_f64()
            + trimmed
            + "\nfn main() { let block = TransformerBlock { attention: SelfAttention, ff: FeedForward { w1: vec![], w2: vec![] } }; let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; let _ = block.forward(&x); }\n"
        )
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:1":
        return block + "\nfn main() { let v = Vector::new(vec![1.0, 2.0]); let u = Vector::new(vec![3.0, 4.0]); let mut m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]); let _ = (v.len(), v.dot(&u), v.add(&u), v.map(|x| x), v.as_slice(), m.get(0, 0), m.mul_vec(&u)); m.set(0, 1, 5.0); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:2":
        return prelude_vector_matrix() + block + "\nfn main() { let layer = Linear { weight: Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), bias: Vector::new(vec![0.5, 0.5]) }; let x = Vector::new(vec![2.0, -3.0]); let _ = (layer.forward(&x), relu(&x)); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:3":
        return prelude_vector() + block + "\nfn main() { let _seq = Sequence::new(vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])]); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:4":
        return prelude_attention() + block + "\nfn main() { let id = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]); let zero = Vector::new(vec![0.0, 0.0]); let layer = Linear::new(id.clone(), zero.clone()); let seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]); let attn = SelfAttention { w_q: layer.clone(), w_k: layer.clone(), w_v: layer }; let _ = attn.forward(&seq); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:5":
        return prelude_vector() + block + "\nfn main() { let v = Vector::new(vec![-1.0, 0.0, 2.0]); let _ = phi(&v); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:6":
        return prelude_linear_attention() + block + "\nfn main() { let id = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]); let zero = Vector::new(vec![0.0, 0.0]); let layer = Linear::new(id.clone(), zero.clone()); let seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]); let attn = LinearAttention { w_q: layer.clone(), w_k: layer.clone(), w_v: layer }; let _ = attn.forward(&seq); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:7":
        return prelude_transformer() + block + "\nfn main() { let id = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]); let zero = Vector::new(vec![0.0, 0.0]); let layer = Linear::new(id.clone(), zero.clone()); let attention = LinearAttention::new(layer.clone(), layer.clone(), layer.clone()); let ff = FeedForward::new(layer.clone(), layer); let block = TransformerBlock { attention, ff }; let seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]); let _ = block.forward(&seq); }\n"
    if key == "lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md:8":
        return block + "\nfn main() { let m = MatrixMN { data: [[1.0, 0.0], [0.0, 1.0]] }; let v = VectorN { data: [2.0, 3.0] }; let _ = m.mul_vec(&v); }\n"
    if key == "lessons/07-transformer/exercises.md:1":
        return common_f64() + f"fn main() {{ let weights = vec![0.2_f64, 0.8_f64]; {block} }}\n"
    if key == "lessons/07-transformer/exercises.md:2":
        return common_f64() + f"fn main() {{ let _v = {block}; }}\n"
    if key == "lessons/07-transformer/exercises.md:3":
        return common_f64() + "fn main() {\n" + textwrap.indent(block, "    ") + "\n    let _ = identity_vec(&[1.0, 2.0]);\n}\n"
    raise KeyError(key)


def compile_general_snippets(temp_dir: Path) -> int:
    authored_paths = [
        Path("lessons/01-foundations/01-core-idea.md"),
        Path("lessons/01-foundations/02-reading-algebra-like-a-programmer.md"),
        Path("lessons/01-foundations/03-rust-syntax-for-ml.md"),
        Path("lessons/01-foundations/solutions.md"),
        Path("lessons/02-vectors/01-scalars-vectors-matrices.md"),
        Path("lessons/02-vectors/02-sum-dot-product-and-mat-vec.md"),
        Path("lessons/02-vectors/03-sigmoid-loss-and-gradient-descent.md"),
        Path("lessons/02-vectors/exercises.md"),
        Path("lessons/02-vectors/solutions.md"),
        Path("lessons/07-transformer/01-tiny-transformer-from-first-principles.md"),
        Path("lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md"),
        Path("lessons/07-transformer/exercises.md"),
    ]

    failures: list[str] = []
    count = 0
    for rel_path in authored_paths:
        full_path = ROOT / rel_path
        for idx, block in enumerate(extract_blocks(full_path), start=1):
            src = wrap_known_block(str(rel_path), idx, block)
            rs_path = temp_dir / f"general_{count:02d}.rs"
            rs_path.write_text(src, encoding="utf-8")
            result = subprocess.run(
                ["rustc", "--edition=2024", "--emit=metadata", str(rs_path), "-o", str(rs_path.with_suffix(".rmeta"))],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                failures.append(
                    f"--- {rel_path} block {idx} ---\n{result.stderr}"
                )
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from authored foundational and Transformer lessons.")
    return 0


def compile_chunked_transformer_snippets(temp_dir: Path) -> int:
    parts = chunk_parts()
    lesson = ROOT / "lessons/07-transformer/03-transformer-encoder-in-small-chunks.md"
    blocks = extract_blocks(lesson)

    def main_wrap(body: str, suffix: str = "") -> str:
        body_text = textwrap.indent(body.strip(), "    ")
        suffix_text = textwrap.indent(suffix, "    ")
        return "fn main() {\n" + body_text + ("\n" if body.strip() else "") + suffix_text + "\n}\n"

    full_env = combine(
        parts,
        "vector",
        "matrix",
        "linear",
        "relu",
        "sequence",
        "attention_score",
        "scaled_attention_score",
        "softmax",
        "weighted_sum",
        "attention_head",
        "concat",
        "mha_struct",
        "mha_impl",
        "pos_struct",
        "pos_encode",
        "pos_add",
        "residual",
        "layernorm",
        "layernorm_seq",
        "feedforward_struct",
        "feedforward_token",
        "feedforward_seq",
        "block_struct",
        "block_impl",
        "encoder",
    )

    failures: list[str] = []
    for i, block in enumerate(blocks, start=1):
        if i == 1:
            src = combine(parts) + block + "\n" + main_wrap("let _ = Vector::new(vec![1.0, 2.0, 3.0]).len();")
        elif i == 2:
            src = combine(parts, "vector") + "\n" + main_wrap(block, "let _ = x.len();")
        elif i == 3:
            src = (
                combine(parts, "vector_no_dot")
                + "\n"
                + block
                + "\n"
                + main_wrap(
                    "let a = Vector::new(vec![1.0, 2.0]);\nlet b = Vector::new(vec![3.0, 4.0]);\nlet _ = a.dot(&b);"
                )
            )
        elif i == 4:
            src = combine(parts, "vector") + "\n" + main_wrap(block, "let _ = a.dot(&b);")
        elif i == 5:
            src = combine(parts) + block + "\n" + main_wrap("let m = Matrix::new(1, 1, vec![2.0]);\nlet _ = m.get(0, 0);")
        elif i == 6:
            src = (
                combine(parts, "vector", "matrix_no_mul")
                + "\n"
                + block
                + "\n"
                + main_wrap(
                    "let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);\nlet x = Vector::new(vec![5.0, 6.0]);\nlet _ = m.mul_vec(&x);"
                )
            )
        elif i == 7:
            src = combine(parts, "vector", "matrix") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet x = Vector::new(vec![1.0, 2.0]);\nlet _ = layer.forward(&x);"
            )
        elif i == 8:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let x = Vector::new(vec![-1.0, 2.0]);\nlet _ = relu(&x);"
            )
        elif i == 9:
            src = combine(parts, "vector", "matrix", "linear", "relu") + "\n" + block + "\n"
        elif i == 10:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let seq = Sequence::new(vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])]);\nlet _ = seq.len();"
            )
        elif i == 11:
            src = combine(parts, "vector", "matrix", "linear") + "\n" + block + "\n" + main_wrap(
                "let proj = AttentionProjections {\n    w_q: Linear::new(Matrix::new(1, 1, vec![1.0]), Vector::new(vec![0.0])),\n    w_k: Linear::new(Matrix::new(1, 1, vec![1.0]), Vector::new(vec![0.0])),\n    w_v: Linear::new(Matrix::new(1, 1, vec![1.0]), Vector::new(vec![0.0])),\n};\nlet x = Vector::new(vec![1.0]);\nlet _ = proj.project(&x);"
            )
        elif i == 12:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let q = Vector::new(vec![1.0, 2.0]);\nlet k = Vector::new(vec![3.0, 4.0]);\nlet _ = attention_score(&q, &k);"
            )
        elif i == 13:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let q = Vector::new(vec![1.0, 2.0]);\nlet k = Vector::new(vec![3.0, 4.0]);\nlet _ = scaled_attention_score(&q, &k);"
            )
        elif i == 14:
            src = combine(parts) + block + "\n" + main_wrap("let _ = softmax(&[1.0, 2.0, 3.0]);")
        elif i == 15:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let weights = vec![0.25, 0.75];\nlet values = vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])];\nlet _ = weighted_sum(&weights, &values);"
            )
        elif i == 16:
            src = combine(parts, "vector", "matrix", "linear", "sequence", "scaled_attention_score", "softmax", "weighted_sum") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet head = AttentionHead { w_q: layer.clone(), w_k: layer.clone(), w_v: layer };\nlet seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]);\nlet _ = head.forward(&seq);"
            )
        elif i == 17:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let _ = concat_vectors(&[Vector::new(vec![1.0]), Vector::new(vec![2.0])]);"
            )
        elif i == 18:
            src = combine(parts, "vector", "matrix", "linear", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet head = AttentionHead { w_q: layer.clone(), w_k: layer.clone(), w_v: layer.clone() };\nlet _mha = MultiHeadAttention { heads: vec![head], w_o: layer };"
            )
        elif i == 19:
            src = combine(parts, "vector", "matrix", "linear", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head", "concat", "mha_struct") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet head = AttentionHead { w_q: layer.clone(), w_k: layer.clone(), w_v: layer.clone() };\nlet mha = MultiHeadAttention {\n    heads: vec![head.clone(), head],\n    w_o: Linear::new(Matrix::new(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), Vector::new(vec![0.0, 0.0])),\n};\nlet seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]);\nlet _ = mha.forward(&seq);"
            )
        elif i == 20:
            src = combine(parts) + block + "\n" + main_wrap("let _ = PositionalEncoding::new(4);")
        elif i == 21:
            src = combine(parts, "vector", "sequence", "pos_struct") + "\n" + block + "\n" + main_wrap(
                "let pe = PositionalEncoding::new(4);\nlet _ = pe.encode_position(2);"
            )
        elif i == 22:
            src = combine(parts, "vector", "sequence", "pos_struct", "pos_encode") + "\n" + block + "\n" + main_wrap(
                "let pe = PositionalEncoding::new(2);\nlet seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]);\nlet _ = pe.add_to(&seq);"
            )
        elif i == 23:
            src = combine(parts, "vector", "sequence") + "\n" + block + "\n" + main_wrap(
                "let a = Sequence::new(vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])]);\nlet b = Sequence::new(vec![Vector::new(vec![3.0]), Vector::new(vec![4.0])]);\nlet _ = residual(&a, &b);"
            )
        elif i == 24:
            src = combine(parts, "vector") + "\n" + block + "\n" + main_wrap(
                "let norm = LayerNorm::new();\nlet x = Vector::new(vec![1.0, 2.0, 3.0]);\nlet _ = norm.forward(&x);"
            )
        elif i == 25:
            src = combine(parts, "vector", "sequence", "layernorm") + "\n" + block + "\n" + main_wrap(
                "let norm = LayerNorm::new();\nlet seq = Sequence::new(vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])]);\nlet _ = norm.forward_sequence(&seq);"
            )
        elif i == 26:
            src = combine(parts, "vector", "matrix", "linear") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(1, 1, vec![1.0]), Vector::new(vec![0.0]));\nlet _ = FeedForward { linear1: layer.clone(), linear2: layer };"
            )
        elif i == 27:
            src = combine(parts, "vector", "matrix", "linear", "relu", "feedforward_struct") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet ff = FeedForward { linear1: layer.clone(), linear2: layer };\nlet x = Vector::new(vec![1.0, -2.0]);\nlet _ = ff.forward_token(&x);"
            )
        elif i == 28:
            src = combine(parts, "vector", "matrix", "linear", "relu", "sequence", "feedforward_struct", "feedforward_token") + "\n" + block + "\n" + main_wrap(
                "let layer = Linear::new(Matrix::new(1, 1, vec![1.0]), Vector::new(vec![0.0]));\nlet ff = FeedForward { linear1: layer.clone(), linear2: layer };\nlet seq = Sequence::new(vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])]);\nlet _ = ff.forward_sequence(&seq);"
            )
        elif i == 29:
            src = combine(parts, "vector", "matrix", "linear", "relu", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head", "concat", "mha_struct", "mha_impl", "pos_struct", "pos_encode", "pos_add", "residual", "layernorm", "layernorm_seq", "feedforward_struct", "feedforward_token", "feedforward_seq") + "\n" + block + "\n" + main_wrap("let _ = None::<TransformerEncoderBlock>;")
        elif i == 30:
            src = combine(parts, "vector", "matrix", "linear", "relu", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head", "concat", "mha_struct", "mha_impl", "residual", "layernorm", "layernorm_seq", "feedforward_struct", "feedforward_token", "feedforward_seq", "block_struct") + "\n" + block + "\n" + main_wrap(
                "let layer2 = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet head = AttentionHead { w_q: layer2.clone(), w_k: layer2.clone(), w_v: layer2.clone() };\nlet mha = MultiHeadAttention {\n    heads: vec![head.clone(), head],\n    w_o: Linear::new(Matrix::new(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), Vector::new(vec![0.0, 0.0])),\n};\nlet ff = FeedForward { linear1: layer2.clone(), linear2: layer2 };\nlet block = TransformerEncoderBlock { attention: mha, norm1: LayerNorm::new(), feed_forward: ff, norm2: LayerNorm::new() };\nlet seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]);\nlet _ = block.forward(&seq);"
            )
        elif i == 31:
            src = combine(parts, "vector", "matrix", "linear", "relu", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head", "concat", "mha_struct", "mha_impl", "residual", "layernorm", "layernorm_seq", "feedforward_struct", "feedforward_token", "feedforward_seq", "block_struct", "block_impl") + "\n" + block + "\n" + main_wrap(
                "let layer2 = Linear::new(Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]), Vector::new(vec![0.0, 0.0]));\nlet head = AttentionHead { w_q: layer2.clone(), w_k: layer2.clone(), w_v: layer2.clone() };\nlet mha = MultiHeadAttention {\n    heads: vec![head.clone(), head],\n    w_o: Linear::new(Matrix::new(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), Vector::new(vec![0.0, 0.0])),\n};\nlet ff = FeedForward { linear1: layer2.clone(), linear2: layer2 };\nlet block = TransformerEncoderBlock { attention: mha, norm1: LayerNorm::new(), feed_forward: ff, norm2: LayerNorm::new() };\nlet encoder = Encoder { blocks: vec![block] };\nlet seq = Sequence::new(vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])]);\nlet _ = encoder.forward(&seq);"
            )
        elif i == 32:
            src = combine(parts, "vector", "sequence") + "\n" + main_wrap(block, "let _ = seq.len();")
        elif i == 33:
            src = combine(parts, "vector", "sequence", "pos_struct", "pos_encode", "pos_add") + "\n" + main_wrap(
                "let seq = Sequence::new(vec![\n    Vector::new(vec![1.0, 0.0, 1.0, 0.0]),\n    Vector::new(vec![0.0, 1.0, 0.0, 1.0]),\n    Vector::new(vec![1.0, 1.0, 0.0, 0.0]),\n]);\n" + block,
                "let _ = seq_with_pos.len();",
            )
        elif i == 34:
            src = combine(parts, "vector", "matrix", "linear", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head") + "\n" + main_wrap(
                block,
                "let _ = head.forward(&Sequence::new(vec![\n    Vector::new(vec![1.0, 0.0, 1.0, 0.0]),\n    Vector::new(vec![0.0, 1.0, 0.0, 1.0]),\n]));",
            )
        elif i == 35:
            src = combine(parts, "vector", "matrix", "linear", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head", "concat", "mha_struct", "mha_impl") + "\n" + main_wrap(
                "let projector = Linear::new(\n    Matrix::new(\n        4,\n        4,\n        vec![\n            1.0, 0.0, 0.0, 0.0,\n            0.0, 1.0, 0.0, 0.0,\n            0.0, 0.0, 1.0, 0.0,\n            0.0, 0.0, 0.0, 1.0,\n        ],\n    ),\n    Vector::new(vec![0.0, 0.0, 0.0, 0.0]),\n);\nlet head = AttentionHead {\n    w_q: projector.clone(),\n    w_k: projector.clone(),\n    w_v: projector,\n};\n"
                    + block,
                "let seq = Sequence::new(vec![\n    Vector::new(vec![1.0, 0.0, 1.0, 0.0]),\n    Vector::new(vec![0.0, 1.0, 0.0, 1.0]),\n]);\nlet _ = mha.forward(&seq);",
            )
        elif i == 36:
            src = combine(parts, "vector", "matrix", "linear", "relu", "sequence", "layernorm", "layernorm_seq", "feedforward_struct", "feedforward_token", "feedforward_seq") + "\n" + main_wrap(
                block,
                "let x = Sequence::new(vec![Vector::new(vec![1.0, 2.0, 3.0, 4.0])]);\nlet _ = (ff.forward_sequence(&x), norm1.forward_sequence(&x), norm2.forward_sequence(&x));",
            )
        elif i == 37:
            src = combine(parts, "vector", "matrix", "linear", "relu", "sequence", "scaled_attention_score", "softmax", "weighted_sum", "attention_head", "concat", "mha_struct", "mha_impl", "residual", "layernorm", "layernorm_seq", "feedforward_struct", "feedforward_token", "feedforward_seq", "block_struct", "block_impl") + "\n" + main_wrap(
                "let projector = Linear::new(\n    Matrix::new(\n        4,\n        4,\n        vec![\n            1.0, 0.0, 0.0, 0.0,\n            0.0, 1.0, 0.0, 0.0,\n            0.0, 0.0, 1.0, 0.0,\n            0.0, 0.0, 0.0, 1.0,\n        ],\n    ),\n    Vector::new(vec![0.0, 0.0, 0.0, 0.0]),\n);\nlet head = AttentionHead { w_q: projector.clone(), w_k: projector.clone(), w_v: projector.clone() };\nlet mha = MultiHeadAttention {\n    heads: vec![head.clone(), head],\n    w_o: Linear::new(\n        Matrix::new(\n            4,\n            8,\n            vec![\n                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,\n            ],\n        ),\n        Vector::new(vec![0.0, 0.0, 0.0, 0.0]),\n    ),\n};\nlet ff = FeedForward { linear1: projector.clone(), linear2: projector };\nlet norm1 = LayerNorm::new();\nlet norm2 = LayerNorm::new();\n"
                    + block,
                "let seq = Sequence::new(vec![\n    Vector::new(vec![1.0, 0.0, 1.0, 0.0]),\n    Vector::new(vec![0.0, 1.0, 0.0, 1.0]),\n]);\nlet _ = block.forward(&seq);",
            )
        elif i == 38:
            src = full_env + "\n" + main_wrap(
                "let seq = Sequence::new(vec![\n    Vector::new(vec![1.0, 0.0, 1.0, 0.0]),\n    Vector::new(vec![0.0, 1.0, 0.0, 1.0]),\n    Vector::new(vec![1.0, 1.0, 0.0, 0.0]),\n]);\nlet pe = PositionalEncoding::new(4);\nlet seq_with_pos = pe.add_to(&seq);\nlet projector = Linear::new(\n    Matrix::new(\n        4,\n        4,\n        vec![\n            1.0, 0.0, 0.0, 0.0,\n            0.0, 1.0, 0.0, 0.0,\n            0.0, 0.0, 1.0, 0.0,\n            0.0, 0.0, 0.0, 1.0,\n        ],\n    ),\n    Vector::new(vec![0.0, 0.0, 0.0, 0.0]),\n);\nlet head = AttentionHead { w_q: projector.clone(), w_k: projector.clone(), w_v: projector.clone() };\nlet mha = MultiHeadAttention {\n    heads: vec![head.clone(), head],\n    w_o: Linear::new(\n        Matrix::new(\n            4,\n            8,\n            vec![\n                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,\n                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,\n            ],\n        ),\n        Vector::new(vec![0.0, 0.0, 0.0, 0.0]),\n    ),\n};\nlet ff = FeedForward { linear1: projector.clone(), linear2: projector };\nlet block = TransformerEncoderBlock { attention: mha, norm1: LayerNorm::new(), feed_forward: ff, norm2: LayerNorm::new() };\n"
                + block
            )
        else:
            raise ValueError(f"Unhandled chunked Transformer block {i}")

        rs_path = temp_dir / f"chunked_{i:02d}.rs"
        rs_path.write_text(src, encoding="utf-8")
        result = subprocess.run(
            ["rustc", "--edition=2024", "--emit=metadata", str(rs_path), "-o", str(rs_path.with_suffix(".rmeta"))],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"--- lessons/07-transformer/03-transformer-encoder-in-small-chunks.md block {i} ---\n{result.stderr}")

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {len(blocks)} Rust snippets from the chunked Transformer lesson.")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rust-ml-snippets-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        general = compile_general_snippets(temp_dir)
        chunked = compile_chunked_transformer_snippets(temp_dir)
        if general or chunked:
            return 1
    print("All authored Rust lesson snippets compiled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
