#!/usr/bin/env python3
"""Compile-check Rust snippets embedded in authored lesson Markdown files."""

from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_HELPER_SIGNATURE_RE = re.compile(
    r"("
    r"\bVec\s*<"
    r"|&\s*\["
    r"|&\s*str\b"
    r"|&\s*'static\s+str\b"
    r"|\bf32\b"
    r"|\bf64\b"
    r"|\busize\b"
    r"|\bu64\b"
    r"|\bu128\b"
    r"|\bi64\b"
    r"|\bbool\b"
    r")"
)
FUNCTION_START_RE = re.compile(r"^\s*fn\s+(\w+)\s*[<(]")


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

            pub fn components(&self) -> &[f32] {
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
                        sum += self.get(r, c) * x.components()[c];
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
            let mean: f32 = x.components().iter().sum::<f32>() / n;
            let var: f32 = x
                .components()
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / n;

            let eps = 1e-5;
            Vector::new(
                x.components()
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


def collect_function_signatures(lines: list[str]) -> list[tuple[int, str, str]]:
    signatures: list[tuple[int, str, str]] = []
    collecting = False
    start_line = 0
    function_name = ""
    current: list[str] = []
    paren_depth = 0

    for line_number, line in enumerate(lines, start=1):
        if not collecting:
            function = FUNCTION_START_RE.search(line)
            if function is None:
                continue

            collecting = True
            start_line = line_number
            function_name = function.group(1)
            current = []
            paren_depth = 0

        current.append(line.strip())
        paren_depth += line.count("(") - line.count(")")

        if paren_depth <= 0 and ("{" in line or ";" in line):
            signatures.append((start_line, function_name, " ".join(current)))
            collecting = False

    return signatures


def extract_blocks_with_lines(markdown_file: Path) -> list[tuple[str, int]]:
    text = markdown_file.read_text(encoding="utf-8")
    return [
        (match.group(1).strip(), text.count("\n", 0, match.start()) + 2)
        for match in re.finditer(r"```rust\n(.*?)```", text, re.S)
    ]


def extract_blocks(markdown_file: Path) -> list[str]:
    return [block for block, _start_line in extract_blocks_with_lines(markdown_file)]


def check_typed_crate_helper_signatures(
    rel_path: Path,
    block_index: int,
    block_start_line: int,
    block: str,
    crate_label: str,
) -> list[str]:
    errors: list[str] = []

    for line_number, function_name, signature in collect_function_signatures(block.splitlines()):
        if function_name == "main":
            continue

        if RAW_HELPER_SIGNATURE_RE.search(signature):
            errors.append(
                f"ERROR: {rel_path}:{block_start_line + line_number - 1} "
                f"block {block_index} helper `{function_name}` has a raw primitive/container signature; "
                f"{crate_label} lesson snippets should validate raw literals inline with TryFrom and keep helpers typed"
            )

    return errors


def check_transformer_helper_signatures(
    rel_path: Path,
    block_index: int,
    block_start_line: int,
    block: str,
) -> list[str]:
    return check_typed_crate_helper_signatures(
        rel_path,
        block_index,
        block_start_line,
        block,
        "Transformer",
    )


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
    if key == "lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md:1":
        return block + "\n"
    if key == "lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md:2":
        return block + "\n"
    if key == "lessons/03-neuron/02-neuron-as-a-chain-of-functions.md:1":
        return block + "\n"
    if key == "lessons/03-neuron/02-neuron-as-a-chain-of-functions.md:2":
        return block + "\n"
    if key == "lessons/04-learning/01-training-step-as-feedback.md:1":
        return block + "\n"
    if key == "lessons/04-learning/02-epochs-and-loss-traces.md:1":
        return block + "\n"
    if key == "lessons/05-mlp/01-hidden-layers-as-representations.md:1":
        return block + "\n"
    if key == "lessons/05-mlp/02-shape-flow-through-an-mlp.md:1":
        return block + "\n"
    if key == "lessons/05-mlp/exercises.md:1":
        return block + "\n"
    if key == "lessons/05-mlp/solutions.md:1":
        return block + "\n"
    if key == "lessons/06-attention/01-tokens-as-vectors-in-a-sequence.md:1":
        return block + "\n"
    if key == "lessons/06-attention/02-query-key-value-roles.md:1":
        return block + "\n"
    if key == "lessons/06-attention/03-scores-weights-and-value-mixing.md:1":
        return block + "\n"
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
        return block + "\nfn main() { let v = Vector::new(vec![1.0, 2.0]); let u = Vector::new(vec![3.0, 4.0]); let mut m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]); let _ = (v.len(), v.dot(&u), v.add(&u), v.map(|x| x), v.components(), m.get(0, 0), m.mul_vec(&u)); m.set(0, 1, 5.0); }\n"
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
        Path("lessons/02-vectors/exercises.md"),
        Path("lessons/03-neuron/exercises.md"),
        Path("lessons/03-neuron/solutions.md"),
        Path("lessons/04-learning/exercises.md"),
        Path("lessons/04-learning/solutions.md"),
        Path("lessons/05-mlp/exercises.md"),
        Path("lessons/05-mlp/solutions.md"),
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

    print(
        f"Compiled {count} Rust snippets from the authored vectors, neuron practice, learning, and MLP practice lessons."
    )
    return 0


def compile_learning_lens_snippets(temp_dir: Path) -> int:
    rel_path = Path("lessons/00-learning-lens.md")
    full_path = ROOT / rel_path
    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "learning-lens-snippet-target"
    crate_path = (ROOT / "code/category_lens").resolve()

    for idx, (block, block_start_line) in enumerate(
        extract_blocks_with_lines(full_path), start=1
    ):
        failures.extend(
            check_typed_crate_helper_signatures(
                rel_path,
                idx,
                block_start_line,
                block,
                "Learning lens",
            )
        )

        snippet_dir = temp_dir / f"learning_lens_{count:03d}"
        src_dir = snippet_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (snippet_dir / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""
                [package]
                name = "learning_lens_snippet_{count:03d}"
                version = "0.1.0"
                edition = "2024"

                [dependencies]
                rust_ml_category_lens = {{ path = "{crate_path}" }}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        source = block + "\n"
        if "fn main" not in block:
            source += "fn main() {}\n"
        (src_dir / "main.rs").write_text(source, encoding="utf-8")

        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(target_dir)
        env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

        result = subprocess.run(
            [
                "cargo",
                "check",
                "--quiet",
                "--manifest-path",
                str(snippet_dir / "Cargo.toml"),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
        count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the Learning Lens against the local crate.")
    return 0


def compile_neuron_crate_snippets(temp_dir: Path) -> int:
    neuron_paths = [
        Path("lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md"),
        Path("lessons/03-neuron/02-neuron-as-a-chain-of-functions.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "neuron-snippet-target"
    crate_path = (ROOT / "code/neuron").resolve()

    for rel_path in neuron_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Neuron",
                )
            )

            snippet_dir = temp_dir / f"neuron_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "neuron_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_neuron = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the Neuron module against the local crate.")
    return 0


def compile_intro_crate_snippets(temp_dir: Path) -> int:
    intro_paths = [
        Path("lessons/01-foundations/01-core-idea.md"),
        Path("lessons/01-foundations/02-reading-algebra-like-a-programmer.md"),
        Path("lessons/01-foundations/03-rust-syntax-for-ml.md"),
        Path("lessons/01-foundations/solutions.md"),
        Path("lessons/02-vectors/01-scalars-vectors-matrices.md"),
        Path("lessons/02-vectors/02-sum-dot-product-and-mat-vec.md"),
        Path("lessons/02-vectors/03-sigmoid-loss-and-gradient-descent.md"),
        Path("lessons/02-vectors/solutions.md"),
        Path("lessons/04-learning/01-training-step-as-feedback.md"),
        Path("lessons/04-learning/02-epochs-and-loss-traces.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "intro-snippet-target"
    neuron_path = (ROOT / "code/neuron").resolve()
    transformer_path = (ROOT / "code/transformer").resolve()

    for rel_path in intro_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Intro",
                )
            )

            snippet_dir = temp_dir / f"intro_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "intro_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_neuron = {{ path = "{neuron_path}" }}
                    rust_ml_transformer = {{ path = "{transformer_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the intro modules against local crates.")
    return 0


def compile_mlp_crate_snippets(temp_dir: Path) -> int:
    mlp_paths = [
        Path("lessons/05-mlp/01-hidden-layers-as-representations.md"),
        Path("lessons/05-mlp/02-shape-flow-through-an-mlp.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "mlp-snippet-target"
    crate_path = (ROOT / "code/mlp").resolve()

    for rel_path in mlp_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "MLP",
                )
            )

            snippet_dir = temp_dir / f"mlp_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "mlp_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_mlp = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the MLP module against the local crate.")
    return 0


def compile_attention_crate_snippets(temp_dir: Path) -> int:
    attention_paths = [
        Path("lessons/06-attention/01-tokens-as-vectors-in-a-sequence.md"),
        Path("lessons/06-attention/02-query-key-value-roles.md"),
        Path("lessons/06-attention/03-scores-weights-and-value-mixing.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "attention-snippet-target"
    crate_path = (ROOT / "code/attention").resolve()

    for rel_path in attention_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Attention",
                )
            )

            snippet_dir = temp_dir / f"attention_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "attention_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_attention = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the Attention module against the local crate.")
    return 0


def compile_lm_basics_crate_snippets(temp_dir: Path) -> int:
    lm_basics_paths = [
        Path("lessons/08-language-modeling/01-text-to-token-ids.md"),
        Path("lessons/08-language-modeling/02-next-token-batches-loss-and-update.md"),
        Path("lessons/08-language-modeling/03-public-text-boundary.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "lm-basics-snippet-target"
    crate_path = (ROOT / "code/lm_basics").resolve()

    for rel_path in lm_basics_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Language-modeling basics",
                )
            )

            snippet_dir = temp_dir / f"lm_basics_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "lm_basics_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_lm_basics = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(
        f"Compiled {count} Rust snippets from the Language Modeling module against the local crate."
    )
    return 0


def compile_systems_crate_snippets(temp_dir: Path) -> int:
    systems_paths = [
        Path("lessons/09-systems/01-shapes-elements-bytes-and-flops.md"),
        Path("lessons/09-systems/02-timing-intensity-and-memory-hierarchy.md"),
        Path("lessons/09-systems/03-public-systems-report-boundary.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "systems-snippet-target"
    crate_path = (ROOT / "code/systems").resolve()

    for rel_path in systems_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Systems",
                )
            )

            snippet_dir = temp_dir / f"systems_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "systems_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_systems = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the Systems module against the local crate.")
    return 0


def compile_kernels_crate_snippets(temp_dir: Path) -> int:
    kernels_paths = [
        Path("lessons/10-kernels/01-elementwise-maps-and-reductions.md"),
        Path("lessons/10-kernels/02-tiling-a-matrix-vector-kernel.md"),
        Path("lessons/10-kernels/03-public-kernel-report-boundary.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "kernels-snippet-target"
    crate_path = (ROOT / "code/kernels").resolve()

    for rel_path in kernels_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Kernels",
                )
            )

            snippet_dir = temp_dir / f"kernels_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "kernels_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_kernels = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the Kernels module against the local crate.")
    return 0


def compile_inference_crate_snippets(temp_dir: Path) -> int:
    inference_paths = [
        Path("lessons/11-inference/01-autoregressive-decoding-state-trace.md"),
        Path("lessons/11-inference/02-public-decode-boundary-and-latency.md"),
        Path("lessons/11-inference/exercises.md"),
        Path("lessons/11-inference/solutions.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "inference-snippet-target"
    crate_path = (ROOT / "code/inference").resolve()

    for rel_path in inference_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_typed_crate_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                    "Inference",
                )
            )

            snippet_dir = temp_dir / f"inference_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "inference_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_inference = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(
        "Compiled "
        f"{count} Rust snippets from the Inference module against the local crate."
    )
    return 0


def compile_chunked_transformer_snippets(temp_dir: Path) -> int:
    transformer_paths = [
        Path("lessons/07-transformer/01-tiny-transformer-from-first-principles.md"),
        Path("lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md"),
        Path("lessons/07-transformer/03-transformer-encoder-in-small-chunks.md"),
        Path("lessons/07-transformer/exercises.md"),
    ]

    failures: list[str] = []
    count = 0
    target_dir = temp_dir / "transformer-snippet-target"
    crate_path = (ROOT / "code/transformer").resolve()

    for rel_path in transformer_paths:
        full_path = ROOT / rel_path
        for idx, (block, block_start_line) in enumerate(
            extract_blocks_with_lines(full_path), start=1
        ):
            failures.extend(
                check_transformer_helper_signatures(
                    rel_path,
                    idx,
                    block_start_line,
                    block,
                )
            )

            snippet_dir = temp_dir / f"transformer_{count:03d}"
            src_dir = snippet_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (snippet_dir / "Cargo.toml").write_text(
                textwrap.dedent(
                    f"""
                    [package]
                    name = "transformer_snippet_{count:03d}"
                    version = "0.1.0"
                    edition = "2024"

                    [dependencies]
                    rust_ml_transformer = {{ path = "{crate_path}" }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (src_dir / "main.rs").write_text(block + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(target_dir)
            env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"

            result = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--quiet",
                    "--manifest-path",
                    str(snippet_dir / "Cargo.toml"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                failures.append(f"--- {rel_path} block {idx} ---\n{result.stderr}")
            count += 1

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Compiled {count} Rust snippets from the Transformer module against the local crate.")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rust-ml-snippets-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        general = compile_general_snippets(temp_dir)
        learning_lens = compile_learning_lens_snippets(temp_dir)
        neuron = compile_neuron_crate_snippets(temp_dir)
        intro = compile_intro_crate_snippets(temp_dir)
        mlp = compile_mlp_crate_snippets(temp_dir)
        attention = compile_attention_crate_snippets(temp_dir)
        lm_basics = compile_lm_basics_crate_snippets(temp_dir)
        systems = compile_systems_crate_snippets(temp_dir)
        kernels = compile_kernels_crate_snippets(temp_dir)
        inference = compile_inference_crate_snippets(temp_dir)
        chunked = compile_chunked_transformer_snippets(temp_dir)
        if (
            general
            or learning_lens
            or neuron
            or intro
            or mlp
            or attention
            or lm_basics
            or systems
            or kernels
            or inference
            or chunked
        ):
            return 1
    print("All authored Rust lesson snippets compiled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
