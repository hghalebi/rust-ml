# Lesson 19: Transformer Encoder in Small Chunks

## Overview

This lesson takes the standard Transformer encoder path and breaks it into deliberately small pieces.

The rule is simple:

- one idea
- one tiny code snippet
- one takeaway

Do not try to hold the whole Transformer in your head at once. Stack small truths until the architecture becomes readable.

## Learning Goals

- explain how a Transformer grows out of vectors, matrices, and linear layers
- trace one attention head from queries and keys to weighted value mixing
- describe why multi-head attention, positional encoding, residual paths, and layer normalization exist
- assemble one Transformer encoder block in Rust-sized pieces
- place linear attention in the correct part of the design space

## Plain-English Explanation

The Transformer is not one giant magical object.

It is a disciplined stack of smaller ideas:

1. vectors represent tokens
2. matrices transform vectors
3. linear layers learn projections
4. attention lets one token score all other tokens
5. multi-head attention repeats that scoring in parallel
6. positional encoding tells the model where tokens are
7. residual paths preserve earlier signal
8. layer normalization keeps values stable
9. feed-forward layers refine each token independently
10. stacking blocks deepens context

This lesson follows that order.

## Algebra Form

Neural-network core:

```math
\text{input} \rightarrow \text{some math} \rightarrow \text{output}
```

Dot product:

```math
a \cdot b = a_1b_1 + a_2b_2 + a_3b_3 + \dots
```

Linear layer:

```math
y = Wx + b
```

Attention score:

```math
\text{score} = q \cdot k
```

Scaled attention score:

```math
\frac{q \cdot k}{\sqrt{d_k}}
```

Self-attention output:

```math
\text{output}_i = \sum_j \alpha_{ij} v_j
```

Residual connection:

```math
x + f(x)
```

Token plus position:

```math
\text{token representation} = \text{word embedding} + \text{position encoding}
```

Encoder block shape:

```math
\text{input}
\rightarrow \text{attention}
\rightarrow \text{residual + norm}
\rightarrow \text{feed-forward}
\rightarrow \text{residual + norm}
```

## Rust Form

### Chunk 0. A neural network is just input, math, output

At first principles level, a neural network is just a learned function.

- input goes in
- arithmetic happens
- output comes out

The arithmetic is mostly matrix multiplication, bias addition, and a nonlinearity.

### Chunk 1. A vector is a list of numbers

```rust
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
```

Example:

```rust
let x = Vector::new(vec![1.0, 2.0, 3.0]);
```

Takeaway: a token representation starts life as a plain numeric list.

### Chunk 2. Dot product

```rust
impl Vector {
    fn dot(&self, other: &Vector) -> f32 {
        assert_eq!(self.len(), other.len());

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
```

Example:

```rust
let a = Vector::new(vec![1.0, 2.0, 3.0]);
let b = Vector::new(vec![4.0, 5.0, 6.0]);

println!("{}", a.dot(&b)); // 32.0
```

Takeaway: dot product measures alignment. Later, attention turns that alignment into a score.

### Chunk 3. A matrix is a table of numbers

```rust
#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>, // row-major
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
```

Takeaway: matrices are learned tables that reshape and combine features.

### Chunk 4. Matrix times vector

```rust
impl Matrix {
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
```

Takeaway: this is the core computation behind linear layers.

### Chunk 5. Linear layer

```rust
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
```

Takeaway: a linear layer is just a learned projection plus a bias.

### Chunk 6. ReLU

```rust
fn relu(x: &Vector) -> Vector {
    Vector::new(x.data.iter().map(|v| v.max(0.0)).collect())
}
```

Takeaway: nonlinearities keep a deep network from collapsing into one big linear map.

### Chunk 7. Tiny neural network

```rust
fn main() {
    let layer1 = Linear::new(
        Matrix::new(
            2,
            3,
            vec![
                1.0, 0.0, 2.0,
                0.0, 1.0, 3.0,
            ],
        ),
        Vector::new(vec![0.5, -1.0]),
    );

    let x = Vector::new(vec![1.0, 2.0, 3.0]);

    let y = layer1.forward(&x);
    let z = relu(&y);

    println!("y = {:?}", y.data);
    println!("z = {:?}", z.data);
}
```

Takeaway: even a tiny network already has the main pattern of learned transformation plus activation.

### Chunk 8. A Transformer starts with a sequence

```rust
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
```

Takeaway: a Transformer does not process one vector. It processes a list of token vectors.

### Chunk 9. Query, key, value projections

```rust
#[derive(Debug, Clone)]
struct AttentionProjections {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl AttentionProjections {
    fn project(&self, x: &Vector) -> (Vector, Vector, Vector) {
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);

        (q, k, v)
    }
}
```

Takeaway:

- query = what am I looking for?
- key = what do I contain?
- value = what information do I contribute?

### Chunk 10. One attention score

```rust
fn attention_score(q: &Vector, k: &Vector) -> f32 {
    q.dot(k)
}
```

Takeaway: one query-key dot product measures how strongly two token views match.

### Chunk 11. Scale the score

```rust
fn scaled_attention_score(q: &Vector, k: &Vector) -> f32 {
    let dk = q.len() as f32;
    q.dot(k) / dk.sqrt()
}
```

Takeaway: scaling keeps large hidden dimensions from blowing up score magnitudes.

### Chunk 12. Softmax turns scores into probabilities

```rust
fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    exps.into_iter().map(|e| e / sum).collect()
}
```

Takeaway: softmax turns raw compatibility scores into weights that sum to 1.

### Chunk 13. Weighted sum of values

```rust
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
```

Takeaway: one attention output is a mixture of value vectors, not a hard choice of one token.

### Chunk 14. One full attention head

```rust
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
            let scores: Vec<f32> = keys
                .iter()
                .map(|k| scaled_attention_score(q, k))
                .collect();

            let weights = softmax(&scores);
            let out = weighted_sum(&weights, &values);

            outputs.push(out);
        }

        Sequence::new(outputs)
    }
}
```

Takeaway: this is self-attention. Each token compares against every token, then mixes values.

### Chunk 15. One head is not enough

One head gives one learned way to compare tokens.

Language has many relationships:

- syntax
- meaning
- reference
- agreement
- position-sensitive behavior

That is why the paper uses multiple heads.

### Chunk 16. Concatenation helper

```rust
fn concat_vectors(vectors: &[Vector]) -> Vector {
    let mut out = Vec::new();

    for v in vectors {
        out.extend_from_slice(&v.data);
    }

    Vector::new(out)
}
```

Takeaway: concatenation just stitches multiple head outputs together end to end.

### Chunk 17. Multi-head attention container

```rust
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    w_o: Linear,
}
```

Takeaway: multi-head attention is many attention heads plus one output projection.

### Chunk 18. Multi-head forward pass

```rust
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
```

Takeaway: each head attends differently, then the model recombines those views.

### Chunk 19. Why positional encoding is needed

Attention by itself does not know order. Without positional information, the model is missing the difference between:

- `dog bites man`
- `man bites dog`

That is why the Transformer adds positional encoding.

### Chunk 20. Positional encoding struct

```rust
#[derive(Debug, Clone)]
struct PositionalEncoding {
    d_model: usize,
}

impl PositionalEncoding {
    fn new(d_model: usize) -> Self {
        Self { d_model }
    }
}
```

### Chunk 21. Sinusoidal encoding for one position

```rust
impl PositionalEncoding {
    fn encode_position(&self, pos: usize) -> Vector {
        let mut values = vec![0.0; self.d_model];

        for (i, slot) in values.iter_mut().enumerate() {
            let exponent = (2 * (i / 2)) as f32 / self.d_model as f32;
            let angle = pos as f32 / 10000_f32.powf(exponent);

            *slot = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }

        Vector::new(values)
    }
}
```

Takeaway: the original paper uses sine and cosine patterns so nearby positions have related encodings.

### Chunk 22. Add positions to a sequence

```rust
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
```

Takeaway: a token representation is token meaning plus position signal.

### Chunk 23. Residual connection

```rust
fn residual(x: &Sequence, fx: &Sequence) -> Sequence {
    x.add(fx)
}
```

Takeaway: residual connections let the model preserve the original signal and learn a correction on top.

### Chunk 24. Why normalization exists

Deep networks can become numerically unstable. Layer normalization keeps activations in a more stable range.

### Chunk 25. Layer norm for one vector

```rust
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
```

Takeaway: layer norm centers and rescales each token vector independently.

### Chunk 26. Apply layer norm to a full sequence

```rust
impl LayerNorm {
    fn forward_sequence(&self, seq: &Sequence) -> Sequence {
        Sequence::new(seq.tokens.iter().map(|token| self.forward(token)).collect())
    }
}
```

### Chunk 27. Feed-forward network intuition

Attention mixes information between tokens.

Feed-forward layers then transform each token by itself. That division is core to the architecture.

### Chunk 28. Feed-forward struct

```rust
#[derive(Debug, Clone)]
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}
```

### Chunk 29. Feed-forward for one token

```rust
impl FeedForward {
    fn forward_token(&self, x: &Vector) -> Vector {
        let hidden = relu(&self.linear1.forward(x));
        self.linear2.forward(&hidden)
    }
}
```

### Chunk 30. Feed-forward for a full sequence

```rust
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
```

Takeaway: attention is cross-token mixing. Feed-forward is per-token transformation.

### Chunk 31. One encoder block: the heartbeat

One encoder block does:

1. multi-head attention
2. residual
3. layer norm
4. feed-forward
5. residual
6. layer norm

### Chunk 32. Encoder block struct

```rust
#[derive(Debug, Clone)]
struct TransformerEncoderBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    feed_forward: FeedForward,
    norm2: LayerNorm,
}
```

### Chunk 33. Full encoder block forward pass

```rust
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
```

Takeaway: this is the core encoder pattern from the paper.

### Chunk 34. Why stacking blocks matters

One block gives shallow context. Multiple blocks let the model refine meaning across layers.

### Chunk 35. Stack encoder blocks

```rust
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
```

Takeaway: the Transformer becomes powerful by repeating the same block many times.

### Chunk 36. The decoder exists, but the encoder is the right first stop

The original paper also has a decoder with masked self-attention and encoder-decoder attention.

For first-principles learning, the encoder is the right stopping point because it already contains the main invention.

### Chunk 37. Tiny demo sequence

```rust
let seq = Sequence::new(vec![
    Vector::new(vec![1.0, 0.0, 1.0, 0.0]),
    Vector::new(vec![0.0, 1.0, 0.0, 1.0]),
    Vector::new(vec![1.0, 1.0, 0.0, 0.0]),
]);
```

### Chunk 38. Add positional encoding

```rust
let pe = PositionalEncoding::new(4);
let seq_with_pos = pe.add_to(&seq);
```

### Chunk 39. Build one tiny attention head

```rust
let projector = Linear::new(
    Matrix::new(
        4,
        4,
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    ),
    Vector::new(vec![0.0, 0.0, 0.0, 0.0]),
);

let head = AttentionHead {
    w_q: projector.clone(),
    w_k: projector.clone(),
    w_v: projector,
};
```

### Chunk 40. Build multi-head attention

```rust
let output_projection = Linear::new(
    Matrix::new(
        4,
        8,
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ],
    ),
    Vector::new(vec![0.0, 0.0, 0.0, 0.0]),
);

let mha = MultiHeadAttention {
    heads: vec![head.clone(), head],
    w_o: output_projection,
};
```

### Chunk 41. Build feed-forward and norms

```rust
let ff = FeedForward {
    linear1: Linear::new(
        Matrix::new(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        Vector::new(vec![0.0, 0.0, 0.0, 0.0]),
    ),
    linear2: Linear::new(
        Matrix::new(
            4,
            4,
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        Vector::new(vec![0.0, 0.0, 0.0, 0.0]),
    ),
};

let norm1 = LayerNorm::new();
let norm2 = LayerNorm::new();
```

### Chunk 42. Build one encoder block

```rust
let block = TransformerEncoderBlock {
    attention: mha,
    norm1,
    feed_forward: ff,
    norm2,
};
```

### Chunk 43. Run the block

```rust
let output = block.forward(&seq_with_pos);
println!("{}", output.len());
```

Takeaway: that one line represents attention, residual structure, normalization, and feed-forward refinement.

### Chunk 44. Where linear attention fits

The original paper uses scaled dot-product attention.

Linear attention is a later efficient-attention family. The block shape stays similar, but the attention module is replaced with a different computation strategy.

### Chunk 45. What Liger is not

Keep the categories clean:

- `Attention Is All You Need` = the original Transformer architecture paper
- linear attention = later efficient attention methods
- Liger Kernel = runtime and training optimization tooling

These are different layers of the stack.

### Chunk 46. What this lesson does not build

This lesson does not build:

- training
- backpropagation
- optimizers
- masking
- decoder cross-attention
- batching
- dropout
- GPU kernels

That is intentional. Architecture comes first.

## Why This Matters

This chunked path gives you a reliable mental model:

- `Vector`, `Matrix`, and `Linear` explain the arithmetic substrate
- `AttentionHead` explains how token-to-token interaction works
- `MultiHeadAttention` explains why the model can look through multiple learned lenses at once
- `PositionalEncoding` explains how order enters the system
- `TransformerEncoderBlock` explains the exact structural loop that later models repeat many times

Once this becomes familiar, later topics stop looking like symbol soup. They become variations on a known scaffold.

## Short Practice

Use the smallest possible learning loop:

1. Read only the `Vector` and `Matrix` chunks.
2. Compile a tiny `Linear` example.
3. Print one query, one key, and one attention score.
4. Print softmax weights for a three-token example.
5. Run one attention head on a fake sequence.
6. Run one encoder block and print the output length.
7. Compare this lesson with Lesson 18 and explain where linear attention swaps into the block.

Best next milestone:

- implement `Vector`
- implement `Matrix`
- implement `Linear`
- implement one `AttentionHead`
- print queries, keys, scores, weights, and outputs

That is the point where the Transformer stops feeling mystical and starts feeling architectural.
