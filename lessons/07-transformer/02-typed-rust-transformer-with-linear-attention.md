# Lesson 18: Typed Rust Transformer with Linear Attention

## Overview

This lesson takes the next step after the tiny Transformer block. Instead of only reading the architecture conceptually, we ask a more engineering-shaped question:

How do we model a Transformer block clearly in Rust, with types that make the structure easier to reason about?

## Learning Goals

- explain attention and linear attention from first principles
- describe why Transformers replaced many recurrent models
- model vectors, matrices, sequences, and layers as explicit Rust types
- trace a tiny linear-attention Transformer block in English, algebra, and Rust
- understand what this educational implementation omits compared with production systems

## Plain-English Explanation

### What is a neural network?

A neural network is just a function with parameters.

It takes an input:

```math
x
```

and transforms it into an output:

```math
\hat{y}
```

The parameters are usually weights and biases. Training means adjusting those parameters so the output gets closer to the correct answer.

At a small scale, a neural network keeps repeating the same pattern:

1. multiply by numbers
2. add numbers
3. apply a nonlinearity
4. repeat

The power comes from composing many simple steps.

### Why did Transformers appear?

Before Transformers, many sequence models used recurrent neural networks.

The RNN idea is simple:

- read one token at a time
- keep a hidden memory
- update that memory as you move through the sequence

That works, but it causes two big problems:

- sequential bottleneck: tokens depend on earlier tokens step by step, which makes parallelism hard
- long-range dependency pain: information from early tokens must survive many updates

Transformers replace that recurrence with attention.

Instead of forcing information to travel through many recurrent steps, each token can directly ask:

- which other tokens matter for me?
- how much do they matter?
- how should I combine their information?

### What is attention?

Suppose each token is represented by a vector.

For each token vector $x$, we compute:

- a query vector
- a key vector
- a value vector

Intuitively:

- query = what this token is looking for
- key = what this token offers
- value = the information this token carries

Then we compare one token's query against all keys, turn those scores into weights, and use the weights to mix the value vectors.

### Why does linear attention exist?

Standard attention compares every token with every other token.

If the sequence length is $n$, the full token-to-token interaction pattern is roughly:

```math
O(n^2)
```

That becomes expensive for long sequences.

Linear attention tries to rewrite attention so we do not have to build the full pairwise interaction matrix explicitly. Instead, we push the computation into precomputed summaries of keys and values.

That changes the engineering story from:

- compare every token to every token

to:

- build global summaries once
- let each query consult those summaries

### Why typed Rust helps

Rust forces us to name things precisely. That is useful here.

Instead of throwing nested `Vec<Vec<f32>>` everywhere, we can create types that make the concepts visible:

- `Vector`
- `Matrix`
- `Linear`
- `Sequence`
- `SelfAttention`
- `LinearAttention`
- `FeedForward`
- `TransformerBlock`

That makes the code read more like the math and less like accidental plumbing.

### The small mental model

The educational crate for this lesson builds the following path:

1. vectors and matrices
2. linear layers and nonlinearities
3. sequences of token vectors
4. standard self-attention
5. linear attention
6. a Transformer block with residuals and layer norm

### What is simplified?

This lesson and its crate are intentionally educational.

We skip:

- token embeddings
- positional encoding
- causal masking
- multi-head attention
- autograd
- optimizers
- dropout
- GPU kernels

The goal is understanding, not production throughput.

## Algebra Form

Neural-network view:

```math
\hat{y} = f(x; \theta)
```

Attention projections:

```math
q = xW_Q,\quad k = xW_K,\quad v = xW_V
```

Standard self-attention:

```math
\text{output}_i = \sum_j \alpha_{ij} v_j
```

```math
\alpha_{ij} = \text{softmax}_j\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)
```

Linear-attention form:

```math
\text{output}_i =
\frac{
\phi(q_i)^T \left(\sum_j \phi(k_j) v_j^T\right)
}{
\phi(q_i)^T \left(\sum_j \phi(k_j)\right)
}
```

Layer normalization:

```math
\text{LayerNorm}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

Residual structure:

```math
\text{output} = x + f(x)
```

Transformer block view:

```math
H = \text{LayerNorm}(X + \text{Attention}(X))
```

```math
\text{Output} = \text{LayerNorm}(H + F(H))
```

## Rust Form

Vectors and matrices:

```rust
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
        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c) * x.as_slice()[c];
            }
            out[r] = sum;
        }
        Vector::new(out)
    }
}
```

Linear layer and ReLU:

```rust
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Matrix,
    bias: Vector,
}

impl Linear {
    pub fn forward(&self, x: &Vector) -> Vector {
        self.weight.mul_vec(x).add(&self.bias)
    }
}

pub fn relu(v: &Vector) -> Vector {
    v.map(|x| x.max(0.0))
}
```

Sequence type:

```rust
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
}
```

Standard self-attention:

```rust
#[derive(Debug, Clone)]
pub struct SelfAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl SelfAttention {
    pub fn forward(&self, seq: &Sequence) -> Sequence {
        let qs: Vec<Vector> = seq.tokens().iter().map(|x| self.w_q.forward(x)).collect();
        let ks: Vec<Vector> = seq.tokens().iter().map(|x| self.w_k.forward(x)).collect();
        let vs: Vec<Vector> = seq.tokens().iter().map(|x| self.w_v.forward(x)).collect();

        let d_k = qs[0].len() as f32;
        let scale = d_k.sqrt();

        let mut outputs = Vec::with_capacity(seq.len());

        for q in &qs {
            let scores: Vec<f32> = ks.iter().map(|k| q.dot(k) / scale).collect();
            let weights = softmax(&scores);

            let mut out = vec![0.0; vs[0].len()];
            for (weight, v) in weights.iter().zip(vs.iter()) {
                for (j, value) in v.as_slice().iter().enumerate() {
                    out[j] += weight * value;
                }
            }

            outputs.push(Vector::new(out));
        }

        Sequence::new(outputs)
    }
}
```

Feature map for linear attention:

```rust
pub fn phi(v: &Vector) -> Vector {
    let eps = 1e-6;
    v.map(|x| x.max(0.0) + eps)
}
```

Linear attention:

```rust
#[derive(Debug, Clone)]
pub struct LinearAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl LinearAttention {
    pub fn forward(&self, seq: &Sequence) -> Sequence {
        let qs: Vec<Vector> = seq.tokens().iter().map(|x| phi(&self.w_q.forward(x))).collect();
        let ks: Vec<Vector> = seq.tokens().iter().map(|x| phi(&self.w_k.forward(x))).collect();
        let vs: Vec<Vector> = seq.tokens().iter().map(|x| self.w_v.forward(x)).collect();

        let d_k = qs[0].len();
        let d_v = vs[0].len();
        let mut s = Matrix::zeros(d_k, d_v);
        let mut z = vec![0.0; d_k];

        for (k, v) in ks.iter().zip(vs.iter()) {
            for i in 0..d_k {
                z[i] += k.as_slice()[i];
                for j in 0..d_v {
                    let current = s.get(i, j);
                    s.set(i, j, current + k.as_slice()[i] * v.as_slice()[j]);
                }
            }
        }

        let mut outputs = Vec::with_capacity(seq.len());
        for q in &qs {
            let mut numerator = vec![0.0; d_v];
            for j in 0..d_v {
                let mut sum = 0.0;
                for i in 0..d_k {
                    sum += q.as_slice()[i] * s.get(i, j);
                }
                numerator[j] = sum;
            }

            let denominator: f32 = q
                .as_slice()
                .iter()
                .zip(z.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                .max(1e-6);

            let out = numerator.into_iter().map(|x| x / denominator).collect();
            outputs.push(Vector::new(out));
        }

        Sequence::new(outputs)
    }
}
```

Transformer block:

```rust
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention: LinearAttention,
    ff: FeedForward,
}

impl TransformerBlock {
    pub fn forward(&self, seq: &Sequence) -> Sequence {
        let attn_out = self.attention.forward(seq);

        let after_attn: Vec<Vector> = seq
            .tokens()
            .iter()
            .zip(attn_out.tokens().iter())
            .map(|(x, a)| layer_norm(&x.add(a)))
            .collect();

        let seq2 = Sequence::new(after_attn);

        let after_ff: Vec<Vector> = seq2
            .tokens()
            .iter()
            .map(|x| {
                let ff = self.ff.forward(x);
                layer_norm(&x.add(&ff))
            })
            .collect();

        Sequence::new(after_ff)
    }
}
```

Compile-time-sized teaching types:

```rust
#[derive(Debug, Clone, Copy)]
pub struct VectorN<const N: usize> {
    pub data: [f32; N],
}

#[derive(Debug, Clone, Copy)]
pub struct MatrixMN<const R: usize, const C: usize> {
    pub data: [[f32; C]; R],
}

impl<const R: usize, const C: usize> MatrixMN<R, C> {
    pub fn mul_vec(&self, x: &VectorN<C>) -> VectorN<R> {
        let mut out = [0.0; R];
        for r in 0..R {
            let mut sum = 0.0;
            for c in 0..C {
                sum += self.data[r][c] * x.data[c];
            }
            out[r] = sum;
        }
        VectorN { data: out }
    }
}
```

## Why This Matters

This lesson connects three levels at once:

- machine-learning intuition
- Transformer architecture
- Rust code structure

If you understand this lesson, you no longer have to think of Transformers as mystical giant models. You can think of them as stacks of understandable pieces:

- attention builds relationships between tokens
- feed-forward layers transform representations
- residuals and normalization keep learning stable
- linear attention is an efficiency-minded reformulation of the same core idea

## Short Practice

1. In plain English, what is the difference between standard attention and linear attention?
2. Why is `Sequence` a useful type instead of a raw `Vec<Vector>` everywhere?
3. What role does the denominator play in the linear-attention formula?
4. Why do residual connections make optimization easier?
5. Which parts of a production Transformer are still missing from this educational implementation?
