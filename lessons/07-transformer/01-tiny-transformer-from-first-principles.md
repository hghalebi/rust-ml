# Lesson 17: A Tiny Transformer (From First Principles)

## Overview

At this point you already understand vectors, dot products, neurons, gradients, computation graphs, and attention. This lesson assembles those parts into something bigger: a Transformer block.

## Learning Goals

- explain the structure of a Transformer block from first principles
- describe how attention, feed-forward layers, and residual connections interact
- read the block in English, algebra, and Rust
- recognize what this toy block omits compared with a production Transformer

## Plain-English Explanation

### The core idea

A Transformer processes a sequence of vectors:

```math
X = [x_1, x_2, \ldots, x_n]
```

Each $x_i$ is the embedding of one token.

Each token does four things:

1. looks at other tokens through attention
2. updates its representation with that context
3. passes through a small neural network
4. keeps some of its original information through residual connections

### The block structure

```text
Input X
   ↓
Self-Attention
   ↓
Add (Residual)
   ↓
Feed-Forward
   ↓
Add (Residual)
   ↓
Output
```

### Self-attention recap

Attention still does the same job as before:

- create queries, keys, and values
- compare queries to keys
- turn scores into weights
- use the weights to mix the value vectors

The important change is where this sits: it is now one sublayer inside a larger block.

### Feed-forward layer

After attention, each token goes through a small neural network independently. This is usually a two-layer projection with a nonlinearity in the middle.

You can think of it like this:

- attention mixes information across tokens
- feed-forward transforms each token on its own

### Residual connections

Residual connections mean we add the original signal back to the transformed signal.

That gives the block two useful properties:

- the model can preserve information instead of constantly overwriting it
- optimization becomes easier because each sublayer only needs to learn an improvement, not a full replacement

### Step-by-step intuition

Follow one token $x_i$:

1. attention builds a contextualized version of the token
2. residual addition preserves the original token alongside that new context
3. the feed-forward network transforms the token representation
4. a second residual addition keeps both the earlier state and the new nonlinear transformation

### Why this architecture works

This design separates responsibilities cleanly:

- attention gives global context and token-to-token interaction
- feed-forward gives nonlinear representation power
- residual connections give stability and easier learning

### What we are simplifying

This lesson is intentionally not the full Transformer. We leave out:

- multi-head attention
- positional encoding
- layer normalization
- masking
- training

That is deliberate. Structure first. Details next.

## Algebra Form

Self-attention:

```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Feed-forward:

```math
F(x) = W_2 \cdot \text{ReLU}(W_1 x)
```

Residual structure:

```math
H = X + \text{Attention}(X)
```

```math
\text{Output} = H + F(H)
```

Mental model:

```math
\text{look around} \rightarrow \text{transform} \rightarrow \text{keep original} \rightarrow \text{repeat}
```

## Rust Form

Core attention score:

```rust
scores[j] = dot(&queries[i], &keys[j]) / scale;
```

Feed-forward core:

```rust
let hidden = mat_vec_mul(&self.w1, x);
let activated = relu_vec(&hidden);
mat_vec_mul(&self.w2, &activated)
```

Residual core:

```rust
vec_add(&x[i], &attention_out[i])
```

Toy block:

```rust
type Vector = Vec<f64>;

struct TransformerBlock {
    attention: SelfAttention,
    ff: FeedForward,
}

impl TransformerBlock {
    fn forward(&self, x: &[Vector]) -> Vec<Vector> {
        // 1. Attention
        let attention_out = self.attention.forward(x);

        // 2. Residual after attention
        let mut after_attention = Vec::new();
        for i in 0..x.len() {
            after_attention.push(vec_add(&x[i], &attention_out[i]));
        }

        // 3. Feed-forward
        let ff_out = self.ff.forward(&after_attention);

        // 4. Final residual
        let mut final_out = Vec::new();
        for i in 0..x.len() {
            final_out.push(vec_add(&after_attention[i], &ff_out[i]));
        }

        final_out
    }
}
```

## Why This Matters

A neuron computes a dot product:

```math
z = w \cdot x
```

Attention also relies on dot products:

```math
Q \cdot K
```

The deep connection is that modern language models are still built from the same ingredients you already know. They are just composed into a richer structure.

If you understand this lesson, you understand the skeleton behind GPT-, BERT-, and Transformer-style models:

- tokens interact
- context is built
- recurrence is replaced by structure

## Short Practice

1. Explain a Transformer block in one sentence without using the word "Transformer."
2. Why does the block need both attention and a feed-forward layer?
3. What information is preserved by the residual connections?
4. Which parts of the full Transformer are still missing from this toy version?
