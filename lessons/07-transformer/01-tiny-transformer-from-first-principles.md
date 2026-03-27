# Lesson 17: What Problem the Transformer Solves

## Overview

This lesson resets the Transformer story from the front door:

- what problem it solves
- what the core math is actually doing
- what we will build in Rust
- why this course uses semantic types instead of anonymous tensor soup

The goal is not to memorize the paper. The goal is to stop the architecture from looking mystical.

## Learning Goals

- explain why the Transformer replaced many recurrent sequence models
- read scaled dot-product attention in plain English, algebra, and Rust
- identify the core encoder-side parts of the original 2017 architecture
- explain why this module uses semantic runtime types before compile-time shape machinery
- run a tiny forward-pass encoder demo in the companion crate

## 1. What problem is the Transformer solving?

### English

Suppose you have a sentence:

`"the cat sat"`

A model wants to understand that:

- `"the"` modifies `"cat"`
- `"cat"` relates to `"sat"`
- words influence each other across the sequence

Older sequence models often processed tokens one by one. The Transformer changes the rule:

let every token look at every other token directly

That mechanism is attention.

### Algebra

If the sequence is:

```math
X = [x_1, x_2, \ldots, x_n]
```

then token `x_i` should be able to use information from every `x_j`, not only from the previous recurrent state.

### Rust

```rust
use rust_ml_transformer::{DenseVector, ModelError, TokenEmbedding, TokenSequence};

fn main() -> Result<(), ModelError> {
    let sentence = TokenSequence::new(vec![
        TokenEmbedding(DenseVector::new(vec![1.0, 0.0, 1.0, 0.0])?),
        TokenEmbedding(DenseVector::new(vec![0.0, 1.0, 0.0, 1.0])?),
        TokenEmbedding(DenseVector::new(vec![1.0, 1.0, 0.0, 0.0])?),
    ])?;

    println!("tokens = {}", sentence.len());
    println!("d_model = {}", sentence.d_model());
    Ok(())
}
```

## 2. The core math, explained like a patient adult

### English

For each token vector `x`, we build three new views of that token:

- query: what am I looking for?
- key: what do I offer?
- value: what information do I carry?

Then we:

1. compare one query with all keys
2. turn those comparisons into weights
3. use the weights to mix the values

### Algebra

For one token:

```math
q = xW_Q,\quad k = xW_K,\quad v = xW_V
```

For a whole sequence:

```math
\mathrm{Attention}(Q, K, V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Rust

```rust
use rust_ml_transformer::{
    scaled_attention_score, softmax, AttentionScores, DenseVector, Key, ModelError, Query,
};

fn main() -> Result<(), ModelError> {
    let query = Query(DenseVector::new(vec![1.0, 2.0])?);
    let key_a = Key(DenseVector::new(vec![3.0, 4.0])?);
    let key_b = Key(DenseVector::new(vec![1.0, 0.0])?);

    let scores = AttentionScores(vec![
        scaled_attention_score(&query, &key_a)?,
        scaled_attention_score(&query, &key_b)?,
    ]);
    let weights = softmax(&scores)?;

    println!("scores = {:?}", scores.0);
    println!("weights = {:?}", weights.0);
    Ok(())
}
```

## 3. What we will build in Rust

We will build a forward-pass teaching implementation of one Transformer encoder block:

- `DenseVector`
- `DenseMatrix`
- `TokenSequence`
- `QueryLayer`
- `KeyLayer`
- `ValueLayer`
- `AttentionHead`
- `MultiHeadAttention`
- `PositionalEncodingTable`
- `LayerNorm`
- `FeedForward`
- `TransformerEncoderBlock`

This is not a training framework.

- no autograd
- no optimizer
- no GPU kernel wizardry

Just the architecture, cleanly.

## 4. Design choice: what "typed" means here

There are two kinds of "typed" Rust designs you could use here.

### English

Option A is strict compile-time shapes:

- `Matrix<ROWS, COLS>`
- `Vector<N>`

Option B is semantic types with runtime shape checks:

- `TokenEmbedding`
- `Query`
- `Key`
- `Value`
- `TokenSequence`

For a beginner-friendly Transformer, this module starts with option B.

Why?

Because you are learning the model, not auditioning for a role as a hostage negotiator with the compiler.

### Algebra

We still care about dimensions:

```math
x \in \mathbb{R}^{d_{model}},\quad
q,k \in \mathbb{R}^{d_k},\quad
v \in \mathbb{R}^{d_v}
```

We just check them at runtime while giving the roles explicit names.

### Rust

```rust
use rust_ml_transformer::{DenseVector, Key, ModelError, Query, TokenEmbedding, Value};

fn main() -> Result<(), ModelError> {
    let token = TokenEmbedding(DenseVector::new(vec![0.2, 0.7, -0.1, 0.4])?);
    let query = Query(DenseVector::new(vec![0.3, 0.1])?);
    let key = Key(DenseVector::new(vec![0.2, 0.5])?);
    let value = Value(DenseVector::new(vec![1.0, -1.0])?);

    println!("token width = {}", token.len());
    println!("query width = {}", query.len());
    println!("key width = {}", key.len());
    println!("value width = {}", value.len());
    Ok(())
}
```

## 5. Run the full forward-pass demo

The companion crate includes a runnable encoder demo:

```bash
cargo run --example encoder_demo --manifest-path code/transformer/Cargo.toml
```

That example builds:

- a 3-token input sequence
- sinusoidal positional encodings
- two attention heads
- a multi-head output projection
- a feed-forward network
- one encoder block

and then prints the final contextualized token vectors.

## 6. Read the encoder block in plain English

One encoder block does this:

1. each token looks at the whole sequence through self-attention
2. the block adds the original token representation back through a residual path
3. layer normalization stabilizes the result
4. each token goes through a small feed-forward network on its own
5. the block adds another residual path
6. layer normalization stabilizes the result again

That is the rhythm.

## 7. What comes directly from *Attention Is All You Need*?

These parts are directly aligned with the original architecture:

- scaled dot-product attention
- multi-head attention
- sinusoidal positional encodings
- feed-forward layers
- residual connections
- layer normalization

This lesson does not build:

- decoder masking
- encoder-decoder cross-attention
- dropout
- learned token embeddings
- training
- backpropagation

Because that would turn a teaching lesson into a hostage situation.

## 8. Where linear attention fits

The original paper does **not** use linear attention.

The slot looks like this:

```text
Transformer block
  -> attention module
```

In the original paper:

```text
attention module = scaled dot-product multi-head attention
```

In an efficient later variant:

```text
attention module = linear attention
```

So the block shape stays similar. The attention mechanism changes.

## 9. Why this Rust style is clean

This module chooses:

- explicit domain structs
- small methods
- shape checks close to the bug
- function signatures that tell the story
- expressive `thiserror` diagnostics instead of panic soup

It avoids:

- giant nested `Vec<Vec<Vec<f32>>>`
- mystery tensor gymnastics
- cleverness that hides the architecture

For a first implementation, readability beats cleverness.

## 10. What to study next

Study the encoder path in this order:

1. `DenseVector`
2. `DenseMatrix`
3. `TokenSequence`
4. `Query`, `Key`, `Value`
5. one `AttentionHead`
6. `MultiHeadAttention`
7. `PositionalEncodingTable`
8. `LayerNorm`
9. `FeedForward`
10. `TransformerEncoderBlock`

Once you can explain each of those out loud, the Transformer stops looking magical and starts looking like disciplined engineering.

## 11. One-sentence summary

A Transformer encoder block lets each token look at other tokens, mix useful information, refine itself with a small neural network, and stay stable through residual connections and normalization.
