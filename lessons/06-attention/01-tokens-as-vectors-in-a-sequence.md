# Tokens as Vectors in a Sequence

## Overview

Attention starts with a simple problem:

```text
a token needs information from other tokens
```

In an MLP, each input example is usually handled as one fixed vector. In a sequence model, we have several token vectors at once.

The first attention question is not "what is the final answer?" It is:

```text
which other token vectors should this token read from?
```

## Learning Goals

- explain why attention starts from a sequence of token vectors
- distinguish token position from token vector width
- explain why a sequence needs a shared width invariant
- read a token sequence as a list of meaningful objects, not a raw matrix
- connect sequence shape to later Transformer shape flow

## Plain-English Explanation

### A sequence is not one vector

A sentence-like input has multiple token positions.

Each position stores a vector:

```text
token 0 -> [1.0, 0.0]
token 1 -> [0.0, 1.0]
token 2 -> [1.0, 1.0]
```

The sequence has two important sizes:

```text
number of tokens
width of each token vector
```

Those are different ideas.

### Shared width is the composition rule

Attention compares token vectors. That comparison only makes sense when the vectors live in compatible spaces.

So the beginner invariant is:

```text
all token embeddings in one sequence must have the same width
```

This is the same structural habit from MLPs:

```text
maps compose only when their shapes line up
```

## Algebra Form

A token sequence can be written as:

```math
X = [x_0, x_1, ..., x_{n-1}]
```

where each token vector has the same width:

```math
x_i \in R^{d_model}
```

The shape story is:

```text
sequence length = n
token width = d_model
```

The category-theory lens is:

```text
TokenEmbedding is an object
TokenSequence is an ordered container of compatible TokenEmbedding values
attention will build maps from each token to a mixed output
```

## Rust Form

```rust
use rust_ml_attention::{TokenComponent, TokenEmbedding, TokenSequence};

fn main() -> Result<(), rust_ml_attention::Error> {
    let sequence = TokenSequence::from_tokens(vec![
        TokenEmbedding::from_values([
            TokenComponent::try_from(1.0)?,
            TokenComponent::try_from(0.0)?,
        ])?,
        TokenEmbedding::from_values([
            TokenComponent::try_from(0.0)?,
            TokenComponent::try_from(1.0)?,
        ])?,
        TokenEmbedding::from_values([
            TokenComponent::try_from(1.0)?,
            TokenComponent::try_from(1.0)?,
        ])?,
    ])?;

    println!("tokens={}, width={}", sequence.len(), sequence.token_width());
    Ok(())
}
```

The companion crate makes the constructor stricter: it rejects empty sequences and inconsistent token widths.

## Why This Matters

The Transformer does not process one token in isolation.

It repeatedly asks how tokens should exchange information.

If learners confuse sequence length with token width, later ideas such as attention heads, context length, and positional encoding become much harder than they need to be.

## Concept Trace

- **Object/newtype:** `TokenEmbedding`, `TokenSequence`, `TokenCount`, and `VectorWidth`.
- **Invariant:** every token in a sequence must share the same model width.
- **Map:** same-width token list -> valid sequence object.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace`.
- **Failure signal:** you confuse adding tokens with changing the width of each token vector.

## Short Practice

1. What is the difference between sequence length and token width?
2. Why should all token vectors in one sequence have the same width?
3. In `TokenSequence -> AttentionOutput`, which object stores the ordered tokens?
4. How does this shape habit connect back to the MLP module?
