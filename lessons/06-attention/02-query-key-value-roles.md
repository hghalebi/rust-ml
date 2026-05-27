# Query, Key, and Value Roles

## Overview

Attention uses three views of each token:

```text
query: what am I looking for?
key: what do I offer as a match target?
value: what information do I carry if selected?
```

These are different roles, even when they are all vectors.

## Learning Goals

- explain query, key, and value in plain English
- connect projections to role-specific maps
- explain why query and key widths must match
- explain why value width controls the output width
- read attention as typed composition rather than untyped vector arithmetic

## Plain-English Explanation

### One token, three roles

A token embedding is a general representation.

Attention projects it into role-specific views:

```text
TokenEmbedding -> Query
TokenEmbedding -> Key
TokenEmbedding -> Value
```

The query asks a question. The key is a label that other queries can match against. The value is the information that gets copied into the mixture.

### Query and key must meet in the same space

The attention score is a dot product:

```text
query dot key
```

That dot product requires equal widths.

The value vector does not need to have the same width as the query and key. The value vector controls what kind of information gets mixed into the output.

## Algebra Form

For one token vector:

```math
q_i = W_Q x_i + b_Q
```

```math
k_i = W_K x_i + b_K
```

```math
v_i = W_V x_i + b_V
```

Then a score between token `i` and token `j` is:

```math
score(i, j) = \frac{q_i \cdot k_j}{\sqrt{d_k}}
```

The structural map is:

```text
TokenEmbedding -> Query
TokenEmbedding -> Key
TokenEmbedding -> Value
Query * Key -> AttentionScore
```

## Rust Form

```rust
use rust_ml_attention::{Key, KeyComponent, Query, QueryComponent};

fn main() -> Result<(), rust_ml_attention::Error> {
    let query = Query::from_values([
        QueryComponent::try_from(1.0)?,
        QueryComponent::try_from(1.0)?,
    ])?;
    let key = Key::from_values([
        KeyComponent::try_from(1.0)?,
        KeyComponent::try_from(0.0)?,
    ])?;

    println!("{:.4}", (&query * &key)?);
    Ok(())
}
```

The crate version uses the same idea with richer names and shape errors, so a query and key with different widths cannot silently produce nonsense.

## Why This Matters

Query, key, and value are one of the most overloaded parts of Transformer explanations.

The safest beginner model is:

```text
query decides what to search for
key decides what can be matched
value decides what content is mixed
```

Typed Rust helps because `Query`, `Key`, and `Value` are different names at the boundary where meaning matters.

## Concept Trace

- **Object/newtype:** `Query`, `Key`, `Value`, and `ProjectionWeight`.
- **Invariant:** role-specific vectors may share width but still have different meaning.
- **Map:** token embedding -> query/key/value projections.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair`.
- **Failure signal:** you see three vectors and cannot say which one asks, which one answers, and which one carries content.

## Short Practice

1. Which role asks "what am I looking for?"
2. Which role is used as the match target?
3. Which role carries the information that gets mixed?
4. Why do query and key widths need to match?
