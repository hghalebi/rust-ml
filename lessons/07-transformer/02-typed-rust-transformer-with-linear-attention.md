# Typed Rust Transformer with Expressive Errors

## Overview

Now that the encoder shape is on the table, this lesson focuses on the engineering choices behind the companion crate:

- semantic newtypes instead of anonymous vectors
- `thiserror` instead of vague failures
- typed projection layers instead of one giant generic blob
- a clean place to talk about linear attention without confusing it with the 2017 paper

## Learning Goals

- explain why semantic newtypes make attention code easier to read and harder to misuse
- explain why the crate returns `Result<_, ModelError>` instead of panicking on every mistake
- trace typed projection layers from `TokenEmbedding` to `Query`, `Key`, and `Value`
- explain where `LinearAttentionHead` fits relative to standard multi-head attention
- understand the crate layout that backs the written lessons

## 1. One checked math layer, many semantic model roles

### English

At the bottom, the model is still made of scalar values.

So the crate keeps one checked dense-math layer:

- `DenseVector`
- `DenseMatrix`

Then it wraps important model roles in newtypes:

- `TokenEmbedding`
- `Query`
- `Key`
- `Value`
- `AttentionOutput`

That way Rust can tell the difference between concepts that happen to share the same storage.
Raw learner literals still enter at the edge through `ModelScalar::try_from(...)`.
After that boundary, the lesson names the object, the invariant, and the map.

### Algebra

The checked dense values might all live in:

```math
\mathbb{R}^n
```

but the roles are different:

```math
x \neq q \neq k \neq v
```

even when each one is represented by a vector.

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseVector, Key, ModelError, Query, TokenEmbedding, Value,
};

fn main() -> Result<(), ModelError> {
    let token = TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?);
    let query = Query::from_vector(DenseVector::new([ModelScalar::try_from(0.2)?, ModelScalar::try_from(0.5)?])?);
    let key = Key::from_vector(DenseVector::new([ModelScalar::try_from(0.1)?, ModelScalar::try_from(0.4)?])?);
    let value = Value::from_vector(DenseVector::new([ModelScalar::try_from(2.0)?, ModelScalar::try_from(-1.0)?])?);

    println!("{:?}", token.scalar_values());
    println!("{:?}", query.scalar_values());
    println!("{:?}", key.scalar_values());
    println!("{:?}", value.scalar_values());
    Ok(())
}
```

## 2. Use `thiserror`, not panic-heavy shortcuts

### English

Most beginner Transformer bugs are shape bugs.

So the crate does not hide them behind `"something went wrong"`.

It returns `ModelError` values that say:

- which operation failed
- what the shapes were
- what relationship should have held

That helps both developers and researchers.

### Algebra

If you want:

```math
y = Wx
```

then the matrix width and vector length must match.

If they do not, the operation is invalid.

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseMatrix, DenseVector, ModelError};

fn main() -> Result<(), ModelError> {
    let matrix = DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?], [ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?]])?;
    let vector = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?;

    match &matrix * &vector {
        Ok(_) => println!("unexpected success"),
        Err(error) => println!("{error}"),
    }

    Ok(())
}
```

## 3. `TokenSequence` keeps the sequence honest

### English

A Transformer does not process a random list of vectors.

It processes a sequence of token embeddings with one shared model width.

`TokenSequence::new` checks that invariant immediately.

### Algebra

If:

```math
X \in \mathbb{R}^{n \times d_{model}}
```

then every token row must have width:

```math
d_{model}
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError, TokenEmbedding, TokenSequence};

fn main() -> Result<(), ModelError> {
    let sequence = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
    ])?;

    println!("tokens = {}", sequence.len());
    println!("d_model = {}", sequence.d_model());
    Ok(())
}
```

## 4. Projection layers tell the story in their type signatures

### English

Instead of a single generic `Linear` that can mean anything, the crate uses typed projection layers:

- `QueryLayer`
- `KeyLayer`
- `ValueLayer`
- `OutputLayer`

That makes the function signature teach the architecture.

The underlying newtypes also implement Rust operation traits. The expression
`&projection * &token` is not generic matrix arithmetic: its output type is
`Query`. Adding a `ProjectionBias` keeps the same semantic role.

### Algebra

For one token embedding `x`:

```math
q = W_Q x + b_Q
```

```math
k = W_K x + b_K
```

```math
v = W_V x + b_V
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseMatrix, DenseVector, ModelError, ProjectionBias, QueryProjection, TokenEmbedding,
};

fn main() -> Result<(), ModelError> {
    let projection = QueryProjection::from_matrix(DenseMatrix::from_rows([
        [ModelScalar::try_from(0.2)?, ModelScalar::try_from(0.1)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.3)?],
        [ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.4)?, ModelScalar::try_from(0.1)?, ModelScalar::try_from(0.2)?],
    ])?);
    let bias = ProjectionBias::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?);
    let token = TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?);

    let projected = (&projection * &token)?;
    let query = (&projected + &bias)?;

    println!("{:?}", query.scalar_values());
    Ok(())
}
```

## 5. One typed attention head

### English

A standard attention head does exactly three things:

1. project every token to query, key, value
2. compare each query with all keys
3. mix the values using the normalized scores

The important part is that the types keep the roles separate all the way through.

### Algebra

For the whole sequence:

```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```

```math
\mathrm{head}(X) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    AttentionHead, DenseMatrix, DenseVector, KeyLayer, KeyProjection, ModelError,
    ProjectionBias, QueryLayer, QueryProjection, TokenEmbedding, TokenSequence, ValueLayer,
    ValueProjection, VectorLength,
};

fn identity_projection() -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?], [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?]])
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn main() -> Result<(), ModelError> {
    let width = VectorLength::try_from(2)?;
    let head = AttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;

    let sequence = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;

    let outputs = head.forward(&sequence)?;
    println!("{:?}", outputs.output(rust_ml_transformer::TokenIndex::try_from(0)?)?.scalar_values());
    Ok(())
}
```

## 6. Where linear attention plugs in

### English

`LinearAttentionHead` is not the original paper.

It is the same architecture slot with a different internal computation.

That keeps the categories clean:

- original Transformer: exact scaled dot-product attention
- linear attention: efficiency-oriented reformulation

### Algebra

Standard attention:

```math
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Linear attention rewrites the computation through feature maps and summary terms:

```math
\phi(q_i)^T \left(\sum_j \phi(k_j)v_j^T\right)
```

with a matching normalizer.

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseMatrix, DenseVector, KeyLayer, KeyProjection, LinearAttentionHead, ModelError,
    ProjectionBias, QueryLayer, QueryProjection, TokenEmbedding, TokenSequence, ValueLayer,
    ValueProjection, VectorLength,
};

fn identity_projection() -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?], [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?]])
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn main() -> Result<(), ModelError> {
    let width = VectorLength::try_from(2)?;
    let head = LinearAttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;

    let sequence = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;

    let outputs = head.forward(&sequence)?;
    println!("{:?}", outputs.output(rust_ml_transformer::TokenIndex::try_from(0)?)?.scalar_values());
    Ok(())
}
```

## 7. Why this module stops before const generics

### English

You could encode dimensions at compile time:

- `Vector<const N: usize>`
- `Matrix<const R: usize, const C: usize>`

That is a good second step.

It is not the best first step for most people learning attention from scratch.

### Algebra

The important thing is still the relationship:

```math
W \in \mathbb{R}^{m \times n},\quad x \in \mathbb{R}^{n}
```

not the exact syntax used to enforce it.

### Rust

```text
// Future direction, not the first lesson:
//
// pub struct Vector<const N: usize> {
//     data: [f32; N],
// }
//
// pub struct Matrix<const R: usize, const C: usize> {
//     data: [[f32; C]; R],
// }
```

## 8. Crate layout

The executable teaching crate is split like this:

```text
code/transformer/src/
  error.rs
  math.rs
  types.rs
  attention.rs
  transformer.rs
  lib.rs
```

That separation is deliberate:

- `math.rs` owns raw dense math
- `types.rs` owns semantic model roles
- `attention.rs` owns attention-specific logic
- `transformer.rs` owns encoder-side assembly

## Concept Trace

- **Object/newtype:** `DenseVector` is the checked dense-math object, while `Query`, `Key`, `Value`, and `AttentionOutput` are semantic model roles.
- **Invariant:** shared storage does not imply shared meaning; role wrappers prevent architectural swaps.
- **Map:** checked vector -> semantic projection role -> attention output.
- **Runnable proof:** `cargo test --manifest-path code/Cargo.toml -p rust_ml_transformer --all-targets`.
- **Failure signal:** you treat `Query`, `Key`, and `Value` as interchangeable because they all contain vectors.

## Short Practice

1. Why is `Query` a different type from `Value` even though both wrap a `DenseVector`?
2. What kind of bug becomes easier to diagnose once the crate returns `ModelError` instead of panicking everywhere?
3. Why is `LinearAttentionHead` a different concept from the original paper even if it occupies the same architectural slot?
