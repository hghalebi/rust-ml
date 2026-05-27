# Shapes, Elements, Bytes, And FLOPs

## Overview

Systems intuition starts with counting.

Before asking whether a model is fast, ask what it must store and what it must
compute. In typed Rust, those are not anonymous numbers. They are objects such as
`ActivationShape`, `ElementCount`, `Bytes`, `AttentionEstimate`, and `Flops`.

## Learning Goals

- compute activation elements from batch, sequence length, and model width
- convert element counts into bytes with an explicit element size
- estimate dense attention score FLOPs and value-mixing FLOPs
- keep bytes and FLOPs as different types
- read systems estimates as maps from semantic shapes to resource evidence

## Plain-English Explanation

An activation shape answers:

```text
How many scalar values are alive here?
```

For:

```text
batch size = 2
sequence length = 8
model width = 16
```

the element count is:

```text
2 * 8 * 16 = 256
```

If each element is a `float32`, each scalar needs 4 bytes:

```text
256 * 4 = 1024 bytes
```

Attention has another cost. Dense self-attention compares every token with every
other token, so the score work grows with:

```text
sequence length * sequence length * model width
```

The systems lesson is to name these quantities before optimizing. If bytes and
FLOPs share one raw number type in the learner's head, the resource story becomes
muddy.

## Algebra Form

Activation memory:

```text
elements = batch_size * sequence_length * model_width
bytes = elements * element_size
```

Dense attention estimate:

```text
score_flops = sequence_length^2 * model_width * 2
value_mix_flops = sequence_length^2 * model_width * 2
total_flops = score_flops + value_mix_flops
score_matrix_bytes = sequence_length^2 * element_size
```

The composition is:

```text
ActivationShape -> ElementCount -> Bytes
AttentionEstimate -> Flops + Bytes
```

## Rust Form

```rust
use rust_ml_systems::{
    ActivationShape, AttentionEstimate, BatchSize, ElementSize, ModelWidth, SequenceLength,
};

fn main() -> Result<(), rust_ml_systems::Error> {
    let shape = ActivationShape::new(
        BatchSize::try_from(2)?,
        SequenceLength::try_from(8)?,
        ModelWidth::try_from(16)?,
    );
    let estimate = AttentionEstimate::new(
        SequenceLength::try_from(8)?,
        ModelWidth::try_from(16)?,
    );

    println!("activation elements = {}", shape.elements()?);
    println!(
        "activation memory   = {}",
        shape.activation_bytes(ElementSize::float32())?
    );
    println!("attention FLOPs     = {}", estimate.total_flops()?);
    println!(
        "score matrix bytes  = {}",
        estimate.score_matrix_bytes(ElementSize::float32())?
    );

    Ok(())
}
```

The raw literals enter through `TryFrom` adapters. After that boundary, the
resource path moves through semantic types.

## Why This Matters

Performance discussions often fail because the evidence is unnamed.

This module makes the evidence explicit. `Bytes` means storage or movement.
`Flops` means arithmetic work. `ElementSize` means representation width. The
types force the learner to ask which resource is changing.

## Concept Trace

- **Object/newtype:** `ActivationShape`, `ElementCount`, `Bytes`, `ElementSize`, `AttentionEstimate`, and `Flops`.
- **Invariant:** dimensions and resource counts are positive, and bytes and FLOPs remain different units.
- **Map:** semantic shape -> element count -> bytes, and attention shape -> FLOP and score-matrix estimates.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting`.
- **Failure signal:** you claim a systems result without naming whether the evidence is bytes, FLOPs, or both.

## Short Practice

1. For batch `2`, sequence length `8`, and model width `16`, how many activation elements are there?
2. Why does `ElementSize::float32()` turn `256` elements into `1024 bytes`?
3. Why are `Bytes` and `Flops` separate types?
4. Which object estimates dense attention score and value-mixing FLOPs?
