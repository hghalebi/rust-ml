# Scalars, Vectors, and Matrices

## Overview

This lesson introduces the data shapes that show up everywhere in machine learning.

## Learning Goals

- define scalar, vector, and matrix in plain language
- recognize the same structures in algebra and Rust
- understand why shape is a first-class idea in ML

## Plain-English Explanation

### Scalar

A scalar is one number.

### Vector

A vector is an ordered list of numbers.

### Matrix

A matrix is a rectangular table of numbers.

In ML, these are not abstract decorations. They are the containers that hold:

- inputs
- parameters
- activations
- embeddings

## Algebra Form

Scalar:

```math
x = 2.5
```

Vector:

```math
x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
```

Matrix:

```math
W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
```

## Rust Form

```rust
use rust_ml_transformer::{DenseMatrix, DenseVector, ModelError, ModelScalar};

fn main() -> Result<(), ModelError> {
    let x = ModelScalar::try_from(2.5)?;

    let v = DenseVector::new([
        ModelScalar::try_from(1.0)?,
        ModelScalar::try_from(2.0)?,
        ModelScalar::try_from(3.0)?,
    ])?;

    let w = DenseMatrix::from_rows([
        [ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?],
        [ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?],
    ])?;

    println!("scalar = {x}");
    println!("vector width = {}", v.len());
    println!("matrix shape = {} x {}", w.rows(), w.cols());
    Ok(())
}
```

## Why This Matters

The word "shape" matters because ML computations only make sense when the structures line up.

If you know:

- vector length
- matrix rows
- matrix columns

then you can predict what an operation is allowed to do.

## Concept Trace

- **Object/newtype:** later crates name shapes with types such as `VectorWidth`, `RowCount`, `ColumnCount`, and `MatrixShape`.
- **Invariant:** vector lengths and matrix dimensions must be known before composition.
- **Map:** shape description -> allowed operation.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow`.
- **Failure signal:** you calculate with numbers before naming whether the value is a scalar, vector, or matrix.

## Short Practice

1. Is `[1.0, 2.0, 3.0]` a scalar, vector, or matrix?
2. How many rows and columns does this matrix have?

```text
DenseMatrix::from_rows([
    [ModelScalar::try_from(0.1)?, ModelScalar::try_from(0.2)?, ModelScalar::try_from(0.3)?],
    [ModelScalar::try_from(0.4)?, ModelScalar::try_from(0.5)?, ModelScalar::try_from(0.6)?],
])?
```
