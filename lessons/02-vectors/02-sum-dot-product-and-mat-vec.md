# Sum, Dot Product, and Matrix-Vector Multiplication

## Overview

These are the core arithmetic moves that later models keep reusing.

## Learning Goals

- expand sum notation into ordinary arithmetic
- compute a dot product in English, algebra, and Rust
- explain matrix-vector multiplication as repeated dot products

## Plain-English Explanation

### Sum

Add all the numbers in a collection.

### Dot product

Multiply matching positions, then add the results.

If:

- $a = [1, 2]$
- $b = [3, 4]$

then:

- first pair: `1 * 3`
- second pair: `2 * 4`
- total: `11`

### Matrix-vector multiplication

Each row of the matrix takes a dot product with the vector.

That means the output has one number per row.

## Algebra Form

Sum:

```math
\sum_{i=1}^{n} a_i
```

Dot product:

```math
a \cdot b = \sum_{i=1}^{n} a_i b_i
```

Matrix-vector multiplication:

```math
Wx
```

Example:

```math
W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}

x = \begin{bmatrix} 5 \\ 6 \end{bmatrix}

Wx = \begin{bmatrix} 17 \\ 39 \end{bmatrix}
```

## Rust Form

```rust
use rust_ml_transformer::{DenseMatrix, DenseVector, ModelError, ModelScalar};

fn main() -> Result<(), ModelError> {
    let a = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
    let b = DenseVector::new([ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?])?;

    let dot_product = (&a * &b)?;

    let matrix = DenseMatrix::from_rows([
        [ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?],
        [ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?],
    ])?;
    let vector = DenseVector::new([ModelScalar::try_from(5.0)?, ModelScalar::try_from(6.0)?])?;

    let output = (&matrix * &vector)?;

    println!("dot product = {dot_product}");
    println!("matrix-vector output width = {}", output.len());
    Ok(())
}
```

## Why This Matters

The dot product is one of the main bridges between "math-looking ML" and "loop-looking Rust."

Later, a neuron uses a dot product.
Later still, attention uses dot products too.

The surrounding architecture changes. The arithmetic family does not.

## Concept Trace

- **Object/newtype:** dot products later connect `FeatureVector`, `WeightVector`, `Query`, `Key`, and `AttentionScore`.
- **Invariant:** paired vectors must have compatible widths before multiplication and summation.
- **Map:** aligned vectors -> weighted evidence -> one scalar score.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair`.
- **Failure signal:** your result has the wrong shape because you did not reduce pairwise products into one score.

## Short Practice

1. Compute $[2, 1] \cdot [4, 3]$ by hand.
2. If a matrix has 3 rows, how long is the output of `mat_vec_mul`?
3. Explain matrix-vector multiplication in one sentence without using the word "matrix."
