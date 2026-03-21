# Lesson 1: Scalars, Vectors, and Matrices

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
let x: f64 = 2.5;

let v: Vec<f64> = vec![1.0, 2.0, 3.0];

let w: Vec<Vec<f64>> = vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
];
```

## Why This Matters

The word "shape" matters because ML computations only make sense when the structures line up.

If you know:

- vector length
- matrix rows
- matrix columns

then you can predict what an operation is allowed to do.

## Short Practice

1. Is `[1.0, 2.0, 3.0]` a scalar, vector, or matrix?
2. How many rows and columns does this matrix have?

```rust
let w = vec![
    vec![0.1, 0.2, 0.3],
    vec![0.4, 0.5, 0.6],
];
```
