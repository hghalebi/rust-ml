# Lesson 2: Sum, Dot Product, and Matrix-Vector Multiplication

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

- `a = [1, 2]`
- `b = [3, 4]`

then:

- first pair: `1 * 3`
- second pair: `2 * 4`
- total: `11`

### Matrix-vector multiplication

Each row of the matrix takes a dot product with the vector.

That means the output has one number per row.

## Algebra Form

```text
sum_{i=1}^n a_i

a . b = sum_{i=1}^n a_i b_i

W x
```

Example:

```text
W = [ [1, 2],
      [3, 4] ]

x = [5, 6]

W x = [17, 39]
```

## Rust Form

```rust
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

fn mat_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let mut result = vec![0.0; rows];

    for r in 0..rows {
        let mut sum = 0.0;

        for c in 0..vector.len() {
            sum += matrix[r][c] * vector[c];
        }

        result[r] = sum;
    }

    result
}
```

## Why This Matters

The dot product is one of the main bridges between "math-looking ML" and "loop-looking Rust."

Later, a neuron uses a dot product.
Later still, attention uses dot products too.

The surrounding architecture changes. The arithmetic family does not.

## Short Practice

1. Compute `dot([2, 1], [4, 3])` by hand.
2. If a matrix has 3 rows, how long is the output of `mat_vec_mul`?
3. Explain matrix-vector multiplication in one sentence without using the word "matrix."
