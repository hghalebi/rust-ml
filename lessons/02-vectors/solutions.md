# Vectors Solutions

## Solution 1: Classify the shape

1. `3.14` is a scalar.
2. `[1.0, 2.0, 3.0]` is a vector.
3. `[[1.0, 2.0], [3.0, 4.0]]` is a matrix.

## Solution 2: Dot product by hand

```math
[1, 2, 3] \cdot [4, 5, 6] = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32
```

Rust:

```rust
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}
```

## Solution 3: Matrix-vector multiplication by hand

```math
Wx = \begin{bmatrix} 2 \cdot 4 + 0 \cdot 5 \\ 1 \cdot 4 + 3 \cdot 5 \end{bmatrix}
= \begin{bmatrix} 8 \\ 19 \end{bmatrix}
```

## Solution 4: Sigmoid and loss

1. $\sigma(0) = 0.5$
2. Squared error:

```math
(0.25 - 1.0)^2 = (-0.75)^2 = 0.5625
```

## Solution 5: Read the update

Ordinary English:

> Replace `w` with its old value minus a small step in the direction of the loss gradient.

That is what "move opposite the gradient" means in code.
