# Foundations Solutions

## Solution 1: One sentence, three forms

Algebra:

```text
z = w1x1 + w2x2 + b
```

Rust:

```rust
let z = w1 * x1 + w2 * x2 + b;
```

## Solution 2: Fix the indexing mismatch

```rust
let x = vec![10.0, 20.0, 30.0];

let x1 = x[0];
let x3 = x[2];
```

Math starts at 1. Rust starts at 0.

## Solution 3: Read the notation

1. `x^(2)` usually means "training example number two" when the parentheses are part of the notation.
2. `sum_{i=1}^n a_i` means "add all the `a_i` terms from the first one to the n-th one."
3. `y_hat` means "predicted `y`."
4. `partial L / partial w` means "how much the loss changes when `w` changes a little."

## Solution 4: Write a dot product

```rust
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}
```

## Solution 5: Model as a struct

```rust
struct Neuron {
    w1: f64,
    w2: f64,
    b: f64,
}

impl Neuron {
    fn forward(&self, x1: f64, x2: f64) -> f64 {
        self.w1 * x1 + self.w2 * x2 + self.b
    }
}
```

The important idea is not the syntax alone. The important idea is that the struct holds the parameters, and the method expresses the arithmetic.
