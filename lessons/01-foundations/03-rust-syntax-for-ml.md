# Lesson 3: Rust Syntax for ML

## Overview

You do not need all of Rust before starting ML. You need a small, practical subset.

## Learning Goals

- read the Rust syntax used in the next modules
- know when data should be mutable
- recognize the role of functions, structs, methods, and loops in model code

## Plain-English Explanation

These are the core building blocks we will use:

### Variables and mutability

- `let` creates a binding
- `mut` means the value can change

Weights change during training, so parameters are often mutable.

### Functions

Functions turn named inputs into outputs.

### Arrays and vectors

- arrays have fixed size
- `Vec<T>` is growable and common in beginner ML code

### Structs

A struct groups related data into one model-shaped object.

### Methods and `impl`

Methods attach behavior to a struct.

- use `&self` when reading the model
- use `&mut self` when training updates the model

### Loops

Loops are how we spell repeated arithmetic in ordinary beginner Rust.

## Algebra Form

```text
parameter update:
w := w - eta * (partial L / partial w)

model structure:
neuron = {w1, w2, b}

forward pass:
z = w1x1 + w2x2 + b
```

## Rust Form

```rust
let x = 3.0;
let mut y = 5.0;
y = y + 1.0;

fn add(a: f64, b: f64) -> f64 {
    a + b
}

let pair: [f64; 2] = [1.0, 2.0];
let values: Vec<f64> = vec![1.0, 2.0, 3.0];

struct Neuron {
    w1: f64,
    w2: f64,
    b: f64,
}

impl Neuron {
    fn forward(&self, x1: f64, x2: f64) -> f64 {
        self.w1 * x1 + self.w2 * x2 + self.b
    }

    fn update_bias(&mut self, step: f64) {
        self.b = self.b - step;
    }
}

for i in 0..values.len() {
    println!("{}", values[i]);
}
```

## Why This Matters

Rust is a strict lab partner.

That strictness is useful for ML because it makes you be explicit about:

- what data belongs together
- what can change
- what only gets read
- what the repeated computation actually is

## Short Practice

1. Which fields in a `Neuron` struct should be mutable during training?
2. Why does `forward` usually take `&self` instead of `&mut self`?
3. Rewrite "sum all values in a vector" as a `for` loop.
