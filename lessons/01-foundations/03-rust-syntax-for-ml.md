# Rust Syntax for ML

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
- growable vector storage is an implementation detail; in this course, learner-facing ML values are wrapped in semantic types such as `FeatureVector`

### Structs

A struct groups related data into one model-shaped object.

### Methods and `impl`

Methods attach behavior to a struct.

- use `&self` when reading the model
- use `&mut self` when training updates the model

### Loops

Loops are how we spell repeated arithmetic in ordinary beginner Rust.

## Algebra Form

```math
\begin{aligned}
w &\leftarrow w - \eta \frac{\partial L}{\partial w} \\
\text{neuron} &= \{w_1, w_2, b\} \\
z &= w_1 x_1 + w_2 x_2 + b
\end{aligned}
```

## Rust Form

```rust
use rust_ml_neuron::{Bias, FeatureVector, InputValue, TinyNeuron, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let features = FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(2.0)?);
    let weights = WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?);
    let mut neuron = TinyNeuron::new(weights, Bias::try_from(0.1)?);

    let first_prediction = neuron.predict(&features)?;

    neuron = TinyNeuron::new(
        WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?),
        Bias::try_from(0.0)?,
    );

    for value in features.values() {
        println!("feature = {value}");
    }

    println!("before bias update = {first_prediction:.4}");
    println!("after bias update  = {:.4}", neuron.predict(&features)?);
    Ok(())
}
```

## Why This Matters

Rust is a strict lab partner.

That strictness is useful for ML because it makes you be explicit about:

- what data belongs together
- what can change
- what only gets read
- what the repeated computation actually is

## Concept Trace

- **Object/newtype:** raw syntax introduces the storage shape that later becomes semantic types such as `FeatureVector`, `Weight`, and `LearningRate`.
- **Invariant:** values that can change during training should be separated from values that are only read during prediction.
- **Map:** struct state + method inputs -> method output.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass`.
- **Failure signal:** you can write a struct but cannot explain which fields are parameters or why `forward` should not mutate them.

## Short Practice

1. Which fields in a `Neuron` struct should be mutable during training?
2. Why does `forward` usually take `&self` instead of `&mut self`?
3. Rewrite "sum all values in a vector" as a `for` loop.
