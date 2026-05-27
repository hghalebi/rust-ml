# The Single Most Important Idea

## Overview

Machine learning can look mysterious because the notation is compact. The underlying loop is much simpler than it looks.

## Learning Goals

- say the basic ML loop in one sentence
- recognize the same idea in English, algebra, and Rust
- understand why this course keeps translating between those three forms

## Plain-English Explanation

Take some numbers in, do arithmetic with parameters, get an output, measure how wrong it is, then update the parameters to be less wrong.

That sentence explains a surprising amount of machine learning.

Here is a tiny version of that loop:

- inputs: `x1`, `x2`
- parameters: `w1`, `w2`, `b`
- output: `z`

In words:

- multiply each input by a weight
- add the results together
- add a bias

## Algebra Form

```math
z = w_1 x_1 + w_2 x_2 + b
```

This says the same thing in compressed mathematical form.

## Rust Form

```rust
use rust_ml_neuron::{Bias, FeatureVector, InputValue, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let inputs = FeatureVector::two(InputValue::try_from(2.0)?, InputValue::try_from(4.0)?);
    let weights = WeightVector::two(Weight::try_from(1.0)?, Weight::try_from(3.0)?);
    let bias = Bias::try_from(5.0)?;

    let z = ((&inputs * &weights)? + bias)?;

    println!("score = {z}");
    Ok(())
}
```

This says the same thing in executable form.

## Why This Matters

The rest of the course is mostly this one skill repeated:

- read a formula
- ask what the inputs are
- ask what the parameters are
- ask what arithmetic is happening
- write the same thing as Rust

If you can do that, ML stops looking magical.

## Concept Trace

- **Object/newtype:** later code names the roles as `InputValue`, `Weight`, `Bias`, and `PreActivation`.
- **Invariant:** input values, parameters, and score roles should not be mentally interchangeable.
- **Map:** `inputs + parameters -> score`.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum`.
- **Failure signal:** you can write `w1 * x1 + w2 * x2 + b` but cannot say which values are data and which values are learned.

## Short Practice

Translate this sentence into algebra and then typed Rust:

> Take two inputs, weight them, add a bias, and compute a score.

Expected target shape:

```math
z = \dots
```

```text
let z = ((&inputs * &weights)? + bias)?;
```
