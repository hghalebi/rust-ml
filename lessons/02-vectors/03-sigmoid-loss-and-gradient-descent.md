# Sigmoid, Loss, and Gradient Descent

## Overview

This lesson adds three essential ideas:

- a nonlinear activation
- a numerical measure of wrongness
- an update rule for improving parameters

## Learning Goals

- explain what sigmoid does
- compute a simple squared-error loss
- read a gradient-descent update as a sentence

## Plain-English Explanation

### Sigmoid

Sigmoid takes any real number and squashes it into the range from 0 to 1.

That makes it useful when you want output that behaves like a score or probability-like value.

### Loss

Loss is a number that says how wrong the model is.

A simple one-example loss is squared error:

- prediction minus truth
- then square the difference

### Gradient descent

If changing a parameter makes the loss go up, move in the opposite direction.

That is the intuition behind:

- compute the gradient
- step against it

## Algebra Form

Sigmoid:

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

Loss:

```math
L = (\hat{y} - y)^2
```

Gradient descent:

```math
w \leftarrow w - \eta \frac{\partial L}{\partial w}
```

## Rust Form

```rust
use rust_ml_neuron::{FeatureVector, InputValue, LearningRate, Target, TinyNeuron, TrainingExample};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let mut neuron = TinyNeuron::lesson_seed()?;
    let example = TrainingExample::new(
        FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
        Target::try_from(1.0)?,
    );

    let loss_before = neuron.loss(&example)?;
    let step = neuron.train_one_step(&example, LearningRate::try_from(0.5)?)?;

    println!("loss before = {loss_before:.4}");
    println!("loss after  = {:.4}", step.loss_after());
    Ok(())
}
```

## Why This Matters

These three lines are the smallest version of learning:

- turn a raw score into an output
- compare output to truth
- update the parameter so the future output is less wrong

That is the backbone of the neuron module that comes next.

## Concept Trace

- **Object/newtype:** `PreActivation`, `Prediction`, `Target`, `Loss`, `Gradient`, and `LearningRate`.
- **Invariant:** predictions stay in a valid range, and learning rates must be positive.
- **Map:** score -> prediction -> loss -> parameter adjustment.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training`.
- **Failure signal:** you can compute loss but cannot explain which value tells the parameter how to move.

## Short Practice

1. What is $\sigma(0)$?
2. If $\hat{y} = 0.8$ and $y = 1.0$, what is the squared error?
3. Read `w = w - learning_rate * d_loss_d_w` aloud in ordinary English.
