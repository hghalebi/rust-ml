# A Training Step Is A Feedback Story

## Overview

A training step is the smallest unit of learning.

It has one job: use one example to make the parameters slightly better for next time.

The story is:

```text
predict -> measure -> adjust
```

This is the same as the neuron story, but now the backward direction is part of the main loop.

## Learning Goals

- explain a training step in plain English
- connect loss to parameter updates
- read the update rule as a typed operation
- explain why the target is not changed during learning
- identify `LearningRate`, `Gradient`, and `Adjustment` as different roles

## Plain-English Explanation

### The target is the teacher, not the student

When the model is wrong, we do not change the target.

The target is the answer we want the model to approach. The parameters are the student. The update changes the student, not the answer.

### Loss is feedback

Loss says how wrong the prediction was.

But loss alone is not enough. It says "wrong" without saying which parameter should move.

The gradient gives direction. It says how the loss changes when a parameter changes.

### Learning rate controls step size

The learning rate answers a practical question:

```text
how much should we trust this gradient right now?
```

A large step can move fast but overshoot. A small step can be stable but slow.

## Algebra Form

One training step uses:

```math
\hat{y} = model(x)
```

```math
L = (\hat{y} - y)^2
```

```math
w := w - \eta \frac{dL}{dw}
```

Read the update rule as:

```text
new weight = old weight - learning rate * gradient
```

The category-theory lens is:

```text
the model is a map, and training changes the parameters inside that map
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

    let step = neuron.train_one_step(&example, LearningRate::try_from(0.5)?)?;

    println!("prediction before = {:.4}", step.prediction_before());
    println!("loss before       = {:.4}", step.loss_before());
    println!("bias gradient     = {:.4}", step.gradients().bias());
    println!("loss after        = {:.4}", step.loss_after());
    Ok(())
}
```

The code separates `LearningRate`, `Gradient`, `Adjustment`, and `Weight` because they mean different things.

## Why This Matters

Without separate roles, training code turns into raw arithmetic:

```text
w = w - a * b
```

That line runs, but it does not teach.

Typed roles let the learner see the update as a meaningful transformation:

```text
(LearningRate, Gradient) -> Adjustment
Weight -> Weight
```

## Concept Trace

- **Object/newtype:** `Loss`, `Gradient`, `LearningRate`, `Adjustment`, `Weight`, and `Bias`.
- **Invariant:** feedback values are not parameters, and step-size controls are not gradients.
- **Map:** prediction error -> gradient -> adjustment -> updated parameter.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training`.
- **Failure signal:** you can report that loss changed but cannot explain which typed value caused the parameter update.

## Short Practice

1. Why does the target stay fixed during a training step?
2. What is the difference between a gradient and an adjustment?
3. If the gradient is negative, why can subtracting it increase the weight?
4. In category-theory language, what map is changed during training?
