# Backpropagation as Local Gradient Bookkeeping

## Overview

Backpropagation sounds large and dramatic, but the local mechanism is simple:

- every function contributes one local derivative
- those local derivatives multiply along the dependency path
- the result is the gradient for the parameter we care about

This lesson zooms in on that bookkeeping step.

## Learning Goals

- explain backpropagation as chained local derivatives
- identify the local derivative contributed by each stage of the neuron
- distinguish upstream gradient from local gradient
- read a Rust function that computes gradient factors explicitly

## Plain-English Explanation

### Think locally, not globally

The whole loss depends on each parameter, but each function in the chain only
knows about its immediate local relationship.

That is a feature, not a weakness.

Backpropagation works because each stage can contribute its local derivative,
then pass the blame signal backward to the stage before it.

### Upstream blame times local sensitivity

When the gradient reaches a stage, that stage asks:

- how much blame already arrived from later in the chain?
- how sensitive is my output to my input right here?

Multiply those together and you get the new upstream signal for the earlier stage.

### Why this scales

The neuron is tiny, but the same pattern survives in deeper models:

- activation layers
- matrix multiplies
- attention blocks
- output projections

The graph gets bigger, but the local rule stays the same.

## Algebra Form

For the neuron:

```math
L(\hat{y}(z(w_1, w_2, b)))
```

The backward path for `w1` is:

```math
\frac{dL}{dw_1}
=
\frac{dL}{d\hat{y}}
\cdot
\frac{d\hat{y}}{dz}
\cdot
\frac{dz}{dw_1}
```

Local pieces:

```math
\frac{dL}{d\hat{y}} = 2(\hat{y} - y)
```

```math
\frac{d\hat{y}}{dz} = \hat{y}(1 - \hat{y})
```

```math
\frac{dz}{dw_1} = x_1,\quad
\frac{dz}{dw_2} = x_2,\quad
\frac{dz}{db} = 1
```

## Rust Form

```rust
#[derive(Debug, Clone, Copy)]
struct GradientBreakdown {
    d_loss_d_prediction: f64,
    d_prediction_d_z: f64,
    d_z_d_w1: f64,
    d_z_d_w2: f64,
    d_z_d_b: f64,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn gradient_breakdown(x1: f64, x2: f64, y_hat: f64, target: f64) -> GradientBreakdown {
    GradientBreakdown {
        d_loss_d_prediction: 2.0 * (y_hat - target),
        d_prediction_d_z: y_hat * (1.0 - y_hat),
        d_z_d_w1: x1,
        d_z_d_w2: x2,
        d_z_d_b: 1.0,
    }
}

fn main() {
    let x1 = 1.0;
    let x2 = 0.5;
    let z = 0.8 * x1 + (-0.4) * x2 + 0.1;
    let y_hat = sigmoid(z);
    let target = 1.0;

    let parts = gradient_breakdown(x1, x2, y_hat, target);
    let upstream = parts.d_loss_d_prediction * parts.d_prediction_d_z;
    let d_loss_d_w1 = upstream * parts.d_z_d_w1;
    let d_loss_d_w2 = upstream * parts.d_z_d_w2;
    let d_loss_d_b = upstream * parts.d_z_d_b;

    println!(
        "dw1={d_loss_d_w1:.6}, dw2={d_loss_d_w2:.6}, db={d_loss_d_b:.6}"
    );
}
```

## Why This Matters

If you can separate upstream blame from local sensitivity, larger training
systems stop looking like symbol soup.

That is the mental move that survives all the way from one neuron to next-token
prediction in sequence models.

## Concept Trace

- **Object/newtype:** `PredictionError`, `Gradient`, `NeuronGradients`, `Adjustment`
- **Invariant:** each gradient factor must belong to the local map that produced it
- **Map:** `Loss blame -> local derivative -> upstream gradient -> parameter gradient`
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training`
- **Failure signal:** a gradient appears without a named local derivative or upstream path

## Short Practice

1. Which local derivative comes from the activation stage?
2. Why is `dz/db = 1` instead of depending on the inputs?
3. In words, what does the upstream gradient mean before it reaches `w1`?
