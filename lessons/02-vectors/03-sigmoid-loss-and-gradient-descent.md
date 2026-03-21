# Lesson 3: Sigmoid, Loss, and Gradient Descent

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
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

let loss = (y_hat - y) * (y_hat - y);

w = w - learning_rate * d_loss_d_w;
```

## Why This Matters

These three lines are the smallest version of learning:

- turn a raw score into an output
- compare output to truth
- update the parameter so the future output is less wrong

That is the backbone of the neuron module that comes next.

## Short Practice

1. What is $\sigma(0)$?
2. If $\hat{y} = 0.8$ and $y = 1.0$, what is the squared error?
3. Read `w = w - learning_rate * d_loss_d_w` aloud in ordinary English.
