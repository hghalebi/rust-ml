# Lesson 5: A Neuron as a Chain of Functions

## Overview

This lesson rewrites the one-neuron model in the most teachable shape possible:

- one function for the weighted sum
- one function for the activation
- one function for the loss
- one backward path that traces responsibility through that chain

The metaphor is a tiny factory with three rooms:

```text
inputs + parameters
        ↓
   [Mixing room]
        ↓
         z
        ↓
   [Shaping room]
        ↓
       y_hat
        ↓
    [Judge room]
        ↓
         L
```

## Learning Goals

- explain the neuron as `loss(activation(pre_activation(...)))`
- compute and interpret `z`, `\hat{y}`, and `L`
- explain backpropagation as structured blame assignment through functions
- explain why `dz/dw1 = x1`, `dz/dw2 = x2`, and `dz/db = 1`
- read a full training step in Rust without losing the chain-rule story

## Plain-English Explanation

### The tiny factory

Imagine a tiny factory with three rooms:

1. mixing room
   takes inputs and parameters and produces a raw score `z`
2. shaping room
   turns `z` into a prediction between `0` and `1`
3. judge room
   compares the prediction to the true answer and assigns a penalty

That is the forward pass:

- mix
- squash
- judge

### The chain of functions view

The neuron is easier to reason about if we treat it as a composition of functions rather than as a random pile of variables:

- `pre_activation` makes `z`
- `activation` turns `z` into `y_hat`
- `loss` compares `y_hat` to `y`

So the real shape is:

```text
loss(activation(pre_activation(...)))
```

### Backpropagation as structured blame

For a parameter such as `w1`, the dependency path is:

```text
w1 -> pre_activation -> activation -> loss
```

Backpropagation is just tracing that path backward.

The complaint from the judge room travels backward through the factory, telling each earlier room how much it contributed to the mistake.

That is why the gradient factors multiply:

- how much loss changes with prediction
- how much prediction changes with pre-activation
- how much pre-activation changes with the parameter

### Cleaner Rust with operator traits

The noisy version of this lesson uses `.0` constantly because every newtype wraps an `f64`.

For the teaching code, a better compromise is:

- keep the semantic wrappers
- overload the specific operators that are mathematically valid
- avoid `Deref<Target = f64>`, because it makes every type feel raw again

That gives us readable lines such as:

```text
x1 * w1 + x2 * w2 + b
prediction - target
self.w1 = self.w1 - lr * d_loss_d_w1
```

The wrappers still mean something. The code just stops shouting about their storage layout.

### Why `dz/dw1 = x1`

Start from:

```math
z = w_1x_1 + w_2x_2 + b
```

If only `w1` changes, the only relevant term is `w1x1`.

So the rate of change is just:

```math
\frac{dz}{dw_1} = x_1
```

Metaphor: if `w1` is a volume knob and `x1` is the incoming sound, the effect of turning the knob depends on how much signal is already there.

No signal, no effect.

## Algebra Form

Forward pass:

```math
z = w_1x_1 + w_2x_2 + b
```

```math
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
```

```math
L = (\hat{y} - y)^2
```

Whole function chain:

```math
L(\hat{y}(z(w_1, w_2, b)))
```

Backward pass for `w1`:

```math
\frac{dL}{dw_1}
=
\frac{dL}{d\hat{y}}
\cdot
\frac{d\hat{y}}{dz}
\cdot
\frac{dz}{dw_1}
```

Backward pass for the local pieces:

```math
\frac{dz}{dw_1} = x_1,\quad
\frac{dz}{dw_2} = x_2,\quad
\frac{dz}{db} = 1
```

Parameter update:

```math
w := w - \eta \frac{dL}{dw}
```

## Rust Form

```rust
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy)]
struct Input(f64);

#[derive(Debug, Clone, Copy)]
struct Weight(f64);

#[derive(Debug, Clone, Copy)]
struct Bias(f64);

#[derive(Debug, Clone, Copy)]
struct Prediction(f64);

#[derive(Debug, Clone, Copy)]
struct Target(f64);

impl Mul<Weight> for Input {
    type Output = f64;

    fn mul(self, rhs: Weight) -> Self::Output {
        self.0 * rhs.0
    }
}

impl Add<Bias> for f64 {
    type Output = f64;

    fn add(self, rhs: Bias) -> Self::Output {
        self + rhs.0
    }
}

impl Sub<Target> for Prediction {
    type Output = f64;

    fn sub(self, rhs: Target) -> Self::Output {
        self.0 - rhs.0
    }
}

impl From<Prediction> for f64 {
    fn from(value: Prediction) -> Self {
        value.0
    }
}

fn pre_activation(x1: Input, x2: Input, w1: Weight, w2: Weight, b: Bias) -> f64 {
    x1 * w1 + x2 * w2 + b
}

fn activation(z: f64) -> Prediction {
    Prediction(1.0 / (1.0 + (-z).exp()))
}

fn loss(prediction: Prediction, target: Target) -> f64 {
    (prediction - target).powi(2)
}

fn main() {
    let target = Target(1.0);
    let z = pre_activation(Input(1.0), Input(0.0), Weight(0.8), Weight(-0.4), Bias(0.1));
    let y_hat = activation(z);
    let loss_value = loss(y_hat, target);

    println!(
        "z = {:.4}, y_hat = {:.4}, loss = {:.4}",
        z,
        f64::from(y_hat),
        loss_value
    );
}
```

```rust
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy)]
struct Input(f64);

#[derive(Debug, Clone, Copy)]
struct Weight(f64);

#[derive(Debug, Clone, Copy)]
struct Bias(f64);

#[derive(Debug, Clone, Copy)]
struct Prediction(f64);

#[derive(Debug, Clone, Copy)]
struct Target(f64);

#[derive(Debug, Clone, Copy)]
struct LearningRate(f64);

#[derive(Debug, Clone, Copy)]
struct Gradient(f64);

#[derive(Debug, Clone, Copy)]
struct Adjustment(f64);

#[derive(Debug, Clone, Copy)]
struct Neuron {
    w1: Weight,
    w2: Weight,
    b: Bias,
}

impl Mul<Weight> for Input {
    type Output = f64;

    fn mul(self, rhs: Weight) -> Self::Output {
        self.0 * rhs.0
    }
}

impl Add<Bias> for f64 {
    type Output = f64;

    fn add(self, rhs: Bias) -> Self::Output {
        self + rhs.0
    }
}

impl Sub<Target> for Prediction {
    type Output = f64;

    fn sub(self, rhs: Target) -> Self::Output {
        self.0 - rhs.0
    }
}

impl From<Input> for f64 {
    fn from(value: Input) -> Self {
        value.0
    }
}

impl From<Prediction> for f64 {
    fn from(value: Prediction) -> Self {
        value.0
    }
}

impl Mul<Gradient> for LearningRate {
    type Output = Adjustment;

    fn mul(self, rhs: Gradient) -> Self::Output {
        Adjustment(self.0 * rhs.0)
    }
}

impl Sub<Adjustment> for Weight {
    type Output = Weight;

    fn sub(self, rhs: Adjustment) -> Self::Output {
        Weight(self.0 - rhs.0)
    }
}

impl Sub<Adjustment> for Bias {
    type Output = Bias;

    fn sub(self, rhs: Adjustment) -> Self::Output {
        Bias(self.0 - rhs.0)
    }
}

fn pre_activation(x1: Input, x2: Input, w1: Weight, w2: Weight, b: Bias) -> f64 {
    x1 * w1 + x2 * w2 + b
}

fn activation(z: f64) -> Prediction {
    Prediction(1.0 / (1.0 + (-z).exp()))
}

fn sigmoid_derivative_from_output(output: f64) -> f64 {
    output * (1.0 - output)
}

fn loss(prediction: Prediction, target: Target) -> f64 {
    (prediction - target).powi(2)
}

impl Neuron {
    fn forward(&self, x1: Input, x2: Input) -> Prediction {
        let z = pre_activation(x1, x2, self.w1, self.w2, self.b);
        activation(z)
    }

    fn train_one_step(
        &mut self,
        x1: Input,
        x2: Input,
        target: Target,
        lr: LearningRate,
    ) -> f64 {
        let z = pre_activation(x1, x2, self.w1, self.w2, self.b);
        let prediction = activation(z);
        let current_loss = loss(prediction, target);

        let d_loss_d_prediction = 2.0 * (prediction - target);
        let d_prediction_d_pre_activation =
            sigmoid_derivative_from_output(f64::from(prediction));

        let d_pre_activation_d_w1 = f64::from(x1);
        let d_pre_activation_d_w2 = f64::from(x2);
        let d_pre_activation_d_b = 1.0;

        let d_loss_d_w1 = Gradient(
            d_loss_d_prediction
            * d_prediction_d_pre_activation
            * d_pre_activation_d_w1,
        );

        let d_loss_d_w2 = Gradient(
            d_loss_d_prediction
            * d_prediction_d_pre_activation
            * d_pre_activation_d_w2,
        );

        let d_loss_d_b = Gradient(
            d_loss_d_prediction
            * d_prediction_d_pre_activation
            * d_pre_activation_d_b,
        );

        self.w1 = self.w1 - lr * d_loss_d_w1;
        self.w2 = self.w2 - lr * d_loss_d_w2;
        self.b = self.b - lr * d_loss_d_b;

        current_loss
    }
}

fn main() {
    let mut neuron = Neuron {
        w1: Weight(0.5),
        w2: Weight(-0.3),
        b: Bias(0.1),
    };

    let before = neuron.forward(Input(1.0), Input(0.0));
    let loss_before = neuron.train_one_step(
        Input(1.0),
        Input(0.0),
        Target(1.0),
        LearningRate(0.1),
    );
    let after = neuron.forward(Input(1.0), Input(0.0));

    println!(
        "before={:.4}, loss={:.4}, after={:.4}",
        f64::from(before),
        loss_before,
        f64::from(after)
    );
}
```

## Why This Matters

The chain-of-functions view is the important upgrade.

Once you see the neuron as:

```text
mix -> squash -> judge
```

and the backward pass as:

```text
blame -> trace -> adjust
```

backpropagation stops looking mystical. It becomes disciplined bookkeeping over a composition of functions.

That is the right mental model before you move to larger networks.

## Short Practice

1. Write the dependency path from `w2` to the final loss in words.
2. If `x1 = 0`, what happens to `dz/dw1`, and what does that imply for the update?
3. Read this line aloud in ordinary English: `self.w1 = self.w1 - lr * d_loss_d_w1;`
