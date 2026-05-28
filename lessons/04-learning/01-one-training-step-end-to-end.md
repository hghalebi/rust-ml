# One Training Step, End to End

## Overview

This lesson turns the neuron into a full learning step:

- make a prediction
- measure the loss
- compute gradients
- update the parameters

The point is not to memorize formulas in isolation. The point is to see the
whole loop once, end to end, without hand-waving.

## Learning Goals

- explain the four stages of one training step
- compute a squared-error loss for one example
- trace how the loss produces gradients for `w1`, `w2`, and `b`
- read an SGD update in Rust as a direct translation of the algebra

## Plain-English Explanation

### Forward pass first, always

Training begins with the same thing inference begins with: a forward pass.

The neuron takes inputs and parameters and turns them into a prediction. That
prediction is just a claim. It is not yet a learning signal.

### Loss turns a claim into a complaint

The loss compares the prediction to the target.

If the prediction is close to the target, the complaint is small. If the
prediction is far away, the complaint is large.

Loss is what makes "wrong" measurable.

### Backward pass turns the complaint into blame

Once we know how wrong the prediction was, we ask a more useful question:

which parameter contributed how much to that wrongness?

That is the backward pass. It turns one scalar complaint into one gradient per
parameter.

### The optimizer turns gradients into motion

The optimizer is the rule that actually changes the parameters.

For plain stochastic gradient descent, the rule is simple:

- if a gradient says "moving this parameter upward would increase loss"
- then move the parameter downward a little

That "a little" is controlled by the learning rate.

## Algebra Form

Forward pass:

```math
z = w_1x_1 + w_2x_2 + b
```

```math
\hat{y} = \sigma(z)
```

Loss:

```math
L = (\hat{y} - y)^2
```

Backward pass:

```math
\frac{dL}{dw_1}
=
\frac{dL}{d\hat{y}}
\cdot
\frac{d\hat{y}}{dz}
\cdot
\frac{dz}{dw_1}
```

SGD update:

```math
w := w - \eta \frac{dL}{dw}
```

## Rust Form

```rust
#[derive(Debug, Clone, Copy)]
struct Input(f64);

#[derive(Debug, Clone, Copy)]
struct Weight(f64);

#[derive(Debug, Clone, Copy)]
struct Bias(f64);

#[derive(Debug, Clone, Copy)]
struct Target(f64);

#[derive(Debug, Clone, Copy)]
struct Gradients {
    d_w1: f64,
    d_w2: f64,
    d_b: f64,
}

#[derive(Debug, Clone, Copy)]
struct Neuron {
    w1: Weight,
    w2: Weight,
    b: Bias,
}

impl Neuron {
    fn pre_activation(&self, x1: Input, x2: Input) -> f64 {
        self.w1.0 * x1.0 + self.w2.0 * x2.0 + self.b.0
    }

    fn predict(&self, x1: Input, x2: Input) -> f64 {
        let z = self.pre_activation(x1, x2);
        1.0 / (1.0 + (-z).exp())
    }

    fn backward(&self, x1: Input, x2: Input, target: Target) -> Gradients {
        let prediction = self.predict(x1, x2);
        let d_loss_d_prediction = 2.0 * (prediction - target.0);
        let d_prediction_d_z = prediction * (1.0 - prediction);
        let upstream = d_loss_d_prediction * d_prediction_d_z;

        Gradients {
            d_w1: upstream * x1.0,
            d_w2: upstream * x2.0,
            d_b: upstream,
        }
    }
}

fn squared_error(prediction: f64, target: Target) -> f64 {
    (prediction - target.0).powi(2)
}

fn main() {
    let x1 = Input(1.0);
    let x2 = Input(0.0);
    let target = Target(1.0);
    let learning_rate = 0.5;

    let mut neuron = Neuron {
        w1: Weight(0.8),
        w2: Weight(-0.4),
        b: Bias(0.1),
    };

    let before = neuron.predict(x1, x2);
    let before_loss = squared_error(before, target);
    let gradients = neuron.backward(x1, x2, target);

    neuron.w1.0 -= learning_rate * gradients.d_w1;
    neuron.w2.0 -= learning_rate * gradients.d_w2;
    neuron.b.0 -= learning_rate * gradients.d_b;

    let after = neuron.predict(x1, x2);
    let after_loss = squared_error(after, target);

    println!(
        "before={before:.4} loss={before_loss:.4} | after={after:.4} loss={after_loss:.4}"
    );
}
```

## Why This Matters

Once one training step is clear, the bigger training loop stops being mystical.

An epoch is not a fundamentally new idea. It is just this one step repeated
across many examples.

## Short Practice

1. In one sentence, what new job does the loss perform that the forward pass does not?
2. Why does the optimizer need a learning rate instead of applying the full gradient directly?
3. If `x2 = 0`, what happens immediately to `dL/dw2` in this example?
