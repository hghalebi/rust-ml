# A Neuron as a Chain of Functions

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

The raw-storage version of this lesson peels values apart constantly. That
teaches the computer's storage detail instead of the learner's mental model.

The crate version uses a better compromise:

- keep the semantic wrappers
- overload only the specific operators that are mathematically valid
- return checked errors when an operation can fail
- avoid `Deref<Target = f64>`, because it makes every type feel raw again

That gives us readable typed lines such as:

```text
&features * &weights
weighted_sum + bias
prediction - target
self.w1 = self.w1 - lr * d_loss_d_w1
```

The wrappers still mean something. The code just reads like typed composition.

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
use rust_ml_neuron::{
    Bias, FeatureVector, InputValue, Target, TinyNeuron, TrainingExample, Weight, WeightVector,
};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let neuron = TinyNeuron::new(
        WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?),
        Bias::try_from(0.1)?,
    );
    let example = TrainingExample::new(
        FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
        Target::try_from(1.0)?,
    );

    let z = neuron.raw_score(example.features())?;
    let prediction = neuron.predict(example.features())?;
    let loss = neuron.loss(&example)?;

    println!("z = {z:.4}");
    println!("prediction = {prediction:.4}");
    println!("loss = {loss:.4}");

    Ok(())
}
```

```rust
use rust_ml_neuron::{
    FeatureVector, InputValue, LearningRate, Target, TinyNeuron, TrainingExample,
};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let mut neuron = TinyNeuron::lesson_seed()?;
    let example = TrainingExample::new(
        FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
        Target::try_from(1.0)?,
    );

    let before = neuron.predict(example.features())?;
    let step = neuron.train_one_step(&example, LearningRate::try_from(0.1)?)?;
    let after = neuron.predict(example.features())?;

    println!("before={before:.4}");
    println!("loss before={:.4}", step.loss_before());
    println!("loss after={:.4}", step.loss_after());
    println!("after={after:.4}");

    Ok(())
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

## Concept Trace

- **Object/newtype:** `FeatureVector`, `WeightedSum`, `Prediction`, `Loss`, `Gradient`, and `Adjustment`.
- **Invariant:** the forward chain and feedback chain must preserve each role instead of collapsing into anonymous numbers.
- **Map:** `FeatureVector -> WeightedSum -> Prediction -> Loss`, then feedback maps loss back to parameter adjustments.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training`.
- **Failure signal:** you can say "backpropagation" but cannot name the path from one parameter to the loss.

## Short Practice

1. Write the dependency path from `w2` to the final loss in words.
2. If `x1 = 0`, what happens to `dz/dw1`, and what does that imply for the update?
3. Read this line aloud in ordinary English: `self.w1 = self.w1 - lr * d_loss_d_w1;`
