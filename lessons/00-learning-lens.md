# The Learning Lens

Before the first module, learn the lens this repository uses.

Machine learning often feels hard because one idea appears in too many forms:

- a sentence
- a formula
- a code block
- a diagram

This course treats those as translations of the same structure.

```text
plain English <-> algebra <-> Rust newtypes <-> composable maps
```

## What You Already Know

You already know that programs transform inputs into outputs.

You also know that Rust uses types to distinguish kinds of values.

That is enough to start reading ML structurally.

## One Tiny Story

A neuron does three things:

```text
mix -> squash -> judge
```

It mixes inputs with weights, squashes the raw score into a prediction, and judges the prediction against a target.

The same story in algebra is:

```text
z = w dot x + b
y_hat = sigmoid(z)
L = (y_hat - y)^2
```

The same story in Rust roles is:

```text
FeatureVector -> PreActivation -> Prediction -> Loss
```

The category-theory reading is:

```text
small typed maps compose into one model
```

## Newtypes: Meaning In The Type System

Many ML values are stored as numbers, but they do not mean the same thing.

`0.1` might be:

- a learning rate
- a bias
- a loss
- a prediction
- a weight

Raw numbers cannot protect those meanings.

Newtypes can:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRate(f64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bias(f64);
```

Both values store an `f64`. They do not play the same role.

## Category-Theory Intuition Without Jargon

For this repo, start with three useful ideas.

### Objects

Objects are meaningful spaces of values.

In Rust, types are a practical version of this idea:

```text
FeatureVector
PreActivation
Prediction
Loss
```

### Arrows

Arrows are transformations from one meaningful space to another.

In Rust, functions and methods are practical arrows:

```text
raw_score: FeatureVector -> PreActivation
sigmoid: PreActivation -> Prediction
squared_error: Prediction + Target -> Loss
```

### Composition

Composition means connecting arrows.

The neuron forward pass is a composition:

```text
FeatureVector -> PreActivation -> Prediction -> Loss
```

This is the central habit. Ask what value you have, what value you need next, and what map owns that transformation.

## Worked Example

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
```

You should see a raw score and a prediction.

Read them like this:

```text
z is not just a number. It is a PreActivation.
prediction is not just a number. It is a Prediction.
sigmoid is the map from one role to the next.
```

## Self-Check

Answer these without looking ahead:

1. Why should `Bias` and `LearningRate` be separate types?
2. In the neuron forward pass, what are two objects and one arrow?
3. What does composition mean in the phrase "a model is a composition of maps"?

## Where This Leaves Us

Now the foundations module can stay concrete. When a formula appears, ask for its English story, its Rust roles, and its map structure.
