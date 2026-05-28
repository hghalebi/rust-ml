# Hidden Layers as Representations

## Overview

A single neuron draws one soft boundary.

A multi-layer perceptron starts to become more interesting because it can create a hidden representation before it predicts.

The story is:

```text
input signals -> hidden features -> output score -> prediction
```

The hidden layer is not magic. It is a small set of learned detectors. Each detector asks a simpler question, and the output layer combines the answers.

## Learning Goals

- explain a hidden activation in plain English
- connect hidden units to intermediate features
- read a two-layer MLP as a composition of maps
- explain why XOR needs a hidden representation
- connect `HiddenPreActivation` and `HiddenActivation` to the ReLU step

## Plain-English Explanation

### Why one neuron is not enough

A single neuron computes one weighted sum and then squashes it.

That means it tries to solve the task with one direct boundary in the input space.

Some patterns are not shaped like one boundary. XOR is the classic tiny example:

```text
0 xor 0 -> false
1 xor 0 -> true
0 xor 1 -> true
1 xor 1 -> false
```

The true cases are separated from each other. A single straight boundary cannot isolate both true corners while rejecting both false corners.

### What the hidden layer changes

A hidden layer creates new features.

For XOR, one hidden unit can detect "left is on and right is off". Another hidden unit can detect "right is on and left is off".

The output layer no longer has to understand the raw input directly. It only has to ask:

```text
did either hidden detector fire?
```

That is the first deep-learning intuition: depth changes the representation before the final decision.

## Algebra Form

A tiny MLP with one hidden layer can be written as:

```math
h_pre = W_1 x + b_1
```

```math
h = ReLU(h_pre)
```

```math
z = W_2 h + b_2
```

```math
\hat{y} = sigmoid(z)
```

Read this as a composition:

```text
InputVector -> HiddenPreActivation -> HiddenActivation -> OutputLogit -> Prediction
```

The category-theory lens is gentle but useful:

```text
each named type is a space of meaningful values
each layer is a map between those spaces
the model is the composition of those maps
```

## Rust Form

```rust
use rust_ml_mlp::{InputValue, InputVector, TinyMlp};

fn main() -> Result<(), rust_ml_mlp::Error> {
    let mlp = TinyMlp::xor_seed()?;
    let input = InputVector::from_values([
        InputValue::try_from(1.0)?,
        InputValue::try_from(0.0)?,
    ])?;
    let trace = mlp.forward(&input)?;

    for value in trace.hidden_activation().values() {
        println!("hidden detector = {value:.1}");
    }

    Ok(())
}
```

This small code block shows the role change:

```text
HiddenPreActivation -> HiddenActivation
```

The numbers are still numbers underneath, but the type names tell the learner what the numbers mean.

## Why This Matters

Hidden activations are the first place where "representation" becomes concrete.

The hidden vector is not the input, and it is not the final prediction. It is a new view of the input created by learned maps.

That idea scales all the way to Transformers:

```text
raw tokens -> embeddings -> contextual states -> logits
```

The names change, but the structural story remains the same.

## Concept Trace

- **Object/newtype:** `InputVector`, `HiddenPreActivation`, `HiddenActivation`, `OutputLogit`, and `Prediction`.
- **Invariant:** a hidden representation has its own role and width; it is not raw input.
- **Map:** input space -> hidden representation space -> prediction space.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 01_hidden_features`.
- **Failure signal:** you describe hidden layers as "more neurons" but cannot say what representation they produce.

## Short Practice

1. Why can a hidden layer solve patterns a single neuron cannot solve?
2. In the XOR story, what does each hidden detector notice?
3. What is the difference between `HiddenPreActivation` and `HiddenActivation`?
4. What are the maps in `InputVector -> HiddenActivation -> Prediction`?
