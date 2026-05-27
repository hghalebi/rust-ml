# Shape Flow Through an MLP

## Overview

An MLP is a chain of maps.

Each map has an input width and an output width. If those widths do not line up, the model is not merely inaccurate. It is structurally invalid.

Shape flow is the habit of asking:

```text
what shape enters this map, and what shape leaves it?
```

## Learning Goals

- trace input, hidden, and output widths through a tiny MLP
- explain why weight rows and bias length must match
- explain why weight columns must match the incoming vector width
- read shape errors as boundary errors, not random runtime failures
- connect shape flow to typed composition

## Plain-English Explanation

### A layer is a translator between spaces

Suppose the input has two numbers:

```text
[x_left, x_right]
```

If the hidden layer has two units, then it maps:

```text
2 input signals -> 2 hidden features
```

If the output layer produces one logit, then it maps:

```text
2 hidden features -> 1 output score
```

The full shape story is:

```text
2 -> 2 -> 1
```

That small chain tells us whether composition is legal.

### Bias belongs to the output side of a layer

A layer with two output units needs two bias values.

The bias does not belong to the input width. It belongs to the layer's output width because each output unit gets one offset.

This is why the companion crate checks:

```text
weight rows == bias length
```

### The output layer must accept the hidden width

The hidden layer returns a hidden activation. The output layer consumes it.

Therefore:

```text
hidden output width == output input width
```

If this is false, the two maps cannot compose.

## Algebra Form

For a dense layer:

```math
y = W x + b
```

If:

```math
W \in R^{out \times in}
```

then:

```math
x \in R^{in}
```

and:

```math
b \in R^{out}, \quad y \in R^{out}
```

For the tiny MLP:

```text
InputVector(2)
  -> HiddenPreActivation(2)
  -> HiddenActivation(2)
  -> OutputLogit(1)
  -> Prediction
```

The category-theory lens is:

```text
maps compose only when the output object of one map matches the input object of the next map
```

## Rust Form

```rust
use rust_ml_mlp::TinyMlp;

fn main() -> Result<(), rust_ml_mlp::Error> {
    let mlp = TinyMlp::xor_seed()?;

    println!("InputVector width      = {}", mlp.input_dim());
    println!("HiddenActivation width = {}", mlp.hidden_dim());
    println!("OutputLogit width      = 1");
    println!("composition: InputVector -> HiddenActivation -> OutputLogit -> Prediction");

    Ok(())
}
```

This is the shape-flow idea without matrix arithmetic.

The real crate adds the arithmetic and returns expressive errors when the layer widths do not line up.

## Why This Matters

Shape mistakes are common in ML code because raw arrays do not explain their role.

Newtypes do not remove every shape problem, but they make the intended boundary visible:

```text
InputVector is not HiddenActivation
WeightMatrix rows are not WeightMatrix columns
Prediction is not OutputLogit
```

That discipline prepares the learner for attention, where shape flow becomes more demanding:

```text
sequence length, embedding width, head count, head width
```

## Concept Trace

- **Object/newtype:** `InputWidth`, `HiddenWidth`, `OutputWidth`, and `MatrixShape` in the learner's shape vocabulary.
- **Invariant:** the output width of one layer must match the input width of the next layer.
- **Map:** layer shape -> legal composition -> full MLP shape flow.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow`.
- **Failure signal:** you can run a layer alone but cannot explain why two adjacent layers fail to compose.

## Short Practice

1. A layer has weight shape `3 x 2`. What input width does it expect?
2. How many bias values does that layer need?
3. If a hidden layer outputs width `4`, what input width must the output layer accept?
4. Why is a shape mismatch a composition problem?
