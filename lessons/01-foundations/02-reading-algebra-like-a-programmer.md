# Reading Algebra Like a Programmer

## Overview

Most beginners do not fail on the mathematics first. They fail on the notation. This lesson turns the notation into a readable dictionary.

## Learning Goals

- read common ML notation without mistaking symbols for something else
- map algebraic notation to sensible Rust names
- notice where math and Rust use different conventions

## Plain-English Explanation

Below are the symbols you will see again and again.

### Variables

A variable is a named quantity.

### Subscripts

$x_1$, $x_2$, and $x_i$ mean "the first", "the second", or "the i-th" element of $x$.

They do not mean multiplication.

### Superscripts

Superscripts can mean different things depending on context.

- $x^2$ means "x squared"
- $x^{(1)}$ often means "training example 1"

The parentheses matter.

### Sum notation

$\sum_{i=1}^{n} a_i$ means "add all the terms from the first to the n-th."

### Function notation

$f(x)$ means "apply function $f$ to input $x$."

In ML, $\hat{y} = f(x; \theta)$ often means:

- $x$ is the input
- $\theta$ is the set of learned parameters
- $\hat{y}$ is the prediction

### Hat notation

$\hat{y}$ means predicted $y$, not true $y$.

### Dot product

$w \cdot x$ means multiply matching entries and add them all together.

### Matrix

A matrix is a rectangular table of numbers.

### Shape notation

$x \in \mathbb{R}^3$ means $x$ is a vector of length 3 with real-valued entries.

$W \in \mathbb{R}^{2 \times 3}$ means $W$ is a matrix with 2 rows and 3 columns.

### Transpose

$K^T$ means rows become columns and columns become rows.

### Derivative

A derivative tells you how fast one quantity changes when another quantity changes.

### Partial derivative

$dL/dw$ in code-friendly form means "how much the loss changes if I nudge $w$."

## Algebra Form

```math
\begin{aligned}
x &= 3 \\
x &= [x_1, x_2, x_3] \\
x^2 \\
x^{(1)}, x^{(2)}, x^{(3)} \\
\sum_{i=1}^{n} a_i \\
f(x) \\
\hat{y} &= f(x; \theta) \\
w \cdot x &= \sum_i w_i x_i \\
W &\in \mathbb{R}^{2 \times 3} \\
K^T \\
\frac{d}{dx} x^2 &= 2x \\
\frac{\partial L}{\partial w}
\end{aligned}
```

## Rust Form

```rust
use rust_ml_neuron::{
    Bias, FeatureVector, InputValue, Target, TinyNeuron, Weight, WeightVector,
};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let features = FeatureVector::from_values([
        InputValue::try_from(10.0)?,
        InputValue::try_from(20.0)?,
        InputValue::try_from(30.0)?,
    ])?;
    let weights = WeightVector::from_values([
        Weight::try_from(0.1)?,
        Weight::try_from(0.2)?,
        Weight::try_from(0.3)?,
    ])?;

    let weighted_evidence = (&features * &weights)?;

    let first_example = FeatureVector::two(InputValue::try_from(0.0)?, InputValue::try_from(1.0)?);
    let neuron = TinyNeuron::new(
        WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?),
        Bias::try_from(0.1)?,
    );
    let y = Target::try_from(1.0)?;
    let y_hat = neuron.predict(&first_example)?;

    println!("weighted evidence = {weighted_evidence}");
    println!("target = {y}, prediction = {y_hat}");
    Ok(())
}
```

## Why This Matters

Notation is compressed meaning.

If you can unpack symbols into:

- data
- parameters
- arithmetic

then algebra becomes readable instead of intimidating.

## Concept Trace

- **Object/newtype:** notation later becomes names such as `TokenIndex`, `VectorWidth`, `Prediction`, and `Gradient`.
- **Invariant:** every symbol should keep its role when translated into Rust.
- **Map:** compact notation -> explicit names -> checked computation.
- **Runnable proof:** compile the lesson snippets with `python3 scripts/check_lesson_rust_snippets.py`.
- **Failure signal:** you can pronounce a formula but cannot identify the data, parameter, output, or feedback role.

## Short Practice

Translate each item into English:

1. $x_2$
2. $\hat{y}$
3. $\sum_{i=1}^{n} a_i$
4. $w \cdot x$
5. $\frac{\partial L}{\partial b}$
