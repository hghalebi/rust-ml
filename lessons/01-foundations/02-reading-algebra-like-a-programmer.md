# Lesson 2: Reading Algebra Like a Programmer

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
let mut x = 3.0;
x = 4.0;
let same = x == 4.0;

let values = vec![10.0, 20.0, 30.0];
let x1 = values[0];
let x2 = values[1];

let squared = x * x;

let training_examples = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
];
let first_example = &training_examples[0];

let mut sum = 0.0;
for i in 0..values.len() {
    sum += values[i];
}

fn model(x: f64, theta: f64) -> f64 {
    x * theta
}

let y = 1.0;
let y_hat = 0.82;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

let matrix = vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
];

let d_loss_d_w = 0.123;
```

## Why This Matters

Notation is compressed meaning.

If you can unpack symbols into:

- data
- parameters
- arithmetic

then algebra becomes readable instead of intimidating.

## Short Practice

Translate each item into English:

1. $x_2$
2. $\hat{y}$
3. $\sum_{i=1}^{n} a_i$
4. $w \cdot x$
5. $\frac{\partial L}{\partial b}$
