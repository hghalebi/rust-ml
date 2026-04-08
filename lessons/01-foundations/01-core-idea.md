# The Single Most Important Idea

## Overview

Machine learning can look mysterious because the notation is compact. The underlying loop is much simpler than it looks.

## Learning Goals

- say the basic ML loop in one sentence
- recognize the same idea in English, algebra, and Rust
- understand why this course keeps translating between those three forms

## Plain-English Explanation

Take some numbers in, do arithmetic with parameters, get an output, measure how wrong it is, then update the parameters to be less wrong.

That sentence explains a surprising amount of machine learning.

Here is a tiny version of that loop:

- inputs: `x1`, `x2`
- parameters: `w1`, `w2`, `b`
- output: `z`

In words:

- multiply each input by a weight
- add the results together
- add a bias

## Algebra Form

```math
z = w_1 x_1 + w_2 x_2 + b
```

This says the same thing in compressed mathematical form.

## Rust Form

```rust
let z = w1 * x1 + w2 * x2 + b;
```

This says the same thing in executable form.

## Why This Matters

The rest of the course is mostly this one skill repeated:

- read a formula
- ask what the inputs are
- ask what the parameters are
- ask what arithmetic is happening
- write the same thing as Rust

If you can do that, ML stops looking magical.

## Short Practice

Translate this sentence into algebra and then Rust:

> Take two inputs, weight them, add a bias, and compute a score.

Expected target shape:

```math
z = \dots
```

```rust
let z = /* your expression here */ 0.0;
```
