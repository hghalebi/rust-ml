# 03 Neuron

Status: active.

This folder maps to course Module 2.

This module turns the vector and loss intuition from Module 1 into the smallest trainable model: one typed neuron with a forward pass, a loss, and a backward-pass story that does not hide behind variable soup.

## Role In The Course

This module is the bridge between algebraic ingredients and the first complete model. It is where the course stops talking only about parts and starts talking about a learnable system.

## Outcomes

After this module, you should be able to:

- explain a neuron as a chain of functions instead of a bag of symbols
- map `Input`, `Weight`, `Bias`, `Prediction`, and `Target` to Rust newtypes
- trace one forward pass from weighted sum to sigmoid output to loss
- explain backpropagation as structured blame assignment through a function chain
- read `dz/dw1 = x1` and explain why it is true

## Lessons

1. [Rust essentials for a tiny neuron](01-rust-essentials-for-a-tiny-neuron.md)
2. [A neuron as a chain of functions](02-neuron-as-a-chain-of-functions.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Code Artifact

- Runnable companion crate shared with Module 3: [`code/neuron`](../../code/neuron/README.md)

## Prerequisite

- Complete [02 Scalars, vectors, and dot products](../02-vectors/README.md)

## Before You Move On

You are ready for the learning module if you can:

- explain the neuron as `mix -> squash -> judge`
- describe the backward pass as `blame -> trace -> adjust`
- compute a single gradient factor such as `dz/dw1` by hand
- read the update rule `w = w - learning_rate * gradient` in ordinary English

## Current Focus

The authored material in this module is deliberately narrow:

- enough Rust syntax to read the neuron code honestly
- a typed single-neuron model with semantic newtypes
- one full training-step walkthrough as a preview of the dedicated learning module
