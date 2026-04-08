# 01 Foundations

Status: active.

This folder maps to course Module 0.

The job of this module is narrow and important: teach the smallest set of notation and Rust syntax needed to read beginner machine-learning material without panic.

## Role In The Course

This module is the orientation layer. It teaches the notation and Rust surface area that all later modules reuse.

## Outcomes

After this module, you should be able to:

- explain the basic ML loop in plain English
- read variables, subscripts, hats, sums, and derivatives without freezing
- map common algebra notation into Rust variable names and loops
- recognize the Rust syntax needed for later model code

## Lessons

1. [The single most important idea](01-core-idea.md)
2. [Reading algebra like a programmer](02-reading-algebra-like-a-programmer.md)
3. [Rust syntax for ML](03-rust-syntax-for-ml.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Code Artifact

- No dedicated crate yet. This module is intentionally lightweight and prepares the learner for the executable code that starts later in the course.

## Prerequisite

- None. This is the course entry point.

## Before You Move On

You are ready for the next module if you can explain all three of these without notes:

- what `y_hat` means
- why `x_1` in algebra becomes `x[0]` in Rust
- why `&self` and `&mut self` fit the difference between inference and training
