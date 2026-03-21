# Foundations Exercises

Work these without opening the solutions first.

## Exercise 1: One sentence, three forms

Translate this idea into algebra and Rust:

> Multiply two inputs by two weights, add a bias, and compute a score.

## Exercise 2: Fix the indexing mismatch

Suppose algebra says:

```math
x = [x_1, x_2, x_3]
```

Write Rust that stores those values in a vector and reads the first and third entries.

## Exercise 3: Read the notation

Explain each expression in one short English sentence:

1. $x^{(2)}$
2. $\sum_{i=1}^{n} a_i$
3. $\hat{y}$
4. $\frac{\partial L}{\partial w}$

## Exercise 4: Write a dot product

Write a Rust function signature and loop body for:

```math
w \cdot x = \sum_i w_i x_i
```

## Exercise 5: Model as a struct

Write a `Neuron` struct with fields `w1`, `w2`, and `b`, and add a `forward` method that computes:

```math
z = w_1 x_1 + w_2 x_2 + b
```
