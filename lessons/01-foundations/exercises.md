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

Write typed Rust that stores those values in a `FeatureVector`, then explain which semantic feature is first and which is third.

## Exercise 3: Read the notation

Explain each expression in one short English sentence:

1. $x^{(2)}$
2. $\sum_{i=1}^{n} a_i$
3. $\hat{y}$
4. $\frac{\partial L}{\partial w}$

## Exercise 4: Write a dot product

Write typed Rust that computes this map with `FeatureVector`, `WeightVector`, and `weighted_sum`:

```math
w \cdot x = \sum_i w_i x_i
```

## Exercise 5: Model as a struct

Build a `TinyNeuron` with two weights and a bias, then call the method that computes:

```math
z = w_1 x_1 + w_2 x_2 + b
```

## Failure Signals

- You can write the Rust expression but cannot say which algebra symbol each field represents.
- You treat `x_1` as Rust index `1` instead of remembering that Rust vectors start at index `0`.
- Your dot-product loop adds weights or inputs separately instead of adding pairwise products.
- Your struct stores values but does not expose the map from inputs to score.

## Debugging Hints

- Label every value before calculating: input, weight, bias, or score.
- Translate one symbol at a time instead of translating the whole formula at once.
- For indexing, write the algebra position and the Rust index side by side.
- Read your `forward` method aloud as a sentence: "take inputs, multiply by weights, add bias, return score."
