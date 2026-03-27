# Neuron Exercises

## Exercise 1: Label the factory

For each room in the tiny-factory metaphor, name:

1. the input it receives
2. the output it produces
3. the mathematical symbol for that output

## Exercise 2: One forward pass by hand

Given:

```math
x_1 = 1,\quad x_2 = 0,\quad w_1 = 0.8,\quad w_2 = -0.4,\quad b = 0.1
```

1. Compute `z`
2. Compute `\hat{y} = \sigma(z)`
3. If the target is `y = 1`, compute the squared-error loss

## Exercise 3: Read the dependency chain

Write the parameter-to-loss path for:

1. `w1`
2. `w2`
3. `b`

using words, not only symbols.

## Exercise 4: Explain the local derivatives

Why are these true?

```math
\frac{dz}{dw_1} = x_1,\quad
\frac{dz}{dw_2} = x_2,\quad
\frac{dz}{db} = 1
```

Use both:

- one algebra sentence
- one metaphor sentence

## Exercise 5: Read the update rule in English

Explain this in one plain sentence:

```math
w := w - \eta \frac{dL}{dw}
```

## Exercise 6: Memory hook

Without looking back, write the two three-word summaries for:

1. forward pass
2. backward pass
