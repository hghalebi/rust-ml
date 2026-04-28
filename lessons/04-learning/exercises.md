# Learning Module Exercises

## Exercise 1: Name the four stages

Write the four stages of one training step in order.

Then explain, in one short sentence each, what new information appears at every stage.

## Exercise 2: One backward pass by hand

Given:

```math
x_1 = 1,\quad x_2 = 0,\quad w_1 = 0.8,\quad w_2 = -0.4,\quad b = 0.1,\quad y = 1
```

1. Compute `z`
2. Compute `\hat{y}`
3. Compute the squared-error loss
4. Explain why `dL/dw2 = 0` in this case

## Exercise 3: Upstream versus local

For `dL/dw1`, label which factor belongs to:

1. the judge room
2. the shaping room
3. the mixing room

Use the chain-rule form, not only words.

## Exercise 4: Read the optimizer rule

Explain this sentence in plain English:

```math
\theta := \theta - \eta \nabla_\theta L
```

Then say what would happen if `\eta` were far too large.

## Exercise 5: Dataset loop reasoning

Suppose a dataset has 8 examples and you train for 50 epochs.

1. How many times does the model see each example?
2. What does it mean if the average loss falls at first, then stops changing much?

## Exercise 6: Token targets

A model outputs one logit per vocabulary token.

1. Why is the target now an index instead of one scalar value like `0` or `1`?
2. Why does this make training feel like a larger system, even though gradients still drive it?
