# Vectors Exercises

## Exercise 1: Classify the shape

For each item, say whether it is a scalar, vector, or matrix:

1. `3.14`
2. `[1.0, 2.0, 3.0]`
3. `[[1.0, 2.0], [3.0, 4.0]]`

## Exercise 2: Dot product by hand

Compute:

```text
[1, 2, 3] . [4, 5, 6]
```

Then write the equivalent Rust loop.

## Exercise 3: Matrix-vector multiplication by hand

Compute:

```text
W = [ [2, 0],
      [1, 3] ]

x = [4, 5]
```

Find `W x`.

## Exercise 4: Sigmoid and loss

1. What is `sigmoid(0.0)`?
2. If `y_hat = 0.25` and `y = 1.0`, what is the squared error loss?

## Exercise 5: Read the update

Explain this line in one sentence:

```rust
w = w - learning_rate * d_loss_d_w;
```
