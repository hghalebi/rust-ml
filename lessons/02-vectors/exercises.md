# Vectors Exercises

## Exercise 1: Classify the shape

For each item, say whether it is a scalar, vector, or matrix:

1. `3.14`
2. `[1.0, 2.0, 3.0]`
3. `[[1.0, 2.0], [3.0, 4.0]]`

## Exercise 2: Dot product by hand

Compute:

```math
[1, 2, 3] \cdot [4, 5, 6]
```

Then write the equivalent typed Rust operation using `DenseVector` and `&a * &b`.

## Exercise 3: Matrix-vector multiplication by hand

Compute:

```math
W = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}

x = \begin{bmatrix} 4 \\ 5 \end{bmatrix}
```

Find $Wx$.

## Exercise 4: Sigmoid and loss

1. What is $\sigma(0)$?
2. If $\hat{y} = 0.25$ and $y = 1.0$, what is the squared error loss?

## Exercise 5: Read the update

Explain this line in one sentence:

```text
let step = neuron.train_one_step(&example, learning_rate)?;
```

## Failure Signals

- You call every list a vector without checking whether it is a row, column, or matrix.
- Your dot product gives a vector result instead of one scalar.
- Your matrix-vector multiplication ignores row-by-row structure.
- You explain the update as "subtract a number" without naming learning rate or gradient.

## Debugging Hints

- Write shapes before values. Most vector mistakes become obvious when the shape is named.
- For a dot product, multiply aligned positions first, then add those products.
- For matrix-vector multiplication, compute one output entry per matrix row.
- For the update rule, ask which value is the old parameter and which value controls step size.
