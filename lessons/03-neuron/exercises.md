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

## Exercise 7: Run the executable ladder

Run these examples in order:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
```

For each example, write one sentence that connects the printed numbers to the lesson vocabulary.

## Failure Signals

- You can compute `z` but cannot explain why the sigmoid map comes next.
- You describe the backward pass as "magic correction" instead of tracing loss back to each parameter.
- You mix up `Prediction`, `Target`, `Loss`, `Gradient`, and `Adjustment` as if they were the same kind of number.
- You run the examples but cannot connect the printed values to the type names in `code/neuron`.

## Debugging Hints

- Draw the forward path first: `FeatureVector -> WeightedSum -> Prediction -> Loss`.
- For each parameter, ask which intermediate value it directly changes.
- Keep the three-word hooks nearby: forward is mix, squash, judge; backward is blame, trace, adjust.
- When an example prints a value, write the Rust newtype name beside the printed number.
