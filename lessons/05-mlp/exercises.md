# MLP Exercises

## Exercise 1: Explain the hidden layer

Explain this chain in your own words:

```text
InputVector -> HiddenPreActivation -> HiddenActivation
```

Use the words:

- weighted sum
- ReLU
- representation

## Exercise 2: Trace the XOR detectors

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 01_hidden_features
```

For each input, write down which hidden unit fires.

Then answer: why do the two true XOR cases produce different hidden activations?

## Exercise 3: Follow shape flow

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow
```

Write the shape chain as:

```text
Input width -> hidden width -> output width
```

Then explain which two widths must match for the hidden layer and output layer to compose.

## Exercise 4: Predict before running

Before running the example, predict which inputs should have a high output:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 04_xor_table
```

After running it, compare your prediction to the printed values.

## Exercise 5: Complete the role map

Fill in the missing role names:

```rust
fn main() {
    let chain = [
        "InputVector",
        "_______",
        "HiddenActivation",
        "_______",
        "Prediction",
    ];

    println!("{}", chain.join(" -> "));
}
```

Then explain which arrows are linear maps and which arrows are activation maps.

## Failure Signals

- You say a hidden layer is "more neurons" but cannot name the representation it produces.
- You cannot explain why XOR needs hidden features while a single linear boundary is not enough.
- You mix up hidden width, output width, and input width when composing layers.
- You predict outputs only after running the code, so the example becomes observation without intuition.

## Debugging Hints

- Treat each hidden unit as a detector and ask which input pattern makes it positive.
- Write the shape chain before following the numeric values.
- For composition, check that the previous output width equals the next input width.
- Before running `04_xor_table`, write the expected high-output rows and compare afterward.
