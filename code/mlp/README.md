# mlp

Status: active.

This crate is the executable companion for [05 MLP](../../lessons/05-mlp/README.md).

It teaches the first step beyond a single neuron: hidden activations create intermediate representations.

## Owns

- lesson module: [05 MLP](../../lessons/05-mlp/README.md)

## Current State

- active teaching crate
- deterministic XOR-shaped forward pass
- explicit `TryFrom` adapters for raw learner numbers
- typed layer roles and shape checks
- typed `std::ops` arithmetic for dot-product pieces and bias addition
- no training claim; this crate focuses on representation and shape flow

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_hidden_features.rs
  02_shape_flow.rs
  03_forward_trace.rs
  04_xor_table.rs
```

## Learning Ladder

1. `01_hidden_features` shows what the hidden units detect.
2. `02_shape_flow` names the typed map through the network.
3. `03_forward_trace` prints every important intermediate value.
4. `04_xor_table` shows why hidden representation can solve a pattern one straight-line neuron cannot.

## Category Lens

Read the MLP as a composition of representation maps:

```text
InputVector -> HiddenPreActivation
HiddenPreActivation -> HiddenActivation
HiddenActivation -> OutputLogit
OutputLogit -> Prediction
```

The composition rule is shape agreement. A `WeightMatrix` can follow an
`InputVector` or `HiddenActivation` only when the vector width matches the
matrix input width.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_mlp --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 01_hidden_features
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 03_forward_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 04_xor_table
```

## Scope

This crate intentionally does not implement backpropagation for MLPs.

Module 04 teaches the update loop with a single neuron. This crate teaches the next structural idea:

```text
InputVector -> HiddenActivation -> OutputLogit -> Prediction
```

The public API avoids raw domain primitives. Examples parse raw numbers at the edge into `InputValue`, `WeightValue`, and `BiasValue`, then the forward pass moves through typed vectors, matrix rows, dimensions, logits, and predictions.
