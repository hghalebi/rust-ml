# 05 MLP

Status: active.

This folder maps to course Module 4.

This module scales the neuron story into layers, hidden activations, and shape flow through a small network.

## Outcomes

After this module, you should be able to:

- explain why a hidden layer can represent intermediate features
- trace `InputVector -> HiddenActivation -> OutputLogit -> Prediction`
- read a small MLP as a composition of typed maps
- read typed operations such as `InputValue * WeightValue` and `WeightedSum + BiasValue`
- explain why shape checks belong at layer boundaries
- run the companion MLP examples and predict the printed trace

## Lessons

1. [Hidden Layers as Representations](01-hidden-layers-as-representations.md)
2. [Shape Flow Through an MLP](02-shape-flow-through-an-mlp.md)

## Practice

- [MLP exercises](exercises.md)
- [MLP solutions](solutions.md)

## Code Artifact

- Active crate: [`code/mlp`](../../code/mlp/README.md)

Run the examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 01_hidden_features
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 03_forward_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 04_xor_table
```

## Prerequisite

- Complete [04 Learning](../04-learning/README.md)

## Before You Move On

You are ready for attention when you can explain this sentence without guessing:

```text
The hidden layer changes the representation space before the output layer makes the final decision.
```

The companion crate now keeps raw numbers at explicit `TryFrom` boundaries, then uses semantic newtypes and `std::ops` implementations for the actual MLP operations.
