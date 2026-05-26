# Code

This directory is reserved for runnable examples that follow the lesson progression.

It is the executable companion layer for the course, not the canonical teaching layer.

## Strategy

- `lessons/` stays canonical for authored teaching content.
- `code/` is the executable companion layer once a topic earns real runnable examples.
- Active crates must have tests and examples that line up with the lesson sequence.
- Planned topic directories stay as honest roadmap placeholders until they are ready.

## Packaging Rule

When executable lessons start, each topic directory becomes a Cargo workspace crate.

Do not use loose standalone `.rs` files as the long-term structure.

## Workspace Commands

Run all active teaching crates:

```bash
cargo test --manifest-path code/Cargo.toml --workspace --all-targets
```

Run the beginner neuron examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

Run the advanced Transformer example:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example encoder_demo
```

## Topics

| Topic | Status | Purpose |
| --- | --- | --- |
| [neuron](neuron/README.md) | Active crate | The first typed trainable model companion crate. |
| [mlp](mlp/README.md) | Planned crate | Will become the small multi-layer network companion crate. |
| [attention](attention/README.md) | Planned crate | Will become the explicit attention-mechanics companion crate. |
| [transformer](transformer/README.md) | Active crate | Real tested encoder-path teaching crate for the current advanced preview module. |
