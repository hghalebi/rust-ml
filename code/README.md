# Code

This directory is reserved for runnable examples that follow the lesson progression.

It is the executable companion layer for the course, not the canonical teaching layer.

## Strategy

- `lessons/` stays canonical for authored teaching content.
- `code/` will become the executable companion once each topic earns a real runnable example.
- These topic directories are placeholders now, not empty fake crates.

## Packaging Rule

When executable lessons start, each topic directory becomes a Cargo workspace crate.

Do not use loose standalone `.rs` files as the long-term structure.

## Topics

| Topic | Status | Purpose |
| --- | --- | --- |
| [neuron](neuron/README.md) | Planned crate | Will become the first trainable model companion crate. |
| [mlp](mlp/README.md) | Planned crate | Will become the small multi-layer network companion crate. |
| [attention](attention/README.md) | Planned crate | Will become the explicit attention-mechanics companion crate. |
| [transformer](transformer/README.md) | Active crate | Real tested encoder-path teaching crate for the current advanced preview module. |
