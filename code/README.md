# Code

This directory is reserved for runnable examples that follow the lesson progression.

## Strategy

- `lessons/` stays canonical for authored teaching content.
- `code/` will become the executable companion once each topic earns a real runnable example.
- These topic directories are placeholders now, not empty fake crates.

## Packaging Rule

When executable lessons start, each topic directory becomes a Cargo workspace crate.

Do not use loose standalone `.rs` files as the long-term structure.

## Topics

- [neuron](neuron/README.md)
- [mlp](mlp/README.md)
- [attention](attention/README.md)
- [transformer](transformer/README.md)
