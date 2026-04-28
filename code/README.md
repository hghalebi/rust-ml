# Code

This directory is reserved for runnable teaching crates that follow the lesson progression.

It is the executable companion layer for the course, not the canonical teaching layer.

## Strategy

- `lessons/` stays canonical for authored teaching content.
- `code/` grows as modules earn real runnable examples.
- Some topic directories may still be placeholders, but active topics should be real Cargo crates with tests.

## Packaging Rule

When executable lessons start, each topic directory becomes a real Cargo crate.

Do not use loose standalone `.rs` files as the long-term structure.

## Teaching Contract

- Active crates must compile, run tests, and have examples that match the lesson module they support.
- Planned crate directories should stay as roadmaps until the matching lesson module is authored.
- Crate READMEs should explain scope limits so learners do not mistake teaching code for production ML infrastructure.

## Topics

| Topic | Status | Purpose |
| --- | --- | --- |
| [neuron](neuron/README.md) | Active crate | Teaches one-neuron training, manual gradients, SGD, datasets, and a tiny next-token bridge. |
| [mlp](mlp/README.md) | Planned crate | Will become the small multi-layer network companion crate. |
| [attention](attention/README.md) | Planned crate | Will become the explicit attention-mechanics companion crate. |
| [transformer](transformer/README.md) | Active crate | Real tested encoder-path teaching crate for the current advanced preview module. |
