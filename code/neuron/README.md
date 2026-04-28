# neuron

Status: active.

This crate is the executable companion for [03 Neuron](../../lessons/03-neuron/README.md)
and [04 Learning](../../lessons/04-learning/README.md).

It models the first honest training system in the course with:

- semantic newtypes for inputs, weights, bias, prediction, target, loss, and gradients
- a single-neuron forward pass with manual backpropagation
- SGD updates and epoch-level dataset training helpers
- tiny boolean datasets such as OR and AND
- token-target utilities that bridge scalar supervision to next-token loss
- a tiny bigram next-token model using `token -> embedding -> lm_head -> logits`

## Layout

```text
src/
  bigram.rs
  dataset.rs
  neuron.rs
  optimizer.rs
  token_targets.rs
  lib.rs
examples/
  train_bigram_cycle.rs
  train_or_gate.rs
  token_targets.rs
```

## Run

```bash
cargo test --manifest-path code/neuron/Cargo.toml
```

```bash
cargo run --example train_or_gate --manifest-path code/neuron/Cargo.toml
```

```bash
cargo run --example train_bigram_cycle --manifest-path code/neuron/Cargo.toml
```

```bash
cargo run --example token_targets --manifest-path code/neuron/Cargo.toml
```

## Scope

This crate is intentionally educational, not production-grade. It does not include:

- mini-batch loaders
- momentum or Adam
- autograd
- matrix-based multi-layer backpropagation
- full tokenizer or vocabulary tooling
- Transformer training

The goal is to make the first training loop and the token-target bridge explicit.
