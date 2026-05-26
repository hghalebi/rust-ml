# neuron

Status: active.

This crate is the executable companion for [03 Neuron](../../lessons/03-neuron/README.md) and the first runnable bridge into [04 Learning](../../lessons/04-learning/README.md).

It keeps the beginner model explicit:

- semantic scalar types: `InputValue`, `Weight`, `Bias`, `Target`, `Prediction`, `LearningRate`
- vector wrappers: `FeatureVector`, `WeightVector`
- typed model: `TinyNeuron`
- explicit errors through `NeuronError`
- learner-visible `TrainingStep` values for gradients, loss before, and loss after

## Owns

- lesson module: [03 Neuron](../../lessons/03-neuron/README.md)
- related training concepts: [04 Learning](../../lessons/04-learning/README.md)

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_weighted_sum.rs
  02_forward_pass.rs
  03_one_step_training.rs
  04_and_gate_epoch.rs
```

## Learning Ladder

1. `01_weighted_sum` shows the dot product as one feature per weight.
2. `02_forward_pass` adds bias and sigmoid: `mix -> squash`.
3. `03_one_step_training` exposes `blame -> trace -> adjust` for one labeled example.
4. `04_and_gate_epoch` repeats updates across a tiny AND dataset so learners can watch average loss move.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_neuron --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

## Scope

This crate is intentionally small. It does not include autograd, optimizers, tensors, batching, GPU kernels, or generic neural-network layers.

The goal is one complete mental model:

```text
weighted sum -> sigmoid -> loss -> gradient update
```
