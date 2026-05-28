# neuron

Status: active.

This crate is the executable companion for [03 Neuron](../../lessons/03-neuron/README.md) and the first runnable bridge into [04 Learning](../../lessons/04-learning/README.md).

It keeps the beginner model explicit:

## Current State

- semantic scalar types: `InputValue`, `Weight`, `Bias`, `Target`, `Prediction`, `LearningRate`
- explicit `TryFrom` adapters for raw learner literals
- readable typed arithmetic through `std::ops` traits, such as `&FeatureVector * &WeightVector`, `InputValue * Weight`, `WeightedSum + Bias`, and `Weight - Adjustment`
- vector wrappers: `FeatureVector`, `WeightVector`
- typed model: `TinyNeuron`
- explicit errors through `NeuronError`
- learner-visible `TrainingStep` values for gradients, loss before, and loss after
- public training-step review boundary for learner-facing update evidence

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
  05_public_training_step.rs
  token_targets.rs
  train_bigram_cycle.rs
  train_or_gate.rs
```

## Learning Ladder

1. `01_weighted_sum` shows the dot product as one feature per weight.
2. `02_forward_pass` adds bias and sigmoid: `mix -> squash`.
3. `03_one_step_training` exposes `blame -> trace -> adjust` for one labeled example.
4. `04_and_gate_epoch` repeats updates across a tiny AND dataset so learners can watch average loss move.
5. `05_public_training_step` shows how reviewed update evidence becomes publishable learner-facing material.
6. `token_targets` derives token-level probabilities and gradients for cross-entropy intuition.
7. `train_bigram_cycle` shows self-contained bigram-style training with a compact two-step language loop.
8. `train_or_gate` trains the tiny neuron on OR truth-table data for several epochs and prints predictions.

## Category Lens

Read the neuron as a composition of tiny maps:

```text
FeatureVector * WeightVector -> WeightedSum
WeightedSum + Bias -> PreActivation
PreActivation -> Prediction
Prediction + Target -> Loss
Loss -> Gradient -> Adjustment
ReviewedTrainingStep -> PublicTrainingStep
```

The composition rule is alignment. Every input feature needs exactly one
weight, and each update keeps the parameter role separate from the observed
training example.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_neuron --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 05_public_training_step
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example token_targets
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example train_bigram_cycle
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example train_or_gate
```

## Scope

This crate is intentionally small. It does not include autograd, optimizers, tensors, batching, GPU kernels, or generic neural-network layers.

The goal is one complete mental model:

```text
weighted sum -> sigmoid -> loss -> gradient update
ReviewedTrainingStep -> PublicTrainingStep
```

The public API avoids raw domain primitives: examples parse raw numbers at the edge with `TryFrom`, then the model code moves through semantic newtypes and checked operations.

The public training-step boundary keeps update evidence separate from release
permission: a `TrainingStep` explains the learning move, while a
`PublicTrainingStep` proves that evidence was reviewed for learner-facing use.
