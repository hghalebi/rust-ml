# 04 Learning

Status: active.

This folder maps to course Module 3.

This module takes the neuron from "one step makes sense" to "a full training loop makes sense."

## Role In The Course

The neuron module showed one forward pass and one backward-pass story. This module repeats that story across examples and epochs so learning becomes a process, not a single formula.

## Outcomes

After this module, you should be able to:

- explain a training step as `predict -> measure -> adjust`
- read a gradient update without treating it as magic
- explain why a dataset loop repeats the same local update across many examples
- describe an epoch as one pass over the training data
- interpret a loss trace as feedback about whether learning is moving in a useful direction

## Lessons

1. [A training step is a feedback story](01-training-step-as-feedback.md)
2. [Epochs turn one update into a learning process](02-epochs-and-loss-traces.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Code Artifact

- Current bridge crate: [`code/neuron`](../../code/neuron/README.md)
- The `03_one_step_training` and `04_and_gate_epoch` examples are the executable anchors:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

## Prerequisite

- Complete [03 Neuron](../03-neuron/README.md)

## Before You Move On

You are ready for the MLP module if you can:

- explain why the target stays fixed while parameters change
- say what one epoch means
- describe why loss can go down unevenly rather than smoothly
- connect `LearningRate`, `Gradient`, and `Adjustment` to the update rule
