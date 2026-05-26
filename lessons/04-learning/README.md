# 04 Learning

Status: planned.

This folder maps to course Module 3.

This module will take the neuron from "one step makes sense" to "a full training loop makes sense."

## Goal

Turn a forward pass into a learning loop:

- compute loss
- compute gradients
- update parameters
- repeat across data

## Planned Lesson Ladder

1. one training step, end to end
2. chain rule by hand for the neuron parameters
3. dataset loops, epochs, and reading loss over time

## Planned Practice

- derive one update by hand
- run a tiny dataset loop
- explain what it means for loss to go down or fail to go down

## Code Artifact

- Current bridge crate: [`code/neuron`](../../code/neuron/README.md)
- The `04_and_gate_epoch` example already shows the first full dataset loop:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

## Prerequisite

- Complete [03 Neuron](../03-neuron/README.md)

## Planned Outcome

Be able to derive and explain parameter updates from first principles.
