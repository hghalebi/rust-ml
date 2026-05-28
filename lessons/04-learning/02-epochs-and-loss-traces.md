# Epochs Turn One Update Into A Learning Process

## Overview

One training step is local feedback.

An epoch repeats that feedback across the dataset.

The key idea is simple:

```text
one step teaches from one example
one epoch teaches from every example once
many epochs let the model keep improving
```

## Learning Goals

- explain what an epoch is
- read a dataset loop as repeated feedback
- interpret average loss over time
- explain why loss may go down unevenly
- connect the `04_and_gate_epoch` example to the training-loop idea

## Plain-English Explanation

### A dataset is a sequence of teaching moments

Each example tells the model something small.

One example may push one weight up. Another example may push it down. The dataset loop lets the model receive all of those signals.

### An epoch is one full pass

An epoch means:

```text
visit every training example once
```

After one epoch, the model has had one chance to learn from each example.

### Loss traces are feedback about the process

Average loss is not a guarantee of wisdom. It is a signal.

If average loss moves down, the updates are probably helping. If it moves up or becomes unstable, the learning rate, data, model shape, or gradient calculation may be wrong.

## Algebra Form

For a dataset with examples:

```math
(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
```

one epoch applies the training step to each example:

```math
\theta := update(\theta, x_i, y_i)
```

for every example `i`.

Average loss is:

```math
\frac{1}{n}\sum_i L_i
```

The category-theory lens is:

```text
the same parameterized map is reused across many examples
```

## Rust Form

```rust
use rust_ml_neuron::{Dataset, LearningRate, TinyNeuron};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let dataset = Dataset::and_gate()?;
    let mut neuron = TinyNeuron::lesson_seed()?;
    let rate = LearningRate::try_from(0.8)?;

    for epoch in 1..=5 {
        neuron.train_epoch(&dataset, rate)?;
        println!(
            "checkpoint {epoch}: average loss = {:.4}",
            neuron.average_loss(&dataset)?
        );
    }

    Ok(())
}
```

The real crate example computes the losses from a neuron. This small snippet teaches how to read the trace.

Run the executable version:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

## Why This Matters

Learning is not a single dramatic moment.

It is a repeated process:

```text
example -> prediction -> loss -> gradient -> update
```

The serious skill is watching that process and knowing what the numbers mean.

## Concept Trace

- **Object/newtype:** `TrainingExample`, `Epoch`, `Loss`, and `AverageLoss` in the learner's mental model.
- **Invariant:** one-example loss and dataset-average loss answer different questions.
- **Map:** many examples -> many updates -> epoch-level loss trace.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch`.
- **Failure signal:** you judge training from one example while ignoring the epoch average.

## Short Practice

1. What does one epoch mean?
2. Why can average loss go down even if one example briefly gets worse?
3. What are two possible explanations if average loss goes up for many epochs?
4. In the typed view, why should a dataset not be just unlabelled nested numeric storage?
