# Datasets, Epochs, and Token Targets

## Overview

One training step teaches the mechanism.

Real training adds repetition and scale:

- many examples instead of one
- many parameter updates instead of one
- losses averaged across datasets instead of inspected one at a time
- targets that may be a token index instead of a single scalar label

That is why training is a second beast.

## Learning Goals

- define dataset, example, batch, and epoch in plain English
- explain why average loss across data matters more than one-example loss
- compare scalar targets with token targets
- read the cross-entropy shape used in next-token training

## Plain-English Explanation

### A dataset is repeated evidence

One example can teach you the mechanics of learning, but it cannot define a
useful model by itself.

A dataset is the repeated evidence that forces the parameters to become good for
many cases, not just one lucky case.

### An epoch is one full pass over the evidence

If a dataset has four examples and you process all four once, that is one epoch.

If you do it again, that is a second epoch.

The model learns because the optimizer keeps making small corrections as it sees
the dataset over and over.

### Token targets change the output shape

In the neuron lessons, the target is one scalar such as `0` or `1`.

In language modeling, the target is often "which token should come next?"

That means the model output is no longer one number. It becomes a whole score
vector over the vocabulary, and the target selects one index inside that vector.

### The first next-token model can still be tiny

You do not need full self-attention to introduce token-target training.

A clean bridge is:

```text
token id -> embedding vector -> linear lm_head -> logits -> softmax loss
```

That model is still small enough to train with an explicit SGD loop, but it
already teaches the real sequence-learning shift:

- the input is discrete token identity
- the model must produce one score per vocabulary token
- the target picks the correct next token, not one scalar regression value

### Why this feels like a different animal

The local learning rule is still gradients plus updates.

What changes is the scale:

- bigger outputs
- larger datasets
- more parameters
- more structure in the target

Once token embeddings, position embeddings, attention projections,
feed-forward layers, and the language-model head all become trainable together,
manual gradient bookkeeping stops being a pleasant teaching tool and starts
becoming a strong argument for autograd.

The training principle is continuous. The workload is not.

## Algebra Form

Average loss across a dataset:

```math
J = \frac{1}{N}\sum_{i=1}^{N} L_i
```

Token probabilities from logits:

```math
p_i = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}
```

Cross-entropy loss for the correct token:

```math
L = -\log p_{\text{target}}
```

Gradient shortcut for logits:

```math
\frac{dL}{d\text{logits}} = p - \text{one\_hot(target)}
```

## Rust Form

```rust
#[derive(Debug, Clone, Copy)]
struct Example {
    x: f64,
    target: f64,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn average_squared_error(weight: f64, dataset: &[Example]) -> f64 {
    let total: f64 = dataset
        .iter()
        .map(|example| {
            let prediction = sigmoid(weight * example.x);
            (prediction - example.target).powi(2)
        })
        .sum();

    total / dataset.len() as f64
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|value| value / sum).collect()
}

fn cross_entropy_loss(logits: &[f64], target_index: usize) -> f64 {
    let probabilities = softmax(logits);
    -probabilities[target_index].ln()
}

fn main() {
    let dataset = vec![
        Example { x: 0.0, target: 0.0 },
        Example { x: 1.0, target: 1.0 },
        Example { x: 2.0, target: 1.0 },
    ];

    let average_loss = average_squared_error(0.9, &dataset);

    let logits = vec![2.4, 0.3, -1.1];
    let token_target = 0;
    let token_loss = cross_entropy_loss(&logits, token_target);

    println!(
        "dataset average loss = {average_loss:.4}, token loss = {token_loss:.4}"
    );
}
```

## Why This Matters

This lesson is the conceptual bridge between "a neuron learns from one labeled
example" and "a sequence model learns to predict the next token from a whole
distribution."

The mechanism is still learning by gradients. The first size jump is a tiny
embedding-plus-head model. The next jump is a full trainable Transformer.

## Short Practice

1. In one sentence, what does an epoch count?
2. Why is average dataset loss more informative than loss on one hand-picked example?
3. What is the key difference between a scalar target and a token target?
