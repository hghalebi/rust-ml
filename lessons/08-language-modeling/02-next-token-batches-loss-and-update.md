# Next-Token Batches, Loss, And Updates

## Overview

Language modeling asks one repeated question:

```text
Given the current token, what token should come next?
```

The tiny sequence:

```text
red blue red
```

creates two next-token pairs:

```text
red  -> blue
blue -> red
```

In ID form:

```text
inputs  = [0, 1]
targets = [1, 0]
```

## Learning Goals

- build next-token pairs from adjacent IDs
- explain why inputs and targets must have the same length
- read uniform loss as an honest baseline
- explain why one update can lower loss on the same tiny batch
- read `TinyBigramLanguageModel * NextTokenBatch -> Loss` as typed composition

## Plain-English Explanation

A next-token batch is a supervised training example made from a sequence.

Each input token needs exactly one target token:

```text
input at position 0 -> target at position 1
input at position 1 -> target at position 2
```

The model first starts uncertain. With two possible next tokens and equal logits,
the loss is:

```text
ln(2) = 0.6931...
```

After one tiny update, the model gives more weight to the observed next-token
pairs, so the loss on the same batch falls.

## Algebra Form

For a token ID sequence:

```text
x = [x_0, x_1, x_2]
```

build:

```text
inputs  = [x_0, x_1]
targets = [x_1, x_2]
```

The loss map is:

```text
Model * NextTokenBatch -> Loss
```

One update changes the model parameters:

```text
(Model, NextTokenBatch, LearningRate) -> TrainingStepTrace
```

The trace records:

```text
loss_before -> loss_after
```

## Rust Form

```rust
use rust_ml_lm_basics::{
    LearningRate, NextTokenBatch, RawText, TinyBigramLanguageModel, Vocabulary,
    WhitespaceTokenizer,
};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let text = RawText::try_from("red blue red")?;
    let tokens = WhitespaceTokenizer.tokenize(&text)?;
    let vocabulary = Vocabulary::from_tokens(&tokens)?;
    let ids = (&vocabulary * &tokens)?;
    let batch = NextTokenBatch::from_sequence(&ids)?;

    let mut model = TinyBigramLanguageModel::uniform(vocabulary.size())?;
    let baseline_loss = (&model * &batch)?;
    let trace = model.train_one_step(&batch, LearningRate::try_from(0.5)?)?;

    println!("baseline loss = {baseline_loss:.4}");
    println!("loss before   = {:.4}", trace.loss_before());
    println!("loss after    = {:.4}", trace.loss_after());

    Ok(())
}
```

The model and batch must agree on vocabulary size before the loss map composes.
The learning rate is also a semantic value, so a non-positive step size is
rejected before it can update the model.

## Why This Matters

This is the smallest complete language-modeling training loop in the repo.

It does not pretend to be a production Transformer. It gives the learner a
visible path from checked tokens to loss, then from loss to a parameter update.
Once this path is clear, the larger Transformer and systems modules have a
stable foundation.

## Concept Trace

- **Object/newtype:** `NextTokenBatch`, `TinyBigramLanguageModel`, `Loss`, `LearningRate`, and `TrainingStepTrace`.
- **Invariant:** every input token needs exactly one target token, and model and batch must share vocabulary size.
- **Map:** token IDs -> next-token batch -> loss -> training-step trace.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step`.
- **Failure signal:** you build inputs and targets with different lengths or update with an unvalidated learning-rate literal.

## Short Practice

1. For `[0, 1, 0]`, what are the inputs and targets?
2. Why does a uniform two-token model have loss near `0.6931`?
3. Which type records loss before and after one update?
4. Which map computes loss from a model and a batch?
