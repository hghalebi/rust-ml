# R1 Basics: A Tiny Language Model Core

## Goal

Build the first CPU-first Rust path from text to loss.

This assignment turns the language-modeling problem into typed maps:

```text
RawText -> TokenSequence -> TokenIds -> Logits -> Loss
```

## What You Build

Create an original Rust implementation with these parts:

- a tiny tokenizer interface
- a vocabulary type with checked token IDs
- a batch type that separates input tokens from target tokens
- a minimal Transformer or Transformer-like forward pass
- a cross-entropy loss calculation
- a small optimizer step over a toy parameter set

## Active Starter Crate

The first executable artifact is [`code/lm_basics`](../../code/lm_basics/README.md).

It intentionally starts with a tiny bigram language model before moving to Transformer-scale code. That keeps the first language-modeling path inspectable:

```text
RawText -> TokenTextSequence -> Vocabulary -> TokenIdSequence -> NextTokenBatch -> Logits -> Loss -> Update
```

Run it with:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 02_next_token_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 03_uniform_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step
```

## Object/Map Preflight

Before implementation, write this preflight in your assignment notes:

- **Objects:** `RawText`, `TokenTextSequence`, `Vocabulary`, `TokenIdSequence`, `NextTokenBatch`, `Logits`, `Loss`.
- **Maps:** tokenize text, build vocabulary, encode tokens, build next-token pairs, score logits, compute loss, apply update.
- **Composition path:** `RawText -> TokenTextSequence -> Vocabulary -> TokenIdSequence -> NextTokenBatch -> Logits -> Loss -> Update`.
- **Invariant to protect with newtypes:** a token ID is valid only inside the vocabulary that produced it.

## Expected Deliverables

- a short note explaining the typed path from `RawText` to `Loss`
- a tokenizer and vocabulary path that rejects invalid token IDs
- a next-token batch constructor that keeps inputs and targets aligned
- a hand-computable loss fixture with the expected value written down
- one runnable example that shows loss before and after a tiny update

## Newtype And Category-Theory Lens

Use newtypes for:

- `RawText`
- `Token`
- `TokenIndex`
- `TokenId`
- `TokenCount`
- `VocabularySize`
- `ContextLength`
- `Position`
- `Logit`
- `Loss`
- `LearningRate`

Raw literals should enter through explicit `TryFrom` adapters. For example, a tiny input string becomes `RawText` before tokenization, and a learning-rate literal becomes `LearningRate` before optimization.

Read the model as composition:

```text
TokenIds -> Embeddings -> ContextualStates -> Logits -> Loss
```

## Required Checks

- reject token IDs outside the vocabulary
- reject batches whose input and target lengths do not match
- test loss on a tiny hand-computable example
- run one training step and print loss before and after

## Assessment Rubric

- **Typed meaning:** token text, token IDs, positions, logits, loss, and learning rate are not interchangeable raw primitives.
- **Compositional clarity:** each map in the pipeline has a clear input type, output type, and validation point.
- **Numerical honesty:** the toy loss result is small enough for a learner to verify by hand.
- **Executable learning:** examples run locally and show the learner what changed after each step.

## Failure Signals

- a raw integer can cross from tokenization into the model as if it were a valid token ID
- batch construction can silently drop or pad a target without saying why
- the loss fixture only checks that code runs, not that the number is correct
- the update example changes parameters without showing whether loss moved

## Suggested Repo Integration

Start from the existing typed style in `code/neuron`, `code/mlp`, and `code/transformer`.

Do not optimize yet. Correct shape flow and clear types matter first.
