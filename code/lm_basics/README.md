# lm_basics

Status: active.

This crate is the first executable companion for [R1 Basics](../../assignments/cs336-rust/01-basics.md) in the CS336 Rust equivalent track.

It teaches the first language-modeling boundary:

```text
RawText -> TokenTextSequence -> Vocabulary -> TokenIdSequence -> NextTokenBatch -> Logits -> Loss
```

## Owns

- assignment: [R1 Basics](../../assignments/cs336-rust/01-basics.md)
- track: [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)

## Current State

- active teaching crate
- whitespace tokenizer for tiny public examples
- raw literals enter through explicit `TryFrom` edge adapters
- checked vocabulary and token IDs
- next-token batch construction
- tiny trainable bigram language model
- cross-entropy loss and one gradient step

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_tokenize_and_encode.rs
  02_next_token_batch.rs
  03_uniform_loss.rs
  04_training_step.rs
```

## Learning Ladder

1. `01_tokenize_and_encode` turns raw text into checked token IDs.
2. `02_next_token_batch` creates input/target next-token pairs.
3. `03_uniform_loss` computes cross-entropy for a uniform model.
4. `04_training_step` applies one update and prints loss before and after.

## Category Lens

Read the language-modeling path as maps between typed text objects:

```text
RawText -> TokenTextSequence
TokenTextSequence + Vocabulary -> TokenIdSequence
TokenIdSequence -> NextTokenBatch
TokenId -> LogitRow -> Loss
NextTokenBatch + LearningRate -> TrainingStepTrace
```

The composition rule is vocabulary agreement. Token IDs, batches, logits, and
loss all have to live over the same `VocabularySize` before the maps compose.
In code, the first maps can be read as typed operations:

```rust
let ids = (&vocabulary * &tokens)?;
let loss = (&model * &batch)?;
```

The `*` operator is not magic multiplication here. It is the learner-facing
shape of a checked map: vocabulary application encodes tokens, and model
application measures the batch loss.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_lm_basics --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 02_next_token_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 03_uniform_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step
```

## Scope

This is intentionally a bigram model, not a Transformer.

It gives learners a small executable version of the language-modeling pipeline before they move into Transformer-scale architecture.

The public API is intentionally newtype-first: raw strings, indexes, and floats are accepted only at validation boundaries, then the rest of the pipeline moves through semantic Rust types.
