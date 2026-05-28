# attention

Status: active.

This crate is the executable companion for [06 Attention](../../lessons/06-attention/README.md).

It teaches one scaled dot-product attention head before the full Transformer module.

## Owns

- lesson module: [06 Attention](../../lessons/06-attention/README.md)

## Current State

- active teaching crate
- typed query, key, value, score, weight, and output roles
- explicit `TryFrom` adapters where raw learner literals enter the system
- readable typed arithmetic through `std::ops` for projection products, score contributions, and weighted value mixing
- stable softmax and weighted-sum helpers
- learner-visible attention traces
- public trace review boundary for learner-facing attention evidence
- no multi-head claim; this crate teaches the single-head mechanism first

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_score_one_pair.rs
  02_softmax_focus.rs
  03_weighted_sum.rs
  04_attention_trace.rs
  05_public_trace.rs
```

## Learning Ladder

1. `01_score_one_pair` computes one scaled query-key score.
2. `02_softmax_focus` turns raw scores into normalized focus weights.
3. `03_weighted_sum` mixes value vectors.
4. `04_attention_trace` prints the whole score-to-output path for one query token.
5. `05_public_trace` shows how a reviewed trace becomes publishable learner-facing material.

Each example follows the same discipline:

```text
raw numbers -> TryFrom boundary -> semantic newtypes -> typed operations
```

That means learners see the mathematical roles directly in the code. A query
component can multiply a key component to create a score contribution, but it is
not interchangeable with a value component or a projection bias.

## Category Lens

Read one attention head as a composition of token-role maps:

```text
TokenEmbedding -> Query
TokenEmbedding -> Key
TokenEmbedding -> Value
Query * Key -> AttentionScore
AttentionScores -> AttentionWeights
AttentionWeights * ValueSequence -> AttentionOutput
ReviewedAttentionTrace -> PublicAttentionTrace
```

The composition rule is role agreement. Queries compare with keys, weights mix
values, and the output keeps the value width.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_attention --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 02_softmax_focus
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 03_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace
```

## Scope

This crate intentionally stays smaller than `code/transformer`.

Its job is to make this chain inspectable:

```text
TokenEmbedding -> Query
TokenEmbedding -> Key
TokenEmbedding -> Value
Query * Key -> AttentionScore
AttentionScores -> AttentionWeights
AttentionWeights * Values -> AttentionOutput
ReviewedAttentionTrace -> PublicAttentionTrace
```

## Typed Operation Model

The crate deliberately avoids a generic `Vec<f64>` public API for attention
roles. Instead, it uses newtypes such as `QueryComponent`, `KeyComponent`,
`AttentionWeight`, and `AttentionOutputComponent`.

The arithmetic is implemented with standard Rust operation traits:

```text
TokenComponent * ProjectionWeight -> ProjectionProduct
ProjectionOutput + ProjectionProduct -> ProjectionOutput
QueryComponent * KeyComponent -> ScoreContribution
AttentionScore + ScoreContribution -> AttentionScore
&Query * &Key -> AttentionScore
AttentionWeight * ValueComponent -> WeightedValueComponent
AttentionOutputComponent + WeightedValueComponent -> AttentionOutputComponent
&AttentionWeights * &ValueSequence -> AttentionOutput
ReviewedAttentionTrace -> PublicAttentionTrace
```

This keeps the category intuition concrete: attention is a composition of small
maps between meaningful objects, not a bag of interchangeable floating point
arrays.

The public trace boundary keeps that same discipline at the publication edge:
an `AttentionTrace` is evidence, while a `PublicAttentionTrace` is evidence that
has been explicitly reviewed for learner-facing release.
