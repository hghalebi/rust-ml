# 06 Attention

Status: active.

This folder maps to course Module 5.

This module builds the conceptual bridge from vector-space models to token-to-token interaction.

## Outcomes

After this module, you should be able to:

- explain attention as a typed information-mixing map
- distinguish query, key, and value without collapsing them into one generic vector
- compute one scaled dot-product score by hand
- explain softmax as normalized focus
- trace `scores -> weights -> weighted value mixture`
- explain why raw numbers enter through `TryFrom` adapters before becoming attention roles
- read the crate's `std::ops` arithmetic as typed composition between newtypes
- run the companion attention examples and predict their outputs

## Lessons

1. [Tokens as Vectors in a Sequence](01-tokens-as-vectors-in-a-sequence.md)
2. [Query, Key, and Value Roles](02-query-key-value-roles.md)
3. [Scores, Weights, and Value Mixing](03-scores-weights-and-value-mixing.md)

## Practice

- [Attention exercises](exercises.md)
- [Attention solutions](solutions.md)

## Code Artifact

- Active crate: [`code/attention`](../../code/attention/README.md)

Run the examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 02_softmax_focus
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 03_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace
```

While reading the examples, keep the same mental translation in view:

```text
learner literal -> checked newtype -> typed operation -> next semantic object
```

For example, `QueryComponent * KeyComponent` creates a score contribution, and
`AttentionWeight * ValueComponent` creates one weighted value contribution. The
names are part of the lesson: they prevent attention from looking like untyped
array arithmetic.

## Reference Material

- [Transformer video transcript](../../references/transcripts/attention-is-all-you-need-transformer-model-explanation-inference-training.md)
- [Bahdanau et al. (2014)](../../references/papers/bahdanau-neural-machine-translation-2014.pdf)
- [Luong et al. (2015)](../../references/papers/luong-effective-approaches-attention-nmt-2015.pdf)
- [Vaswani et al. (2017)](../../references/papers/vaswani-attention-is-all-you-need-2017.pdf)
- [Raschka: LLMs From Scratch](../../references/repos/llms-from-scratch.md)

## Prerequisite

- Complete [05 MLP](../05-mlp/README.md)

## Before You Move On

You are ready for the Transformer module when you can explain this chain without treating it as magic:

```text
Query * Keys -> AttentionScores -> AttentionWeights
AttentionWeights * Values -> AttentionOutput
```
