# 07 Transformer

Status: active.

This folder maps to course Module 6.

This module now teaches the Transformer encoder path in three complementary modes:

1. a narrative lesson about the problem, the paper, and the full encoder rhythm
2. a typed Rust lesson about semantic newtypes, `thiserror`, and architecture visibility
3. a low-cognitive-load chunk ladder using `English -> Algebra -> Rust`

## Goal

Understand what the Transformer is solving, how the encoder math works, and how to model that architecture cleanly in Rust.

## Role In The Course

This module is the current capstone after the attention bridge. It applies the same newtype-and-map discipline at Transformer encoder scale.

## Outcomes

After this module, you should be able to:

- explain why attention replaces one-token-at-a-time recurrence
- read scaled dot-product attention in English, algebra, and Rust
- trace one encoder block from token sequence to contextualized output
- explain why semantic newtypes help attention code stay readable
- explain where linear attention fits without confusing it with the 2017 paper
- validate core architecture hyperparameters before they become model code
- describe top-1 expert routing as a typed token-to-expert map

## Code Artifact

- Runnable crate: [`code/transformer`](../../code/transformer/README.md)

## Reference Material

- [Transformer video transcript](../../references/transcripts/attention-is-all-you-need-transformer-model-explanation-inference-training.md)
- [Bahdanau et al. (2014)](../../references/papers/bahdanau-neural-machine-translation-2014.pdf)
- [Luong et al. (2015)](../../references/papers/luong-effective-approaches-attention-nmt-2015.pdf)
- [Vaswani et al. (2017)](../../references/papers/vaswani-attention-is-all-you-need-2017.pdf)
- [Raschka: LLMs From Scratch](../../references/repos/llms-from-scratch.md)

## Prerequisite

- Complete [06 Attention](../06-attention/README.md)

## Lessons

1. [What Problem the Transformer Solves](01-tiny-transformer-from-first-principles.md)
2. [Typed Rust Transformer with Expressive Errors](02-typed-rust-transformer-with-linear-attention.md)
3. [Transformer Encoder in Small Chunks](03-transformer-encoder-in-small-chunks.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Current Focus

The current authored material covers:

- the motivation for attention-first sequence modeling
- the encoder-side math from query/key/value through residuals and normalization
- a semantic-newtype Rust crate with expressive shape errors
- a typed `TransformerConfig` that checks head width and estimates encoder parameters
- a typed `TopExpertRouter` that routes token scores to expert choices
- a runnable encoder demo
- a chunked study path for low-cognitive-load review

## Before You Move On

You are ready to leave this preview module when you can:

- explain the encoder block as `attention -> residual -> norm -> feed-forward -> residual -> norm`
- describe the roles of query, key, and value without collapsing them into one generic vector
- read the scaled dot-product attention equation and point to the matching Rust code shape
- explain why positional encodings are required once recurrence is removed
- explain why `d_model / head_count` must be validated before multi-head attention is built
- explain why a router must receive one score for every available expert
