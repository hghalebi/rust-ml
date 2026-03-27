# 07 Transformer

Status: active.

This module now teaches the Transformer encoder path in three complementary modes:

1. a narrative lesson about the problem, the paper, and the full encoder rhythm
2. a typed Rust lesson about semantic newtypes, `thiserror`, and architecture visibility
3. an ADHD-friendly chunk ladder using `English -> Algebra -> Rust`

## Goal

Understand what the Transformer is solving, how the encoder math works, and how to model that architecture cleanly in Rust.

## Outcomes

After this module, you should be able to:

- explain why attention replaces one-token-at-a-time recurrence
- read scaled dot-product attention in English, algebra, and Rust
- trace one encoder block from token sequence to contextualized output
- explain why semantic newtypes help attention code stay readable
- explain where linear attention fits without confusing it with the 2017 paper

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

1. [Lesson 17: What Problem the Transformer Solves](01-tiny-transformer-from-first-principles.md)
2. [Lesson 18: Typed Rust Transformer with Expressive Errors](02-typed-rust-transformer-with-linear-attention.md)
3. [Lesson 19: Transformer Encoder in Small Chunks](03-transformer-encoder-in-small-chunks.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Current Focus

The current authored material covers:

- the motivation for attention-first sequence modeling
- the encoder-side math from query/key/value through residuals and normalization
- a semantic-newtype Rust crate with expressive shape errors
- a runnable encoder demo
- a chunked study path for low-cognitive-load review
