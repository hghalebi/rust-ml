# 07 Transformer

Status: started.

This folder maps to course Modules 6 and 7. The first authored lesson here focuses on the Transformer block assembly step.

## Goal

Assemble a tiny transformer-style block from the ingredients introduced earlier.

## Outcomes

After this module starts taking shape, you should be able to:

- describe a Transformer block as attention plus feed-forward layers plus residual connections
- explain what each sublayer contributes
- trace a toy Transformer block in English, algebra, and Rust
- explain why linear attention exists and how typed Rust can model Transformer concepts clearly
- walk through a standard Transformer encoder in small, compile-sized chunks

## Code Artifact

- Runnable crate: [`code/transformer`](../../code/transformer/README.md)

## Reference Material

- [Transformer video transcript](../../references/transcripts/attention-is-all-you-need-transformer-model-explanation-inference-training.md)
- [Bahdanau et al. (2014)](../../references/papers/bahdanau-neural-machine-translation-2014.pdf)
- [Luong et al. (2015)](../../references/papers/luong-effective-approaches-attention-nmt-2015.pdf)
- [Vaswani et al. (2017)](../../references/papers/vaswani-attention-is-all-you-need-2017.pdf)

## Prerequisite

- Complete [06 Attention](../06-attention/README.md)

## Lessons

1. [Lesson 17: A Tiny Transformer (From First Principles)](01-tiny-transformer-from-first-principles.md)
2. [Lesson 18: Typed Rust Transformer with Linear Attention](02-typed-rust-transformer-with-linear-attention.md)
3. [Lesson 19: Transformer Encoder in Small Chunks](03-transformer-encoder-in-small-chunks.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Current Focus

This module is still partial. The current lesson gives you the first full Transformer-block assembly step:

- self-attention recap
- feed-forward sublayer
- residual connections
- a toy end-to-end block in Rust
- a typed Rust path from vectors and linear layers to linear attention
- a standard encoder walkthrough broken into short, low-cognitive-load chunks
