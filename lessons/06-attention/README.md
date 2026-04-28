# 06 Attention

Status: planned.

This folder maps to course Module 5.

This module will build the conceptual bridge from vector-space models to token-to-token interaction.

## Role In The Course

This planned module will connect the MLP shape-flow story to token interaction. It should make the Transformer preview feel like a continuation rather than a jump.

## Goal

Introduce token representations, projections, masks, attention scores, normalized weights, and weighted sums.

## Planned Lesson Ladder

1. tokens as vectors in a sequence
2. query, key, and value projections
3. causal masks and why future tokens are hidden during next-token training
4. attention scores, weights, and weighted sums

## Planned Practice

- compute one attention score by hand
- explain softmax as normalized focus
- explain why a causal mask blocks future positions
- trace how one token mixes information from other tokens

## Code Artifact

- Future crate: [`code/attention`](../../code/attention/README.md)

## Reference Material

- [Transformer video transcript](../../references/transcripts/attention-is-all-you-need-transformer-model-explanation-inference-training.md)
- [Bahdanau et al. (2014)](../../references/papers/bahdanau-neural-machine-translation-2014.pdf)
- [Luong et al. (2015)](../../references/papers/luong-effective-approaches-attention-nmt-2015.pdf)
- [Vaswani et al. (2017)](../../references/papers/vaswani-attention-is-all-you-need-2017.pdf)
- [Raschka: LLMs From Scratch](../../references/repos/llms-from-scratch.md)

## Prerequisite

- Complete [05 MLP](../05-mlp/README.md)

## Planned Outcome

Be able to explain query, key, value, masks, attention weights, and weighted sums in plain English and as Rust loops.
