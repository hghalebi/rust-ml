# LLMs From Scratch

External repository reference:

- Repo: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main)
- Author: Sebastian Raschka
- Context: official code repository for the book *Build a Large Language Model (From Scratch)*

## Why it matters here

This repository is useful as an inspiration source for the `rust-ml` course because it takes a step-by-step, first-principles path through:

- text data and tokenization
- attention mechanisms
- GPT-style model assembly
- pretraining and finetuning workflows
- educational bonus material around architecture variants and efficiency

## How to use it in `rust-ml`

Use it as:

- curriculum inspiration for later modules
- sequencing inspiration for attention, Transformer, and training topics
- implementation-shape inspiration when comparing toy educational code to fuller model code

Do not use it as a direct code template for this repo. It is a Python/PyTorch project, while `rust-ml` is teaching the same ideas through plain English, algebra, and Rust.

## Especially relevant areas

Based on the current upstream repository structure, the most relevant parts for this course are:

- `ch03/` for attention mechanisms
- `ch04/` for GPT-style model assembly
- `ch05/` for pretraining workflows
- `appendix-A/` for PyTorch setup and foundations
- bonus material around efficient attention and architecture alternatives

## Note on provenance

This is an external source of inspiration, not bundled source material. The canonical course content for this repo remains in `lessons/`.
