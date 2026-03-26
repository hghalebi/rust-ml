# transformer

Status: active.

This directory is now a real Rust crate that implements a tiny typed Transformer-style block with both standard self-attention and a simplified linear-attention variant.

The current crate also uses a newtype-heavy public API so meaning-bearing primitives are modeled explicitly instead of being passed around as raw `f32` and `usize` values.

## Owns

- lesson module: [07 Transformer](../../lessons/07-transformer/README.md)

## Current Purpose

- provide a small, readable Rust implementation of:
  - vectors and matrices
  - newtype-wrapped scalars, dimensions, counts, and indexes
  - linear layers
  - sequences
  - standard self-attention
  - linear attention
  - feed-forward layers
  - a minimal Transformer block
- give the lessons a concrete code target that can be tested with Cargo

## Run

```bash
cargo test --manifest-path code/transformer/Cargo.toml
```

## Scope

This crate is intentionally educational, not production-grade. It does not include:

- token embeddings
- positional encoding
- masking
- multi-head attention
- autograd
- optimizers
- GPU kernels

The goal is clarity first.
