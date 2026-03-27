# transformer

Status: active.

This crate is the executable companion for [07 Transformer](../../lessons/07-transformer/README.md).

It models the encoder-side Transformer path with:

- raw math types: `DenseVector`, `DenseMatrix`
- semantic model types: `TokenEmbedding`, `Query`, `Key`, `Value`, `TokenSequence`
- expressive `thiserror` diagnostics through `ModelError`
- standard scaled dot-product attention
- a simplified `LinearAttentionHead` for architecture comparison
- multi-head attention
- sinusoidal positional encodings
- layer normalization
- position-wise feed-forward layers
- one `TransformerEncoderBlock`

## Layout

```text
src/
  error.rs
  math.rs
  types.rs
  attention.rs
  transformer.rs
  lib.rs
examples/
  encoder_demo.rs
```

## Run

```bash
cargo test --manifest-path code/transformer/Cargo.toml
```

```bash
cargo run --example encoder_demo --manifest-path code/transformer/Cargo.toml
```

## Scope

This crate is intentionally educational, not production-grade. It does not include:

- tokenization
- learned embeddings from vocabulary ids
- masking
- decoder cross-attention
- dropout
- autograd
- optimizers
- GPU kernels

The goal is architecture clarity first.
