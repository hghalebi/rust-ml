# transformer

Status: active.

This crate is the executable companion for [07 Transformer](../../lessons/07-transformer/README.md).

## Owns

- lesson module: [07 Transformer](../../lessons/07-transformer/README.md)
- package: `rust_ml_transformer`

## Current State

It models the encoder-side Transformer path with:

- typed math containers: `DenseVector`, `DenseMatrix`
- typed architecture configuration: `TransformerConfig`, `LayerCount`, `HeadCount`, `FeedForwardWidth`
- typed expert routing: `ExpertScores`, `TopExpertRouter`, `ExpertRoute`, `ExpertBank`
- semantic model types: `TokenEmbedding`, `Query`, `Key`, `Value`, `TokenSequence`
- readable `std::ops` arithmetic for typed projection, score, positional, and residual operations
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
  architecture.rs
  error.rs
  experts.rs
  math.rs
  types.rs
  attention.rs
  transformer.rs
  lib.rs
examples/
  encoder_demo.rs
```

## Learning Ladder

1. `architecture_config` validates a tiny encoder architecture and estimates its repeated-block parameter budget.
2. `expert_routing` routes one token to the highest-scoring typed expert and applies that expert.
3. `encoder_demo` builds a tiny typed encoder block.
4. The unit tests exercise architecture checks, expert routing, vectors, matrices, attention heads, positional encodings, layer normalization, feed-forward maps, and encoder blocks.
5. The Transformer lessons explain the same path in English, algebra, and Rust.

## Category Lens

Read the encoder as composed maps over token sequences:

```text
TokenSequence + PositionEncoding -> TokenSequence
TokenSequence -> MultiHeadAttention -> TokenSequence
TokenSequence -> LayerNorm -> TokenSequence
TokenSequence -> FeedForward -> TokenSequence
EncoderBlock -> Encoder
```

The composition rule is `d_model`. Residual addition, attention output,
normalization, and feed-forward output must all return to the same token object
shape before the next block can run.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_transformer
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example encoder_demo
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example architecture_config
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example expert_routing
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

## Typed Operation Model

The crate keeps named functions such as `add_token_embeddings` and
`add_sequences` for learners who want explicit signposts. Internally, the same
ideas are also implemented with Rust operation traits:

```text
&TokenEmbedding + &PositionEncoding -> TokenEmbedding
&TokenEmbedding + &TokenEmbedding -> TokenEmbedding
&TokenSequence + &TokenSequence -> TokenSequence
&QueryProjection * &TokenEmbedding -> Query
&KeyProjection * &TokenEmbedding -> Key
&ValueProjection * &TokenEmbedding -> Value
&OutputProjection * &ConcatenatedHeadOutput -> TokenEmbedding
&FeedForwardProjection1 * &TokenEmbedding -> HiddenPreActivation
&FeedForwardProjection2 * &HiddenActivation -> TokenEmbedding
&Query * &Key -> AttentionScore
&AttentionWeights * &ValueSequence -> AttentionOutput
&Query + &ProjectionBias -> Query
&Key + &ProjectionBias -> Key
&Value + &ProjectionBias -> Value
&TokenEmbedding + &ProjectionBias -> TokenEmbedding
&DenseMatrix * &DenseVector -> DenseVector
&DenseVector * &DenseVector -> ModelScalar
&DenseVector * ModelScalar -> Result<DenseVector, ModelError>
```

This makes the Transformer path read closer to the algebra while preserving the
newtype boundary: projections produce the role they are meant to produce,
query-key multiplication returns a scaled attention score, bias addition stays
typed, residual addition is checked, and mismatched shapes still return
`ModelError`.

Raw learner literals enter through explicit scalar validation such as
`ModelScalar::try_from(...)` and dimension validation such as
`VectorLength::try_from(...)`. Vectors and matrices are then built with typed
constructors such as `DenseVector::new([ModelScalar, ...])` and
`DenseMatrix::from_rows([[ModelScalar, ...], ...])`. Once values cross that
boundary, public constructors and operations use typed values rather than
generic raw primitive or raw container adapters.

Architecture hyperparameters follow the same rule. A learner validates layer
counts, head counts, model width, and feed-forward width before they become a
`TransformerConfig`. The config then exposes maps such as:

```text
VectorLength / HeadCount -> VectorLength
TransformerConfig -> ParameterCount
```

That keeps the CS336-style architecture discussion concrete: hyperparameters are
not loose numbers, they are typed choices with invariants.

The expert-routing example adds a small attention-alternatives bridge:

```text
ExpertScores -> ExpertChoice
TokenIndex + ExpertChoice -> ExpertRoute
ExpertBank + ExpertRoute + TokenEmbedding -> TokenEmbedding
```

It intentionally teaches top-1 routing before distributed experts, load
balancing, or auxiliary losses. The invariant is simple: the router must receive
exactly one score per available expert, and the chosen route must point to an
expert that exists in the bank.
