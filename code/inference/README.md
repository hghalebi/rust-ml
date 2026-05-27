# inference

Status: active.

This crate is the first executable companion for CS336-style inference in the [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md) track.

It teaches autoregressive inference as a typed trace:

```text
PromptTokens + DecodeRequest -> DecodeTrace
```

## Owns

- lecture direction: inference in [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)
- package: `rust_ml_inference`

## Current State

- active teaching crate
- typed prompt tokens, token IDs, vocabulary sizes, context windows, sampling controls, decode steps, KV-cache entries, and latency budgets
- deterministic greedy and top-k selection over a tiny next-token model
- KV-cache trace that separates prompt-prefix entries from generated-token entries
- typed `std::ops` arithmetic for context growth, generated-token accumulation, cache insertion, temperature scaling, token budgets, and latency estimates
- expressive `thiserror` diagnostics through `InferenceError`

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_greedy_decode.rs
  02_sampling_controls.rs
  03_kv_cache_trace.rs
  04_latency_budget.rs
```

## Learning Ladder

1. `01_greedy_decode` generates a tiny deterministic phrase with greedy decoding.
2. `02_sampling_controls` shows top-k and temperature as semantic controls, not loose numbers.
3. `03_kv_cache_trace` prints prompt and generated entries in the toy KV cache.
4. `04_latency_budget` estimates prefill plus per-token generation latency with typed units.

## Category Lens

Read inference as a composition that grows a state:

```text
ContextTokens + TokenId -> ContextTokens
KvCache + KvCacheEntry -> KvCache
GeneratedTokens + TokenId -> GeneratedTokens
LatencyMillis + LatencyMillis -> LatencyMillis
```

The composition rule is state preservation. Each generated token must update
the context, the generated sequence, and the cache together, or the trace stops
being trustworthy.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_inference --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 01_greedy_decode
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 02_sampling_controls
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 03_kv_cache_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 04_latency_budget
```

## Scope

This crate intentionally does not implement a production serving engine, random
sampling, paged attention, speculative decoding, or GPU kernels.

The goal is to teach the invariants first: prompt tokens must fit the context
window, token IDs must fit the vocabulary, sampling controls must be typed, the
KV cache must stay aligned with generated steps, and latency estimates should
carry units.
