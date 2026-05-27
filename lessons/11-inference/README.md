# Inference and Public Decode Traces

Status: active.

This folder maps to course Module 10.

## Outcomes

After this module, a learner should be able to:

- describe autoregressive decoding as repeated state updates,
- explain why the cache and generated tokens must stay aligned step by step,
- use typed sampling controls instead of raw `top_k` and `temperature` numbers,
- estimate latency with explicit budget and generated-token units,
- enforce a public boundary that blocks restricted or private inference traces.

## Lessons

1. [Autoregressive Decoding as a Typed State Trace](01-autoregressive-decoding-state-trace.md)
2. [The Public Decode Boundary and Typed Latency](02-public-decode-boundary-and-latency.md)

## Practice

- [exercises.md](exercises.md)
- [solutions.md](solutions.md)

## Code Artifact

This module maps to [`code/inference`](../../code/inference/README.md), the Rust crate that models:

```text
DecodeRequest + ContextTokens -> DecodeTrace
DecodeTrace -> DecodeLoopState -> LatencyReport
ReviewedDecodeTrace -> PublicDecodeTrace
```

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 01_greedy_decode
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 02_sampling_controls
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 03_kv_cache_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 04_latency_budget
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 05_public_trace
```

## Prerequisite

- [08 Language Modeling](../08-language-modeling/README.md) for token types and next-token batching
- [09 Systems](../09-systems/README.md) for time and budget intuition
- [10 Kernels](../10-kernels/README.md) for typed arithmetic discipline

## Before You Move On

- Can you explain how `ContextTokens`, `GeneratedTokens`, and `KvCache` are updated together in each decode step?
- Can you explain why `LatencyMillis + LatencyMillis` and `LatencyMillis * GeneratedTokenCount` are safer than mixing raw `u64` values?
- Can you state the exact condition that blocks `PublicDecodeTrace::from_reviewed_trace`?

If all three are clear, you are ready to continue into alignment and evaluation tracks in the CS336 path.
