# Inference Exercises

## Exercise 1: Read the typed state loop

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 03_kv_cache_trace
```

Answer:

1. How many generated tokens are produced?
2. How many cache entries are produced?
3. What role does the first cache entry have in a prompt continuation?
4. Why does context growth happen via `ContextTokens + TokenId -> ContextTokens`?

## Exercise 2: Validate the public release rule

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 05_public_trace
```

Answer:

1. Which visibility value allows publication?
2. Which value blocks publication?
3. What exact error behavior should you expect from a blocked review?

## Exercise 3: Compare two latency assemblies

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 04_latency_budget
```

Answer:

1. Which two typed values are added to estimate total latency?
2. Why must the per-token term depend on `GeneratedTokenCount`?
3. What does `WithinBudget` mean in one sentence?

## Failure Signals

- You treat `LatencyMillis` values as interchangeable raw integers.
- You compare public readiness by checking trace text instead of `PublicDecodeTrace::from_reviewed_trace`.
- You let generated count and cache-entry count drift without checking alignment.
- You skip the visibility check and reason from `TraceVisibility` as if it were a UI flag.

## Debugging Hints

- Read `prompt`, `context`, and `cache` together from one trace print block.
- In publication errors, always look at the constructor path that returned `Err(...)`.
- For latency mismatch bugs, print only typed units (`LatencyMillis`, `GeneratedTokenCount`, and `LatencyBudget`).
- If values seem inconsistent, rebuild the same call using one example file from `code/inference/examples` and compare behavior.
