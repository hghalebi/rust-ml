# systems

Status: active.

This crate is the first executable companion for [R2 Systems](../../assignments/cs336-rust/02-systems.md) in the CS336 Rust equivalent track.

It teaches systems measurement as typed resource accounting:

```text
shape -> elements -> bytes
shape -> operations -> FLOPs
repeated timings -> median timing
FLOPs / bytes -> arithmetic intensity
bytes / bandwidth -> transfer time
ReviewedStageMeasurement* -> PublicSystemsReport
```

## Owns

- assignment: [R2 Systems](../../assignments/cs336-rust/02-systems.md)
- track: [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)

## Current State

- active teaching crate
- typed dimensions for batch size, sequence length, model width, rows, and columns
- typed resource values for bytes, FLOPs, elapsed time, and arithmetic intensity
- activation memory estimates
- matrix-vector FLOP and byte estimates
- dense self-attention FLOP and score-matrix memory estimates
- median timing over repeated stage measurements
- accelerator memory-tier transfer estimates without GPU-specific APIs
- reviewed stage measurements and public reports that reject restricted or private measurements

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_memory_accounting.rs
  02_attention_flops.rs
  03_median_timing.rs
  04_arithmetic_intensity.rs
  05_memory_hierarchy.rs
  06_public_report.rs
```

## Learning Ladder

1. `01_memory_accounting` turns a typed activation shape into element and byte counts.
2. `02_attention_flops` estimates dense attention score and value-mixing work.
3. `03_median_timing` shows why repeated runs need a median rather than one lucky timing.
4. `04_arithmetic_intensity` connects FLOPs and bytes moved.
5. `05_memory_hierarchy` compares the same byte movement through different memory tiers.
6. `06_public_report` checks that only public measurements can become learner-facing systems reports.

## Category Lens

Read systems work as maps from model shapes to resource traces:

```text
ActivationShape -> ElementCount -> Bytes
MatrixVectorShape -> Flops
AttentionEstimate -> Flops + Bytes
StageMeasurements -> ElapsedNanos
Flops + Bytes -> ArithmeticIntensity
Bytes + BytesPerSecond + MemoryLevel -> ElapsedNanos
ReviewedStageMeasurement* -> PublicSystemsReport
```

The composition rule is units. You can change the implementation schedule, but
the typed resource map must still say which shape, work count, byte movement,
and timing produced the measurement.

## Three Lenses

**Rust syntax:** `ReviewedStageMeasurement` pairs a `StageMeasurement` with a
`MeasurementVisibility` enum. `PublicSystemsReport::from_reviewed_measurements`
rejects restricted or private measurements before computing the public median.

**ML systems concept:** public performance notes must be reproducible teaching
evidence. A private host profile or restricted benchmark can be useful internally
without becoming part of the public course surface.

**Category-theory concept:** the report is a typed map from reviewed measurement
objects into a public object. The map exists only when each input object carries
the right publication class.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_systems --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 02_attention_flops
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 03_median_timing
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 05_memory_hierarchy
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 06_public_report
```

## Scope

This crate is CPU-first resource reasoning, not accelerator programming. The
memory-hierarchy example names accelerator-like tiers, but it stays a typed
estimate: no CUDA, Triton, device drivers, or vendor-specific APIs.

The goal is to name the systems quantities before optimizing anything. A learner should be able to say which mathematical map stayed the same, which implementation schedule changed, and which resource trace improved.
Public examples add one more rule: restricted or private measurements do not
enter learner-facing systems reports.
