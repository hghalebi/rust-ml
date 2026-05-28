# 09 Systems

Status: active.

This folder maps to course Module 8.

This module makes model systems measurable without turning performance into
guesswork. The learner names resource quantities as typed objects before trying
to optimize anything.

## Outcomes

After this module, you should be able to:

- explain activation memory as `shape -> elements -> bytes`
- estimate dense attention score and value-mixing FLOPs
- explain why repeated timings need a median
- compute arithmetic intensity as `FLOPs / bytes`
- compare the same byte movement through different memory tiers
- explain why public systems reports require reviewed public measurements
- run the companion systems examples and predict their outputs

## Lessons

1. [Shapes, Elements, Bytes, And FLOPs](01-shapes-elements-bytes-and-flops.md)
2. [Timing, Arithmetic Intensity, And Memory Hierarchy](02-timing-intensity-and-memory-hierarchy.md)
3. [The Public Systems Report Boundary](03-public-systems-report-boundary.md)

## Practice

- [Systems exercises](exercises.md)
- [Systems solutions](solutions.md)

## Code Artifact

- Active crate: [`code/systems`](../../code/systems/README.md)
- Assignment bridge: [R2 Systems](../../assignments/cs336-rust/02-systems.md)

Run the examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 02_attention_flops
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 03_median_timing
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 05_memory_hierarchy
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 06_public_report
```

Keep this translation in view:

```text
semantic shape -> resource count -> unit-safe report -> public review
```

Systems work is not separate from the model path. It is the same map viewed
through resource evidence.

## Prerequisite

- Complete [08 Language Modeling](../08-language-modeling/README.md)

## Before You Move On

You are ready for the kernels and parallelism parts of the CS336 Rust path when
you can explain these chains without replacing the units with loose numbers:

```text
ActivationShape -> ElementCount -> Bytes
AttentionEstimate -> Flops + Bytes
StageMeasurements -> ElapsedNanos
Flops / Bytes -> ArithmeticIntensity
Bytes / BytesPerSecond -> ElapsedNanos
ReviewedStageMeasurement -> PublicSystemsReport
```

You should also be able to say why an optimization note must name the resource
trace that improved, not only claim that code became faster.
