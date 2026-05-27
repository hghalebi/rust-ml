# R2 Systems: Measure Before Optimizing

## Goal

Make the tiny model measurable.

The learner should stop treating performance as vague and start naming the resources:

```text
time, memory, bytes moved, FLOPs, arithmetic intensity
```

## What You Build

Extend the R1 core with:

- timing around each model stage
- memory-size estimates for parameters, activations, and batches
- FLOP estimates for matrix-vector or matrix-matrix operations
- an attention baseline and a more cache-friendly attention variant
- a short parallelism design note for data, tensor, and pipeline splits

## Active Starter Crate

The first executable artifact is [`code/systems`](../../code/systems/README.md).

It starts with typed resource accounting before any real optimization:

```text
ActivationShape -> ElementCount -> Bytes
AttentionEstimate -> Flops
StageMeasurements -> median elapsed time
StageMeasurement -> ArithmeticIntensity
MemoryTransfer -> ElapsedNanos
```

Run it with:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 02_attention_flops
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 03_median_timing
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 05_memory_hierarchy
```

## Object/Map Preflight

Before implementation, write this preflight in your assignment notes:

- **Objects:** `ActivationShape`, `ElementCount`, `Bytes`, `BytesPerSecond`, `MemoryLevel`, `AttentionEstimate`, `Flops`, `StageMeasurements`, `ArithmeticIntensity`.
- **Maps:** count elements, estimate bytes, estimate attention FLOPs, collect repeated timings, summarize median time, compute arithmetic intensity, estimate memory-transfer time.
- **Composition path:** `ActivationShape -> ElementCount -> Bytes`, `Bytes + BytesPerSecond + MemoryLevel -> ElapsedNanos`, and `StageMeasurements -> MedianElapsed -> ArithmeticIntensity`.
- **Invariant to protect with newtypes:** bytes, bandwidth, FLOPs, elapsed time, and intensity are different units and must not be interchangeable numbers.

## Expected Deliverables

- a table or note that reports bytes, FLOPs, elapsed time, and arithmetic intensity with units
- at least three measurements for the same stage and input shape
- a median timing calculation that is visible in code or test output
- one hand-computable arithmetic-intensity fixture
- a short systems note that names the mathematical map that stayed unchanged
- a memory-hierarchy note that compares the same byte movement through two memory tiers

## Newtype And Category-Theory Lens

Use newtypes for:

- `Bytes`
- `Flops`
- `ElapsedNanos`
- `ArithmeticIntensity`
- `BytesPerSecond`
- `MemoryLevel`
- `BatchSize`
- `SequenceLength`
- `ModelWidth`
- `StageName`

Systems work is still composition:

```text
same mathematical map
  -> different implementation schedule
  -> different resource trace

same bytes
  -> different memory tier
  -> different transfer time
```

## Required Checks

- benchmark the same input size at least three times
- report median time instead of one lucky run
- test arithmetic-intensity calculations on a hand-computable case
- test memory-transfer calculations on a hand-computable bandwidth case
- document which optimization changes the implementation, not the mathematical result

## Assessment Rubric

- **Unit safety:** bytes, FLOPs, elapsed time, and intensity cannot be accidentally added or compared as the same value.
- **Measurement discipline:** repeated runs are summarized with a median, not a single convenient result.
- **Systems intuition:** the learner can explain whether a stage is compute-heavy or bandwidth-heavy.
- **Hierarchy intuition:** the learner can explain why the same bytes cost different time at different memory levels.
- **Mathematical preservation:** optimization changes the schedule or memory pattern without changing the intended function.

## Failure Signals

- timing is reported without the input shape or stage name that produced it
- one benchmark run is treated as a conclusion
- FLOPs and bytes are mixed through untyped arithmetic
- an optimization note claims speedup without saying which resource trace improved
- accelerator discussion names GPU/TPU hardware without tying it to bytes, bandwidth, or transfer time

## Suggested Repo Integration

Start from the active `code/systems` crate. Keep new resource quantities typed before connecting them to larger model code.

Keep CPU-first correctness before any accelerator-specific work.
