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
```

Run it with:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 02_attention_flops
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 03_median_timing
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
```

## Expected Deliverables

- a table or note that reports bytes, FLOPs, elapsed time, and arithmetic intensity with units
- at least three measurements for the same stage and input shape
- a median timing calculation that is visible in code or test output
- one hand-computable arithmetic-intensity fixture
- a short systems note that names the mathematical map that stayed unchanged

## Newtype And Category-Theory Lens

Use newtypes for:

- `Bytes`
- `Flops`
- `ElapsedNanos`
- `ArithmeticIntensity`
- `BatchSize`
- `SequenceLength`
- `ModelWidth`
- `StageName`

Systems work is still composition:

```text
same mathematical map
  -> different implementation schedule
  -> different resource trace
```

## Required Checks

- benchmark the same input size at least three times
- report median time instead of one lucky run
- test arithmetic-intensity calculations on a hand-computable case
- document which optimization changes the implementation, not the mathematical result

## Assessment Rubric

- **Unit safety:** bytes, FLOPs, elapsed time, and intensity cannot be accidentally added or compared as the same value.
- **Measurement discipline:** repeated runs are summarized with a median, not a single convenient result.
- **Systems intuition:** the learner can explain whether a stage is compute-heavy or bandwidth-heavy.
- **Mathematical preservation:** optimization changes the schedule or memory pattern without changing the intended function.

## Failure Signals

- timing is reported without the input shape or stage name that produced it
- one benchmark run is treated as a conclusion
- FLOPs and bytes are mixed through untyped arithmetic
- an optimization note claims speedup without saying which resource trace improved

## Suggested Repo Integration

Start from the active `code/systems` crate. Keep new resource quantities typed before connecting them to larger model code.

Keep CPU-first correctness before any accelerator-specific work.
