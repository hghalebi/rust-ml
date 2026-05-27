# Timing, Arithmetic Intensity, And Memory Hierarchy

## Overview

After counting bytes and FLOPs, systems work asks how expensive those quantities
are in time.

One timing is weak evidence. Repeated measurements are stronger. FLOPs divided
by bytes gives arithmetic intensity. Bytes divided by bandwidth gives an
estimated transfer time.

## Learning Goals

- explain why repeated measurements need a median
- compute arithmetic intensity from FLOPs and bytes
- compare the same byte movement through two memory tiers
- distinguish elapsed time from byte movement and bandwidth
- read memory hierarchy as a typed map, not hardware folklore

## Plain-English Explanation

A timing trace should name:

```text
what stage was measured
how long it took
how many FLOPs it represents
how many bytes moved
```

Arithmetic intensity asks:

```text
How much arithmetic work happened per byte moved?
```

The example:

```text
1024 FLOPs / 2240 bytes = 0.4571 FLOPs/byte
```

Memory hierarchy asks a different question:

```text
If the same bytes move through a faster or slower tier, how does time change?
```

The same `16384 bytes` can cost `3 ns` through a high-bandwidth local path and
`512 ns` through a slower host-memory path.

## Algebra Form

Median timing:

```text
StageMeasurements -> median elapsed time
```

Arithmetic intensity:

```text
ArithmeticIntensity = Flops / Bytes
```

Transfer estimate:

```text
ElapsedNanos = Bytes / BytesPerSecond
```

The composition is:

```text
StageMeasurement -> ArithmeticIntensity
MemoryTransfer -> ElapsedNanos
```

## Rust Form

```rust
use rust_ml_systems::{
    Bytes, BytesPerSecond, ColumnCount, ElapsedNanos, ElementSize, MatrixVectorShape,
    MemoryLevel, MemoryTransfer, RowCount, StageMeasurement, StageName,
};

fn main() -> Result<(), rust_ml_systems::Error> {
    let shape = MatrixVectorShape::new(RowCount::try_from(16)?, ColumnCount::try_from(32)?);
    let measurement = StageMeasurement::new(
        StageName::try_from("matvec")?,
        ElapsedNanos::try_from(90_000_u128)?,
        shape.multiply_add_flops()?,
        shape.bytes_moved(ElementSize::float32())?,
    );
    let transfer = MemoryTransfer::new(
        MemoryLevel::HostMemory,
        Bytes::try_from(16_384_u64)?,
        BytesPerSecond::try_from(32_000_000_000_u128)?,
    );

    println!("stage      = {}", measurement.name());
    println!("intensity  = {}", measurement.arithmetic_intensity()?);
    println!(
        "{} transfer -> {}",
        transfer.level(),
        transfer.estimated_elapsed()?
    );

    Ok(())
}
```

The division operators are typed. `Flops / Bytes` produces
`ArithmeticIntensity`. `Bytes / BytesPerSecond` produces `ElapsedNanos`.

## Why This Matters

Optimization is not a magic label. It should point to a changed resource trace.

If a stage is bandwidth-heavy, reducing bytes moved may matter more than
reducing arithmetic. If a stage is compute-heavy, reducing FLOPs or improving the
compute schedule may matter more. Typed systems evidence makes those claims
inspectable.

## Concept Trace

- **Object/newtype:** `StageMeasurement`, `StageMeasurements`, `ElapsedNanos`, `ArithmeticIntensity`, `BytesPerSecond`, `MemoryLevel`, and `MemoryTransfer`.
- **Invariant:** elapsed time, byte movement, bandwidth, and FLOPs-per-byte stay separate units.
- **Map:** repeated measurements -> median elapsed time, FLOPs/bytes -> intensity, and bytes/bandwidth -> transfer time.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity`.
- **Failure signal:** you report speed without saying whether timing, FLOPs, bytes moved, or bandwidth changed.

## Short Practice

1. Why is a median stronger than a single timing?
2. What does `0.4571 FLOPs/byte` say about the example stage?
3. Which map produces `ElapsedNanos` from `Bytes` and `BytesPerSecond`?
4. Why can the same byte count have different transfer times?
