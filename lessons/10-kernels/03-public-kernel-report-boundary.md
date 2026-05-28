# The Public Kernel Report Boundary

## Overview

A kernel trace can be correct without being publishable public course evidence.

This module separates the computation trace from the learner-facing report:

```text
TiledMatVecTrace -> ReviewedTiledMatVecTrace -> PublicKernelReport
```

The public boundary prevents restricted or private kernel evidence from entering
the public repo.

## Learning Goals

- distinguish `TiledMatVecTrace` from `PublicKernelReport`
- explain why kernel visibility is an enum, not a boolean
- read `PublicKernelReport::from_reviewed_trace` as the public-release map
- explain why public kernel reports still expose tile, FLOP, and byte evidence
- predict why restricted or private traces are rejected

## Plain-English Explanation

`TiledMatVecTrace` answers:

```text
What output, tile plan, FLOPs, and HBM bytes did this kernel produce?
```

`PublicKernelReport` answers:

```text
May this kernel trace appear in learner-facing public material?
```

Those are different questions. The public example prints:

```text
public tile windows = 4
public FLOPs = 12 FLOPs
public HBM bytes = 44 bytes
```

Then it blocks a non-public trace at the report boundary.

## Algebra Form

Computation map:

```text
KernelMatrix + KernelVector + TileShape -> TiledMatVecTrace
```

Public-release map:

```text
ReviewedTiledMatVecTrace -> PublicKernelReport
```

Release invariant:

```text
visibility == Public
```

If visibility is `ResearchRestricted` or `Private`, the map does not produce a
public report.

## Rust Form

```rust
use rust_ml_kernels::{
    ElementSize, KernelMatrix, KernelScalar, KernelTraceVisibility, KernelVector,
    PublicKernelReport, ReviewedTiledMatVecTrace, TileColumns, TileRows, TileShape,
    TiledMatVecTrace,
};

fn main() -> Result<(), rust_ml_kernels::Error> {
    let matrix = KernelMatrix::from_rows([
        KernelVector::from_values([
            KernelScalar::try_from(1.0)?,
            KernelScalar::try_from(2.0)?,
        ])?,
        KernelVector::from_values([
            KernelScalar::try_from(3.0)?,
            KernelScalar::try_from(4.0)?,
        ])?,
    ])?;
    let vector = KernelVector::from_values([
        KernelScalar::try_from(1.0)?,
        KernelScalar::try_from(0.5)?,
    ])?;
    let tile_shape = TileShape::new(TileRows::try_from(1)?, TileColumns::try_from(1)?);
    let trace = TiledMatVecTrace::run(matrix, vector, tile_shape, ElementSize::float32())?;
    let report = PublicKernelReport::from_reviewed_trace(ReviewedTiledMatVecTrace::new(
        trace,
        KernelTraceVisibility::Public,
    ))?;

    println!("public tile windows = {}", report.tile_plan().windows().count());
    println!("public FLOPs = {}", report.flops());
    println!("public HBM bytes = {}", report.hbm_bytes());

    Ok(())
}
```

The owning constructor is:

```text
PublicKernelReport::from_reviewed_trace
```

It keeps the public rule at the boundary instead of scattering checks through
the examples.

## Why This Matters

Kernel work often contains environment-specific traces. A public teaching repo
should use tiny reviewed examples, not private benchmark evidence.

The type model makes that rule visible: a trace is computation evidence, while a
public report is reviewed learner-facing evidence.

## Concept Trace

- **Object/newtype:** `TiledMatVecTrace`, `ReviewedTiledMatVecTrace`, `KernelTraceVisibility`, and `PublicKernelReport`.
- **Invariant:** learner-facing kernel reports can use only reviewed public tiled traces.
- **Map:** reviewed tiled trace -> public kernel report.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 05_public_report`.
- **Failure signal:** you treat a valid tiled trace as automatically publishable public evidence.

## Short Practice

1. Why is `TiledMatVecTrace` not enough to publish a kernel report?
2. Which type carries the release classification?
3. Which constructor rejects restricted or private traces?
4. Why should the public report still expose FLOPs and HBM bytes?
