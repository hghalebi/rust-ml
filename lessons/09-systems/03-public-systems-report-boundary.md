# The Public Systems Report Boundary

## Overview

A systems measurement can be useful without being public teaching material.

The repo separates measured evidence from public learner-facing evidence:

```text
StageMeasurement -> ReviewedStageMeasurement -> PublicSystemsReport
```

That boundary keeps private or restricted benchmark evidence out of the public
course surface.

## Learning Goals

- distinguish a valid systems measurement from a public systems report
- explain why `MeasurementVisibility` is an enum, not a boolean
- read `PublicSystemsReport::from_reviewed_measurements` as the release map
- explain why the public report still computes a typed median
- predict why restricted or private measurements are rejected

## Plain-English Explanation

`StageMeasurement` answers a systems question:

```text
What happened for this stage, timing, FLOPs, and byte movement?
```

`PublicSystemsReport` answers a publication question:

```text
May these measurements appear in public learner-facing material?
```

Those are different questions. A private machine profile may be valid evidence
for internal work but still not belong in a public lesson.

## Algebra Form

The systems measurement map is:

```text
StageName + ElapsedNanos + Flops + Bytes -> StageMeasurement
StageMeasurements -> median elapsed time
StageMeasurement -> ArithmeticIntensity
```

The public-release map is:

```text
ReviewedStageMeasurement* -> PublicSystemsReport
```

The release invariant is:

```text
visibility == Public
```

If any reviewed measurement is `ResearchRestricted` or `Private`, the public
report is rejected.

## Rust Form

```rust
use rust_ml_systems::{
    Bytes, ElapsedNanos, Flops, MeasurementVisibility, PublicSystemsReport,
    ReviewedStageMeasurement, StageMeasurement, StageName,
};

fn main() -> Result<(), rust_ml_systems::Error> {
    let public_measurement = StageMeasurement::new(
        StageName::try_from("matvec")?,
        ElapsedNanos::try_from(120_000_u128)?,
        Flops::try_from(1024_u64)?,
        Bytes::try_from(2240_u64)?,
    );
    let report = PublicSystemsReport::from_reviewed_measurements([
        ReviewedStageMeasurement::new(public_measurement, MeasurementVisibility::Public),
    ])?;

    println!("public median elapsed = {}", report.median_elapsed()?);

    let private_measurement = StageMeasurement::new(
        StageName::try_from("private-host-profile")?,
        ElapsedNanos::try_from(100_000_u128)?,
        Flops::try_from(1024_u64)?,
        Bytes::try_from(2240_u64)?,
    );
    let private_report = PublicSystemsReport::from_reviewed_measurements([
        ReviewedStageMeasurement::new(private_measurement, MeasurementVisibility::Private),
    ]);

    match private_report {
        Ok(_) => println!("unexpected public systems report"),
        Err(error) => println!("blocked from public systems report: {error}"),
    }

    Ok(())
}
```

The constructor that owns the public invariant is:

```text
PublicSystemsReport::from_reviewed_measurements
```

It does not weaken the measurement. It protects the public learning surface.

## Why This Matters

This repo is public learner material. Systems examples must not leak private
host details, restricted benchmark traces, or maintainer-only context.

The type design makes that boundary visible. `MeasurementVisibility` names the
possible release classes, and `PublicSystemsReport` exists only after the review
map succeeds.

## Concept Trace

- **Object/newtype:** `StageMeasurement`, `ReviewedStageMeasurement`, `MeasurementVisibility`, and `PublicSystemsReport`.
- **Invariant:** learner-facing systems reports can use only reviewed public measurements.
- **Map:** reviewed measurements -> public systems report.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 06_public_report`.
- **Failure signal:** you treat a valid benchmark as automatically publishable public evidence.

## Short Practice

1. Why is `StageMeasurement` not enough to publish a systems report?
2. Which type carries the release classification?
3. Which constructor rejects restricted or private measurements?
4. Why is this boundary separate from arithmetic intensity?
