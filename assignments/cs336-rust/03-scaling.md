# R3 Scaling: Experiments As Typed Evidence

## Goal

Turn experiments into structured evidence.

The learner should be able to compare model size, data size, compute, and loss without mixing their meanings.

## What You Build

Create a small scaling-study harness:

- run several tiny training configurations
- record model width, depth, token count, step count, compute estimate, and final loss
- fit a simple curve to the logged results
- write a short interpretation of what the curve can and cannot justify

## Active Starter Crate

The first executable artifact is [`code/scaling`](../../code/scaling/README.md).

It starts with typed scaling evidence:

```text
ExperimentConfig -> TrainingRun -> MetricRecord -> ScalingFit
```

Run it with:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 01_record_runs
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 02_fit_power_law
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 03_forecast_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 04_report_limitations
```

## Object/Map Preflight

Before implementation, write this preflight in your assignment notes:

- **Objects:** `ExperimentConfig`, `TrainingRun`, `MetricRecord`, `ComputeBudgetFlops`, `ValidationLoss`, `ScalingFit`.
- **Maps:** configure a run, record metrics, fit a curve, forecast loss, attach limitation notes.
- **Composition path:** `ExperimentConfig -> TrainingRun -> MetricRecord -> ScalingFit -> Forecast`.
- **Invariant to protect with newtypes:** validation loss, parameter count, token count, and compute budget carry different evidence and cannot be swapped.

## Expected Deliverables

- at least three typed metric records from distinct tiny configurations
- a fitted curve over compute and validation loss
- one forecast that uses a larger typed compute budget
- one written limitation explaining what the tiny curve cannot prove
- a trace from each validation loss back to the run that produced it

## Newtype And Category-Theory Lens

Use newtypes for:

- `RunId`
- `ParameterCount`
- `TokenCount`
- `TrainingStep`
- `ComputeBudgetFlops`
- `ValidationLoss`
- `ScalingExponent`

The experimental composition is:

```text
ExperimentConfig -> TrainingRun -> MetricRecord -> ScalingFit
```

Each arrow should preserve enough information for another learner to inspect the result.

## Required Checks

- validate that metric records include the config that produced them
- reject incomplete records
- test the curve-fitting helper on synthetic points
- include one negative result or limitation in the report

## Assessment Rubric

- **Evidence structure:** every metric record keeps run identity, token count, parameter estimate, compute estimate, and loss together.
- **Typed comparison:** model size, data size, compute, and loss cannot be swapped by accident.
- **Interpretive restraint:** the report distinguishes a teaching curve from a real frontier scaling law.
- **Reproducibility:** a learner can rerun the examples and inspect where each number came from.

## Failure Signals

- a validation loss appears without the configuration that produced it
- compute budget is recorded as a vague number with no unit or type
- a curve is fitted to incomplete or single-point evidence
- the report overclaims from toy data instead of naming limitations

## Suggested Repo Integration

Start from the active `code/scaling` crate. Keep the first version small enough to run locally.

The point is not to produce a frontier scaling law. The point is to teach disciplined experiment evidence.
