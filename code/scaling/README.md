# scaling

Status: active.

This crate is the first executable companion for [R3 Scaling](../../assignments/cs336-rust/03-scaling.md) in the CS336 Rust equivalent track.

It teaches scaling as typed evidence:

```text
ExperimentConfig -> TrainingRun -> MetricRecord -> ScalingFit
ScalingFit + TrainingRun -> ScalingCandidate -> ScalingTradeoff
```

## Owns

- assignment: [R3 Scaling](../../assignments/cs336-rust/03-scaling.md)
- track: [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)

## Current State

- active teaching crate
- typed run identifiers, dimensions, token counts, step counts, parameter counts, compute budgets, losses, coefficients, and exponents
- checked parameter and compute estimates for tiny dense Transformer-style runs
- metric records that keep loss attached to the run that produced it
- log-log power-law fitting over compute and validation loss
- forecast errors and learner-facing limitation reports
- typed tradeoff decisions between candidate runs

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_record_runs.rs
  02_fit_power_law.rs
  03_forecast_loss.rs
  04_report_limitations.rs
  05_tradeoff_decision.rs
```

## Learning Ladder

1. `01_record_runs` turns one typed experiment config into a complete metric record.
2. `02_fit_power_law` fits a tiny synthetic power law in log-log space.
3. `03_forecast_loss` uses the fitted curve to forecast a larger run.
4. `04_report_limitations` packages the result with an explicit limitation.
5. `05_tradeoff_decision` compares a baseline and candidate run with typed loss and compute tradeoffs.

## Category Lens

Read scaling as maps from experimental choices to evidence:

```text
ExperimentConfig -> TrainingRun
TrainingRun + ValidationLoss -> MetricRecord
MetricRecords -> ScalingFit
ScalingFit + ComputeBudgetFlops -> ForecastLoss
ForecastLoss + ValidationLoss -> LossDelta
ScalingCandidate + ScalingCandidate -> ScalingTradeoff
```

The composition rule is accountability. A fitted curve is meaningful only when
each loss still points back to the run, token count, parameter estimate, and
compute budget that produced it.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_scaling --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 01_record_runs
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 02_fit_power_law
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 03_forecast_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 04_report_limitations
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 05_tradeoff_decision
```

## Scope

This crate does not claim to discover a real frontier scaling law.

The goal is to teach disciplined evidence: every loss should point back to the config, step count, token count, parameter estimate, and compute estimate that produced it.
