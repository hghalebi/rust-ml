# evaluation

Status: active.

This crate is the first executable companion for CS336-style evaluation in the [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md) track.

It teaches evaluation as a typed harness:

```text
EvalExample + ModelPrediction -> ScoredPrediction -> EvalReport
```

## Owns

- lecture direction: evaluation in [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)
- package: `rust_ml_evaluation`

## Current State

- active teaching crate
- typed example IDs, run IDs, prompts, expected answers, model answers, correctness labels, counts, accuracies, and deltas
- deterministic exact-match scoring after whitespace and case normalization
- report construction that rejects duplicate example IDs
- typed `std::ops` arithmetic for correct-count division and accuracy deltas
- expressive `thiserror` diagnostics through `EvaluationError`

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_score_prediction.rs
  02_accuracy_report.rs
  03_reject_mismatched_ids.rs
  04_compare_runs.rs
```

## Learning Ladder

1. `01_score_prediction` scores one prediction against one reference.
2. `02_accuracy_report` builds an exact-match report from scored predictions.
3. `03_reject_mismatched_ids` shows why predictions must stay attached to the correct example.
4. `04_compare_runs` compares two runs through a typed accuracy delta.

## Category Lens

Read evaluation as a composition that preserves evidence:

```text
EvalExample + ModelPrediction -> ScoredPrediction
ScoredPrediction* -> EvalReport
EvalReport + EvalReport -> AccuracyDelta
```

The composition rule is identity preservation. A metric is only meaningful when
the prediction, reference answer, example ID, and run ID remain visible.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_evaluation --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 01_score_prediction
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 02_accuracy_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 03_reject_mismatched_ids
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 04_compare_runs
```

## Scope

This crate intentionally starts with exact match. Exact match is not a complete
language-model evaluation strategy.

The goal is to teach the invariants first: a metric has a named behavior, every
prediction belongs to one example, every report belongs to one run, and every
comparison preserves the baseline being compared.

