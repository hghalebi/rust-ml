# parallelism

Status: active.

This crate is the first executable companion for CS336-style distributed parallelism in the [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md) track.

It teaches parallelism as typed partitioning:

```text
WorldSize + RankIndex -> RankId
GlobalBatchSize / WorldSize -> LocalBatchSize
ModelWidth / WorldSize -> ShardWidth
LayerCount / WorldSize -> LayersPerRank
CollectiveTrace + ParallelTraceVisibility -> PublicParallelismReport
```

## Owns

- lecture direction: parallelism in [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)
- package: `rust_ml_parallelism`

## Current State

- active teaching crate
- typed ranks, world sizes, batch sizes, model widths, layer counts, micro-batches, shard starts, shard lengths, and communication bytes
- data-parallel, tensor-parallel, and pipeline-parallel layout summaries
- tiny all-reduce trace over rank-owned shard sums
- typed `std::ops` arithmetic for exact splits, pipeline schedule length, communication addition, and communication-round multiplication
- public-report review that blocks restricted or private collective traces before they reach learner-facing material
- expressive `thiserror` diagnostics through `ParallelismError`

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_data_parallel_batch.rs
  02_tensor_parallel_width.rs
  03_collective_all_reduce.rs
  04_pipeline_schedule.rs
  05_public_report.rs
```

## Learning Ladder

1. `01_data_parallel_batch` splits a batch axis into rank-owned shards.
2. `02_tensor_parallel_width` splits model width across ranks.
3. `03_collective_all_reduce` shows why data parallelism needs gradient communication.
4. `04_pipeline_schedule` connects layers, stages, micro-batches, and pipeline bubbles.
5. `05_public_report` separates public toy traces from restricted or private distributed-training evidence.

## Category Lens

Read parallelism as a map from one global object into rank-indexed local objects:

```text
GlobalBatchSize / WorldSize -> LocalBatchSize
TensorLine / WorldSize -> RankShard*
RankShard* -> CollectiveTrace
ReviewedCollectiveTrace -> PublicParallelismReport
LayerCount / WorldSize -> PipelineLayout
```

The composition rule is ownership preservation. A distributed plan is only
trustworthy when every rank has a valid identity, every shard has a clear
origin, every communication estimate carries units, and every learner-facing
report has passed an explicit visibility review.

## Three Learner Questions

1. Which global object is being split?
2. Which rank owns each local object?
3. Which reviewed traces are safe for the public learning surface?

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_parallelism --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 01_data_parallel_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 02_tensor_parallel_width
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 03_collective_all_reduce
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 04_pipeline_schedule
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 05_public_report
```

## Scope

This crate intentionally does not use GPUs, NCCL, PyTorch, MPI, or real network communication.

The goal is to teach the invariants first: ranks must fit the world, splits
must divide evenly in the simple examples, communication has units, and each
parallel strategy splits a different axis of the training problem. Public
reports are built only from toy public traces so private cluster details,
benchmarks, and operational evidence cannot leak into the learner-facing repo.
