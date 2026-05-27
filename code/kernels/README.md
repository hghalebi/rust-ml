# kernels

Status: active.

This crate is the first executable companion for CS336-style kernels and tiling in the [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md) track.

It teaches kernels as typed memory and compute maps:

```text
MatrixRows * MatrixColumns -> ElementCount
ElementCount * ElementSize -> Bytes
ElementCount * FlopsPerElement -> FlopCount
Accumulator + KernelProduct -> Accumulator
ReviewedTiledMatVecTrace -> PublicKernelReport
```

## Owns

- lecture direction: kernels and tiling in [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)
- package: `rust_ml_kernels`

## Current State

- active teaching crate
- typed matrix shapes, vector lengths, tile shapes, tile windows, element counts, byte counts, FLOP counts, kernel scalars, products, and accumulators
- elementwise GeLU-style trace for the "one thread per element" intuition
- row-wise reduction trace for the "threads need to communicate" intuition
- tiled matrix-vector trace for the "load a tile, compute, write back" intuition
- typed `std::ops` arithmetic for shape products, byte estimates, FLOP estimates, scalar products, and accumulation
- public-report review that blocks restricted or private tiled traces before publication
- expressive `thiserror` diagnostics through `KernelError`

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_elementwise_gelu.rs
  02_row_sum_reduction.rs
  03_tiled_matvec.rs
  04_kernel_estimate.rs
  05_public_report.rs
```

## Learning Ladder

1. `01_elementwise_gelu` applies one elementwise map and reports bytes/FLOPs.
2. `02_row_sum_reduction` reduces a row through a typed accumulator.
3. `03_tiled_matvec` runs a small tiled matrix-vector kernel.
4. `04_kernel_estimate` separates matrix elements, FLOPs, and HBM bytes.
5. `05_public_report` keeps restricted or private kernel evidence out of public learner reports.

## Category Lens

Read a kernel as a map that changes where work happens:

```text
KernelVector -> ElementwiseTrace
KernelVector -> RowReductionTrace
KernelMatrix + KernelVector + TileShape -> TiledMatVecTrace
ReviewedTiledMatVecTrace -> PublicKernelReport
```

The composition rule is resource preservation. The output alone is not enough:
the trace should also name the tile plan, element count, FLOPs, and memory
movement so learners can connect correctness to performance. Public reports add
one more rule: only reviewed public traces can enter learner-facing material.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_kernels --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 01_elementwise_gelu
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 02_row_sum_reduction
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 03_tiled_matvec
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 04_kernel_estimate
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 05_public_report
```

## Scope

This crate intentionally does not use Triton, CUDA, GPU kernels, shared memory,
warps, or real profiling tools.

The goal is to teach the invariants first: shape products produce element
counts, element counts produce byte/FLOP estimates, reductions need an
accumulator, and tiled kernels carry a visible tile plan.
Public examples use tiny synthetic traces and reject restricted or private
kernel evidence before it reaches the learner-facing repo.
