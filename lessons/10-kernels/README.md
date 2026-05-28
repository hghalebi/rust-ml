# 10 Kernels

Status: active.

This folder maps to course Module 9.

This module turns systems resource accounting into kernel intuition. A kernel is
a small map that decides where work happens, how values accumulate, how memory
is touched, and which trace can become public learner-facing evidence.

## Outcomes

After this module, you should be able to:

- explain an elementwise kernel as one map per element
- explain a row reduction as typed accumulation
- trace tiled matrix-vector multiplication through a visible `TilePlan`
- convert matrix shape into element counts, FLOPs, and HBM bytes
- read `KernelScalar * KernelScalar -> KernelProduct` as typed arithmetic
- explain why a public kernel report requires reviewed public trace evidence
- run the companion kernel examples and predict their outputs

## Lessons

1. [Elementwise Maps And Reductions](01-elementwise-maps-and-reductions.md)
2. [Tiling A Matrix-Vector Kernel](02-tiling-a-matrix-vector-kernel.md)
3. [The Public Kernel Report Boundary](03-public-kernel-report-boundary.md)

## Practice

- [Kernel exercises](exercises.md)
- [Kernel solutions](solutions.md)

## Code Artifact

- Active crate: [`code/kernels`](../../code/kernels/README.md)
- Systems prerequisite: [09 Systems](../09-systems/README.md)

Run the examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 01_elementwise_gelu
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 02_row_sum_reduction
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 03_tiled_matvec
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 04_kernel_estimate
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 05_public_report
```

Keep this translation in view:

```text
semantic values -> kernel map -> trace -> resource estimate -> public review
```

The output vector is not the whole story. A kernel lesson should also preserve
the tile plan, element count, FLOPs, bytes, and public release boundary.

## Prerequisite

- Complete [09 Systems](../09-systems/README.md)

## Before You Move On

You are ready for larger optimization and parallelism topics when you can explain
these chains:

```text
KernelVector -> ElementwiseTrace
KernelVector -> RowReductionTrace
KernelMatrix + KernelVector + TileShape -> TiledMatVecTrace
MatrixRows * MatrixColumns -> ElementCount
ElementCount * ElementSize -> Bytes
ElementCount * FlopsPerElement -> FlopCount
ReviewedTiledMatVecTrace -> PublicKernelReport
```

You should also be able to explain why tiling is a schedule and resource-trace
choice, not a change to the mathematical matrix-vector map.
