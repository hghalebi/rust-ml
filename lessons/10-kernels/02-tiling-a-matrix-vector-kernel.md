# Tiling A Matrix-Vector Kernel

## Overview

Tiling breaks a matrix-shaped job into smaller windows.

The mathematical map stays the same:

```text
matrix * vector -> output vector
```

The schedule changes. Instead of reading the whole matrix as one undifferentiated
object, the kernel walks through a `TilePlan`.

## Learning Goals

- explain a tile as a window over a matrix shape
- read `TileShape` and `TileWindow` as schedule objects
- trace tiled matrix-vector multiplication through `TiledMatVecTrace`
- explain why shape mismatch is rejected before the kernel runs
- connect tiling to element counts, FLOPs, and HBM bytes

## Plain-English Explanation

Matrix-vector multiplication combines each matrix row with the input vector.

For a tiny matrix and vector:

```text
row 0 dot vector -> output 0
row 1 dot vector -> output 1
```

The tiled example prints:

```text
tiles = 4
output value = 8.0000
output value = 18.5000
```

The output values are the mathematical result. The tile count is the schedule
evidence. Both matter for kernel intuition.

## Algebra Form

Matrix-vector map:

```text
out_i = sum_j matrix_{ij} * vector_j
```

Typed schedule:

```text
MatrixShape + TileShape -> TilePlan
TilePlan -> TileWindow*
KernelMatrix + KernelVector + TileShape -> TiledMatVecTrace
```

Resource trace:

```text
MatrixRows * MatrixColumns -> ElementCount
ElementCount * FlopsPerElement -> FlopCount
ElementCount * ElementSize -> Bytes
```

## Rust Form

```rust
use rust_ml_kernels::{
    ElementSize, KernelMatrix, KernelScalar, KernelVector, TileColumns, TileRows, TileShape,
    TiledMatVecTrace,
};

fn main() -> Result<(), rust_ml_kernels::Error> {
    let matrix = KernelMatrix::from_rows([
        KernelVector::from_values([
            KernelScalar::try_from(1.0)?,
            KernelScalar::try_from(2.0)?,
            KernelScalar::try_from(3.0)?,
        ])?,
        KernelVector::from_values([
            KernelScalar::try_from(4.0)?,
            KernelScalar::try_from(5.0)?,
            KernelScalar::try_from(6.0)?,
        ])?,
    ])?;
    let vector = KernelVector::from_values([
        KernelScalar::try_from(1.0)?,
        KernelScalar::try_from(0.5)?,
        KernelScalar::try_from(2.0)?,
    ])?;
    let tile_shape = TileShape::new(TileRows::try_from(1)?, TileColumns::try_from(2)?);
    let trace = TiledMatVecTrace::run(matrix, vector, tile_shape, ElementSize::float32())?;

    println!("tiles = {}", trace.tile_plan().windows().count());
    println!("FLOPs = {}", trace.flops());
    println!("HBM bytes = {}", trace.hbm_bytes());
    for value in trace.output().values() {
        println!("output value = {value}");
    }

    Ok(())
}
```

The tile shape is not the model answer. It is the schedule used to produce the
same mathematical map while keeping the work trace visible.

## Why This Matters

Real kernel optimization often changes memory movement and scheduling before it
changes the math. If the learner cannot separate the mathematical map from the
schedule, optimization becomes vague.

`TiledMatVecTrace` keeps both visible: the output vector and the tile/resource
evidence that produced it.

## Concept Trace

- **Object/newtype:** `KernelMatrix`, `KernelVector`, `TileShape`, `TilePlan`, `TileWindow`, and `TiledMatVecTrace`.
- **Invariant:** matrix columns must match vector length, and every tile window has positive row and column spans.
- **Map:** matrix + vector + tile shape -> tiled matrix-vector trace.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 03_tiled_matvec`.
- **Failure signal:** you describe only the output vector and lose the tile plan that explains the kernel schedule.

## Short Practice

1. What stays the same when a matrix-vector map is tiled?
2. What changes when the tile shape changes?
3. Why must matrix columns match vector length?
4. Which object stores the visible tile windows?
