# Elementwise Maps And Reductions

## Overview

Kernels begin with two small patterns:

```text
one value -> one value
many values -> one value
```

An elementwise map applies the same function to each element. A reduction
combines several elements into one accumulator.

## Learning Goals

- explain elementwise work as one independent map per element
- explain reduction work as typed accumulation
- distinguish `KernelScalar`, `KernelProduct`, and `Accumulator`
- connect element count to FLOPs and HBM bytes
- predict the output of the first kernel examples

## Plain-English Explanation

An elementwise kernel has no communication between elements. Each input scalar
can be transformed independently.

The GeLU-style example processes three values:

```text
[-1.0, 0.0, 1.0]
```

It reports:

```text
3 elements
6 FLOPs
24 HBM bytes
```

The reduction example is different. It reads a row and carries a running sum:

```text
1.0 + 2.0 + 3.5 = 6.5
```

The key idea is that reduction has an accumulator. It is not just three separate
outputs.

## Algebra Form

Elementwise map:

```text
y_i = gelu_like(x_i)
```

Reduction:

```text
sum = x_0 + x_1 + x_2
```

Typed composition:

```text
KernelVector -> ElementwiseTrace
KernelVector -> RowReductionTrace
Accumulator + KernelScalar -> Accumulator
```

Resource estimates:

```text
ElementCount * ElementSize -> Bytes
ElementCount * FlopsPerElement -> FlopCount
```

## Rust Form

```rust
use rust_ml_kernels::{
    ElementSize, ElementwiseTrace, KernelScalar, KernelVector, RowReductionTrace,
};

fn main() -> Result<(), rust_ml_kernels::Error> {
    let values = KernelVector::from_values([
        KernelScalar::try_from(-1.0)?,
        KernelScalar::try_from(0.0)?,
        KernelScalar::try_from(1.0)?,
    ])?;
    let elementwise = ElementwiseTrace::gelu(values.clone(), ElementSize::float32())?;
    let reduction = RowReductionTrace::sum(values)?;

    println!("elementwise elements = {}", elementwise.element_count());
    println!("elementwise FLOPs = {}", elementwise.flops());
    println!("elementwise HBM bytes = {}", elementwise.hbm_bytes());
    println!("row sum = {}", reduction.output());

    Ok(())
}
```

The raw numbers become `KernelScalar` at the boundary. After that, the kernel
path uses semantic objects and typed operations.

## Why This Matters

Elementwise maps and reductions are the simplest kernel building blocks.

Elementwise work teaches parallel independence. Reduction teaches coordination
through an accumulator. Later tiling combines both instincts: split work into
windows, compute locally, and preserve a trace that explains the schedule.

## Concept Trace

- **Object/newtype:** `KernelScalar`, `KernelVector`, `ElementwiseTrace`, `Accumulator`, and `RowReductionTrace`.
- **Invariant:** kernel scalars are finite, vectors are non-empty, and reductions carry a typed accumulator.
- **Map:** vector -> elementwise trace, and vector -> row-reduction trace.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 01_elementwise_gelu`.
- **Failure signal:** you treat a reduction like independent elementwise outputs instead of one accumulated value.

## Short Practice

1. Why can an elementwise map process each value independently?
2. Why does a row reduction need an accumulator?
3. What unit does `FlopCount` protect?
4. Why does the GeLU-style trace report HBM bytes as well as output values?
