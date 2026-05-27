# Kernel Exercises

## Exercise 1: Read an elementwise trace

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 01_elementwise_gelu
```

Write down:

1. element count
2. FLOP count
3. HBM bytes
4. the three output values

Then answer: why can each element be processed independently?

## Exercise 2: Read a reduction trace

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 02_row_sum_reduction
```

Write down:

1. row sum
2. element count
3. FLOP count

Then answer: why does a reduction need an `Accumulator`?

## Exercise 3: Trace tiled matrix-vector work

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 03_tiled_matvec
```

Write down:

1. number of tile windows
2. output values

Then answer:

1. what mathematical map stayed the same?
2. what schedule evidence did the trace add?

## Exercise 4: Estimate kernel resources

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 04_kernel_estimate
```

Write down:

1. matrix elements
2. FLOPs
3. HBM bytes

Then explain why `MatrixRows * MatrixColumns` produces `ElementCount`, not a raw number.

## Exercise 5: Review a public kernel report

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 05_public_report
```

Write down:

1. public tile windows
2. public FLOPs
3. public HBM bytes
4. the constructor that rejects the non-public trace

Then answer: why is a valid tiled trace not automatically public learner evidence?

## Failure Signals

- You describe a kernel output without its resource trace.
- You treat a reduction like independent elementwise outputs.
- You confuse tile shape with matrix shape.
- You mix element count, FLOPs, and bytes as loose numbers.
- You treat a valid tiled trace as automatically publishable public evidence.

## Debugging Hints

- Label each value by role: scalar, product, accumulator, vector, matrix, tile, trace, FLOPs, bytes, or public report.
- For reductions, identify where the running value lives.
- For tiling, ask what mathematical map stayed the same.
- For resource estimates, keep element count, FLOPs, and bytes separate.
- For public reports, ask whether the trace crossed the review boundary.
