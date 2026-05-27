# Kernel Solutions

## Solution 1: Read an elementwise trace

The example prints:

```text
elements = 3
flops = 6 FLOPs
hbm = 24 bytes
gelu value = -0.1542
gelu value = 0.0000
gelu value = 0.8458
```

Each element can be processed independently because the GeLU-style map reads one
input scalar and produces one output scalar. No output needs another element's
value.

## Solution 2: Read a reduction trace

The example prints:

```text
row sum = 6.5000
elements = 3
flops = 3 FLOPs
```

A reduction needs an `Accumulator` because the output is one scalar built from
several input values. The running value has to carry the partial sum.

## Solution 3: Trace tiled matrix-vector work

The example prints:

```text
tiles = 4
output value = 8.0000
output value = 18.5000
```

The mathematical map stayed:

```text
matrix * vector -> output vector
```

The trace added schedule evidence: the matrix was visited through four tile
windows.

## Solution 4: Estimate kernel resources

The example prints:

```text
matrix elements = 4
flops = 8 FLOPs
hbm = 32 bytes
```

`MatrixRows * MatrixColumns` produces `ElementCount` because the result has
domain meaning. It is a count of scalar matrix entries, not a reusable raw
integer.

## Solution 5: Review a public kernel report

The example prints:

```text
public tile windows = 4
public FLOPs = 12 FLOPs
public HBM bytes = 44 bytes
blocked from public kernel report: invalid public report in PublicKernelReport::from_reviewed_trace: public kernel reports cannot include restricted or private tiled traces
```

The rejecting constructor is:

```text
PublicKernelReport::from_reviewed_trace
```

A valid tiled trace proves computation and resource evidence. A public report
proves that the trace was reviewed for learner-facing release. Those are
different boundaries.

## Self-Check

- You can distinguish elementwise maps from reductions.
- You can explain why reductions need an accumulator.
- You can describe tiling as a schedule over the same mathematical map.
- You can keep element counts, FLOPs, and bytes separate.
- You can explain why `PublicKernelReport` is a public-content boundary.
