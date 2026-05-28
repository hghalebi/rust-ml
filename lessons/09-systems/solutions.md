# Systems Solutions

## Solution 1: Count activation memory

The example prints:

```text
activation elements = 256
activation memory   = 1024 bytes
```

The hand calculation is:

```text
2 * 8 * 16 = 256 elements
256 * 4 bytes = 1024 bytes
```

`ActivationShape` owns the shape, `ElementCount` owns the scalar count, and
`Bytes` owns the memory estimate.

## Solution 2: Estimate attention work

The example prints:

```text
score FLOPs      = 2048 FLOPs
value-mix FLOPs  = 2048 FLOPs
total FLOPs      = 4096 FLOPs
score matrix size = 256 bytes
```

Dense attention compares every query token with every key token. That is why the
score map uses:

```text
sequence_length * sequence_length
```

The model width is then used inside each token-pair comparison.

## Solution 3: Read median timing

The example prints:

```text
median elapsed = 120000 ns
```

A median is better than one convenient measurement because one run may be noisy.
The median summarizes repeated evidence without letting one unusually fast or
slow run dominate the teaching signal.

## Solution 4: Compute arithmetic intensity

The example prints:

```text
stage      = matvec
elapsed    = 90000 ns
FLOPs      = 1024 FLOPs
bytes      = 2240 bytes
intensity  = 0.4571 FLOPs/byte
```

The arithmetic-intensity calculation is:

```text
1024 / 2240 = 0.4571 FLOPs/byte
```

If the stage is bandwidth-heavy, inspect bytes moved and memory-tier bandwidth
before looking only at arithmetic count.

## Solution 5: Compare memory tiers

The example prints:

```text
shared memory transfer: 16384 bytes at 8000000000000 bytes/s -> 3 ns
host memory transfer: 16384 bytes at 32000000000 bytes/s -> 512 ns
```

The byte count stayed the same. The memory level and bandwidth changed. That is
why the estimated elapsed time changed.

## Solution 6: Review a public systems report

The example prints:

```text
public median elapsed = 120000 ns
blocked from public systems report: invalid public report in PublicSystemsReport::from_reviewed_measurements: public systems reports cannot include restricted or private measurements
```

The rejecting constructor is:

```text
PublicSystemsReport::from_reviewed_measurements
```

A valid measurement answers a systems question. A public systems report answers
a publication question. Restricted or private evidence must not enter
learner-facing material just because it has valid units.

## Self-Check

- You can compute activation memory from shape and element size.
- You can explain why dense attention has sequence-length-squared work.
- You can explain why repeated measurements need a median.
- You can compute arithmetic intensity from FLOPs and bytes.
- You can compare memory tiers while keeping bytes fixed.
- You can explain why `PublicSystemsReport` is a public-content boundary.
