# Systems Exercises

## Exercise 1: Count activation memory

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting
```

Write down:

1. activation elements
2. activation memory

Then compute the same result by hand:

```text
2 * 8 * 16 = ?
elements * 4 bytes = ?
```

## Exercise 2: Estimate attention work

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 02_attention_flops
```

Write down:

1. score FLOPs
2. value-mix FLOPs
3. total FLOPs
4. score matrix size

Then answer: why does dense attention depend on `sequence_length * sequence_length`?

## Exercise 3: Read median timing

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 03_median_timing
```

Explain why a median timing is a better teaching signal than one convenient
measurement.

## Exercise 4: Compute arithmetic intensity

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
```

Use the printed values to explain:

```text
1024 FLOPs / 2240 bytes = 0.4571 FLOPs/byte
```

Then answer: which resource would you inspect first if the stage seems
bandwidth-heavy?

## Exercise 5: Compare memory tiers

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 05_memory_hierarchy
```

Write down the transfer time for:

1. shared memory
2. host memory

Then answer: what stayed the same, and what changed?

## Exercise 6: Review a public systems report

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 06_public_report
```

Write down:

1. the public median elapsed time
2. the constructor that rejects the non-public report
3. why a valid measurement is not automatically public evidence

## Failure Signals

- You report a number without its unit.
- You mix bytes and FLOPs as if they were the same kind of value.
- You treat one timing as a conclusion.
- You discuss memory hierarchy without naming bytes and bandwidth.
- You treat a valid benchmark as automatically publishable public evidence.

## Debugging Hints

- Label every value by unit: elements, bytes, FLOPs, elapsed time, bandwidth, or intensity.
- Ask which shape produced the resource count.
- For timing, ask how many measurements support the claim.
- For memory hierarchy, keep byte count fixed before comparing bandwidth.
- For public reports, ask whether the measurement crossed the review boundary.
