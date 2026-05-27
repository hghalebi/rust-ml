# CS336 Rust Assignments

Status: original assignment scaffold.

These assignments create a Rust equivalent of the public CS336 language-modeling journey.

They are not copied from Stanford's handouts or repositories. They use the public course structure as a topic map, then express the work in this repo's own style: small typed Rust artifacts, explicit invariants, and learner-visible tests.

## Sequence

| Assignment | Focus | Main artifact |
| --- | --- | --- |
| [R1 Basics](01-basics.md) | tokenizer, tiny model pieces, optimizer, minimal training loop | [`code/lm_basics`](../../code/lm_basics/README.md) |
| [R2 Systems](02-systems.md) | profiling, memory accounting, optimized attention, parallelism | [`code/systems`](../../code/systems/README.md) |
| [R3 Scaling](03-scaling.md) | scaling curves and component ablations | [`code/scaling`](../../code/scaling/README.md) |
| [R4 Data](04-data.md) | corpus ingestion, filtering, deduplication, mixing | [`code/data`](../../code/data/README.md) |
| [R5 Alignment](05-alignment.md) | SFT, preference pairs, verifier feedback, safety | [`code/alignment`](../../code/alignment/README.md) |

## Shared Rules

- Every domain value with meaning should have a type name.
- Every boundary should validate input before downstream logic uses it.
- Every assignment should include at least one runnable example and one test.
- Every metric should say what it measures and what it does not measure.
- Every shortcut should be marked as a teaching simplification.

## Completion Standard

The track is complete when a learner can run the Rust artifacts, read the tests, and explain the full path:

```text
text -> tokens -> batches -> model -> loss -> update -> evaluation -> inference -> alignment signal
```
