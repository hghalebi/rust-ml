# Category Theory for Tiny ML in Rust Source Map

Status: active public repository source map.
Snapshot checked: 2026-06-18.
Source commit: `95b94e534f04a962772e8460b170971980c4e5a0`.

Primary citation: [hghalebi/category_theory_transformer_rs](https://github.com/hghalebi/category_theory_transformer_rs).
Hosted book: [Category Theory for Tiny ML in Rust](https://hghalebi.github.io/category_theory_transformer_rs/).

This source map connects the public `category_theory_transformer_rs` project to
the `rust-ml` course. The external project is a Rust book and lab for learning
tiny ML systems through typed transformations, category-theory vocabulary, and
runnable Rust examples. This repository remains the broader Rust ML curriculum;
the external project is the strongest public source for the category-theory lab
route.

## Public Source Role

Use this public repository as a citation when the `rust-ml` course needs:

- a focused category-theory entry point for tiny ML
- a public book that explains objects, morphisms, products, composition,
  endomorphisms, functors, monoids, and chain rule through Rust
- an executable route from text to token IDs, training examples, prediction,
  loss, and repeated parameter updates
- shareable public challenges for compiler-error learning and paper-to-Rust
  implementation
- public reader-review paths and issue templates for feedback-backed teaching

## Example Ladder

The public source exposes a compact runnable ladder. Use it as supporting
context for `rust-ml` lessons, not as a replacement for this repo's own
examples.

| External command | Public source concept | `rust-ml` anchor |
| --- | --- | --- |
| `cargo run --bin category_ml` | full guided walkthrough from objects to training and chain rule | [`lessons/00-learning-lens.md`](../../lessons/00-learning-lens.md) |
| `cargo run --example 01_token_sequence` | text becomes `TokenSequence`, then adjacent training pairs | [`lessons/08-language-modeling`](../../lessons/08-language-modeling/README.md) |
| `cargo run --example 01_domain_objects` | raw token indices cross into typed domain objects | [`lessons/01-foundations`](../../lessons/01-foundations/README.md) |
| `cargo run --example 02_morphism_composition` | legal composition requires matching middle objects | [`code/category_lens`](../../code/category_lens/README.md) |
| `cargo run --example 03_training_endomorphism` | training repeats `Parameters -> Parameters` | [`lessons/04-learning`](../../lessons/04-learning/README.md) |
| `cargo run --example 04_structure_and_calculus` | functors, naturality, monoids, and local derivatives | [`lessons/CONCEPT-ATLAS.md`](../../lessons/CONCEPT-ATLAS.md) |
| `cargo run --example 05_seven_sketches` | applied category-theory transfer across software and systems examples | [`lessons/00-learning-lens.md`](../../lessons/00-learning-lens.md) |
| `cargo run --example 06_attention_scores` | query/key/value roles, masks, weighted values, residuals, and Transformer state | [`lessons/06-attention`](../../lessons/06-attention/README.md), [`lessons/07-transformer`](../../lessons/07-transformer/README.md) |
| `cargo run --example 07_transformer_training_state` | Transformer updates return a whole typed training state | [`code/transformer`](../../code/transformer/README.md), [`code/alignment`](../../code/alignment/README.md) |
| `cargo run --example challenge_adam` | optimizer memory is part of the typed update boundary | [`assignments/cs336-rust`](../../assignments/cs336-rust/README.md) |

## Validation Snapshot

The inspected public source passed:

```bash
cargo test
cargo run --bin category_ml
cargo run --example 01_token_sequence
mdbook build
```

All public examples listed above were also run once. The current source tree did
not contain a local shell validation script, so the validation snapshot uses
the available Cargo and mdBook gates.

## Mapping Rule

When importing ideas from this source into `rust-ml`:

1. Keep the public citation attached to the source repository or hosted book.
2. Convert each idea into this repo's own lesson language, examples, and tests.
3. Preserve the same object/map/invariant discipline.
4. Do not copy external book prose into authored lessons.
5. Prefer this repo's runnable examples in learner-facing paths unless the
   learner is explicitly being sent to the external category-theory lab.
