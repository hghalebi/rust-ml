# Hands-On Rust ML Workshop

Status: active self-serve route.

This workshop is the short, build-first route through the course. It teaches
machine learning as:

```text
plain English -> algebra -> Rust newtypes -> composable maps -> public trace
```

Use it when you want one coherent path from a typed stage graph to a tiny
training loop, then to attention and a Transformer trace.

## Source Citations

- Public website: [rust-ml.com](https://rust-ml.com/)
- Canonical course repository: [hghalebi/rust-ml](https://github.com/hghalebi/rust-ml)
- Website-to-repo map: [rust-ml.com content source map](../references/websites/rust-ml-com.md)
- Public category-theory lab source: [hghalebi/category_theory_transformer_rs](https://github.com/hghalebi/category_theory_transformer_rs)
- Category-theory source map: [Category Theory for Tiny ML in Rust source map](../references/repos/category-theory-transformer-rs.md)
- Public content boundary: [PUBLIC_CONTENT.md](../PUBLIC_CONTENT.md)

## Workshop Promise

By the end, a learner should be able to:

- name each ML value before using it in arithmetic
- explain a model as a legal composition of typed transformations
- run a one-step training update and describe what changed
- distinguish internal training evidence from reviewed public evidence
- connect the same object/map discipline to attention and Transformers

The workshop intentionally stays small. The goal is durable understanding, not
large-model performance.

## Category-Theory Frame

Objects are meaningful Rust types: feature vectors, weights, scores,
predictions, targets, losses, token IDs, attention traces, and public reports.

Morphisms are the transformations between those objects: `mix`, `squash`,
`judge`, `update`, `tokenize`, `score_attention`, and `review_for_publication`.

Composition is the model path. A valid path has matching meanings at each
boundary. A broken path is a teaching signal: the learner found a composition
that should not be allowed.

The training update is an endomorphism over parameters:

```text
Parameters -> Parameters
```

Loss traces form an accumulation. The empty trace is the identity case, and
adding one reviewed step should preserve order and meaning.

Public traces are a boundary functor from internal evidence to learner-facing
evidence. That mapping is intentionally lossy: it keeps the teaching signal and
drops private or unsafe context.

## How To Run

Run every command from the repository root.

```bash
python3 scripts/check_public_content.py
python3 scripts/check_course_content.py
```

Then run each session command as you reach it. The workshop is self-serve:
read the short lesson, run the proof, modify one thing, and record what changed.

## Session Map

| Session | Outcome | Main source | Runnable proof |
| --- | --- | --- | --- |
| 0. Setup and public boundary | Know what can be published | [PUBLIC_CONTENT.md](../PUBLIC_CONTENT.md) | `python3 scripts/check_public_content.py` |
| 1. Objects and maps | See values as typed objects and functions as maps | [The Learning Lens](../lessons/00-learning-lens.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 01_objects_and_maps` |
| 2. Legal composition | Learn why adjacent meanings must match | [Concept Atlas](../lessons/CONCEPT-ATLAS.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 03_composition_failure` |
| 3. Tiny neuron | Build `mix -> squash -> judge` | [03 Neuron](../lessons/03-neuron/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass` |
| 4. One update | Run one loss-driven parameter change | [One Training Step](../lessons/04-learning/01-one-training-step-end-to-end.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training` |
| 5. Epoch trace | Repeat updates and read movement over time | [Epochs and Loss Traces](../lessons/04-learning/02-epochs-and-loss-traces.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch` |
| 6. Language-modeling bridge | Turn public text into checked token pairs | [08 Language Modeling](../lessons/08-language-modeling/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example` |
| 7. Attention bridge | Read attention as visible routing | [06 Attention](../lessons/06-attention/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace` |
| 8. Transformer trace | See the same discipline at encoder scale | [07 Transformer](../lessons/07-transformer/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example public_encoder_trace` |

## Session 0: Setup And Public Boundary

Outcome: know the learner-facing surface before writing or publishing anything.

Run:

```bash
python3 scripts/check_public_content.py
python3 scripts/check_course_content.py
```

Hands-on task: open [PUBLIC_CONTENT.md](../PUBLIC_CONTENT.md) and write one
sentence in your notebook explaining the difference between internal evidence
and public evidence.

Evidence: the public-content check passes, and the notebook sentence names the
boundary.

Failure signal: a learner writes down a secret-shaped value, a local machine
path, or maintainer-only context as if it belonged in public course material.

## Session 1: Objects And Maps

Outcome: see ML values as objects and Rust functions as maps.

Read:

- [The Learning Lens](../lessons/00-learning-lens.md)
- [01 Foundations](../lessons/01-foundations/README.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 01_objects_and_maps
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 02_compose_neuron_forward
```

Hands-on task: choose one printed value and classify it as an object, a map, or
composition evidence.

Evidence: the learner can say which type owns the meaning and which function
changes it.

Failure signal: the learner explains the example as numbers only and cannot
name the role of each value.

## Session 2: Legal Composition

Outcome: understand that invalid model paths are composition failures, not
syntax accidents.

Read:

- [Concept Atlas](../lessons/CONCEPT-ATLAS.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 03_composition_failure
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 04_public_composition_trace
```

Hands-on task: write the legal path and one illegal path using type names only.

Evidence: the legal path has matching middle objects, and the illegal path has
one clear mismatch.

Failure signal: the learner says the illegal path is wrong only because "Rust
does not like it" instead of naming the failed boundary.

## Session 3: Tiny Neuron

Outcome: build the first ML model as a chain of typed transformations.

Read:

- [Rust Essentials for a Tiny Neuron](../lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md)
- [A Neuron as a Chain of Functions](../lessons/03-neuron/02-neuron-as-a-chain-of-functions.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
```

Hands-on task: change one input in the weighted-sum example and predict the
direction of the score before running it again.

Evidence: the learner can trace `FeatureVector -> PreActivation -> Prediction`.

Failure signal: the learner cannot explain why weights and features are not the
same kind of value.

## Session 4: One Update

Outcome: see learning as a typed parameter transition.

Read:

- [One Training Step End To End](../lessons/04-learning/01-one-training-step-end-to-end.md)
- [Backpropagation As Local Gradient Bookkeeping](../lessons/04-learning/02-backpropagation-as-local-gradient-bookkeeping.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 05_public_training_step
```

Hands-on task: identify which printed line is prediction evidence, which line is
loss evidence, and which line is update evidence.

Evidence: the learner can describe the update as `Parameters -> Parameters`.

Failure signal: the learner treats the loss as the model output instead of the
judgment about the output.

## Session 5: Epoch Trace

Outcome: read repeated training as ordered evidence.

Read:

- [Epochs And Loss Traces](../lessons/04-learning/02-epochs-and-loss-traces.md)
- [Datasets, Epochs, And Token Targets](../lessons/04-learning/03-datasets-epochs-and-token-targets.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

Hands-on task: record the first and last loss values and explain whether the
trace moved in the expected direction.

Evidence: the learner can explain why a trace is ordered and why an empty trace
would be the identity case for accumulation.

Failure signal: the learner judges the model from one row and ignores the
sequence of evidence.

## Session 6: Language-Modeling Bridge

Outcome: turn public text into checked token-target pairs.

Read:

- [From Text To Token IDs](../lessons/08-language-modeling/01-text-to-token-ids.md)
- [Next-Token Batches, Loss, And Updates](../lessons/08-language-modeling/02-next-token-batches-loss-and-update.md)
- [The Public Text Boundary](../lessons/08-language-modeling/03-public-text-boundary.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 02_next_token_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example
```

Hands-on task: write the object sequence from public text to one next-token
training pair.

Evidence: the learner names the public text boundary before tokenization.

Failure signal: the learner treats text as safe merely because it is short.

## Session 7: Attention Bridge

Outcome: understand attention as typed routing between token roles.

Read:

- [Tokens As Vectors In A Sequence](../lessons/06-attention/01-tokens-as-vectors-in-a-sequence.md)
- [Query, Key, And Value Roles](../lessons/06-attention/02-query-key-value-roles.md)
- [Scores, Weights, And Value Mixing](../lessons/06-attention/03-scores-weights-and-value-mixing.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace
```

Hands-on task: explain which object answers "what am I looking for?", which
object answers "what do I contain?", and which object is mixed into the result.

Evidence: the learner separates query, key, and value by role.

Failure signal: the learner says attention is only similarity and does not name
the value-mixing step.

## Session 8: Transformer Trace

Outcome: connect the same discipline to an encoder-scale example.

Read:

- [Tiny Transformer From First Principles](../lessons/07-transformer/01-tiny-transformer-from-first-principles.md)
- [Typed Rust Transformer With Linear Attention](../lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md)
- [Transformer Encoder In Small Chunks](../lessons/07-transformer/03-transformer-encoder-in-small-chunks.md)

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example encoder_demo
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example public_encoder_trace
```

Hands-on task: identify one stage where the Transformer keeps the same shape and
one stage where the semantic role changes.

Evidence: the learner can explain why shape compatibility alone is not enough;
meaning must compose too.

Failure signal: the learner describes the encoder as a collection of operations
without naming the objects that enter and leave each stage.

## Completion Evidence

A learner has completed the workshop when they can produce this short report:

```text
Object:
Invariant:
Map:
Composition path:
Failure signal:
Runnable proof:
Public boundary:
```

The `Runnable proof` line should cite one command from this workshop. The
`Public boundary` line should explain what makes a trace safe to share as
learner-facing evidence.

## Facilitator Notes

Use the workshop in two modes:

- Self-serve mode: one session per sitting, with one modification per session.
- Cohort mode: Sessions 0 through 5 in one half-day, then Sessions 6 through 8
  as follow-up labs.

Keep the teaching rule strict: if a learner cannot name the object and map, slow
down before adding more code.

## Validation

Before publishing workshop edits, run:

```bash
python3 scripts/check_public_content.py
python3 scripts/check_course_content.py
python3 scripts/check_lesson_rust_snippets.py
cargo test --manifest-path code/Cargo.toml --workspace --all-targets
```
