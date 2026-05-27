# Learning Path

This repository teaches machine learning as a structural translation problem:

```text
plain English <-> algebra <-> Rust newtypes <-> composable maps
```

The goal is not to learn isolated formulas. The goal is to build a mental model strong enough that intuition, notation, and code reinforce each other.

## Audience

This repo is for learners who want to understand ML by implementing it.

You do not need to know category theory before starting. You only need to be willing to ask structural questions:

- What kind of value is this?
- What does this value mean?
- What transformation changes it?
- What invariant should the type protect?
- How do small transformations compose into a model?

## The Four Lenses

Each important idea should become clear through four lenses.

### Plain English

Plain English gives purpose.

Example:

```text
The neuron mixes input signals, squashes the score, then judges the prediction.
```

If you cannot explain the idea in ordinary language, the formula is arriving too early.

### Algebra

Algebra compresses the idea.

Example:

```text
z = w dot x + b
y_hat = sigmoid(z)
L = (y_hat - y)^2
```

The symbols are useful only after the roles are understood.

### Rust Newtypes

Rust newtypes preserve meaning in code.

Example:

```text
FeatureVector
WeightVector
Bias
PreActivation
Prediction
Target
Loss
```

These names are not decoration. They stop "just a number" from hiding important differences.

### Category-Theory Intuition

Category theory gives the structural map.

For this repo, the beginner version is enough:

```text
types are spaces of meaningful values
functions are maps between spaces
models are compositions of maps
learning changes maps so loss becomes smaller
```

No heavy abstraction is required at the start.

## Recommended Route

### Step 0: Learn The Lens

Read [lessons/00-learning-lens.md](lessons/00-learning-lens.md).

This gives the core vocabulary used throughout the repo:

- value roles
- newtypes
- maps
- composition
- learning as changing a transformation

Then skim [lessons/CONCEPT-ATLAS.md](lessons/CONCEPT-ATLAS.md).

The atlas is the cross-course map. It shows how the same idea appears as an ML
concept, a Rust newtype, a composable map, and a runnable proof.

### Step 1: Foundations

Read [lessons/01-foundations](lessons/01-foundations/README.md).

Goal: understand how English, algebra, and Rust syntax refer to the same idea.

### Step 2: Vectors

Read [lessons/02-vectors](lessons/02-vectors/README.md).

Goal: understand vectors as typed signals, not only lists of numbers.

### Step 3: Neuron

Read [lessons/03-neuron](lessons/03-neuron/README.md), then run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

Goal: understand a trainable model as `mix -> squash -> judge`, then `blame -> trace -> adjust`.

### Step 4: Learning Loop

Read [lessons/04-learning](lessons/04-learning/README.md), then run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

Goal: understand learning as repeated feedback over examples and epochs.

### Step 5: MLP Bridge

Read [lessons/05-mlp](lessons/05-mlp/README.md), then run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 01_hidden_features
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 03_forward_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 04_xor_table
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 05_public_trace
```

Goal: understand hidden layers as representation maps and shape flow as typed composition.

Also learn where the public boundary lives: `ForwardTrace` is computation
evidence, while `PublicForwardTrace` is reviewed learner-facing evidence.

### Step 6: Attention Bridge

Read [lessons/06-attention](lessons/06-attention/README.md), then run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 02_softmax_focus
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 03_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace
```

Goal: understand token-to-token information mixing through:

- attention scores
- query, key, and value roles
- sequence-to-sequence shape flow
- public trace review before learner-facing release

### Step 7: Transformer

Read [lessons/07-transformer](lessons/07-transformer/README.md), then run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example encoder_demo
```

Goal: see the same type-and-map discipline at Transformer scale.

### Step 8: CS336 Rust Equivalent

Read [CS336 Rust Equivalent](CS336-RUST-EQUIVALENT.md) when you want the full language-modeling systems path.

Then use [the Rust assignment sequence](assignments/cs336-rust/README.md) as the long-form project ladder:

```text
R1 Basics -> R2 Systems -> R3 Scaling -> R4 Data -> R5 Alignment
```

The executable artifacts are [`code/lm_basics`](code/lm_basics/README.md), [`code/systems`](code/systems/README.md), [`code/kernels`](code/kernels/README.md), [`code/scaling`](code/scaling/README.md), [`code/data`](code/data/README.md), [`code/evaluation`](code/evaluation/README.md), [`code/inference`](code/inference/README.md), [`code/parallelism`](code/parallelism/README.md), and [`code/alignment`](code/alignment/README.md).

Goal: rebuild the public CS336 topic journey as original Rust labs with semantic types, explicit invariants, and reproducible checks.

The first executable CS336 Rust artifacts are:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 05_memory_hierarchy
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 06_public_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 04_kernel_estimate
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 05_public_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 04_report_limitations
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 05_tradeoff_decision
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 06_public_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 05_public_manifest
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 05_public_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 05_public_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 04_pipeline_schedule
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 05_public_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 04_audit_record
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 05_alignment_workflow
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 06_public_release
```

## How To Study

For each chapter, keep a small notebook with four headings:

```text
English:
Algebra:
Rust type:
Map:
```

Fill those headings for each new concept.

Example:

```text
English: a prediction is the model output after the raw score is squashed
Algebra: y_hat = sigmoid(z)
Rust type: Prediction
Map: PreActivation -> Prediction
```

This is how intuition becomes durable.

When you finish a module, return to the [Concept Atlas](lessons/CONCEPT-ATLAS.md)
and add one trace of your own. The trace should name an object, a map, the
invariant it protects, and the example or test that proves it.

## Mastery Checks

You are making real progress when you can:

- predict what a tiny example will print before running it
- explain why two values should not share the same Rust type
- trace a value from input to loss
- identify the map that owns each transformation
- explain what invariant a constructor protects
- connect the single-neuron story to the Transformer type names

## Public Resource Standard

This repo is public learner material. It should stay clean, readable, and focused.

Public learning files should contain only:

- learner-facing explanations
- runnable commands
- public references
- source code intended for learners
- exercises and solutions

They should not contain credentials, personal data, local machine paths, deployment notes, or maintainer-only context.
