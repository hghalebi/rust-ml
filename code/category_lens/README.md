# category_lens

Status: active.

Package: `rust_ml_category_lens`

This crate is the smallest executable version of the course lens:

```text
object -> typed map -> legal composition
```

It does not try to teach formal category theory. It gives learners a concrete
Rust object/map vocabulary before they meet neurons, MLPs, attention, and
systems traces.

## Owns

- orientation lesson: [The Learning Lens](../../lessons/00-learning-lens.md)
- cross-module guide: [Concept Atlas](../../lessons/CONCEPT-ATLAS.md)
- code layer: typed objects, typed maps, and composition errors

## Current State

- Active crate with typed object and map names
- Runnable examples for legal and illegal composition
- Unit tests for name validation, map composition, and trace composition

## Layout

```text
category_lens/
├── src/lib.rs
├── src/error.rs
└── examples/
```

## Learning Ladder

1. `01_objects_and_maps` names one ML object and one typed map.
2. `02_compose_neuron_forward` composes the neuron forward path.
3. `03_composition_failure` shows the typed error when maps do not line up.

## Category Lens

Read the crate as the course's structural grammar:

```text
TypedObject --TypedMap--> TypedObject
```

Composition is legal only when the middle object matches:

```text
FeatureVector -> PreActivation -> Prediction
```

That is the same rule learners later use for shapes, token roles, residual
connections, data pipelines, and distributed training plans.

## Run

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 01_objects_and_maps
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 02_compose_neuron_forward
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 03_composition_failure
```

## Scope

This crate is intentionally small. It should stay focused on the object/map
mental model and should not become a general category-theory library.
