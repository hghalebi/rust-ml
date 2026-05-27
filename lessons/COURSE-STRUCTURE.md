# Course Structure

This file is the canonical structure guide for the course.

The repo teaches machine learning as a translation exercise:

`plain English <-> algebra <-> Rust newtypes <-> composable maps`

Everything in `lessons/` should support that goal.

## Translation Contract

Every authored lesson should help the learner move in all three directions:

- from plain English to algebra
- from algebra to Rust
- from Rust back to plain English

That means:

- explain the idea in ordinary language first
- show the compact mathematical form second
- show the runnable or at least compilable Rust form third
- name the Rust roles with semantic types where meaning matters
- identify the map or composition when it clarifies the structure

The order matters. Beginners should not have to decode symbolic syntax before they know what the idea is for.

## Newtype And Category-Theory Spine

This repo uses newtypes and category-theory intuition as teaching tools.

Use newtypes when a value has a domain role, invariant, unit, lifecycle, or boundary meaning.

Use the category-theory lens gently:

- types are spaces of meaningful values
- functions are maps between those spaces
- models are compositions of maps
- learning changes parameters inside maps so loss becomes smaller

Do not use category theory as decoration. Use it only when it makes the learner's mental model clearer.

## Concept Atlas Contract

`lessons/CONCEPT-ATLAS.md` is the maintained cross-module map.

It should connect:

- ML concepts
- semantic Rust types
- protected invariants
- composable maps
- runnable examples or tests

The atlas should stay learner-facing. It should not contain maintainer notes,
private context, deployment details, or unverified claims.

## Strict Rust Teaching Contract

Learner-facing Rust should make meaning visible in the type system.

The public API of active teaching crates must not expose raw domain primitives for meaningful values. Use semantic types such as `RawText`, `Token`, `TokenId`, `VocabularySize`, `ContextLength`, `Logit`, `Loss`, and `LearningRate`.

Raw literals may enter only at explicit edge adapters such as `TryFrom<&str>`, `TryFrom<usize>`, or `TryFrom<f64>`. After that boundary, pass the semantic type. This keeps the mental model simple:

```text
untrusted literal -> validating adapter -> meaningful value -> typed map
```

Rust teaching code must also follow these rules:

- no `.unwrap()` or `.expect()`
- no panic-style macros such as `panic!()`, `todo!()`, `unimplemented!()`, or `unreachable!()` in teaching implementations or tests
- no `Result<_, String>` error surfaces
- all crate error enums use `thiserror`
- public tuple fields stay private unless a type is intentionally transparent and has no invariant
- public enum variants carry semantic payload types, not raw primitives or raw containers
- associated type assignments use semantic types, not raw primitives or raw containers
- public `TryFrom` adapters should validate one raw semantic value at a time, not hide raw `Vec<_>` or slice containers behind a domain type
- public accessor names should teach semantic meaning, so avoid raw-style names such as `as_slice()` on learner-facing APIs
- public Rust snippets follow the same rule: typed errors instead of `unwrap`, `expect`, panic-style macros, or `Result<_, String>`
- diagnostic errors may report the rejected raw value, but the domain path should still use semantic types

Run the guard with:

```bash
python3 scripts/check_rust_teaching_contract.py
```

## Current Learning Paths

The repo currently supports two realistic paths:

### Core Path

This is the recommended order for most learners right now:

0. `00-learning-lens.md` -> orientation lens
1. `01-foundations` -> course Module 0
2. `02-vectors` -> course Module 1
3. `03-neuron` -> course Module 2
4. `04-learning` -> course Module 3
5. `05-mlp` -> course Module 4
6. `06-attention` -> course Module 5
7. `07-transformer` -> course Module 6

### Capstone Path

The Transformer module is the current capstone after the attention bridge:

1. `07-transformer` -> course Module 6

## Naming Rules

### Repo folders

- Folder numbers are sequential for the repository.
- Course modules start at Module 0.
- Therefore repo folder `NN-*` maps to course Module `NN - 1`.

Examples:

- `01-foundations` -> course Module 0
- `03-neuron` -> course Module 2
- `07-transformer` -> course Module 6

### Lesson files

- Use ordered filenames: `01-...md`, `02-...md`, `03-...md`
- Use concept-first titles in the file, not global lesson numbers
- Prefer titles such as `# A Neuron as a Chain of Functions`
- Avoid titles such as `# Lesson 17: ...`

The filename already provides order. The title should provide meaning.

## Module Contract

Every module directory must contain a `README.md`.

### Authored module README

Authored modules must include:

- `Status: active.`
- `This folder maps to course Module N.`
- `## Outcomes`
- `## Lessons`
- `## Practice`
- `## Code Artifact`
- `## Prerequisite`
- `## Before You Move On`

Optional sections such as `## Current Focus` are fine when they help orientation.

### Planned module README

Planned modules must include:

- `Status: planned.`
- `This folder maps to course Module N.`
- `## Goal`
- `## Planned Lesson Ladder`
- `## Planned Practice`
- `## Code Artifact`
- `## Prerequisite`
- `## Planned Outcome`

Planned modules should still read like a credible roadmap, not like empty stubs.

## Lesson Contract

Each authored lesson must include:

- `## Overview`
- `## Learning Goals`
- `## Plain-English Explanation`
- `## Algebra Form`
- `## Rust Form`
- `## Why This Matters`
- `## Concept Trace`
- `## Short Practice`

The concept trace must include:

- `Object/newtype`
- `Invariant`
- `Map`
- `Runnable proof`
- `Failure signal`

The runnable proof must name one checked command. Use one of these forms:

- ``cargo run --manifest-path code/Cargo.toml -p <package> --example <example>``
- ``cargo test --manifest-path code/Cargo.toml -p <package> --all-targets``
- ``python3 scripts/<local-check>.py``

The course-content checker verifies that referenced packages, examples, and scripts exist.

Module `07-transformer` is allowed to use the chunked `English -> Algebra -> Rust` pattern, but it still needs a clear overview, learning goals, and concept trace.

## Practice Contract

Each authored module must contain:

- `exercises.md`
- `solutions.md`

The exercise file should ask the learner to reason, compute, explain, or translate.

The exercise file must include:

- `## Failure Signals`
- `## Debugging Hints`

Failure signals should name the kind of misunderstanding the learner should
notice. Debugging hints should help the learner recover without copying the
solution.

The solution file should:

- show the answer
- explain the reasoning
- keep the answer concise enough for self-checking
- include `## Self-Check`

Solution files should use `## Solution N: ...` headings for individual answers.
Do not label solution sections as exercises.

## Code Contract

`code/` follows the lesson progression.

- Placeholder topic directories may exist before a real crate is ready
- Once a topic earns executable material, it should become a real Cargo crate
- Active topic crates should be members of `code/Cargo.toml`
- Runnable examples should map to lesson concepts in increasing difficulty
- The code README should state whether the topic is active or still planned
- `lessons/` stays the canonical teaching surface even when a companion crate exists

## Pedagogy Rules

- Prefer small lessons over giant walls of mixed concepts
- Introduce one idea at a time, then connect it to the larger system
- Use metaphor only when it reduces confusion, not when it replaces the real mechanism
- Keep examples small enough to inspect without scrolling through noise
- Use semantic Rust types when they improve readability and meaning
- Explain the invariant protected by each newtype when introducing it
- Use the words object, map, and composition only when the concrete type/function relationship is visible
- Keep public learner-facing prose free of maintainer-only context
- Do not treat advanced modules as prerequisites for beginner modules

## Review Checklist

Before considering a lesson coherent and merge-ready, verify:

- the course-module mapping is correct
- the module README follows the correct contract for its status
- lesson titles are concept-first and not globally misnumbered
- the learner path is explicit
- Rust snippets compile or are intentionally structured for the snippet harness
- docs, roadmap, and code artifact references do not contradict each other
