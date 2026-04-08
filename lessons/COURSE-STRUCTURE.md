# Course Structure

This file is the canonical structure guide for the course.

The repo teaches machine learning as a translation exercise:

`plain English <-> algebra <-> Rust`

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

The order matters. Beginners should not have to decode symbolic syntax before they know what the idea is for.

## Current Learning Paths

The repo currently supports two realistic paths:

### Core Path

This is the recommended order for most learners right now:

1. `01-foundations` -> course Module 0
2. `02-vectors` -> course Module 1
3. `03-neuron` -> course Module 2

### Advanced Preview Path

The Transformer module is already authored even though the middle modules are still planned.

Use it as an advanced preview after the core path:

4. `07-transformer` -> course Module 6 preview

Modules `04-learning`, `05-mlp`, and `06-attention` remain the planned bridge that will eventually make the path fully continuous.

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
- `## Short Practice`

Module `07-transformer` is allowed to use the chunked `English -> Algebra -> Rust` pattern, but it still needs a clear overview and learning goals at the top.

## Practice Contract

Each authored module must contain:

- `exercises.md`
- `solutions.md`

The exercise file should ask the learner to reason, compute, explain, or translate.

The solution file should:

- show the answer
- explain the reasoning
- keep the answer concise enough for self-checking

## Code Contract

`code/` follows the lesson progression.

- Placeholder topic directories may exist before a real crate is ready
- Once a topic earns executable material, it should become a real Cargo crate
- The code README should state whether the topic is active or still planned
- `lessons/` stays the canonical teaching surface even when a companion crate exists

## Pedagogy Rules

- Prefer small lessons over giant walls of mixed concepts
- Introduce one idea at a time, then connect it to the larger system
- Use metaphor only when it reduces confusion, not when it replaces the real mechanism
- Keep examples small enough to inspect without scrolling through noise
- Use semantic Rust types when they improve readability and meaning
- Do not treat advanced modules as prerequisites for beginner modules

## Review Checklist

Before considering a lesson coherent and merge-ready, verify:

- the course-module mapping is correct
- the module README follows the correct contract for its status
- lesson titles are concept-first and not globally misnumbered
- the learner path is explicit
- Rust snippets compile or are intentionally structured for the snippet harness
- docs, roadmap, and code artifact references do not contradict each other
