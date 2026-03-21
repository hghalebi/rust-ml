# Rust ML Systems from First Principles

A beginner course for learning machine learning as a translation problem:

`plain English <-> algebra <-> Rust`

The goal is not to memorize symbols. The goal is to learn how to read formulas as programs, and how to read Rust code as precise mathematical structure.

## Who This Is For

- Beginners with little or no machine learning background
- Rust learners who want a concrete reason to use vectors, structs, loops, and functions
- Self-paced learners who want short lessons and small practice steps

## Start Here

1. [Module 0 in repo form: Foundations](lessons/01-foundations/README.md)
2. [Module 1 in repo form: Scalars, vectors, and dot products](lessons/02-vectors/README.md)

The repo uses sequential folder numbers even though the curriculum starts at Module 0:

- Course Module 0 -> Repo folder `lessons/01-foundations`
- Course Module 1 -> Repo folder `lessons/02-vectors`

## What Exists Now

### Authored now

- [Lessons index](lessons/README.md)
- [01 Foundations](lessons/01-foundations/README.md)
- [02 Vectors](lessons/02-vectors/README.md)

### Scaffolded now

- [03 Neuron](lessons/03-neuron/README.md)
- [04 Learning](lessons/04-learning/README.md)
- [05 MLP](lessons/05-mlp/README.md)
- [06 Attention](lessons/06-attention/README.md)
- [07 Transformer](lessons/07-transformer/README.md)
- [Code strategy](code/README.md)
- [Book placeholder](book/README.md)

## Repo Map

```text
rust-ml/
├── lessons/   # canonical course content
├── code/      # future runnable examples and crates
├── book/      # future mdBook/site wrapper
└── README.md
```

## Working Rules For This Repo

- `lessons/` is the source of truth for written teaching content.
- `code/` follows the lesson progression and will later become runnable Cargo workspace crates.
- `book/` is intentionally thin in this pass so the course content does not drift into two competing copies.

## Learning Strategy

Each authored lesson follows the same pattern:

1. Overview
2. Learning goals
3. Plain-English explanation
4. Algebra form
5. Rust form
6. Why this matters
7. Short practice

That repetition is intentional. Repetition is how the translation dictionary becomes automatic.

## Suggested Study Flow

1. Read the module README.
2. Work through the lesson files in order.
3. Do the module exercises without copying from the solutions first.
4. Use the solution files to check reasoning, naming, and Rust syntax.
5. Move to the next module only after you can explain each formula out loud in English.
