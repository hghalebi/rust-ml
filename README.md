<p align="center">
  <img src="assets/logo.svg" alt="rust-ml logo" width="900">
</p>

# Rust ML Systems from First Principles

A beginner course for learning machine learning as a translation problem:

`plain English <-> algebra <-> Rust`

The goal is not to memorize symbols. The goal is to learn how to read formulas as programs, and how to read Rust code as precise mathematical structure.

## Who This Is For

- Beginners with little or no machine learning background
- Rust learners who want a concrete reason to use vectors, structs, loops, and functions
- Self-paced learners who want short lessons and small practice steps

## Start Here

1. Read [01 Foundations](lessons/01-foundations/README.md).
2. Continue with [02 Vectors](lessons/02-vectors/README.md).
3. Use [Lessons index](lessons/README.md) to see the full course map.

If you specifically want the current Transformer material after the fundamentals, jump to [07 Transformer](lessons/07-transformer/README.md).

The repo uses sequential folder numbers even though the curriculum starts at Module 0:

- Course Module 0 -> Repo folder `lessons/01-foundations`
- Course Module 1 -> Repo folder `lessons/02-vectors`

## What Exists Now

### Authored lessons

- [Lessons index](lessons/README.md)
- [01 Foundations](lessons/01-foundations/README.md)
- [02 Vectors](lessons/02-vectors/README.md)
- [07 Transformer](lessons/07-transformer/README.md)

### Transformer track now included

- [Lesson 17: A Tiny Transformer (From First Principles)](lessons/07-transformer/01-tiny-transformer-from-first-principles.md)
- [Lesson 18: Typed Rust Transformer with Linear Attention](lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md)
- [Lesson 19: Transformer Encoder in Small Chunks](lessons/07-transformer/03-transformer-encoder-in-small-chunks.md)
- [Transformer exercises](lessons/07-transformer/exercises.md)
- [Transformer solutions](lessons/07-transformer/solutions.md)

### Executable companion code

- [Code index](code/README.md)
- [transformer crate](code/transformer/README.md)

### Source material and roadmap

- [Reference material](references/README.md)
- [03 Neuron](lessons/03-neuron/README.md)
- [04 Learning](lessons/04-learning/README.md)
- [05 MLP](lessons/05-mlp/README.md)
- [06 Attention](lessons/06-attention/README.md)
- [Book placeholder](book/README.md)

## Repo Map

```text
rust-ml/
├── lessons/    # canonical course content
├── references/ # transcripts and papers used as source material
├── code/       # runnable companion crates
├── book/       # future mdBook/site wrapper
└── README.md
```

## Working Rules For This Repo

- `lessons/` is the source of truth for written teaching content.
- `code/` follows the lesson progression and now includes a real tested `transformer` crate.
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

The Transformer module also includes:

- a typed Rust lesson that bridges architecture and implementation
- a chunked encoder lesson for lower-cognitive-load study
- a small companion crate that turns the lesson concepts into runnable Rust

## Suggested Study Flow

1. Read the module README.
2. Work through the lesson files in order.
3. Do the module exercises without copying from the solutions first.
4. Use the solution files to check reasoning, naming, and Rust syntax.
5. Move to the next module only after you can explain each formula out loud in English.

## Running The Code

The current runnable code artifact is the Transformer teaching crate:

```bash
cargo test --manifest-path code/transformer/Cargo.toml
```

That crate covers:

- vectors and matrices
- newtype-based scalars and structural dimensions
- linear layers
- sequences
- standard self-attention
- simplified linear attention
- feed-forward layers
- a minimal Transformer block

## Quality Automation

The repo now includes two GitHub Actions workflows for quality control:

- `CI` runs deterministic checks for lesson structure, local Markdown links, and authored-section contracts.
- `CI` also compile-checks Rust snippets embedded in lessons and runs `cargo fmt`, `cargo clippy`, and `cargo test` for the Transformer teaching crate.
- `Gemini Writing Review` reviews Markdown content on pull requests for English clarity, technical-teaching quality, structural discipline, and beginner friendliness.

The Gemini review is advisory, not a replacement for human judgment. It is designed to catch weak phrasing, excess cognitive load, mismatches between English and code, and places where the teaching flow violates common technical-writing or technical-instruction best practices.

To enable Gemini review in GitHub Actions, configure:

- repository secret `GEMINI_API_KEY`
- optional repository variable `GEMINI_MODEL` if you want a model other than the default `gemini-2.0-flash`

The workflow writes a review artifact named `gemini-writing-review` so the writing assessment can be read directly from the workflow run.

## References

The repo keeps supporting source material in [references/](references/README.md), including:

- a Transformer explainer transcript
- Bahdanau et al. (2014)
- Luong et al. (2015)
- Vaswani et al. (2017)
- Sebastian Raschka's *LLMs From Scratch* repository as an external inspiration source for attention, GPT, and educational sequencing
