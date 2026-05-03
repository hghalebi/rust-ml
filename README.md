<p align="center">
  <img src="assets/logo.svg" alt="rust-ml logo" width="900">
</p>

# Rust ML Systems from First Principles

A beginner course for learning machine learning as a translation problem:

`plain English <-> algebra <-> Rust`

The goal is not to memorize symbols. The goal is to learn how to read formulas as programs, and how to read Rust code as precise mathematical structure.

For the canonical curriculum layout, see [lessons/Course Structure](lessons/COURSE-STRUCTURE.md).

## Who This Is For

- Beginners with little or no machine learning background
- Rust learners who want a concrete reason to use vectors, structs, loops, and functions
- Self-paced learners who want short lessons and small practice steps

## Start Here

1. Read [01 Foundations](lessons/01-foundations/README.md).
2. Continue with [02 Vectors](lessons/02-vectors/README.md).
3. Continue with [03 Neuron](lessons/03-neuron/README.md).
4. Continue with [04 Learning](lessons/04-learning/README.md).
5. Use [Lessons index](lessons/README.md) to see the full course map and the roadmap modules.

If you specifically want the current advanced preview after the core path, jump to [07 Transformer](lessons/07-transformer/README.md).

The repo uses sequential folder numbers even though the curriculum starts at Module 0:

- Course Module 0 -> Repo folder `lessons/01-foundations`
- Course Module 1 -> Repo folder `lessons/02-vectors`
- Course Module 2 -> Repo folder `lessons/03-neuron`
- Course Module 3 -> Repo folder `lessons/04-learning`
- Course Module 4 -> Repo folder `lessons/05-mlp`
- Course Module 5 -> Repo folder `lessons/06-attention`
- Course Module 6 -> Repo folder `lessons/07-transformer`

## What Exists Now

### Current coherent path

- [Lessons index](lessons/README.md)
- [01 Foundations](lessons/01-foundations/README.md)
- [02 Vectors](lessons/02-vectors/README.md)
- [03 Neuron](lessons/03-neuron/README.md)
- [04 Learning](lessons/04-learning/README.md)

### Neuron track now included

- [Rust Essentials for a Tiny Neuron](lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md)
- [A Neuron as a Chain of Functions](lessons/03-neuron/02-neuron-as-a-chain-of-functions.md)
- [Neuron exercises](lessons/03-neuron/exercises.md)
- [Neuron solutions](lessons/03-neuron/solutions.md)

### Training bridge now included

- [04 Learning](lessons/04-learning/README.md)
- [One training step, end to end](lessons/04-learning/01-one-training-step-end-to-end.md)
- [Backpropagation as local gradient bookkeeping](lessons/04-learning/02-backpropagation-as-local-gradient-bookkeeping.md)
- [Datasets, epochs, and token targets](lessons/04-learning/03-datasets-epochs-and-token-targets.md)
- [Learning exercises](lessons/04-learning/exercises.md)
- [Learning solutions](lessons/04-learning/solutions.md)

### Advanced authored preview

- [07 Transformer](lessons/07-transformer/README.md)
- [What Problem the Transformer Solves](lessons/07-transformer/01-tiny-transformer-from-first-principles.md)
- [Typed Rust Transformer with Expressive Errors](lessons/07-transformer/02-typed-rust-transformer-with-linear-attention.md)
- [Transformer Encoder in Small Chunks](lessons/07-transformer/03-transformer-encoder-in-small-chunks.md)
- [Transformer exercises](lessons/07-transformer/exercises.md)
- [Transformer solutions](lessons/07-transformer/solutions.md)

### Executable companion code

- [Code index](code/README.md)
- [neuron crate](code/neuron/README.md)
- [transformer crate](code/transformer/README.md)

### Source material and roadmap

- [Reference material](references/README.md)
- [05 MLP](lessons/05-mlp/README.md)
- [06 Attention](lessons/06-attention/README.md)
- [Book wrapper placeholder](book/README.md)

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
- `code/` follows the lesson progression and now includes tested `neuron` and `transformer` crates.
- `book/` is intentionally thin in this pass so the course content does not drift into two competing copies.
- `lessons/COURSE-STRUCTURE.md` is the canonical structure guide for module and lesson contracts.

## Learning Strategy

The course keeps the same translation goal everywhere:

`plain English <-> algebra <-> Rust`

The current repo intentionally has two different learning depths:

- a coherent beginner path through Modules 0, 1, 2, and 3
- an advanced Transformer preview in Module 6

## Course Phases

| Phase | Modules | Status | Learner checkpoint |
| --- | --- | --- | --- |
| Orientation | 0 Foundations, 1 Vectors | Active | Read ML notation and vector code without losing the plain-English meaning. |
| First trainable system | 2 Neuron, 3 Learning | Active | Explain one model, one backward pass, one optimizer step, and one token-target bridge. |
| Bridge to architecture | 4 MLP, 5 Attention | Planned | Connect training to layers, hidden activations, token interactions, and attention scores. |
| Architecture preview | 6 Transformer | Active preview | Read the encoder path and understand why full Transformer training needs more machinery. |

Module 6 applies the translation rule in two complementary ways:

- narrative lessons that explain the architecture and the implementation choices
- a chunked encoder lesson where every concept is written as `English -> Algebra -> Rust`

That repetition is intentional. Repetition is how the translation dictionary becomes automatic.

## Suggested Study Flow

1. Read the module README.
2. Work through the lesson files in order.
3. Do the module exercises without copying from the solutions first.
4. Use the solution files to check reasoning, naming, and Rust syntax.
5. Move to the next module only after you can explain each formula out loud in English.

Current recommended sequence:

1. [01 Foundations](lessons/01-foundations/README.md)
2. [02 Vectors](lessons/02-vectors/README.md)
3. [03 Neuron](lessons/03-neuron/README.md)
4. [04 Learning](lessons/04-learning/README.md)
5. [07 Transformer](lessons/07-transformer/README.md) only if you want the advanced preview before the MLP and Attention bridge modules are authored

## Running The Code

The current runnable code artifacts are the neuron and Transformer teaching crates:

```bash
cargo test --manifest-path code/neuron/Cargo.toml
```

```bash
cargo test --manifest-path code/transformer/Cargo.toml
```

The neuron crate covers:

- a single typed neuron
- squared-error loss and manual gradients
- SGD parameter updates
- tiny boolean datasets and epoch loops
- token-target utilities and cross-entropy bridge code
- a tiny bigram next-token model that turns `token -> embedding -> lm_head` into a real training loop

The Transformer crate covers:

- dense vectors and matrices
- semantic model newtypes such as `TokenEmbedding`, `Query`, `Key`, and `Value`
- expressive `thiserror` diagnostics for shape mistakes
- standard self-attention and multi-head attention
- a simplified linear-attention comparison point
- positional encodings, layer norm, feed-forward layers, and an encoder block

## Quality Automation

The repo now includes two GitHub Actions workflows for quality control:

- `CI` runs deterministic checks for lesson structure, local Markdown links, and authored-section contracts.
- `CI` also compile-checks Rust snippets embedded in lessons and runs `cargo fmt`, `cargo clippy`, and `cargo test` for both the neuron and Transformer teaching crates.
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
