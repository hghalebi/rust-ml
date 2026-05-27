# CS336 Rust Equivalent

Status: original Rust curriculum design.

This track is based on the public structure of [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/). It is not Stanford coursework, not a mirror, and not a copy of the lectures, handouts, tests, or code.

The goal is to build an equivalent learning journey in Rust:

```text
language modeling from scratch
  -> typed Rust implementation
  -> newtype-protected meaning
  -> category-theory intuition about maps and composition
```

For the scraped public course map, see [CS336 source map](references/courses/cs336-language-modeling-from-scratch.md).
For the object/map/newtype bridge that connects the core lessons to this track, see [Concept Atlas](lessons/CONCEPT-ATLAS.md).

## Audience

This track is for learners who have finished the core repo path through [06 Attention](lessons/06-attention/README.md) and want a serious language-modeling systems path.

Expected background:

- enough Rust to read structs, enums, traits, `Result`, tests, and iterators
- basic linear algebra from the earlier lessons
- patience for systems details such as memory, shapes, batching, and measurement

## Learning Objectives

After this track, the learner should be able to:

- implement the pieces of a tiny language model in Rust
- explain tokenizer, dataset, Transformer, optimizer, inference, evaluation, and alignment boundaries
- use newtypes to distinguish tokens, positions, logits, losses, metrics, and resource budgets
- read model training as composition plus feedback
- reason about scaling, data quality, and systems constraints without treating them as separate from ML

## Lecture-To-Rust Map

| Unit | Public CS336 topic | Original Rust lab direction |
| --- | --- | --- |
| 1 | Overview, tokenization | build typed `RawText -> TokenSequence -> TokenIds` maps |
| 2 | Resource accounting | count FLOPs, bytes moved, and arithmetic intensity for tiny layers |
| 3 | Architectures, hyperparameters | model `TransformerConfig` with validated dimensions and budgets |
| 4 | Attention alternatives, mixture of experts | compare attention maps and route tokens through typed experts |
| 5 | GPUs, TPUs | explain accelerator memory hierarchy with CPU-first Rust measurements |
| 6 | Kernels and compilation | write tiled CPU kernels and typed resource traces in [`code/kernels`](code/kernels/README.md) before discussing GPU lowering |
| 7 | Parallelism I | split batches and parameters while preserving ownership and shape contracts in [`code/parallelism`](code/parallelism/README.md) |
| 8 | Parallelism II | reason about communication, synchronization, and failure boundaries in [`code/parallelism`](code/parallelism/README.md) |
| 9 | Scaling laws I | log small experiments and fit a simple power-law curve |
| 10 | Inference | implement decoding, typed sampling controls, KV-cache traces, and latency budgets in [`code/inference`](code/inference/README.md) |
| 11 | Scaling laws II | use fitted curves to compare data, model, and compute tradeoffs |
| 12 | Evaluation | build metric newtypes and deterministic evaluation records in [`code/evaluation`](code/evaluation/README.md) |
| 13 | Data sources | stream documents through `DocumentId`, `Source`, and `CorpusShard` types |
| 14 | Data filtering | implement filters, dedup keys, mixture weights, and provenance checks |
| 15 | SFT and RLHF | represent instruction examples and preference pairs with explicit roles |
| 16 | RLVR | model verifier feedback and rollout traces as typed learning signals |
| 17 | Alignment systems | design auditable alignment pipelines with state transitions |
| 18 | Guest systems synthesis | write a design review of the full Rust language-modeling stack |
| 19 | Final synthesis | present a tested Rust artifact and explain the invariants it protects |

## Assignment Sequence

The Rust sequence mirrors the public assignment themes while staying original:

1. [R1 Basics](assignments/cs336-rust/01-basics.md): tokenizer, checked token IDs, next-token batches, loss, and a tiny trainable language-modeling core in [`code/lm_basics`](code/lm_basics/README.md).
2. [R2 Systems](assignments/cs336-rust/02-systems.md): profiling, memory accounting, attention FLOP estimates, median timings, and arithmetic intensity in [`code/systems`](code/systems/README.md).
3. [R3 Scaling](assignments/cs336-rust/03-scaling.md): experiment logs, scaling curves, component ablations, and forecast limitations in [`code/scaling`](code/scaling/README.md).
4. [R4 Data](assignments/cs336-rust/04-data.md): corpus ingestion, filtering, deduplication, and sampling mixtures in [`code/data`](code/data/README.md).
5. [R5 Alignment](assignments/cs336-rust/05-alignment.md): supervised finetuning data, preference pairs, verifier feedback, and safety notes in [`code/alignment`](code/alignment/README.md).

## Repository Integration

The current repo already provides the first conceptual bridge:

- [03 Neuron](lessons/03-neuron/README.md): typed prediction and update roles
- [04 Learning](lessons/04-learning/README.md): feedback loops and epochs
- [05 MLP](lessons/05-mlp/README.md): hidden representations and shape flow
- [06 Attention](lessons/06-attention/README.md): query/key/value roles, scores, weights, and value mixing
- [07 Transformer](lessons/07-transformer/README.md): typed encoder architecture
- [code/lm_basics](code/lm_basics/README.md): first executable CS336 R1 artifact for text-to-loss language modeling
- [code/systems](code/systems/README.md): first executable CS336 R2 artifact for typed resource accounting
- [code/kernels](code/kernels/README.md): first executable kernels artifact for typed tiling, reductions, matrix-vector traces, and resource estimates
- [code/scaling](code/scaling/README.md): first executable CS336 R3 artifact for typed experiment evidence and power-law fitting
- [code/data](code/data/README.md): first executable CS336 R4 artifact for typed corpus preparation
- [code/evaluation](code/evaluation/README.md): first executable evaluation artifact for typed examples, predictions, reports, and run comparison
- [code/inference](code/inference/README.md): first executable inference artifact for typed decoding controls, KV-cache traces, and latency budgets
- [code/parallelism](code/parallelism/README.md): first executable parallelism artifact for typed ranks, sharding plans, collective traces, and pipeline schedules
- [code/alignment](code/alignment/README.md): first executable CS336 R5 artifact for typed post-training signals

## Public-Safe Use

Use the Stanford page as a public reference for topic order and assignment themes. Do not copy handout wording, tests, starter code, private course communication, or non-public materials.

When adding Rust labs, write original prompts, original tests, and original explanations. The learning target can be equivalent; the artifact must be this repo's own work.
