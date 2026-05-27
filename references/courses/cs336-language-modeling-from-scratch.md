# CS336 Language Modeling From Scratch Source Map

Status: public source map.

Last checked: 2026-05-27.

Source: [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/).

This file records the public course structure used to design this repo's original Rust equivalent. It is not a copy of Stanford's lectures, handouts, code, or assignments.

## Public Course Shape

The public CS336 page presents the course as an implementation-heavy language-modeling course that walks through data preparation, Transformer construction, training, evaluation, and deployment-facing concerns.

The current public offering is Spring 2026.

## Public Assignment Tracks

| Public CS336 assignment | Public focus | Rust equivalent in this repo |
| --- | --- | --- |
| [Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics/tree/main) | tokenizer, model architecture, optimizer, minimal language model | [R1 Basics](../../assignments/cs336-rust/01-basics.md) |
| [Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems/tree/main) | profiling, benchmarking, optimized attention, distributed training | [R2 Systems](../../assignments/cs336-rust/02-systems.md) |
| [Assignment 3: Scaling](https://github.com/stanford-cs336/assignment3-scaling/tree/main) | Transformer component understanding and scaling-law fitting | [R3 Scaling](../../assignments/cs336-rust/03-scaling.md) |
| [Assignment 4: Data](https://github.com/stanford-cs336/assignment4-data/tree/main) | raw web data conversion, filtering, deduplication | [R4 Data](../../assignments/cs336-rust/04-data.md) |
| [Assignment 5: Alignment and Reasoning RL](https://github.com/stanford-cs336/assignment5-alignment/tree/main) | supervised finetuning, reinforcement learning for reasoning, optional safety alignment | [R5 Alignment](../../assignments/cs336-rust/05-alignment.md) |

## Public Lecture Map

| Meeting | Public topic | Rust equivalent direction |
| --- | --- | --- |
| 1 | Overview, tokenization | typed text, tokens, vocabulary IDs, and BPE-style merges |
| 2 | PyTorch and resource accounting | Rust tensor loops, FLOPs, memory traffic, and arithmetic intensity |
| 3 | Architectures and hyperparameters | typed Transformer configuration and parameter budgets in [`code/transformer`](../../code/transformer/README.md) |
| 4 | Attention alternatives and mixture of experts | attention variants and typed expert routing in [`code/transformer`](../../code/transformer/README.md) |
| 5 | GPUs and TPUs | accelerator memory hierarchy as typed byte, bandwidth, memory-tier, transfer-time, and public-report maps in [`code/systems`](../../code/systems/README.md) |
| 6 | Kernels, Triton | Rust kernel interfaces, tiling intuition, and typed lowering boundaries in [`code/kernels`](../../code/kernels/README.md) |
| 7 | Parallelism | data, tensor, and pipeline parallelism as ownership, partitioning, and public trace-boundary problems in [`code/parallelism`](../../code/parallelism/README.md) |
| 8 | Parallelism | communication cost, synchronization, distributed shape flow, and public report review in [`code/parallelism`](../../code/parallelism/README.md) |
| 9 | Scaling laws | typed experiment logs and power-law fitting from small runs |
| 10 | Inference | decoding, KV cache roles, batching, latency budgets, and public trace review in [`code/inference`](../../code/inference/README.md) |
| 11 | Scaling laws | using fitted laws and public-report review to reason about model, data, and compute choices in [`code/scaling`](../../code/scaling/README.md) |
| 12 | Evaluation | deterministic eval harnesses, metric newtypes, and public report reviews in [`code/evaluation`](../../code/evaluation/README.md) |
| 13 | Data sources and datasets | streaming corpora, document IDs, and public dataset manifests in [`code/data`](../../code/data/README.md) |
| 14 | Data filtering, deduplication, mixing, synthetic data | typed filters, dedup keys, sampling mixtures, provenance, and public/private source boundaries in [`code/data`](../../code/data/README.md) |
| 15 | Mid/post-training and SFT/RLHF | instruction examples, preference data, and policy updates |
| 16 | Post-training and RLVR | reward signals, verifiers, rollouts, and reasoning traces |
| 17 | Alignment - multimodality and RL systems | alignment pipelines as auditable state machines with public-release review in [`code/alignment`](../../code/alignment/README.md) |
| 18 | Guest lecture | external systems perspective and synthesis notes |
| 19 | Guest lecture | final synthesis and project presentation standard |

## Public Lecture Material Links

These links point to public course materials. They are used as topic references only; this repo does not copy lecture prose, code, slides, private course communication, tests, or handouts.

| Meeting | Public material |
| --- | --- |
| 1 | [lecture_01.py](https://cs336.stanford.edu/lectures/?trace=lecture_01) |
| 2 | [lecture_02.py](https://cs336.stanford.edu/lectures/?trace=lecture_02) |
| 3 | [lecture_03.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) |
| 4 | [lecture_04.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf) |
| 5 | [lecture_05.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf) |
| 6 | [lecture_06.py](https://cs336.stanford.edu/lectures/?trace=lecture_06) |
| 7 | [lecture_07.py](https://cs336.stanford.edu/lectures/?trace=lecture_07) |
| 8 | [lecture_08.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf) |
| 9 | [lecture_09.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf) |
| 10 | [lecture_10.py](https://cs336.stanford.edu/lectures/?trace=lecture_10) |
| 11 | [lecture_11.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf) |
| 12 | [lecture_12.py](https://cs336.stanford.edu/lectures/?trace=lecture_12) |
| 13 | [lecture_13.py](https://cs336.stanford.edu/lectures/?trace=lecture_13) |
| 14 | [lecture_14.py](https://cs336.stanford.edu/lectures/?trace=lecture_14) |
| 15 | [lecture_15.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf) |
| 16 | [lecture_16.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf) |
| 17 | [lecture_17.py](https://cs336.stanford.edu/lectures/?trace=lecture_17) |

## Translation Rule For This Repo

The Rust equivalent uses this translation:

```text
Python/PyTorch assignment idea
  -> original Rust teaching lab
  -> semantic newtypes
  -> explicit shape/resource invariants
  -> small tests and runnable examples
```

This keeps the spirit of learning by implementation while making the material original, public-safe, and consistent with this repo's type-driven pedagogy.
