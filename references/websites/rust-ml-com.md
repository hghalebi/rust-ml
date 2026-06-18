# rust-ml.com Content Source Map

Status: active public source map.
Snapshot checked: 2026-06-18.

Primary citation: [rust-ml.com](https://rust-ml.com/).
Canonical public course repository: [hghalebi/rust-ml](https://github.com/hghalebi/rust-ml).

This note maps the current public website surface back to this repository's
lessons, examples, and workshop-ready material. The website is a live companion
surface; this repository remains the canonical source for the durable course
content and runnable Rust examples.

## Cited Website Routes

Use these public routes when citing the current website surface:

- [Home](https://rust-ml.com/#/)
- [Lab](https://rust-ml.com/#/lab)
- [Category Theory for Tiny ML](https://rust-ml.com/#/lab/category-theory)
- [Workshops](https://rust-ml.com/#/workshop)
- [Regulated Agent Readiness](https://rust-ml.com/#/regulated-agents)

## Website Thesis

The public website frames Rust ML as first-principles machine learning in Rust:
small systems, typed structure, runnable experiments, and a learner path that
moves from meaning to notation, code, and experiments. The site also presents
the lab method as one clear concept, one small runnable system, one Rust
implementation, one structural explanation, and one modification exercise.

Repository alignment:

- Keep authored curriculum in [`lessons/`](../../lessons/README.md).
- Keep executable artifacts in [`code/`](../../code/README.md).
- Keep public/private publication rules in [`PUBLIC_CONTENT.md`](../../PUBLIC_CONTENT.md).
- Keep citation and provenance notes in [`references/`](../README.md).
- Use [`hghalebi/category_theory_transformer_rs`](../repos/category-theory-transformer-rs.md)
  as the public source map for the website's Category Theory lab route.

## Website Categories

The website currently organizes learner intent into four content categories.
Use these categories when adding repo examples, workshop notes, or public
curriculum cross-links.

| Website category | Learner question | Repository anchor | Example commands |
| --- | --- | --- | --- |
| `start-here` | Why do typed stages and legal moves matter before code gets large? | [`lessons/00-learning-lens.md`](../../lessons/00-learning-lens.md), [`lessons/01-foundations`](../../lessons/01-foundations/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 01_objects_and_maps` |
| `tiny-ml` | How do typed transformations become a tiny ML system? | [`lessons/03-neuron`](../../lessons/03-neuron/README.md), [`lessons/04-learning`](../../lessons/04-learning/README.md), [`code/neuron`](../../code/neuron/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training` |
| `transformers` | How does the same lens scale to attention, memory, and generation? | [`lessons/06-attention`](../../lessons/06-attention/README.md), [`lessons/07-transformer`](../../lessons/07-transformer/README.md), [`lessons/11-inference`](../../lessons/11-inference/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace` |
| `production` | How does typed-transition discipline help real agent systems? | [`code/alignment`](../../code/alignment/README.md), [`assignments/cs336-rust/05-alignment.md`](../../assignments/cs336-rust/05-alignment.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 05_alignment_workflow` |

## Workshop Surface

The website's workshop route is an in-person Paris workshop surface for
learning Rust through tiny machine-learning systems. It emphasizes a build-first
format, small code, clear explanations, and real understanding. The stated
prerequisites are intentionally light: basic coding in any language, the line
formula `y = ax + b`, and a loose memory of derivatives.

Repo material that is workshop-ready:

| Workshop segment | Repo source | Runnable proof |
| --- | --- | --- |
| Self-serve workshop route | [`workshops/README.md`](../../workshops/README.md) | `python3 scripts/check_course_content.py` |
| Meaning before syntax | [`lessons/00-learning-lens.md`](../../lessons/00-learning-lens.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 02_compose_neuron_forward` |
| Rust with tiny ML motivation | [`lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md`](../../lessons/03-neuron/01-rust-essentials-for-a-tiny-neuron.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum` |
| One complete update | [`lessons/04-learning/01-one-training-step-end-to-end.md`](../../lessons/04-learning/01-one-training-step-end-to-end.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training` |
| Public learning trace | [`PUBLIC_CONTENT.md`](../../PUBLIC_CONTENT.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 05_public_training_step` |
| Attention as visible routing | [`lessons/06-attention`](../../lessons/06-attention/README.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace` |

## Workshop Example Families

Workshop examples should stay small enough to run and explain live. The current
workshop-ready pattern is:

```text
semantic input -> raw score -> probability -> target comparison -> loss
-> gradient-like correction -> updated parameter
```

That pattern can be taught with three public-safe example families:

| Example family | Teaching object | Repo-aligned lesson | Runnable proof |
| --- | --- | --- | --- |
| Legal stage graph | typed stage transitions where only adjacent meanings compose | [`lessons/00-learning-lens.md`](../../lessons/00-learning-lens.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 03_composition_failure` |
| One-weight classifier | one semantic input, one parameter, one sigmoid prediction, one loss, one update | [`lessons/04-learning/01-one-training-step-end-to-end.md`](../../lessons/04-learning/01-one-training-step-end-to-end.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training` |
| Training trace | repeated examples showing loss and parameter movement over time | [`lessons/04-learning/02-epochs-and-loss-traces.md`](../../lessons/04-learning/02-epochs-and-loss-traces.md) | `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch` |

When turning a workshop example into durable repo content, preserve the same
contract as the rest of the course:

- name every domain value before arithmetic uses it
- make illegal transitions fail at the typed boundary where possible
- show the public learner command that proves the example
- keep private workshop logistics out of public learner files

## Website Curriculum To Repo Curriculum

The home page presents a compact seven-module public curriculum:

1. Foundations
2. Vectors and matrices
3. The neuron
4. Learning
5. Networks
6. Attention
7. Transformers

The repository extends that public website ladder with durable systems modules
and the CS336 Rust equivalent path:

- [`08-language-modeling`](../../lessons/08-language-modeling/README.md)
- [`09-systems`](../../lessons/09-systems/README.md)
- [`10-kernels`](../../lessons/10-kernels/README.md)
- [`11-inference`](../../lessons/11-inference/README.md)
- [`assignments/cs336-rust`](../../assignments/cs336-rust/README.md)

This extension should remain repo-first unless the website is intentionally
updated to expose the longer course path.

## Update Rule

When `rust-ml.com` changes in a way that affects curriculum, workshops, or
public claims:

1. Re-check the public routes above.
2. Update this source map with the changed website category, route, or claim.
3. Link the change to the closest repo lesson, assignment, or runnable example.
4. Run the public content checks before publishing repo changes.

Recommended checks:

```bash
python3 scripts/check_public_content.py
python3 scripts/check_course_content.py
python3 scripts/check_rust_teaching_contract.py
```
