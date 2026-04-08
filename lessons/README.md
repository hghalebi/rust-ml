# Lessons

This directory is the canonical course surface for the repo.

The folder numbers are sequential for the repository, while the course numbering starts at Module 0.

For the canonical structure rules, see [Course Structure](COURSE-STRUCTURE.md).

## Course Map

| Repo folder | Course module | Status | Outcome |
| --- | --- | --- | --- |
| [01-foundations](01-foundations/README.md) | Module 0 | Authored | Read basic ML notation and the minimum Rust syntax needed to follow later lessons. |
| [02-vectors](02-vectors/README.md) | Module 1 | Authored | Work with scalars, vectors, matrices, dot products, sigmoid, loss, and gradient-descent intuition. |
| [03-neuron](03-neuron/README.md) | Module 2 | Authored | Build one typed neuron, read it as a chain of functions, and explain a single backward pass without hand-waving. |
| [04-learning](04-learning/README.md) | Module 3 | Planned | Connect loss, gradients, and parameter updates into one training loop. |
| [05-mlp](05-mlp/README.md) | Module 4 | Planned | Extend one neuron into a small multi-layer network. |
| [06-attention](06-attention/README.md) | Module 5 | Planned | Represent tokens as vectors and learn how attention scores are formed. |
| [07-transformer](07-transformer/README.md) | Module 6 | Authored preview | Learn the encoder path through semantic Rust types, expressive errors, and an English/Algebra/Rust chunk ladder. |

## Current Recommended Paths

### Core path now

Start here if you want the current coherent beginner sequence:

1. [01 Foundations](01-foundations/README.md)
2. [02 Scalars, Vectors, and Dot Products](02-vectors/README.md)
3. [03 Neuron](03-neuron/README.md)

### Advanced preview now

If you want the most complete advanced material already in the repo, continue with:

4. [07 Transformer](07-transformer/README.md)

Treat it as an advanced preview for now. Modules 3, 4, and 5 remain planned and will eventually close the conceptual gap between the neuron lessons and the Transformer module.

## How To Use These Lessons

- Start with the authored modules in order.
- Treat the placeholder modules as the roadmap, not as finished material.
- Read formulas as compressed code.
- Read Rust loops as expanded algebra.
- Use the core path first unless you intentionally want to jump ahead to the Transformer preview.

## Content Contract

Every authored module contains:

- `README.md`
- ordered lesson files
- `exercises.md`
- `solutions.md`

Future modules may exist as `README.md` only until their content is authored.
