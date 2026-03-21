# Lessons

This directory is the canonical course surface for the repo.

The folder numbers are sequential for the repository, while the course numbering starts at Module 0.

## Course Map

| Repo folder | Course module | Status | Outcome |
| --- | --- | --- | --- |
| [01-foundations](01-foundations/README.md) | Module 0 | Authored | Read basic ML notation and the minimum Rust syntax needed to follow later lessons. |
| [02-vectors](02-vectors/README.md) | Module 1 | Authored | Work with scalars, vectors, matrices, dot products, sigmoid, loss, and gradient-descent intuition. |
| [03-neuron](03-neuron/README.md) | Module 2 | Planned | Build the smallest real model: one neuron with weights, bias, and a forward pass. |
| [04-learning](04-learning/README.md) | Module 3 | Planned | Connect loss, gradients, and parameter updates into one training loop. |
| [05-mlp](05-mlp/README.md) | Module 4 | Planned | Extend one neuron into a small multi-layer network. |
| [06-attention](06-attention/README.md) | Module 5 | Planned | Represent tokens as vectors and learn how attention scores are formed. |
| [07-transformer](07-transformer/README.md) | Module 6 | Planned | Understand the compact attention formulation as a batched version of explicit dot-product loops. |
| [07-transformer](07-transformer/README.md) | Module 7 | Planned | Assemble a tiny transformer-style block from projections, attention, and residual structure. |

## How To Use These Lessons

- Start with the authored modules in order.
- Treat the placeholder modules as the roadmap, not as finished material.
- Read formulas as compressed code.
- Read Rust loops as expanded algebra.

## Content Contract

Every authored module contains:

- `README.md`
- ordered lesson files
- `exercises.md`
- `solutions.md`

Future modules may exist as `README.md` only until their content is authored.
