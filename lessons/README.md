# Lessons

This directory is the canonical course surface for the repo.

The folder numbers are sequential for the repository, while the course numbering starts at Module 0.

For the canonical structure rules, see [Course Structure](COURSE-STRUCTURE.md).

Before Module 0, read [The Learning Lens](00-learning-lens.md). It explains the repo's shared vocabulary: newtypes, maps, composition, and category-theory intuition without heavy jargon.

Keep [Concept Atlas](CONCEPT-ATLAS.md) nearby as the cross-module object/map guide. It connects each ML idea to the Rust newtype, invariant, map, and runnable proof that teaches it.

## Course Map

| Repo folder | Course module | Status | Outcome |
| --- | --- | --- | --- |
| [01-foundations](01-foundations/README.md) | Module 0 | Authored | Read basic ML notation and the minimum Rust syntax needed to follow later lessons. |
| [02-vectors](02-vectors/README.md) | Module 1 | Authored | Work with scalars, vectors, matrices, dot products, sigmoid, loss, and gradient-descent intuition. |
| [03-neuron](03-neuron/README.md) | Module 2 | Authored | Build one typed neuron, read it as a chain of functions, and explain a single backward pass without hand-waving. |
| [04-learning](04-learning/README.md) | Module 3 | Authored | Connect loss, gradients, and parameter updates into a dataset-level learning loop. |
| [05-mlp](05-mlp/README.md) | Module 4 | Authored | Extend one neuron into a small multi-layer network with hidden representations and shape flow. |
| [06-attention](06-attention/README.md) | Module 5 | Authored | Represent tokens as vectors and learn how attention scores, weights, and value mixtures are formed. |
| [07-transformer](07-transformer/README.md) | Module 6 | Authored | Learn the encoder path through semantic Rust types, expressive errors, and an English/Algebra/Rust chunk ladder. |
| [08-language-modeling](08-language-modeling/README.md) | Module 7 | Authored | Turn public text into token IDs, next-token batches, loss, and one tiny update. |
| [09-systems](09-systems/README.md) | Module 8 | Authored | Measure bytes, FLOPs, timing, intensity, memory hierarchy, and public systems reports. |
| [10-kernels](10-kernels/README.md) | Module 9 | Authored | Teach elementwise kernels, reductions, tiling, resource estimates, and public kernel reports. |

## Current Recommended Paths

### Core path now

Start here if you want the current coherent beginner sequence:

1. [The Learning Lens](00-learning-lens.md)
2. [Concept Atlas](CONCEPT-ATLAS.md)
3. [01 Foundations](01-foundations/README.md)
4. [02 Scalars, Vectors, and Dot Products](02-vectors/README.md)
5. [03 Neuron](03-neuron/README.md)
6. [04 Learning](04-learning/README.md)
7. [05 MLP](05-mlp/README.md)
8. [06 Attention](06-attention/README.md)
9. [07 Transformer](07-transformer/README.md)
10. [08 Language Modeling](08-language-modeling/README.md)
11. [09 Systems](09-systems/README.md)
12. [10 Kernels](10-kernels/README.md)

## How To Use These Lessons

- Start with the authored modules in order.
- Use the learning lens to connect English, algebra, Rust newtypes, and maps.
- Read formulas as compressed code.
- Read Rust loops as expanded algebra.
- Ask what each type means before asking how it is stored.
- Use the core path first unless you intentionally want to jump ahead to the Transformer module.

## Content Contract

Every authored module contains:

- `README.md`
- ordered lesson files
- `exercises.md`
- `solutions.md`

Future modules may exist as `README.md` only until their content is authored.
