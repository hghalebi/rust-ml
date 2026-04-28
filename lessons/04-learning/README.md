# 04 Learning

Status: active.

This folder maps to course Module 3.

This module turns the neuron story into the first honest training system:
forward pass, loss, gradients, backpropagation, optimizer steps, dataset loops,
and the first bridge from scalar labels to token targets.

## Role In The Course

This module is where the course stops treating learning as one symbolic update
rule and starts treating it as a repeated executable process over data.

## Outcomes

After this module, you should be able to:

- explain one training step as `forward -> loss -> backward -> update`
- compute the gradients for `w1`, `w2`, and `b` without hiding the chain rule
- explain what an optimizer does and why a learning rate matters
- read a dataset loop and an epoch counter in plain English
- explain why token targets make language-model training a different-sized problem

## Lessons

1. [One training step, end to end](01-one-training-step-end-to-end.md)
2. [Backpropagation as local gradient bookkeeping](02-backpropagation-as-local-gradient-bookkeeping.md)
3. [Datasets, epochs, and token targets](03-datasets-epochs-and-token-targets.md)

## Practice

- [Exercises](exercises.md)
- [Solutions](solutions.md)

## Code Artifact

- Runnable crate: [`code/neuron`](../../code/neuron/README.md)

## Prerequisite

- Complete [03 Neuron](../03-neuron/README.md)

## Before You Move On

You are ready for the MLP module if you can:

- explain where the loss is measured and where the gradients come from
- describe backpropagation as local derivative factors multiplied along a path
- explain why one epoch is just one full pass over the dataset
- compare a scalar target such as `0` or `1` with a token target chosen from a vocabulary

## Current Focus

The authored material in this module is intentionally narrow:

- a single neuron with manual gradients
- SGD as the first optimizer
- tiny boolean datasets as the first training data
- a sequence-model bridge that introduces token targets without pretending the full Transformer training story is already covered
