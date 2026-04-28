# 05 MLP

Status: planned.

This folder maps to course Module 4.

This module will scale the neuron story into layers, hidden activations, and shape flow through a small network.

## Role In The Course

This planned module will connect the single trainable neuron from Modules 2 and 3 to a small network made from layers. It is the missing conceptual bridge between manual scalar gradients and the richer shape flow used later by attention and Transformers.

## Goal

Move from one neuron to a small multi-layer perceptron.

## Planned Lesson Ladder

1. from one neuron to one hidden layer
2. activations, layer outputs, and shape flow
3. backpropagation through a two-layer network
4. why depth changes what the model can represent

## Planned Practice

- trace the output shape through each layer
- compare a single neuron to a tiny hidden-layer model
- explain hidden activations in plain English
- identify which gradients belong to each layer

## Code Artifact

- Future crate: [`code/mlp`](../../code/mlp/README.md)

## Prerequisite

- Complete [04 Learning](../04-learning/README.md)

## Planned Outcome

Be able to reason about layers, hidden activations, shape flow, and layer-local gradients through a tiny network.
