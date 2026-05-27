# Neuron Solutions

## Solution 1: Label the factory

1. Mixing room:
   receives inputs and parameters, produces raw score `z`
2. Shaping room:
   receives `z`, produces prediction `\hat{y}`
3. Judge room:
   receives prediction and target, produces loss `L`

## Solution 2: One forward pass by hand

First compute the raw score:

```math
z = 0.8 \cdot 1 + (-0.4) \cdot 0 + 0.1 = 0.9
```

Then the sigmoid output:

```math
\hat{y} = \sigma(0.9) \approx 0.7109
```

Then the squared-error loss with `y = 1`:

```math
L = (\hat{y} - y)^2 \approx (0.7109 - 1)^2 \approx 0.0836
```

## Solution 3: Read the dependency chain

1. `w1 -> pre_activation -> activation -> loss`
2. `w2 -> pre_activation -> activation -> loss`
3. `b -> pre_activation -> activation -> loss`

In words: each parameter changes the raw score first, then the prediction, then the loss.

## Solution 4: Explain the local derivatives

Algebra view:

- `z` is linear in each parameter
- the coefficient of `w1` is `x1`, so `dz/dw1 = x1`
- the coefficient of `w2` is `x2`, so `dz/dw2 = x2`
- bias is added directly, so `dz/db = 1`

Metaphor view:

- each weight is a knob on one signal channel
- the effect of turning the knob depends on how much signal is on that channel
- bias is just a constant offset, so changing it always moves the output by the same immediate amount

## Solution 5: Read the update rule in English

Ordinary English:

> Replace the parameter with its old value minus a small step in the direction suggested by the loss gradient.

## Solution 6: Memory hook

Forward pass:

- mix
- squash
- judge

Backward pass:

- blame
- trace
- adjust

## Solution 7: Run the executable ladder

A strong self-check answer should connect each program to the model chain:

1. `01_weighted_sum` prints the dot product, so it is the mixing step before bias and sigmoid.
2. `02_forward_pass` prints `z` and the sigmoid prediction, so it shows `mix -> squash`.
3. `03_one_step_training` prints loss before and after the update, so it shows `blame -> trace -> adjust`.

The exact gradient values matter less than the direction: for the positive example, the loss after the update should be lower than the loss before the update.

## Self-Check

- You can trace `FeatureVector -> WeightedSum -> Prediction -> Loss` without skipping a role.
- You can explain which parameter each local derivative belongs to.
- You can describe the update as blame, trace, adjust.
- You can connect every printed example value to a Rust newtype in `code/neuron`.
