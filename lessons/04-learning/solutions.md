# Learning Module Solutions

## Solution 1: Name the four stages

The four stages are:

1. forward pass
2. loss
3. backward pass
4. optimizer update

New information at each stage:

- forward pass: the model produces a prediction
- loss: the prediction is compared to the target and scored
- backward pass: the complaint is distributed back into parameter gradients
- optimizer update: the parameters are actually moved

## Solution 2: One backward pass by hand

Raw score:

```math
z = 0.8 \cdot 1 + (-0.4) \cdot 0 + 0.1 = 0.9
```

Prediction:

```math
\hat{y} = \sigma(0.9) \approx 0.7109
```

Loss:

```math
L = (\hat{y} - 1)^2 \approx (0.7109 - 1)^2 \approx 0.0836
```

Why `dL/dw2 = 0` here:

- the local derivative `dz/dw2` is `x2`
- in this example `x2 = 0`
- so the full chain for `w2` is multiplied by zero

## Solution 3: Upstream versus local

For `w1`:

```math
\frac{dL}{dw_1}
=
\frac{dL}{d\hat{y}}
\cdot
\frac{d\hat{y}}{dz}
\cdot
\frac{dz}{dw_1}
```

Role labels:

1. judge room: `dL/d\hat{y}`
2. shaping room: `d\hat{y}/dz`
3. mixing room: `dz/dw1`

## Solution 4: Read the optimizer rule

Plain English:

> Replace the parameter vector with its old value minus a scaled step in the direction the loss says is uphill.

If `\eta` is far too large, the update can overshoot and training may bounce around or even make the loss worse.

## Solution 5: Dataset loop reasoning

1. Each example is seen 50 times.
2. If the average loss falls and then flattens, the model may be approaching the limit of what that architecture and learning-rate setting can improve.

## Solution 6: Token targets

1. The target is an index because the model is choosing among many vocabulary options, not only producing one binary-style scalar output.
2. The local training rule is still gradients plus updates, but the output space, target structure, and amount of computation are much larger.
