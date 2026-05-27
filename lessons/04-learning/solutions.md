# Learning Solutions

## Solution 1: Read the update rule

The update rule means:

> Replace the old weight with the old weight minus a learning-rate-sized step in the gradient direction.

If the gradient is positive, subtracting it lowers the weight. If the gradient is negative, subtracting it raises the weight.

## Solution 2: Separate the roles

1. `Weight`: parameter
2. `Gradient`: feedback about how a parameter affects loss
3. `LearningRate`: step-size control
4. `Adjustment`: the actual amount applied to a parameter
5. `Loss`: feedback about prediction error

## Solution 3: Run one step

A healthy run should show the loss after the update lower than the loss before the update for that same example.

The exact numbers may change if the example seed changes, but the interpretation should not:

```text
prediction before -> loss before -> gradient -> update -> prediction after -> loss after
```

## Solution 4: Read an epoch trace

Average loss is the mean loss across the dataset at a checkpoint.

If the printed average loss moves downward over epochs, the repeated updates are helping the model fit the tiny dataset better.

The important word is average. One example can briefly get worse while the total dataset average improves.

## Solution 5: Use the category-theory lens

The completed mapping is:

```text
FeatureVector -> PreActivation -> Prediction -> Loss
```

Training changes the parameterized map from `FeatureVector` to `PreActivation`, because the weights and bias live inside that transformation.

## Self-Check

- You can separate parameters, feedback values, and step-size controls.
- You can say whether one update reduced loss on the same example.
- You can explain an epoch average without hiding individual-example behavior.
- You can say that learning changes the parameterized map, not the meaning of the input.
