# Learning Exercises

## Exercise 1: Read the update rule

Explain this sentence in your own words:

```math
w := w - \eta \frac{dL}{dw}
```

Use the words:

- old weight
- learning rate
- gradient
- new weight

## Exercise 2: Separate the roles

For each value, say whether it is a parameter, feedback, or a step-size control:

1. `Weight`
2. `Gradient`
3. `LearningRate`
4. `Adjustment`
5. `Loss`

## Exercise 3: Run one step

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
```

Write down:

1. prediction before
2. loss before
3. bias gradient
4. prediction after
5. loss after

Then answer: did this one update help on the same example?

## Exercise 4: Read an epoch trace

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

Explain what the printed average loss means.

## Exercise 5: Use the category-theory lens

Complete this mapping:

```text
FeatureVector -> ______ -> Prediction -> ______
```

Then explain which part training changes.

## Failure Signals

- You explain learning as replacing the example instead of changing parameters.
- You treat `LearningRate` and `Gradient` as interchangeable because both affect the update.
- You report that loss changed without saying whether it went up or down.
- You cannot identify which map is fixed during one forward pass and which values are updated after feedback.

## Debugging Hints

- Separate data, parameters, feedback, and step-size control before writing the update.
- When reading an epoch trace, compare trend and individual examples separately.
- If the loss moved the wrong way, inspect the sign of the gradient and the learning rate.
- Use the category lens: training changes the parameterized map, not the type of the input or target.
