# MLP Solutions

## Solution 1: Explain the hidden layer

`InputVector -> HiddenPreActivation` means the hidden layer computes weighted sums from the input.

`HiddenPreActivation -> HiddenActivation` means ReLU keeps positive detector values and turns negative detector values into zero.

The hidden activation is a representation because it is a new view of the input, not the raw input itself.

## Solution 2: Trace the XOR detectors

For the deterministic seed in the crate:

```text
(0, 0) -> hidden [0, 0]
(1, 0) -> hidden [1, 0]
(0, 1) -> hidden [0, 1]
(1, 1) -> hidden [0, 0]
```

The first true case fires the first detector. The second true case fires the second detector. The output layer can then combine the detector results.

## Solution 3: Follow shape flow

The shape chain is:

```text
2 -> 2 -> 1
```

The hidden layer outputs width `2`, so the output layer must accept input width `2`.

That is the composition rule:

```text
left output width == right input width
```

## Solution 4: Predict before running

The high predictions should be the true XOR cases:

```text
(1, 0)
(0, 1)
```

The low predictions should be:

```text
(0, 0)
(1, 1)
```

The exact printed values are not `0` and `1` because sigmoid returns a smooth probability-like value.

## Solution 5: Complete the role map

One valid completion is:

```rust
fn main() {
    let chain = [
        "InputVector",
        "HiddenPreActivation",
        "HiddenActivation",
        "OutputLogit",
        "Prediction",
    ];

    println!("{}", chain.join(" -> "));
}
```

The linear maps are:

```text
InputVector -> HiddenPreActivation
HiddenActivation -> OutputLogit
```

The activation maps are:

```text
HiddenPreActivation -> HiddenActivation
OutputLogit -> Prediction
```

## Self-Check

- You can explain hidden activations as learned features, not just extra numbers.
- You can trace why the two true XOR inputs activate different detectors.
- You can name which widths must match for layer composition.
- You can classify each arrow as linear, activation, or final prediction map.
