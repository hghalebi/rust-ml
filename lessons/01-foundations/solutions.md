# Foundations Solutions

## Solution 1: One sentence, three forms

Algebra:

```math
z = w_1 x_1 + w_2 x_2 + b
```

Rust:

```text
let z = ((&inputs * &weights)? + bias)?;
```

## Solution 2: Fix the indexing mismatch

```rust
use rust_ml_neuron::{FeatureVector, InputValue};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let x = FeatureVector::from_values([
        InputValue::try_from(10.0)?,
        InputValue::try_from(20.0)?,
        InputValue::try_from(30.0)?,
    ])?;

    println!("feature count = {}", x.len());
    Ok(())
}
```

Math starts at 1. Rust starts at 0.

## Solution 3: Read the notation

1. $x^{(2)}$ usually means "training example number two" when the parentheses are part of the notation.
2. $\sum_{i=1}^{n} a_i$ means "add all the $a_i$ terms from the first one to the n-th one."
3. $\hat{y}$ means "predicted $y$."
4. $\frac{\partial L}{\partial w}$ means "how much the loss changes when $w$ changes a little."

## Solution 4: Write a dot product

```rust
use rust_ml_neuron::{FeatureVector, InputValue, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let x = FeatureVector::from_values([
        InputValue::try_from(1.0)?,
        InputValue::try_from(2.0)?,
    ])?;
    let w = WeightVector::from_values([
        Weight::try_from(3.0)?,
        Weight::try_from(4.0)?,
    ])?;

    println!("{}", (&x * &w)?);
    Ok(())
}
```

## Solution 5: Model as a struct

```rust
use rust_ml_neuron::{Bias, FeatureVector, InputValue, TinyNeuron, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let neuron = TinyNeuron::new(
        WeightVector::two(Weight::try_from(1.0)?, Weight::try_from(2.0)?),
        Bias::try_from(0.5)?,
    );
    let input = FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?);

    println!("{}", neuron.raw_score(&input)?);
    Ok(())
}
```

The important idea is not the syntax alone. The important idea is that the struct holds the parameters, and the method expresses the arithmetic.

## Self-Check

- You can point to the English phrase, algebra symbol, and Rust field for each value.
- You can explain why algebra position `1` maps to Rust index `0`.
- Your dot product returns one score, not another vector.
- Your `forward` method is a map from inputs to score, not only a storage container.
