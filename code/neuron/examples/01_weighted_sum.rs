use rust_ml_neuron::{FeatureVector, InputValue, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let features = FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.5)?);
    let weights = WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?);

    let sum = (&features * &weights)?;

    println!("weighted sum = {sum:.3}");
    println!("meaning: 1.0 * 0.8 + 0.5 * -0.4");

    Ok(())
}
