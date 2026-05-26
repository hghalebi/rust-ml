use rust_ml_neuron::{FeatureVector, InputValue, Weight, WeightVector, weighted_sum};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let features = FeatureVector::two(InputValue::new(1.0)?, InputValue::new(0.5)?);
    let weights = WeightVector::two(Weight::new(0.8)?, Weight::new(-0.4)?);

    let sum = weighted_sum(&features, &weights)?;

    println!("weighted sum = {:.3}", sum.value());
    println!("meaning: 1.0 * 0.8 + 0.5 * -0.4");

    Ok(())
}
