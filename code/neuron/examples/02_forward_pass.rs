use rust_ml_neuron::{Bias, FeatureVector, InputValue, TinyNeuron, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let neuron = TinyNeuron::new(
        WeightVector::two(Weight::new(0.8)?, Weight::new(-0.4)?),
        Bias::new(0.1)?,
    );
    let features = FeatureVector::two(InputValue::new(1.0)?, InputValue::new(0.0)?);

    let z = neuron.raw_score(&features)?;
    let prediction = neuron.predict(&features)?;

    println!("z = {:.4}", z.value());
    println!("prediction = {:.4}", prediction.value());
    println!("meaning: mix the inputs, then squash the score through sigmoid");

    Ok(())
}
