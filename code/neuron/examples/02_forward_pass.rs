use rust_ml_neuron::{Bias, FeatureVector, InputValue, TinyNeuron, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let neuron = TinyNeuron::new(
        WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?),
        Bias::try_from(0.1)?,
    );
    let features = FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?);

    let z = neuron.raw_score(&features)?;
    let prediction = neuron.predict(&features)?;

    println!("z = {z:.4}");
    println!("prediction = {prediction:.4}");
    println!("meaning: mix the inputs, then squash the score through sigmoid");

    Ok(())
}
