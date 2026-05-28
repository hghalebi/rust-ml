use rust_ml_neuron::{
    Bias, Dataset, FeatureVector, InputValue, LearningRate, Target, TinyNeuron, TrainingExample,
    Weight, WeightVector,
};

fn or_dataset() -> Result<Dataset, rust_ml_neuron::Error> {
    Dataset::from_examples([
        TrainingExample::new(
            FeatureVector::two(InputValue::try_from(0.0)?, InputValue::try_from(0.0)?),
            Target::try_from(0.0)?,
        ),
        TrainingExample::new(
            FeatureVector::two(InputValue::try_from(0.0)?, InputValue::try_from(1.0)?),
            Target::try_from(1.0)?,
        ),
        TrainingExample::new(
            FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
            Target::try_from(1.0)?,
        ),
        TrainingExample::new(
            FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(1.0)?),
            Target::try_from(1.0)?,
        ),
    ])
}

fn main() -> Result<(), rust_ml_neuron::Error> {
    let dataset = or_dataset()?;
    let mut neuron = TinyNeuron::new(
        WeightVector::two(Weight::try_from(0.0)?, Weight::try_from(0.0)?),
        Bias::try_from(0.0)?,
    );
    let rate = LearningRate::try_from(0.8)?;

    let before = neuron.average_loss(&dataset)?;

    for _ in 0..500 {
        neuron.train_epoch(&dataset, rate)?;
    }

    let after = neuron.average_loss(&dataset)?;

    println!("average loss: before={before:.4}, after={after:.4}");

    for example in dataset.examples() {
        let prediction = neuron.predict(example.features())?;
        let mut values = example.features().values();
        let x1 = values
            .next()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "0".to_string());
        let x2 = values
            .next()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "0".to_string());

        println!(
            "x1={x1}, x2={x2}, target={:.4}, prediction={prediction:.4}",
            example.target()
        );
    }

    Ok(())
}
