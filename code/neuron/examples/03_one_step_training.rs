use rust_ml_neuron::{
    FeatureVector, InputValue, LearningRate, Target, TinyNeuron, TrainingExample,
};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let mut neuron = TinyNeuron::lesson_seed()?;
    let example = TrainingExample::new(
        FeatureVector::two(InputValue::new(1.0)?, InputValue::new(0.0)?),
        Target::new(1.0)?,
    );

    let step = neuron.train_one_step(&example, LearningRate::new(0.5)?)?;

    println!(
        "prediction before = {:.4}",
        step.prediction_before().value()
    );
    println!("loss before       = {:.4}", step.loss_before().value());
    println!(
        "weight gradients  = [{:.4}, {:.4}]",
        step.gradients().weights()[0].value(),
        step.gradients().weights()[1].value()
    );
    println!("bias gradient     = {:.4}", step.gradients().bias().value());
    println!("prediction after  = {:.4}", step.prediction_after().value());
    println!("loss after        = {:.4}", step.loss_after().value());

    Ok(())
}
