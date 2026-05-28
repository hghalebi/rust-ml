use rust_ml_neuron::{
    FeatureVector, InputValue, LearningRate, Target, TinyNeuron, TrainingExample,
};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let mut neuron = TinyNeuron::lesson_seed()?;
    let example = TrainingExample::new(
        FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
        Target::try_from(1.0)?,
    );

    let step = neuron.train_one_step(&example, LearningRate::try_from(0.5)?)?;
    let weight_gradients = step
        .gradients()
        .weights()
        .map(|gradient| format!("{gradient:.4}"))
        .collect::<Vec<_>>()
        .join(", ");

    println!("prediction before = {:.4}", step.prediction_before());
    println!("loss before       = {:.4}", step.loss_before());
    println!("weight gradients  = [{weight_gradients}]");
    println!("bias gradient     = {:.4}", step.gradients().bias());
    println!("prediction after  = {:.4}", step.prediction_after());
    println!("loss after        = {:.4}", step.loss_after());

    Ok(())
}
