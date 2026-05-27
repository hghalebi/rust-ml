use rust_ml_neuron::{
    FeatureVector, InputValue, LearningRate, PublicTrainingStep, ReviewedTrainingStep, Target,
    TinyNeuron, TrainingExample, TrainingStepVisibility,
};

fn public_example() -> Result<TrainingExample, rust_ml_neuron::Error> {
    Ok(TrainingExample::new(
        FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
        Target::try_from(1.0)?,
    ))
}

fn reviewed_step(
    visibility: TrainingStepVisibility,
) -> Result<ReviewedTrainingStep, rust_ml_neuron::Error> {
    let mut neuron = TinyNeuron::lesson_seed()?;
    let step = neuron.train_one_step(&public_example()?, LearningRate::try_from(0.5)?)?;
    Ok(ReviewedTrainingStep::new(step, visibility))
}

fn main() -> Result<(), rust_ml_neuron::Error> {
    let public_step =
        PublicTrainingStep::from_reviewed_step(reviewed_step(TrainingStepVisibility::Public)?)?;
    let weight_gradients = public_step
        .gradients()
        .weights()
        .map(|gradient| format!("{gradient:.4}"))
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "public prediction before = {:.4}",
        public_step.prediction_before()
    );
    println!(
        "public loss before       = {:.4}",
        public_step.loss_before()
    );
    println!("public weight gradients  = [{weight_gradients}]");
    println!(
        "public bias gradient     = {:.4}",
        public_step.gradients().bias()
    );
    println!(
        "public prediction after  = {:.4}",
        public_step.prediction_after()
    );
    println!("public loss after        = {:.4}", public_step.loss_after());

    let private_step =
        PublicTrainingStep::from_reviewed_step(reviewed_step(TrainingStepVisibility::Private)?);

    if let Err(error) = private_step {
        println!("blocked from public neuron step: {error}");
    }

    Ok(())
}
