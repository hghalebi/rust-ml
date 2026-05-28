use rust_ml_scaling::{
    ExperimentConfig, LayerCount, MetricRecord, ModelWidth, RunId, TokenCount, TrainingStep,
    ValidationLoss,
};

fn main() -> Result<(), rust_ml_scaling::Error> {
    let config = ExperimentConfig::new(
        RunId::try_from("width-4")?,
        ModelWidth::try_from(4)?,
        LayerCount::try_from(2)?,
        TokenCount::try_from(256_u64)?,
    );
    let run = config.plan_run(TrainingStep::try_from(64_u64)?)?;
    let record = MetricRecord::from_run(run, ValidationLoss::try_from(2.40)?);

    println!("run        = {}", record.run().run_id());
    println!("steps      = {}", record.run().training_steps());
    println!("parameters = {}", record.run().parameter_count());
    println!("compute    = {}", record.run().compute_budget());
    println!("loss       = {}", record.validation_loss());

    Ok(())
}
