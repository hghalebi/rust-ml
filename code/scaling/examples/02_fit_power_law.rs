use rust_ml_scaling::{
    ExperimentConfig, LayerCount, MetricRecord, MetricRecords, ModelWidth, RunId, TokenCount,
    TrainingStep, ValidationLoss,
};

fn record(
    run_id: RunId,
    width: ModelWidth,
    loss: ValidationLoss,
) -> Result<MetricRecord, rust_ml_scaling::Error> {
    let config = ExperimentConfig::new(
        run_id,
        width,
        LayerCount::try_from(1)?,
        TokenCount::try_from(1_u64)?,
    );
    let run = config.plan_run(TrainingStep::try_from(1_u64)?)?;

    Ok(MetricRecord::from_run(run, loss))
}

fn main() -> Result<(), rust_ml_scaling::Error> {
    let records = MetricRecords::from_records([
        record(
            RunId::try_from("width-1")?,
            ModelWidth::try_from(1)?,
            ValidationLoss::try_from(10.0 / 72.0_f64.sqrt())?,
        )?,
        record(
            RunId::try_from("width-2")?,
            ModelWidth::try_from(2)?,
            ValidationLoss::try_from(10.0 / 288.0_f64.sqrt())?,
        )?,
        record(
            RunId::try_from("width-4")?,
            ModelWidth::try_from(4)?,
            ValidationLoss::try_from(10.0 / 1152.0_f64.sqrt())?,
        )?,
    ])?;
    let fit = records.fit_power_law()?;

    println!("fit         = {}", fit);
    println!("coefficient = {}", fit.coefficient());
    println!("exponent    = {}", fit.exponent());

    Ok(())
}
