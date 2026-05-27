use rust_ml_scaling::{
    ExperimentConfig, LayerCount, LimitationNote, MetricRecord, MetricVisibility, ModelWidth,
    PublicScalingReport, ReviewedMetricRecord, RunId, TokenCount, TrainingStep, ValidationLoss,
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
    let public_report = PublicScalingReport::from_reviewed_records(
        [
            ReviewedMetricRecord::new(
                record(
                    RunId::try_from("public-width-1")?,
                    ModelWidth::try_from(1)?,
                    ValidationLoss::try_from(10.0 / 72.0_f64.sqrt())?,
                )?,
                MetricVisibility::Public,
            ),
            ReviewedMetricRecord::new(
                record(
                    RunId::try_from("public-width-2")?,
                    ModelWidth::try_from(2)?,
                    ValidationLoss::try_from(10.0 / 288.0_f64.sqrt())?,
                )?,
                MetricVisibility::Public,
            ),
        ],
        LimitationNote::try_from("two public toy runs teach fitting shape, not deployment law")?,
    )?;

    println!("public scaling fit = {}", public_report.fit());

    let blocked = PublicScalingReport::from_reviewed_records(
        [ReviewedMetricRecord::new(
            record(
                RunId::try_from("private-width-4")?,
                ModelWidth::try_from(4)?,
                ValidationLoss::try_from(10.0 / 1152.0_f64.sqrt())?,
            )?,
            MetricVisibility::Private,
        )],
        LimitationNote::try_from("blocked before fitting")?,
    );

    if let Err(error) = blocked {
        println!("blocked from public scaling report: {error}");
    }

    Ok(())
}
