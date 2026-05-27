use rust_ml_systems::{
    Bytes, ElapsedNanos, Flops, MeasurementVisibility, PublicSystemsReport,
    ReviewedStageMeasurement, StageMeasurement, StageName,
};

fn public_stage(
    name: StageName,
    elapsed: ElapsedNanos,
) -> Result<ReviewedStageMeasurement, rust_ml_systems::Error> {
    Ok(ReviewedStageMeasurement::new(
        StageMeasurement::new(
            name,
            elapsed,
            Flops::try_from(2_048_u64)?,
            Bytes::try_from(1_024_u64)?,
        ),
        MeasurementVisibility::Public,
    ))
}

fn main() -> Result<(), rust_ml_systems::Error> {
    let report = PublicSystemsReport::from_reviewed_measurements([
        public_stage(
            StageName::try_from("public-attention-a")?,
            ElapsedNanos::try_from(120_000_u128)?,
        )?,
        public_stage(
            StageName::try_from("public-attention-b")?,
            ElapsedNanos::try_from(110_000_u128)?,
        )?,
        public_stage(
            StageName::try_from("public-attention-c")?,
            ElapsedNanos::try_from(130_000_u128)?,
        )?,
    ])?;

    println!("public median elapsed = {}", report.median_elapsed()?);

    let blocked = PublicSystemsReport::from_reviewed_measurements([ReviewedStageMeasurement::new(
        StageMeasurement::new(
            StageName::try_from("private-host-profile")?,
            ElapsedNanos::try_from(90_000_u128)?,
            Flops::try_from(2_048_u64)?,
            Bytes::try_from(1_024_u64)?,
        ),
        MeasurementVisibility::Private,
    )]);

    if let Err(error) = blocked {
        println!("blocked from public systems report: {error}");
    }

    Ok(())
}
