use rust_ml_systems::{Bytes, ElapsedNanos, Flops, StageMeasurement, StageMeasurements, StageName};

fn stage(elapsed: ElapsedNanos) -> Result<StageMeasurement, rust_ml_systems::Error> {
    Ok(StageMeasurement::new(
        StageName::try_from("attention forward")?,
        elapsed,
        Flops::try_from(2_048_u64)?,
        Bytes::try_from(1_024_u64)?,
    ))
}

fn main() -> Result<(), rust_ml_systems::Error> {
    let repeated = StageMeasurements::from_measurements([
        stage(ElapsedNanos::try_from(120_000_u128)?)?,
        stage(ElapsedNanos::try_from(110_000_u128)?)?,
        stage(ElapsedNanos::try_from(130_000_u128)?)?,
    ])?;

    println!("median elapsed = {}", repeated.median_elapsed()?);

    Ok(())
}
