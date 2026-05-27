use rust_ml_systems::{
    ColumnCount, ElapsedNanos, ElementSize, MatrixVectorShape, RowCount, StageMeasurement,
    StageName,
};

fn main() -> Result<(), rust_ml_systems::Error> {
    let shape = MatrixVectorShape::new(RowCount::try_from(16)?, ColumnCount::try_from(32)?);
    let measurement = StageMeasurement::new(
        StageName::try_from("matvec")?,
        ElapsedNanos::try_from(90_000_u128)?,
        shape.multiply_add_flops()?,
        shape.bytes_moved(ElementSize::float32())?,
    );

    println!("stage      = {}", measurement.name());
    println!("elapsed    = {}", measurement.elapsed());
    println!("FLOPs      = {}", measurement.flops());
    println!("bytes      = {}", measurement.bytes_moved());
    println!("intensity  = {}", measurement.arithmetic_intensity()?);

    Ok(())
}
