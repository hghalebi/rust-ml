use rust_ml_systems::{ActivationShape, BatchSize, ElementSize, ModelWidth, SequenceLength};

fn main() -> Result<(), rust_ml_systems::Error> {
    let shape = ActivationShape::new(
        BatchSize::try_from(2)?,
        SequenceLength::try_from(8)?,
        ModelWidth::try_from(16)?,
    );

    println!("activation elements = {}", shape.elements()?);
    println!(
        "activation memory   = {}",
        shape.activation_bytes(ElementSize::float32())?
    );

    Ok(())
}
