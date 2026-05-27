use rust_ml_attention::{Key, KeyComponent, Query, QueryComponent};

fn main() -> Result<(), rust_ml_attention::Error> {
    let query = Query::from_values([
        QueryComponent::try_from(1.0)?,
        QueryComponent::try_from(1.0)?,
    ])?;
    let key = Key::from_values([KeyComponent::try_from(1.0)?, KeyComponent::try_from(0.0)?])?;

    let score = (&query * &key)?;

    println!("scaled score = {score:.4}");
    Ok(())
}
