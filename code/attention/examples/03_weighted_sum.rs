use std::fmt::Display;

use rust_ml_attention::{AttentionWeight, AttentionWeights, Value, ValueComponent, ValueSequence};

fn main() -> Result<(), rust_ml_attention::Error> {
    let weights = AttentionWeights::from_weights([
        AttentionWeight::try_from(0.75)?,
        AttentionWeight::try_from(0.25)?,
    ])?;
    let values = ValueSequence::from_values([
        Value::from_values([
            ValueComponent::try_from(2.0)?,
            ValueComponent::try_from(0.0)?,
        ])?,
        Value::from_values([
            ValueComponent::try_from(0.0)?,
            ValueComponent::try_from(4.0)?,
        ])?,
    ])?;

    let output = (&weights * &values)?;

    println!("mixed value = [{}]", format_values(output.values()));
    Ok(())
}

fn format_values<'a, T>(values: impl IntoIterator<Item = &'a T>) -> String
where
    T: Display + 'a,
{
    values
        .into_iter()
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}
