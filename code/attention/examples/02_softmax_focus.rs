use std::fmt::Display;

use rust_ml_attention::{AttentionScore, AttentionScores, softmax};

fn main() -> Result<(), rust_ml_attention::Error> {
    let scores = AttentionScores::from_scores([
        AttentionScore::try_from(2.0)?,
        AttentionScore::try_from(1.0)?,
        AttentionScore::try_from(0.0)?,
    ])?;
    let weights = softmax(&scores)?;

    println!("scores  = [{}]", format_values(scores.values()));
    println!("weights = [{}]", format_values(weights.values()));

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
