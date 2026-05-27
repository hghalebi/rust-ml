use std::fmt::Display;

use rust_ml_attention::{
    AttentionHead, TokenComponent, TokenEmbedding, TokenIndex, TokenSequence, VectorWidth,
};

fn main() -> Result<(), rust_ml_attention::Error> {
    let sequence = TokenSequence::from_tokens([
        TokenEmbedding::from_values([
            TokenComponent::try_from(1.0)?,
            TokenComponent::try_from(0.0)?,
        ])?,
        TokenEmbedding::from_values([
            TokenComponent::try_from(0.0)?,
            TokenComponent::try_from(1.0)?,
        ])?,
        TokenEmbedding::from_values([
            TokenComponent::try_from(1.0)?,
            TokenComponent::try_from(1.0)?,
        ])?,
    ])?;
    let head = AttentionHead::identity(VectorWidth::try_from(2)?)?;
    let trace = head.trace_token(&sequence, TokenIndex::try_from(0)?)?;

    println!("query token = {}", trace.query_index());
    println!("scores      = [{}]", format_values(trace.scores().values()));
    println!(
        "weights     = [{}]",
        format_values(trace.weights().values())
    );
    println!("output      = [{}]", format_values(trace.output().values()));

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
