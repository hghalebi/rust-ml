use std::fmt::Display;

use rust_ml_attention::{
    AttentionHead, AttentionTraceVisibility, PublicAttentionTrace, ReviewedAttentionTrace,
    TokenComponent, TokenEmbedding, TokenIndex, TokenSequence, VectorWidth,
};

fn public_sequence() -> Result<TokenSequence, rust_ml_attention::Error> {
    TokenSequence::from_tokens([
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
    ])
}

fn reviewed_trace(
    visibility: AttentionTraceVisibility,
) -> Result<ReviewedAttentionTrace, rust_ml_attention::Error> {
    let head = AttentionHead::identity(VectorWidth::try_from(2)?)?;
    let trace = head.trace_token(&public_sequence()?, TokenIndex::try_from(0)?)?;
    Ok(ReviewedAttentionTrace::new(trace, visibility))
}

fn main() -> Result<(), rust_ml_attention::Error> {
    let public_trace = PublicAttentionTrace::from_reviewed_trace(reviewed_trace(
        AttentionTraceVisibility::Public,
    )?)?;

    println!("public query token = {}", public_trace.query_index());
    println!(
        "public weights     = [{}]",
        format_values(public_trace.weights().values())
    );
    println!(
        "public output      = [{}]",
        format_values(public_trace.output().values())
    );

    let private_trace = PublicAttentionTrace::from_reviewed_trace(reviewed_trace(
        AttentionTraceVisibility::Private,
    )?);

    if let Err(error) = private_trace {
        println!("blocked from public attention trace: {error}");
    }

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
