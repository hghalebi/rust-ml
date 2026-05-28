use std::fmt::Display;

use rust_ml_mlp::{
    InputValue, InputVector, MlpTraceVisibility, PublicForwardTrace, ReviewedForwardTrace, TinyMlp,
};

fn public_input() -> Result<InputVector, rust_ml_mlp::Error> {
    InputVector::from_values([InputValue::try_from(1.0)?, InputValue::try_from(0.0)?])
}

fn reviewed_trace(
    visibility: MlpTraceVisibility,
) -> Result<ReviewedForwardTrace, rust_ml_mlp::Error> {
    let trace = TinyMlp::xor_seed()?.forward(&public_input()?)?;
    Ok(ReviewedForwardTrace::new(trace, visibility))
}

fn main() -> Result<(), rust_ml_mlp::Error> {
    let public_trace =
        PublicForwardTrace::from_reviewed_trace(reviewed_trace(MlpTraceVisibility::Public)?)?;

    println!(
        "public hidden activation = [{}]",
        format_values(public_trace.hidden_activation().values())
    );
    println!(
        "public output logit      = {:.4}",
        public_trace.output_logit()
    );
    println!(
        "public prediction        = {:.4}",
        public_trace.prediction()
    );

    let private_trace =
        PublicForwardTrace::from_reviewed_trace(reviewed_trace(MlpTraceVisibility::Private)?);

    if let Err(error) = private_trace {
        println!("blocked from public MLP trace: {error}");
    }

    Ok(())
}

fn format_values<'a, T>(values: impl IntoIterator<Item = &'a T>) -> String
where
    T: Display + 'a,
{
    values
        .into_iter()
        .map(|value| format!("{value:.1}"))
        .collect::<Vec<_>>()
        .join(", ")
}
