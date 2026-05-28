use rust_ml_mlp::{InputValue, InputVector, TinyMlp};

fn main() -> Result<(), rust_ml_mlp::Error> {
    let mlp = TinyMlp::xor_seed()?;
    let input = InputVector::from_values([InputValue::try_from(1.0)?, InputValue::try_from(0.0)?])?;
    let trace = mlp.forward(&input)?;
    let hidden_pre_activation = trace
        .hidden_pre_activation()
        .values()
        .map(|value| format!("{value:.1}"))
        .collect::<Vec<_>>()
        .join(", ");
    let hidden_activation = trace
        .hidden_activation()
        .values()
        .map(|value| format!("{value:.1}"))
        .collect::<Vec<_>>()
        .join(", ");

    println!("hidden pre-activation = [{hidden_pre_activation}]");
    println!("hidden activation     = [{hidden_activation}]");
    println!("output logit          = {:.4}", trace.output_logit());
    println!("prediction            = {:.4}", trace.prediction());

    Ok(())
}
