use rust_ml_evaluation::{
    EvalExample, ExampleId, ExpectedAnswer, ModelAnswer, ModelPrediction, Prompt, ScoredPrediction,
};

fn main() -> Result<(), rust_ml_evaluation::Error> {
    let example = EvalExample::new(
        ExampleId::try_from("eval-001")?,
        Prompt::try_from("Which language is this repository teaching through ML examples?")?,
        ExpectedAnswer::try_from("Rust")?,
    );
    let prediction = ModelPrediction::new(
        ExampleId::try_from("eval-001")?,
        ModelAnswer::try_from(" rust ")?,
    );

    let scored = ScoredPrediction::exact_match(example, prediction)?;

    println!("outcome = {}", scored.outcome());
    Ok(())
}
