use rust_ml_evaluation::{
    EvalExample, ExampleId, ExpectedAnswer, ModelAnswer, ModelPrediction, Prompt, ScoredPrediction,
};

fn main() -> Result<(), rust_ml_evaluation::Error> {
    let example = EvalExample::new(
        ExampleId::try_from("eval-001")?,
        Prompt::try_from("What should a prediction stay attached to?")?,
        ExpectedAnswer::try_from("the same example id")?,
    );
    let prediction = ModelPrediction::new(
        ExampleId::try_from("eval-999")?,
        ModelAnswer::try_from("the same example id")?,
    );

    match ScoredPrediction::exact_match(example, prediction) {
        Ok(scored) => println!("unexpected outcome = {}", scored.outcome()),
        Err(error) => println!("{error}"),
    }

    Ok(())
}
