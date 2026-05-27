use rust_ml_evaluation::{
    EvalExample, EvalReport, EvalRunId, ExampleId, ExpectedAnswer, ModelAnswer, ModelPrediction,
    Prompt, ScoredPrediction, compare_accuracy,
};

fn main() -> Result<(), rust_ml_evaluation::Error> {
    let baseline = EvalReport::from_records(
        EvalRunId::try_from("baseline")?,
        [
            ScoredPrediction::exact_match(
                EvalExample::new(
                    ExampleId::try_from("eval-001")?,
                    Prompt::try_from("What names a meaningful ML value in Rust?")?,
                    ExpectedAnswer::try_from("newtype")?,
                ),
                ModelPrediction::new(
                    ExampleId::try_from("eval-001")?,
                    ModelAnswer::try_from("newtype")?,
                ),
            )?,
            ScoredPrediction::exact_match(
                EvalExample::new(
                    ExampleId::try_from("eval-002")?,
                    Prompt::try_from("What should an evaluation report preserve?")?,
                    ExpectedAnswer::try_from("run identity")?,
                ),
                ModelPrediction::new(
                    ExampleId::try_from("eval-002")?,
                    ModelAnswer::try_from("only the final number")?,
                ),
            )?,
        ],
    )?;
    let improved = EvalReport::from_records(
        EvalRunId::try_from("improved")?,
        [
            ScoredPrediction::exact_match(
                EvalExample::new(
                    ExampleId::try_from("eval-001")?,
                    Prompt::try_from("What names a meaningful ML value in Rust?")?,
                    ExpectedAnswer::try_from("newtype")?,
                ),
                ModelPrediction::new(
                    ExampleId::try_from("eval-001")?,
                    ModelAnswer::try_from("newtype")?,
                ),
            )?,
            ScoredPrediction::exact_match(
                EvalExample::new(
                    ExampleId::try_from("eval-002")?,
                    Prompt::try_from("What should an evaluation report preserve?")?,
                    ExpectedAnswer::try_from("run identity")?,
                ),
                ModelPrediction::new(
                    ExampleId::try_from("eval-002")?,
                    ModelAnswer::try_from("Run   Identity")?,
                ),
            )?,
        ],
    )?;

    let delta = compare_accuracy(&improved, &baseline)?;

    println!("baseline = {}", baseline.accuracy());
    println!("improved = {}", improved.accuracy());
    println!("delta = {delta}");
    Ok(())
}
