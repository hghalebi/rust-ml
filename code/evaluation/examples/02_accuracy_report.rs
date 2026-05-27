use rust_ml_evaluation::{
    EvalExample, EvalReport, EvalRunId, ExampleId, ExpectedAnswer, ModelAnswer, ModelPrediction,
    Prompt, ScoredPrediction,
};

fn main() -> Result<(), rust_ml_evaluation::Error> {
    let first = EvalExample::new(
        ExampleId::try_from("eval-001")?,
        Prompt::try_from("What protects token identity?")?,
        ExpectedAnswer::try_from("newtypes")?,
    );
    let second = EvalExample::new(
        ExampleId::try_from("eval-002")?,
        Prompt::try_from("What does an evaluation metric measure?")?,
        ExpectedAnswer::try_from("a chosen behavior")?,
    );

    let report = EvalReport::from_records(
        EvalRunId::try_from("typed-eval-demo")?,
        [
            ScoredPrediction::exact_match(
                first,
                ModelPrediction::new(
                    ExampleId::try_from("eval-001")?,
                    ModelAnswer::try_from("NewTypes")?,
                ),
            )?,
            ScoredPrediction::exact_match(
                second,
                ModelPrediction::new(
                    ExampleId::try_from("eval-002")?,
                    ModelAnswer::try_from("a vague feeling")?,
                ),
            )?,
        ],
    )?;

    println!("run = {}", report.run_id());
    println!("examples = {}", report.count());
    println!("correct = {}", report.correct());
    println!("exact_match = {}", report.accuracy());
    Ok(())
}
