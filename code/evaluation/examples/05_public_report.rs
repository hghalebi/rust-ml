use rust_ml_evaluation::{
    EvalExample, EvalRunId, ExampleId, ExampleVisibility, ExpectedAnswer, ModelAnswer,
    ModelPrediction, Prompt, PublicEvalReport, ReviewedScoredPrediction, ScoredPrediction,
};

fn public_record() -> Result<ReviewedScoredPrediction, rust_ml_evaluation::Error> {
    let scored = ScoredPrediction::exact_match(
        EvalExample::new(
            ExampleId::try_from("public-eval-001")?,
            Prompt::try_from("What should a Rust ML evaluation preserve?")?,
            ExpectedAnswer::try_from("run identity")?,
        ),
        ModelPrediction::new(
            ExampleId::try_from("public-eval-001")?,
            ModelAnswer::try_from("Run   Identity")?,
        ),
    )?;
    Ok(ReviewedScoredPrediction::new(
        scored,
        ExampleVisibility::Public,
    ))
}

fn private_record() -> Result<ReviewedScoredPrediction, rust_ml_evaluation::Error> {
    let scored = ScoredPrediction::exact_match(
        EvalExample::new(
            ExampleId::try_from("private-eval-001")?,
            Prompt::try_from("Private evaluation prompt placeholder")?,
            ExpectedAnswer::try_from("not public")?,
        ),
        ModelPrediction::new(
            ExampleId::try_from("private-eval-001")?,
            ModelAnswer::try_from("not public")?,
        ),
    )?;
    Ok(ReviewedScoredPrediction::new(
        scored,
        ExampleVisibility::Private,
    ))
}

fn main() -> Result<(), rust_ml_evaluation::Error> {
    let public_report = PublicEvalReport::from_reviewed_records(
        EvalRunId::try_from("public-eval-run")?,
        [public_record()?],
    )?;
    println!("public accuracy = {}", public_report.accuracy());

    let blocked = PublicEvalReport::from_reviewed_records(
        EvalRunId::try_from("blocked-eval-run")?,
        [private_record()?],
    );

    match blocked {
        Ok(report) => println!("private record published with {}", report.accuracy()),
        Err(error) => println!("blocked from public report: {}", error),
    }

    Ok(())
}
