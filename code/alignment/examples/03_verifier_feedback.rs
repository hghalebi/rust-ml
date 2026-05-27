use rust_ml_alignment::{
    Instruction, ReasoningTrace, Response, SignalSource, VerifierFeedback, VerifierResult,
};

fn main() -> Result<(), rust_ml_alignment::Error> {
    let feedback = VerifierFeedback::new(
        Instruction::try_from("solve 2 + 2 with a visible check")?,
        Response::try_from("2 + 2 = 5")?,
        ReasoningTrace::try_from("the answer adds one extra unit")?,
        VerifierResult::Failed,
        SignalSource::try_from("public-verifier-fixture")?,
    );

    println!("instruction = {}", feedback.instruction());
    println!("response    = {}", feedback.response());
    println!("trace       = {}", feedback.trace());
    println!("result      = {}", feedback.result());
    println!("source      = {}", feedback.source());

    Ok(())
}
