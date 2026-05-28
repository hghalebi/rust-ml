use rust_ml_alignment::{
    AlignmentRunId, AuditNote, AuditRecord, Instruction, ReasoningTrace, Response, SignalSource,
    UpdateSignal, VerifierFeedback, VerifierResult,
};

fn main() -> Result<(), rust_ml_alignment::Error> {
    let signal = UpdateSignal::Verifier(VerifierFeedback::new(
        Instruction::try_from("solve 2 + 2 with a visible check")?,
        Response::try_from("2 + 2 = 5")?,
        ReasoningTrace::try_from("the answer adds one extra unit")?,
        VerifierResult::Failed,
        SignalSource::try_from("public-verifier-fixture")?,
    ));
    let record = AuditRecord::new(
        AlignmentRunId::try_from("align-run-001")?,
        signal,
        AuditNote::try_from("kept failed verifier result visible")?,
    );

    println!("{}", record);

    Ok(())
}
