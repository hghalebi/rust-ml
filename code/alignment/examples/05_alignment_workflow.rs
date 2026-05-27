use rust_ml_alignment::{
    AlignmentRunId, AlignmentWorkflow, AuditNote, AuditRecord, Instruction, ReasoningTrace,
    Response, SignalSource, UpdateSignal, VerifierFeedback, VerifierResult,
};

fn main() -> Result<(), rust_ml_alignment::Error> {
    let run_id = AlignmentRunId::try_from("align-run-001")?;
    let signal = UpdateSignal::Verifier(VerifierFeedback::new(
        Instruction::try_from("solve 2 + 2 with a visible check")?,
        Response::try_from("2 + 2 = 5")?,
        ReasoningTrace::try_from("the answer adds one extra unit")?,
        VerifierResult::Failed,
        SignalSource::try_from("public-verifier-fixture")?,
    ));
    let record = AuditRecord::new(
        run_id.clone(),
        signal,
        AuditNote::try_from("audit kept the verifier failure visible")?,
    );

    let workflow = AlignmentWorkflow::new(run_id)
        .record_signal(record)?
        .approve_audit()?
        .apply_update()?;

    println!("workflow = {}", workflow);
    if let Some(transition) = workflow.latest_transition() {
        println!("last transition = {}", transition);
    }

    Ok(())
}
