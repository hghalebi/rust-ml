use rust_ml_alignment::{
    AlignmentRunId, AlignmentVisibility, AlignmentWorkflow, AuditNote, AuditRecord, Instruction,
    PublicAlignmentRelease, Response, ReviewedAlignmentWorkflow, SignalSource, UpdateSignal,
};

fn main() -> Result<(), rust_ml_alignment::Error> {
    let run_id = AlignmentRunId::try_from("align-run-001")?;
    let signal = UpdateSignal::Supervised(rust_ml_alignment::InstructionExample::new(
        Instruction::try_from("solve 2 + 2 with a visible check")?,
        Response::try_from("2 + 2 = 4")?,
        SignalSource::try_from("public-alignment-fixture")?,
    ));
    let record = AuditRecord::new(
        run_id.clone(),
        signal,
        AuditNote::try_from("audit approved a public toy signal")?,
    );
    let applied_workflow = AlignmentWorkflow::new(run_id)
        .record_signal(record)?
        .approve_audit()?
        .apply_update()?;

    let public_release = PublicAlignmentRelease::from_reviewed_workflow(
        ReviewedAlignmentWorkflow::new(applied_workflow.clone(), AlignmentVisibility::Public),
    )?;
    println!(
        "public alignment release = {} | {}",
        public_release.run_id(),
        public_release.stage()
    );

    let blocked = PublicAlignmentRelease::from_reviewed_workflow(ReviewedAlignmentWorkflow::new(
        applied_workflow,
        AlignmentVisibility::ResearchRestricted,
    ));

    if let Err(error) = blocked {
        println!("blocked from public alignment release: {error}");
    }

    Ok(())
}
