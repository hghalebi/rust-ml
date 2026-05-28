//! Typed post-training signals for the CS336 Rust equivalent track.
//!
//! This crate treats alignment as auditable data flow:
//!
//! ```text
//! PromptedResponse -> PreferenceSignal -> UpdateSignal -> AuditRecord
//! AuditRecord -> AlignmentWorkflow -> AlignmentTransition
//! ReviewedAlignmentWorkflow -> PublicAlignmentRelease
//! ```
//!
//! Raw learner strings and scores enter through `TryFrom` adapters. Once inside
//! the crate, public APIs use semantic values such as [`Instruction`],
//! [`ChosenResponse`], [`RejectedResponse`], [`RewardScore`],
//! [`VerifierFeedback`], [`AlignmentRunId`], [`AlignmentWorkflow`], and
//! [`PublicAlignmentRelease`].

pub mod error;

use std::{
    fmt,
    ops::{Add, Sub},
};

use error::AlignmentError;

pub use error::AlignmentError as Error;

fn validate_nonempty(
    value: &str,
    operation: &'static str,
    details: &'static str,
) -> Result<(), AlignmentError> {
    if value.trim().is_empty() {
        return Err(AlignmentError::empty_input(operation, details));
    }

    Ok(())
}

fn finite(role: &'static str, value: f64) -> Result<f64, AlignmentError> {
    if !value.is_finite() {
        return Err(AlignmentError::non_finite_value(role, value));
    }

    Ok(value)
}

macro_rules! nonempty_text_type {
    ($name:ident, $doc:literal, $operation:literal, $details:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            fn from_owned(value: String) -> Result<Self, AlignmentError> {
                validate_nonempty(&value, $operation, $details)?;
                Ok(Self(value))
            }

            fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<&str> for $name {
            type Error = AlignmentError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::from_owned(value.to_owned())
            }
        }

        impl TryFrom<String> for $name {
            type Error = AlignmentError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::from_owned(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

nonempty_text_type!(
    AlignmentRunId,
    "Stable label for one post-training run.",
    "AlignmentRunId::try_from",
    "alignment run id cannot be empty"
);
nonempty_text_type!(
    SignalSource,
    "Human-readable source of a post-training signal.",
    "SignalSource::try_from",
    "signal source cannot be empty"
);
nonempty_text_type!(
    Instruction,
    "Instruction text for supervised finetuning or preference collection.",
    "Instruction::try_from",
    "instruction cannot be empty"
);
nonempty_text_type!(
    Response,
    "Model response text before it receives a chosen/rejected role.",
    "Response::try_from",
    "response cannot be empty"
);
nonempty_text_type!(
    ReasoningTrace,
    "Short visible reasoning trace for a toy verifier task.",
    "ReasoningTrace::try_from",
    "reasoning trace cannot be empty"
);
nonempty_text_type!(
    AuditNote,
    "Human-readable note attached to an audit record.",
    "AuditNote::try_from",
    "audit note cannot be empty"
);

/// Response selected as better in a preference pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChosenResponse(Response);

impl ChosenResponse {
    /// Labels a response as chosen.
    pub fn from_response(response: Response) -> Self {
        Self(response)
    }

    fn response(&self) -> &Response {
        &self.0
    }
}

impl fmt::Display for ChosenResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Response selected as worse in a preference pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedResponse(Response);

impl RejectedResponse {
    /// Labels a response as rejected.
    pub fn from_response(response: Response) -> Self {
        Self(response)
    }

    fn response(&self) -> &Response {
        &self.0
    }
}

impl fmt::Display for RejectedResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// One instruction-response example for supervised finetuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionExample {
    instruction: Instruction,
    response: Response,
    source: SignalSource,
}

impl InstructionExample {
    /// Creates a supervised instruction-response example.
    pub fn new(instruction: Instruction, response: Response, source: SignalSource) -> Self {
        Self {
            instruction,
            response,
            source,
        }
    }

    /// Returns the instruction.
    pub fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Returns the response.
    pub fn response(&self) -> &Response {
        &self.response
    }

    /// Returns the source.
    pub fn source(&self) -> &SignalSource {
        &self.source
    }
}

/// Pairwise preference for one instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreferencePair {
    instruction: Instruction,
    chosen: ChosenResponse,
    rejected: RejectedResponse,
    source: SignalSource,
}

impl PreferencePair {
    /// Creates a preference pair and rejects identical chosen/rejected responses.
    pub fn new(
        instruction: Instruction,
        chosen: ChosenResponse,
        rejected: RejectedResponse,
        source: SignalSource,
    ) -> Result<Self, AlignmentError> {
        if chosen.response() == rejected.response() {
            return Err(AlignmentError::invalid_pair(
                "PreferencePair::new",
                "chosen and rejected responses must be different",
            ));
        }

        Ok(Self {
            instruction,
            chosen,
            rejected,
            source,
        })
    }

    /// Returns the instruction.
    pub fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Returns the chosen response.
    pub fn chosen(&self) -> &ChosenResponse {
        &self.chosen
    }

    /// Returns the rejected response.
    pub fn rejected(&self) -> &RejectedResponse {
        &self.rejected
    }

    /// Returns the source.
    pub fn source(&self) -> &SignalSource {
        &self.source
    }
}

/// Reward score in a toy preference model.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RewardScore(f64);

impl RewardScore {
    fn from_raw(value: f64) -> Result<Self, AlignmentError> {
        let value = finite("reward score", value)?;
        if !(-1.0..=1.0).contains(&value) {
            return Err(AlignmentError::out_of_range(
                "reward score",
                "[-1, 1]",
                value,
            ));
        }
        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for RewardScore {
    type Error = AlignmentError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for RewardScore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.4}", self.0)
    }
}

/// Difference between chosen and rejected reward scores.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RewardMargin(f64);

impl RewardMargin {
    fn from_raw(value: f64) -> Result<Self, AlignmentError> {
        Ok(Self(finite("reward margin", value)?))
    }
}

impl Sub for RewardScore {
    type Output = Result<RewardMargin, AlignmentError>;

    fn sub(self, right: RewardScore) -> Self::Output {
        RewardMargin::from_raw(self.as_f64() - right.as_f64())
    }
}

impl fmt::Display for RewardMargin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:+.4}", self.0)
    }
}

/// Reward scores attached to the two roles in a preference pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreferenceRewards {
    chosen: RewardScore,
    rejected: RewardScore,
}

impl PreferenceRewards {
    /// Creates role-preserving reward scores for a preference pair.
    pub fn new(chosen: RewardScore, rejected: RewardScore) -> Self {
        Self { chosen, rejected }
    }

    /// Returns the score assigned to the chosen response.
    pub fn chosen(&self) -> RewardScore {
        self.chosen
    }

    /// Returns the score assigned to the rejected response.
    pub fn rejected(&self) -> RewardScore {
        self.rejected
    }

    fn margin(self) -> Result<RewardMargin, AlignmentError> {
        self.chosen - self.rejected
    }
}

/// Rewarded preference signal with typed provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceSignal {
    pair: PreferencePair,
    rewards: PreferenceRewards,
    margin: RewardMargin,
}

impl PreferenceSignal {
    /// Scores a preference pair and keeps the chosen-minus-rejected margin.
    pub fn from_pair(
        pair: PreferencePair,
        chosen_reward: RewardScore,
        rejected_reward: RewardScore,
    ) -> Result<Self, AlignmentError> {
        Self::from_rewards(pair, PreferenceRewards::new(chosen_reward, rejected_reward))
    }

    /// Scores a preference pair from role-preserving reward scores.
    pub fn from_rewards(
        pair: PreferencePair,
        rewards: PreferenceRewards,
    ) -> Result<Self, AlignmentError> {
        let margin = rewards.margin()?;
        Ok(Self {
            pair,
            rewards,
            margin,
        })
    }

    /// Returns the underlying preference pair.
    pub fn pair(&self) -> &PreferencePair {
        &self.pair
    }

    /// Returns chosen reward.
    pub fn chosen_reward(&self) -> RewardScore {
        self.rewards.chosen()
    }

    /// Returns rejected reward.
    pub fn rejected_reward(&self) -> RewardScore {
        self.rewards.rejected()
    }

    /// Returns chosen-minus-rejected reward.
    pub fn margin(&self) -> RewardMargin {
        self.margin
    }
}

impl Add<PreferenceRewards> for PreferencePair {
    type Output = Result<PreferenceSignal, AlignmentError>;

    fn add(self, rewards: PreferenceRewards) -> Self::Output {
        PreferenceSignal::from_rewards(self, rewards)
    }
}

/// Verifier result for a toy reasoning trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierResult {
    /// The verifier accepted the reasoning trace.
    Passed,
    /// The verifier rejected the reasoning trace and the failure remains visible.
    Failed,
}

impl fmt::Display for VerifierResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Passed => formatter.write_str("passed"),
            Self::Failed => formatter.write_str("failed"),
        }
    }
}

/// Verifier feedback tied to the instruction and response it evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierFeedback {
    instruction: Instruction,
    response: Response,
    trace: ReasoningTrace,
    result: VerifierResult,
    source: SignalSource,
}

impl VerifierFeedback {
    /// Creates verifier feedback for a toy reasoning task.
    pub fn new(
        instruction: Instruction,
        response: Response,
        trace: ReasoningTrace,
        result: VerifierResult,
        source: SignalSource,
    ) -> Self {
        Self {
            instruction,
            response,
            trace,
            result,
            source,
        }
    }

    /// Returns the instruction.
    pub fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Returns the response.
    pub fn response(&self) -> &Response {
        &self.response
    }

    /// Returns the visible reasoning trace.
    pub fn trace(&self) -> &ReasoningTrace {
        &self.trace
    }

    /// Returns verifier result.
    pub fn result(&self) -> VerifierResult {
        self.result
    }

    /// Returns the signal source.
    pub fn source(&self) -> &SignalSource {
        &self.source
    }
}

/// Kind of post-training update represented by an audit record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateKind {
    /// Supervised finetuning over instruction-response examples.
    SupervisedFineTuning,
    /// Preference optimization over chosen/rejected pairs.
    PreferenceOptimization,
    /// Verifier-guided reasoning update.
    VerifierGuidedReasoning,
}

impl fmt::Display for UpdateKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SupervisedFineTuning => formatter.write_str("supervised_fine_tuning"),
            Self::PreferenceOptimization => formatter.write_str("preference_optimization"),
            Self::VerifierGuidedReasoning => formatter.write_str("verifier_guided_reasoning"),
        }
    }
}

/// One typed signal that can drive a post-training update.
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateSignal {
    /// Instruction-response supervised signal.
    Supervised(InstructionExample),
    /// Pairwise preference signal.
    Preference(PreferenceSignal),
    /// Verifier feedback signal.
    Verifier(VerifierFeedback),
}

impl UpdateSignal {
    /// Returns the update kind.
    pub fn kind(&self) -> UpdateKind {
        match self {
            Self::Supervised(_) => UpdateKind::SupervisedFineTuning,
            Self::Preference(_) => UpdateKind::PreferenceOptimization,
            Self::Verifier(_) => UpdateKind::VerifierGuidedReasoning,
        }
    }

    /// Returns the signal source.
    pub fn source(&self) -> &SignalSource {
        match self {
            Self::Supervised(example) => example.source(),
            Self::Preference(signal) => signal.pair().source(),
            Self::Verifier(feedback) => feedback.source(),
        }
    }
}

/// Audit record for a post-training signal.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditRecord {
    run_id: AlignmentRunId,
    signal: UpdateSignal,
    note: AuditNote,
}

impl AuditRecord {
    /// Creates an audit record for a signal.
    pub fn new(run_id: AlignmentRunId, signal: UpdateSignal, note: AuditNote) -> Self {
        Self {
            run_id,
            signal,
            note,
        }
    }

    /// Returns the alignment run id.
    pub fn run_id(&self) -> &AlignmentRunId {
        &self.run_id
    }

    /// Returns the update signal.
    pub fn signal(&self) -> &UpdateSignal {
        &self.signal
    }

    /// Returns the audit note.
    pub fn note(&self) -> &AuditNote {
        &self.note
    }
}

impl fmt::Display for AuditRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} | {} | {} | {}",
            self.run_id,
            self.signal.kind(),
            self.signal.source(),
            self.note
        )
    }
}

/// Lifecycle stage for a toy alignment workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentStage {
    /// The workflow is still collecting typed signals.
    CollectingSignals,
    /// A signal has been recorded and is awaiting audit approval.
    AuditingSignal,
    /// The audited signal is ready to drive an update.
    ReadyForUpdate,
    /// The update has been applied.
    UpdateApplied,
}

impl fmt::Display for AlignmentStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CollectingSignals => formatter.write_str("collecting_signals"),
            Self::AuditingSignal => formatter.write_str("auditing_signal"),
            Self::ReadyForUpdate => formatter.write_str("ready_for_update"),
            Self::UpdateApplied => formatter.write_str("update_applied"),
        }
    }
}

/// Semantic event that moves the alignment workflow forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentEvent {
    /// A typed update signal was recorded.
    SignalRecorded(UpdateKind),
    /// The recorded signal passed the audit gate.
    AuditApproved,
    /// The approved update was applied.
    UpdateApplied,
}

impl fmt::Display for AlignmentEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignalRecorded(kind) => write!(formatter, "signal_recorded:{kind}"),
            Self::AuditApproved => formatter.write_str("audit_approved"),
            Self::UpdateApplied => formatter.write_str("update_applied"),
        }
    }
}

/// One audited lifecycle movement in an alignment workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentTransition {
    from: AlignmentStage,
    event: AlignmentEvent,
    to: AlignmentStage,
}

impl AlignmentTransition {
    fn new(from: AlignmentStage, event: AlignmentEvent, to: AlignmentStage) -> Self {
        Self { from, event, to }
    }

    /// Returns the stage before the event.
    pub fn from(&self) -> AlignmentStage {
        self.from
    }

    /// Returns the event that moved the workflow.
    pub fn event(&self) -> AlignmentEvent {
        self.event
    }

    /// Returns the stage after the event.
    pub fn to(&self) -> AlignmentStage {
        self.to
    }
}

impl fmt::Display for AlignmentTransition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} --{}--> {}", self.from, self.event, self.to)
    }
}

/// A tiny auditable alignment workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentWorkflow {
    run_id: AlignmentRunId,
    stage: AlignmentStage,
    latest_record: Option<AuditRecord>,
    latest_transition: Option<AlignmentTransition>,
}

impl AlignmentWorkflow {
    /// Starts a workflow for one alignment run.
    pub fn new(run_id: AlignmentRunId) -> Self {
        Self {
            run_id,
            stage: AlignmentStage::CollectingSignals,
            latest_record: None,
            latest_transition: None,
        }
    }

    /// Returns the run identity.
    pub fn run_id(&self) -> &AlignmentRunId {
        &self.run_id
    }

    /// Returns the current lifecycle stage.
    pub fn stage(&self) -> AlignmentStage {
        self.stage
    }

    /// Returns the latest recorded audit record.
    pub fn latest_record(&self) -> Option<&AuditRecord> {
        self.latest_record.as_ref()
    }

    /// Returns the latest lifecycle transition.
    pub fn latest_transition(&self) -> Option<&AlignmentTransition> {
        self.latest_transition.as_ref()
    }

    /// Records one audited signal and moves into the audit stage.
    pub fn record_signal(mut self, record: AuditRecord) -> Result<Self, AlignmentError> {
        self.require_stage(
            AlignmentStage::CollectingSignals,
            "AlignmentWorkflow::record_signal",
            "signals can only be recorded while collecting signals",
        )?;
        if record.run_id() != &self.run_id {
            return Err(AlignmentError::invalid_transition(
                "AlignmentWorkflow::record_signal",
                "audit record run id must match workflow run id",
            ));
        }

        let event = AlignmentEvent::SignalRecorded(record.signal().kind());
        self.latest_record = Some(record);
        Ok(self.transition(event, AlignmentStage::AuditingSignal))
    }

    /// Approves the recorded signal and marks the workflow ready for update.
    pub fn approve_audit(self) -> Result<Self, AlignmentError> {
        self.require_stage(
            AlignmentStage::AuditingSignal,
            "AlignmentWorkflow::approve_audit",
            "audit can only be approved after a signal is recorded",
        )?;

        Ok(self.transition(
            AlignmentEvent::AuditApproved,
            AlignmentStage::ReadyForUpdate,
        ))
    }

    /// Applies the approved update.
    pub fn apply_update(self) -> Result<Self, AlignmentError> {
        self.require_stage(
            AlignmentStage::ReadyForUpdate,
            "AlignmentWorkflow::apply_update",
            "update can only be applied after audit approval",
        )?;

        Ok(self.transition(AlignmentEvent::UpdateApplied, AlignmentStage::UpdateApplied))
    }

    fn require_stage(
        &self,
        expected: AlignmentStage,
        operation: &'static str,
        details: &'static str,
    ) -> Result<(), AlignmentError> {
        if self.stage != expected {
            return Err(AlignmentError::invalid_transition(operation, details));
        }

        Ok(())
    }

    fn transition(mut self, event: AlignmentEvent, to: AlignmentStage) -> Self {
        let from = self.stage;
        self.stage = to;
        self.latest_transition = Some(AlignmentTransition::new(from, event, to));
        self
    }
}

impl fmt::Display for AlignmentWorkflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} | {}", self.run_id, self.stage)
    }
}

/// Publication class attached to an alignment workflow before public release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentVisibility {
    /// Safe to include in learner-facing public alignment material.
    Public,
    /// Useful for restricted study, but not public learner-facing material.
    ResearchRestricted,
    /// Must stay out of public learner-facing material.
    Private,
}

impl fmt::Display for AlignmentVisibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Public => "public",
            Self::ResearchRestricted => "research-restricted",
            Self::Private => "private",
        };
        formatter.write_str(label)
    }
}

/// Typed decision at the alignment-publication boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicAlignmentDecision {
    /// The workflow can appear in a public learner-facing release.
    Publishable,
    /// The workflow must stay out of public learner-facing releases.
    Blocked,
}

/// Alignment workflow plus explicit public-release review evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewedAlignmentWorkflow {
    workflow: AlignmentWorkflow,
    visibility: AlignmentVisibility,
}

impl ReviewedAlignmentWorkflow {
    /// Creates a reviewed alignment workflow.
    pub fn new(workflow: AlignmentWorkflow, visibility: AlignmentVisibility) -> Self {
        Self {
            workflow,
            visibility,
        }
    }

    /// Returns the reviewed workflow.
    pub fn workflow(&self) -> &AlignmentWorkflow {
        &self.workflow
    }

    /// Returns the publication class.
    pub fn visibility(&self) -> AlignmentVisibility {
        self.visibility
    }

    /// Classifies whether this workflow can enter public learner-facing material.
    pub fn release_decision(&self) -> PublicAlignmentDecision {
        match self.visibility {
            AlignmentVisibility::Public => PublicAlignmentDecision::Publishable,
            AlignmentVisibility::ResearchRestricted | AlignmentVisibility::Private => {
                PublicAlignmentDecision::Blocked
            }
        }
    }

    fn into_workflow(self) -> AlignmentWorkflow {
        self.workflow
    }
}

/// Alignment workflow checked for learner-facing public release.
#[derive(Debug, Clone, PartialEq)]
pub struct PublicAlignmentRelease(AlignmentWorkflow);

impl PublicAlignmentRelease {
    /// Creates a public release only after explicit publication review and audit completion.
    pub fn from_reviewed_workflow(
        reviewed: ReviewedAlignmentWorkflow,
    ) -> Result<Self, AlignmentError> {
        if reviewed.release_decision() == PublicAlignmentDecision::Blocked {
            return Err(AlignmentError::invalid_public_release(
                "PublicAlignmentRelease::from_reviewed_workflow",
                "public alignment releases cannot include restricted or private workflows",
            ));
        }

        if reviewed.workflow().stage() != AlignmentStage::UpdateApplied {
            return Err(AlignmentError::invalid_public_release(
                "PublicAlignmentRelease::from_reviewed_workflow",
                "public alignment releases require an audited and applied workflow",
            ));
        }

        Ok(Self(reviewed.into_workflow()))
    }

    /// Returns the checked alignment workflow.
    pub fn workflow(&self) -> &AlignmentWorkflow {
        &self.0
    }

    /// Returns the alignment run identity.
    pub fn run_id(&self) -> &AlignmentRunId {
        self.0.run_id()
    }

    /// Returns the release stage, which is always update-applied after construction.
    pub fn stage(&self) -> AlignmentStage {
        self.0.stage()
    }

    /// Returns the released audit record.
    pub fn latest_record(&self) -> Option<&AuditRecord> {
        self.0.latest_record()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AlignmentError, AlignmentRunId, AlignmentStage, AlignmentVisibility, AlignmentWorkflow,
        AuditNote, AuditRecord, ChosenResponse, Instruction, InstructionExample, PreferencePair,
        PreferenceRewards, PreferenceSignal, PublicAlignmentRelease, ReasoningTrace,
        RejectedResponse, Response, ReviewedAlignmentWorkflow, RewardScore, SignalSource,
        UpdateKind, UpdateSignal, VerifierFeedback, VerifierResult,
    };

    fn instruction() -> Result<Instruction, AlignmentError> {
        Instruction::try_from("solve 2 + 2 with a visible check")
    }

    fn source() -> Result<SignalSource, AlignmentError> {
        SignalSource::try_from("public-toy-fixture")
    }

    fn applied_workflow() -> Result<AlignmentWorkflow, AlignmentError> {
        let run_id = AlignmentRunId::try_from("align-run-public")?;
        let example =
            InstructionExample::new(instruction()?, Response::try_from("2 + 2 = 4")?, source()?);
        let record = AuditRecord::new(
            run_id.clone(),
            UpdateSignal::Supervised(example),
            AuditNote::try_from("approved public toy signal")?,
        );

        AlignmentWorkflow::new(run_id)
            .record_signal(record)?
            .approve_audit()?
            .apply_update()
    }

    #[test]
    fn preference_pair_rejects_identical_responses() -> Result<(), AlignmentError> {
        let same = Response::try_from("4")?;
        let error = PreferencePair::new(
            instruction()?,
            ChosenResponse::from_response(same.clone()),
            RejectedResponse::from_response(same),
            source()?,
        )
        .err();

        assert!(matches!(error, Some(AlignmentError::InvalidPair { .. })));
        Ok(())
    }

    #[test]
    fn reward_score_ordering_produces_positive_margin() -> Result<(), AlignmentError> {
        let pair = PreferencePair::new(
            instruction()?,
            ChosenResponse::from_response(Response::try_from("2 + 2 = 4")?),
            RejectedResponse::from_response(Response::try_from("2 + 2 = 5")?),
            source()?,
        )?;
        let signal = PreferenceSignal::from_pair(
            pair,
            RewardScore::try_from(0.75)?,
            RewardScore::try_from(-0.25)?,
        )?;

        assert_eq!(signal.margin().to_string(), "+1.0000");
        Ok(())
    }

    #[test]
    fn preference_pair_add_rewards_builds_signal() -> Result<(), AlignmentError> {
        let pair = PreferencePair::new(
            instruction()?,
            ChosenResponse::from_response(Response::try_from("2 + 2 = 4")?),
            RejectedResponse::from_response(Response::try_from("2 + 2 = 5")?),
            source()?,
        )?;
        let rewards =
            PreferenceRewards::new(RewardScore::try_from(0.80)?, RewardScore::try_from(-0.20)?);
        let signal = (pair + rewards)?;

        assert_eq!(signal.chosen_reward(), rewards.chosen());
        assert_eq!(signal.rejected_reward(), rewards.rejected());
        assert_eq!(signal.margin().to_string(), "+1.0000");
        Ok(())
    }

    #[test]
    fn verifier_failures_stay_visible() -> Result<(), AlignmentError> {
        let feedback = VerifierFeedback::new(
            instruction()?,
            Response::try_from("2 + 2 = 5")?,
            ReasoningTrace::try_from("added one extra unit")?,
            VerifierResult::Failed,
            source()?,
        );

        assert_eq!(feedback.result().to_string(), "failed");
        assert_eq!(feedback.trace().to_string(), "added one extra unit");
        Ok(())
    }

    #[test]
    fn every_signal_can_be_audited_with_source() -> Result<(), AlignmentError> {
        let example =
            InstructionExample::new(instruction()?, Response::try_from("2 + 2 = 4")?, source()?);
        let record = AuditRecord::new(
            AlignmentRunId::try_from("align-run-1")?,
            UpdateSignal::Supervised(example),
            AuditNote::try_from("kept public toy instruction response")?,
        );

        assert_eq!(record.signal().kind(), UpdateKind::SupervisedFineTuning);
        assert!(record.to_string().contains("public-toy-fixture"));
        Ok(())
    }

    #[test]
    fn reward_score_rejects_out_of_range_values() {
        let error = RewardScore::try_from(2.0).err();
        assert!(matches!(error, Some(AlignmentError::OutOfRange { .. })));
    }

    #[test]
    fn workflow_records_audits_and_applies_update() -> Result<(), AlignmentError> {
        let run_id = AlignmentRunId::try_from("align-run-1")?;
        let example =
            InstructionExample::new(instruction()?, Response::try_from("2 + 2 = 4")?, source()?);
        let record = AuditRecord::new(
            run_id.clone(),
            UpdateSignal::Supervised(example),
            AuditNote::try_from("approved public toy signal")?,
        );

        let workflow = AlignmentWorkflow::new(run_id)
            .record_signal(record)?
            .approve_audit()?
            .apply_update()?;

        assert_eq!(workflow.stage(), AlignmentStage::UpdateApplied);
        assert_eq!(
            workflow.latest_transition().map(ToString::to_string),
            Some("ready_for_update --update_applied--> update_applied".to_owned())
        );
        Ok(())
    }

    #[test]
    fn workflow_rejects_update_before_audit() -> Result<(), AlignmentError> {
        let workflow = AlignmentWorkflow::new(AlignmentRunId::try_from("align-run-1")?);
        let error = workflow.apply_update().err();

        assert!(matches!(
            error,
            Some(AlignmentError::InvalidTransition { .. })
        ));
        Ok(())
    }

    #[test]
    fn workflow_rejects_record_from_different_run() -> Result<(), AlignmentError> {
        let record = AuditRecord::new(
            AlignmentRunId::try_from("other-run")?,
            UpdateSignal::Verifier(VerifierFeedback::new(
                instruction()?,
                Response::try_from("2 + 2 = 5")?,
                ReasoningTrace::try_from("the answer adds one extra unit")?,
                VerifierResult::Failed,
                source()?,
            )),
            AuditNote::try_from("kept failed verifier result visible")?,
        );

        let error = AlignmentWorkflow::new(AlignmentRunId::try_from("align-run-1")?)
            .record_signal(record)
            .err();

        assert!(matches!(
            error,
            Some(AlignmentError::InvalidTransition { .. })
        ));
        Ok(())
    }

    #[test]
    fn public_alignment_release_accepts_public_applied_workflow() -> Result<(), AlignmentError> {
        let release = PublicAlignmentRelease::from_reviewed_workflow(
            ReviewedAlignmentWorkflow::new(applied_workflow()?, AlignmentVisibility::Public),
        )?;

        assert_eq!(release.stage(), AlignmentStage::UpdateApplied);
        assert_eq!(release.run_id().to_string(), "align-run-public");
        assert!(release.latest_record().is_some());
        Ok(())
    }

    #[test]
    fn public_alignment_release_blocks_restricted_and_private_workflows()
    -> Result<(), AlignmentError> {
        let restricted =
            PublicAlignmentRelease::from_reviewed_workflow(ReviewedAlignmentWorkflow::new(
                applied_workflow()?,
                AlignmentVisibility::ResearchRestricted,
            ))
            .err();
        let private = PublicAlignmentRelease::from_reviewed_workflow(
            ReviewedAlignmentWorkflow::new(applied_workflow()?, AlignmentVisibility::Private),
        )
        .err();

        assert!(matches!(
            restricted,
            Some(AlignmentError::InvalidPublicRelease { .. })
        ));
        assert!(matches!(
            private,
            Some(AlignmentError::InvalidPublicRelease { .. })
        ));
        Ok(())
    }

    #[test]
    fn public_alignment_release_requires_completed_audit_lifecycle() -> Result<(), AlignmentError> {
        let run_id = AlignmentRunId::try_from("align-run-public")?;
        let example =
            InstructionExample::new(instruction()?, Response::try_from("2 + 2 = 4")?, source()?);
        let record = AuditRecord::new(
            run_id.clone(),
            UpdateSignal::Supervised(example),
            AuditNote::try_from("recorded but not yet audit-approved")?,
        );
        let workflow = AlignmentWorkflow::new(run_id).record_signal(record)?;

        let error = PublicAlignmentRelease::from_reviewed_workflow(ReviewedAlignmentWorkflow::new(
            workflow,
            AlignmentVisibility::Public,
        ))
        .err();

        assert!(matches!(
            error,
            Some(AlignmentError::InvalidPublicRelease { .. })
        ));
        Ok(())
    }
}
