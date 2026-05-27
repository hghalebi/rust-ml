//! Typed post-training signals for the CS336 Rust equivalent track.
//!
//! This crate treats alignment as auditable data flow:
//!
//! ```text
//! PromptedResponse -> PreferenceSignal -> UpdateSignal -> AuditRecord
//! ```
//!
//! Raw learner strings and scores enter through `TryFrom` adapters. Once inside
//! the crate, public APIs use semantic values such as [`Instruction`],
//! [`ChosenResponse`], [`RejectedResponse`], [`RewardScore`],
//! [`VerifierFeedback`], and [`AlignmentRunId`].

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

#[cfg(test)]
mod tests {
    use super::{
        AlignmentError, AlignmentRunId, AuditNote, AuditRecord, ChosenResponse, Instruction,
        InstructionExample, PreferencePair, PreferenceRewards, PreferenceSignal, ReasoningTrace,
        RejectedResponse, Response, RewardScore, SignalSource, UpdateKind, UpdateSignal,
        VerifierFeedback, VerifierResult,
    };

    fn instruction() -> Result<Instruction, AlignmentError> {
        Instruction::try_from("solve 2 + 2 with a visible check")
    }

    fn source() -> Result<SignalSource, AlignmentError> {
        SignalSource::try_from("public-toy-fixture")
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
}
