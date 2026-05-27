//! Typed evaluation primitives for the CS336 Rust equivalent track.
//!
//! Evaluation is a checked composition:
//!
//! ```text
//! EvalExample + ModelPrediction -> ScoredPrediction -> EvalReport
//! ```
//!
//! Raw learner text enters through `TryFrom` adapters. Once inside the public
//! API, prompts, answers, example IDs, run IDs, counts, and scores are semantic
//! values that cannot be mixed accidentally. Public release uses a second
//! checked map:
//!
//! ```text
//! ReviewedScoredPrediction* -> PublicEvalReport
//! ```

pub mod error;

use std::{
    collections::BTreeSet,
    fmt,
    ops::{Div, Sub},
};

use error::EvaluationError;

pub use error::EvaluationError as Error;

fn validate_nonempty(
    value: &str,
    operation: &'static str,
    details: &'static str,
) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        return Err(EvaluationError::empty_input(operation, details));
    }

    Ok(())
}

fn finite(role: &'static str, value: f64) -> Result<f64, EvaluationError> {
    if !value.is_finite() {
        return Err(EvaluationError::non_finite_value(role, value));
    }

    Ok(value)
}

fn positive_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, EvaluationError> {
    if value == 0 {
        return Err(EvaluationError::empty_input(operation, role));
    }

    Ok(value)
}

fn normalized_answer(value: &str) -> String {
    value
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

macro_rules! nonempty_text_type {
    ($name:ident, $doc:literal, $operation:literal, $details:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            fn from_owned(value: String) -> Result<Self, EvaluationError> {
                validate_nonempty(&value, $operation, $details)?;
                Ok(Self(value))
            }

            fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<&str> for $name {
            type Error = EvaluationError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::from_owned(value.to_owned())
            }
        }

        impl TryFrom<String> for $name {
            type Error = EvaluationError;

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
    ExampleId,
    "Stable identity for one evaluation example.",
    "ExampleId::try_from",
    "example id cannot be empty"
);
nonempty_text_type!(
    EvalRunId,
    "Stable identity for one evaluation run.",
    "EvalRunId::try_from",
    "evaluation run id cannot be empty"
);
nonempty_text_type!(
    Prompt,
    "Prompt text shown to the model during evaluation.",
    "Prompt::try_from",
    "prompt cannot be empty"
);
nonempty_text_type!(
    ExpectedAnswer,
    "Reference answer used by a deterministic metric.",
    "ExpectedAnswer::try_from",
    "expected answer cannot be empty"
);
nonempty_text_type!(
    ModelAnswer,
    "Answer produced by a model or toy predictor.",
    "ModelAnswer::try_from",
    "model answer cannot be empty"
);

impl ExpectedAnswer {
    fn normalized(&self) -> String {
        normalized_answer(self.as_str())
    }
}

impl ModelAnswer {
    fn normalized(&self) -> String {
        normalized_answer(self.as_str())
    }
}

/// One deterministic evaluation example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalExample {
    id: ExampleId,
    prompt: Prompt,
    expected: ExpectedAnswer,
}

impl EvalExample {
    /// Creates an evaluation example from already validated semantic values.
    pub fn new(id: ExampleId, prompt: Prompt, expected: ExpectedAnswer) -> Self {
        Self {
            id,
            prompt,
            expected,
        }
    }

    /// Returns the example identity.
    pub fn id(&self) -> &ExampleId {
        &self.id
    }

    /// Returns the prompt.
    pub fn prompt(&self) -> &Prompt {
        &self.prompt
    }

    /// Returns the reference answer.
    pub fn expected(&self) -> &ExpectedAnswer {
        &self.expected
    }
}

/// One model prediction attached to an evaluation example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPrediction {
    example_id: ExampleId,
    answer: ModelAnswer,
}

impl ModelPrediction {
    /// Creates a prediction from already validated semantic values.
    pub fn new(example_id: ExampleId, answer: ModelAnswer) -> Self {
        Self { example_id, answer }
    }

    /// Returns the example identity this prediction claims to answer.
    pub fn example_id(&self) -> &ExampleId {
        &self.example_id
    }

    /// Returns the model answer.
    pub fn answer(&self) -> &ModelAnswer {
        &self.answer
    }
}

/// Exact-match outcome after reference and prediction normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Correctness {
    /// The normalized answer matched the normalized reference.
    Correct,
    /// The normalized answer did not match the normalized reference.
    Incorrect,
}

impl Correctness {
    fn correct_count(self) -> CorrectPredictionCount {
        match self {
            Self::Correct => CorrectPredictionCount::one(),
            Self::Incorrect => CorrectPredictionCount::zero(),
        }
    }
}

impl fmt::Display for Correctness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Correct => formatter.write_str("correct"),
            Self::Incorrect => formatter.write_str("incorrect"),
        }
    }
}

/// One prediction after deterministic scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredPrediction {
    example: EvalExample,
    prediction: ModelPrediction,
    outcome: Correctness,
}

impl ScoredPrediction {
    /// Scores one example and prediction using normalized exact match.
    pub fn exact_match(
        example: EvalExample,
        prediction: ModelPrediction,
    ) -> Result<Self, EvaluationError> {
        if example.id() != prediction.example_id() {
            return Err(EvaluationError::example_mismatch(
                "ScoredPrediction::exact_match",
                "prediction example id must match the reference example id",
            ));
        }

        let outcome = if example.expected().normalized() == prediction.answer().normalized() {
            Correctness::Correct
        } else {
            Correctness::Incorrect
        };

        Ok(Self {
            example,
            prediction,
            outcome,
        })
    }

    /// Returns the example identity.
    pub fn example_id(&self) -> &ExampleId {
        self.example.id()
    }

    /// Returns the scored outcome.
    pub fn outcome(&self) -> Correctness {
        self.outcome
    }

    /// Returns the reference example.
    pub fn example(&self) -> &EvalExample {
        &self.example
    }

    /// Returns the prediction.
    pub fn prediction(&self) -> &ModelPrediction {
        &self.prediction
    }
}

/// Non-empty count of evaluated examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExampleCount(usize);

impl ExampleCount {
    fn from_raw(value: usize) -> Result<Self, EvaluationError> {
        Ok(Self(positive_usize(
            "example count must be greater than zero",
            "ExampleCount::from_raw",
            value,
        )?))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ExampleCount {
    type Error = EvaluationError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for ExampleCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Count of correct predictions inside one report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CorrectPredictionCount(usize);

impl CorrectPredictionCount {
    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for CorrectPredictionCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Exact-match accuracy in the closed interval `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ExactMatchAccuracy(f64);

impl ExactMatchAccuracy {
    fn from_raw(value: f64) -> Result<Self, EvaluationError> {
        let value = finite("exact-match accuracy", value)?;
        if !(0.0..=1.0).contains(&value) {
            return Err(EvaluationError::out_of_range(
                "exact-match accuracy",
                "0..=1",
                value,
            ));
        }

        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for ExactMatchAccuracy {
    type Error = EvaluationError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl Div<ExampleCount> for CorrectPredictionCount {
    type Output = Result<ExactMatchAccuracy, EvaluationError>;

    fn div(self, right: ExampleCount) -> Self::Output {
        ExactMatchAccuracy::from_raw(self.as_usize() as f64 / right.as_usize() as f64)
    }
}

impl Sub for ExactMatchAccuracy {
    type Output = Result<AccuracyDelta, EvaluationError>;

    fn sub(self, right: Self) -> Self::Output {
        AccuracyDelta::from_raw(self.as_f64() - right.as_f64())
    }
}

impl fmt::Display for ExactMatchAccuracy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.3}", self.0)
    }
}

/// Difference between two exact-match accuracies.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AccuracyDelta(f64);

impl AccuracyDelta {
    fn from_raw(value: f64) -> Result<Self, EvaluationError> {
        let value = finite("accuracy delta", value)?;
        if !(-1.0..=1.0).contains(&value) {
            return Err(EvaluationError::out_of_range(
                "accuracy delta",
                "-1..=1",
                value,
            ));
        }

        Ok(Self(value))
    }
}

impl fmt::Display for AccuracyDelta {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:+.3}", self.0)
    }
}

/// Evaluation report with a run identity and scored predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    run_id: EvalRunId,
    records: Vec<ScoredPrediction>,
    count: ExampleCount,
    correct: CorrectPredictionCount,
    accuracy: ExactMatchAccuracy,
}

impl EvalReport {
    /// Builds a report from non-empty scored predictions.
    pub fn from_records(
        run_id: EvalRunId,
        records: impl IntoIterator<Item = ScoredPrediction>,
    ) -> Result<Self, EvaluationError> {
        let records = records.into_iter().collect::<Vec<_>>();
        let count = ExampleCount::from_raw(records.len())?;
        let mut seen = BTreeSet::new();
        let mut correct = CorrectPredictionCount::zero();

        for record in &records {
            if !seen.insert(record.example_id().clone()) {
                return Err(EvaluationError::duplicate_example(
                    "EvalReport::from_records",
                    "each example id may appear only once in a report",
                ));
            }

            correct = CorrectPredictionCount(
                correct.as_usize() + record.outcome().correct_count().as_usize(),
            );
        }

        let accuracy = (correct / count)?;

        Ok(Self {
            run_id,
            records,
            count,
            correct,
            accuracy,
        })
    }

    /// Returns the run identity.
    pub fn run_id(&self) -> &EvalRunId {
        &self.run_id
    }

    /// Returns the number of evaluated examples.
    pub fn count(&self) -> ExampleCount {
        self.count
    }

    /// Returns the number of correct predictions.
    pub fn correct(&self) -> CorrectPredictionCount {
        self.correct
    }

    /// Returns exact-match accuracy.
    pub fn accuracy(&self) -> ExactMatchAccuracy {
        self.accuracy
    }

    /// Iterates over scored predictions.
    pub fn records(&self) -> impl ExactSizeIterator<Item = &ScoredPrediction> + '_ {
        self.records.iter()
    }
}

/// Compares two reports by exact-match accuracy.
pub fn compare_accuracy(
    newer: &EvalReport,
    baseline: &EvalReport,
) -> Result<AccuracyDelta, EvaluationError> {
    newer.accuracy() - baseline.accuracy()
}

/// Publication class for an evaluated example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleVisibility {
    /// Safe to include in learner-facing public report examples.
    Public,
    /// Useful for internal study but not safe for the public report surface.
    ResearchRestricted,
    /// Must never appear in public learner-facing report examples.
    Private,
}

impl fmt::Display for ExampleVisibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Public => "public",
            Self::ResearchRestricted => "research-restricted",
            Self::Private => "private",
        };
        formatter.write_str(label)
    }
}

/// Typed decision at the evaluation-publication boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicReportDecision {
    /// The scored prediction can appear in a public learner-facing report.
    Publishable,
    /// The scored prediction must stay out of a public learner-facing report.
    Blocked,
}

/// A scored prediction plus publication evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewedScoredPrediction {
    record: ScoredPrediction,
    visibility: ExampleVisibility,
}

impl ReviewedScoredPrediction {
    /// Creates a reviewed scored prediction.
    pub fn new(record: ScoredPrediction, visibility: ExampleVisibility) -> Self {
        Self { record, visibility }
    }

    /// Returns the scored prediction.
    pub fn record(&self) -> &ScoredPrediction {
        &self.record
    }

    /// Returns the publication class.
    pub fn visibility(&self) -> ExampleVisibility {
        self.visibility
    }

    /// Classifies whether this record can enter a public learner-facing report.
    pub fn release_decision(&self) -> PublicReportDecision {
        match self.visibility {
            ExampleVisibility::Public => PublicReportDecision::Publishable,
            ExampleVisibility::ResearchRestricted | ExampleVisibility::Private => {
                PublicReportDecision::Blocked
            }
        }
    }

    fn into_record(self) -> ScoredPrediction {
        self.record
    }
}

/// Evaluation report checked for public learner-facing release.
#[derive(Debug, Clone, PartialEq)]
pub struct PublicEvalReport(EvalReport);

impl PublicEvalReport {
    /// Builds a public report only from reviewed public examples.
    pub fn from_reviewed_records(
        run_id: EvalRunId,
        records: impl IntoIterator<Item = ReviewedScoredPrediction>,
    ) -> Result<Self, EvaluationError> {
        let mut publishable_records = Vec::new();

        for record in records {
            if record.release_decision() == PublicReportDecision::Blocked {
                return Err(EvaluationError::invalid_public_report(
                    "PublicEvalReport::from_reviewed_records",
                    "public reports cannot include restricted or private evaluation examples",
                ));
            }
            publishable_records.push(record.into_record());
        }

        Ok(Self(EvalReport::from_records(run_id, publishable_records)?))
    }

    /// Returns the checked evaluation report.
    pub fn report(&self) -> &EvalReport {
        &self.0
    }

    /// Returns exact-match accuracy for the public report.
    pub fn accuracy(&self) -> ExactMatchAccuracy {
        self.0.accuracy()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EvalExample, EvalReport, EvalRunId, ExampleId, ExampleVisibility, ExpectedAnswer,
        ModelAnswer, ModelPrediction, Prompt, PublicEvalReport, ReviewedScoredPrediction,
        ScoredPrediction, compare_accuracy,
    };
    use crate::error::EvaluationError;

    fn example(id: ExampleId, prompt: Prompt, expected: ExpectedAnswer) -> EvalExample {
        EvalExample::new(id, prompt, expected)
    }

    fn prediction(id: ExampleId, answer: ModelAnswer) -> ModelPrediction {
        ModelPrediction::new(id, answer)
    }

    fn scored_record(
        id: ExampleId,
        expected: ExpectedAnswer,
        answer: ModelAnswer,
    ) -> Result<ScoredPrediction, EvaluationError> {
        ScoredPrediction::exact_match(
            example(
                id.clone(),
                Prompt::try_from("public learner-safe prompt")?,
                expected,
            ),
            prediction(id, answer),
        )
    }

    #[test]
    fn exact_match_normalizes_case_and_whitespace() -> Result<(), EvaluationError> {
        let scored = ScoredPrediction::exact_match(
            example(
                ExampleId::try_from("ex-1")?,
                Prompt::try_from("name the next token")?,
                ExpectedAnswer::try_from("Rust ML")?,
            ),
            prediction(
                ExampleId::try_from("ex-1")?,
                ModelAnswer::try_from(" rust   ml ")?,
            ),
        )?;

        assert_eq!(scored.outcome(), super::Correctness::Correct);
        Ok(())
    }

    #[test]
    fn exact_match_rejects_mismatched_example_ids() -> Result<(), EvaluationError> {
        let scored = ScoredPrediction::exact_match(
            example(
                ExampleId::try_from("ex-1")?,
                Prompt::try_from("name the next token")?,
                ExpectedAnswer::try_from("rust")?,
            ),
            prediction(ExampleId::try_from("ex-2")?, ModelAnswer::try_from("rust")?),
        );

        assert!(matches!(
            scored,
            Err(EvaluationError::ExampleMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn report_computes_accuracy_from_scored_records() -> Result<(), EvaluationError> {
        let records = [
            ScoredPrediction::exact_match(
                example(
                    ExampleId::try_from("ex-1")?,
                    Prompt::try_from("name the next token")?,
                    ExpectedAnswer::try_from("rust")?,
                ),
                prediction(ExampleId::try_from("ex-1")?, ModelAnswer::try_from("rust")?),
            )?,
            ScoredPrediction::exact_match(
                example(
                    ExampleId::try_from("ex-2")?,
                    Prompt::try_from("name the next token")?,
                    ExpectedAnswer::try_from("types")?,
                ),
                prediction(
                    ExampleId::try_from("ex-2")?,
                    ModelAnswer::try_from("vectors")?,
                ),
            )?,
        ];

        let report = EvalReport::from_records(EvalRunId::try_from("run-a")?, records)?;

        assert_eq!(report.correct().to_string(), "1");
        assert_eq!(report.accuracy().to_string(), "0.500");
        Ok(())
    }

    #[test]
    fn report_rejects_duplicate_examples() -> Result<(), EvaluationError> {
        let records = [
            ScoredPrediction::exact_match(
                example(
                    ExampleId::try_from("ex-1")?,
                    Prompt::try_from("name the next token")?,
                    ExpectedAnswer::try_from("rust")?,
                ),
                prediction(ExampleId::try_from("ex-1")?, ModelAnswer::try_from("rust")?),
            )?,
            ScoredPrediction::exact_match(
                example(
                    ExampleId::try_from("ex-1")?,
                    Prompt::try_from("name the next token")?,
                    ExpectedAnswer::try_from("types")?,
                ),
                prediction(
                    ExampleId::try_from("ex-1")?,
                    ModelAnswer::try_from("types")?,
                ),
            )?,
        ];

        let report = EvalReport::from_records(EvalRunId::try_from("run-a")?, records);

        assert!(matches!(
            report,
            Err(EvaluationError::DuplicateExample { .. })
        ));
        Ok(())
    }

    #[test]
    fn compare_accuracy_reports_typed_delta() -> Result<(), EvaluationError> {
        let baseline = EvalReport::from_records(
            EvalRunId::try_from("baseline")?,
            [
                ScoredPrediction::exact_match(
                    example(
                        ExampleId::try_from("ex-1")?,
                        Prompt::try_from("name the next token")?,
                        ExpectedAnswer::try_from("rust")?,
                    ),
                    prediction(ExampleId::try_from("ex-1")?, ModelAnswer::try_from("rust")?),
                )?,
                ScoredPrediction::exact_match(
                    example(
                        ExampleId::try_from("ex-2")?,
                        Prompt::try_from("name the next token")?,
                        ExpectedAnswer::try_from("types")?,
                    ),
                    prediction(
                        ExampleId::try_from("ex-2")?,
                        ModelAnswer::try_from("vectors")?,
                    ),
                )?,
            ],
        )?;
        let newer = EvalReport::from_records(
            EvalRunId::try_from("newer")?,
            [
                ScoredPrediction::exact_match(
                    example(
                        ExampleId::try_from("ex-1")?,
                        Prompt::try_from("name the next token")?,
                        ExpectedAnswer::try_from("rust")?,
                    ),
                    prediction(ExampleId::try_from("ex-1")?, ModelAnswer::try_from("rust")?),
                )?,
                ScoredPrediction::exact_match(
                    example(
                        ExampleId::try_from("ex-2")?,
                        Prompt::try_from("name the next token")?,
                        ExpectedAnswer::try_from("types")?,
                    ),
                    prediction(
                        ExampleId::try_from("ex-2")?,
                        ModelAnswer::try_from("types")?,
                    ),
                )?,
            ],
        )?;

        let delta = compare_accuracy(&newer, &baseline)?;

        assert_eq!(delta.to_string(), "+0.500");
        Ok(())
    }

    #[test]
    fn public_eval_report_accepts_public_reviewed_records() -> Result<(), EvaluationError> {
        let report = PublicEvalReport::from_reviewed_records(
            EvalRunId::try_from("public-eval")?,
            [
                ReviewedScoredPrediction::new(
                    scored_record(
                        ExampleId::try_from("ex-1")?,
                        ExpectedAnswer::try_from("newtype")?,
                        ModelAnswer::try_from("newtype")?,
                    )?,
                    ExampleVisibility::Public,
                ),
                ReviewedScoredPrediction::new(
                    scored_record(
                        ExampleId::try_from("ex-2")?,
                        ExpectedAnswer::try_from("typed error")?,
                        ModelAnswer::try_from("typed error")?,
                    )?,
                    ExampleVisibility::Public,
                ),
            ],
        )?;

        assert_eq!(report.report().count().to_string(), "2");
        assert_eq!(report.accuracy().to_string(), "1.000");
        Ok(())
    }

    #[test]
    fn public_eval_report_blocks_restricted_and_private_records() -> Result<(), EvaluationError> {
        let restricted = PublicEvalReport::from_reviewed_records(
            EvalRunId::try_from("public-eval")?,
            [ReviewedScoredPrediction::new(
                scored_record(
                    ExampleId::try_from("ex-1")?,
                    ExpectedAnswer::try_from("newtype")?,
                    ModelAnswer::try_from("newtype")?,
                )?,
                ExampleVisibility::ResearchRestricted,
            )],
        );
        let private = PublicEvalReport::from_reviewed_records(
            EvalRunId::try_from("public-eval")?,
            [ReviewedScoredPrediction::new(
                scored_record(
                    ExampleId::try_from("ex-1")?,
                    ExpectedAnswer::try_from("newtype")?,
                    ModelAnswer::try_from("newtype")?,
                )?,
                ExampleVisibility::Private,
            )],
        );

        assert!(matches!(
            restricted,
            Err(EvaluationError::InvalidPublicReport { .. })
        ));
        assert!(matches!(
            private,
            Err(EvaluationError::InvalidPublicReport { .. })
        ));
        Ok(())
    }
}
