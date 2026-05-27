//! Typed scaling-law experiments for the CS336 Rust equivalent track.
//!
//! This crate teaches scaling as evidence management, not curve worship:
//!
//! ```text
//! ExperimentConfig -> TrainingRun -> MetricRecord -> ScalingFit
//! ScalingFit + TrainingRun -> ScalingCandidate -> ScalingTradeoff
//! ```
//!
//! Raw learner literals enter through `TryFrom` adapters. The public teaching
//! path then uses semantic values such as [`ParameterCount`], [`TokenCount`],
//! [`ComputeBudgetFlops`], [`ValidationLoss`], [`ScalingExponent`], and
//! [`ScalingTradeoff`].

pub mod error;

use std::{
    fmt,
    ops::{Add, Mul, Sub},
};

use error::ScalingError;

pub use error::ScalingError as Error;

fn nonzero_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, ScalingError> {
    if value == 0 {
        return Err(ScalingError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u64(
    role: &'static str,
    operation: &'static str,
    value: u64,
) -> Result<u64, ScalingError> {
    if value == 0 {
        return Err(ScalingError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u128(
    role: &'static str,
    operation: &'static str,
    value: u128,
) -> Result<u128, ScalingError> {
    if value == 0 {
        return Err(ScalingError::empty_input(operation, role));
    }

    Ok(value)
}

fn finite(role: &'static str, value: f64) -> Result<f64, ScalingError> {
    if !value.is_finite() {
        return Err(ScalingError::non_finite_value(role, value));
    }

    Ok(value)
}

fn checked_u64_from_usize(operation: &'static str, value: usize) -> Result<u64, ScalingError> {
    u64::try_from(value).map_err(|_| ScalingError::overflow(operation, "value exceeded u64"))
}

macro_rules! positive_usize_type {
    ($name:ident, $doc:literal, $role:literal, $operation:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(usize);

        impl TryFrom<usize> for $name {
            type Error = ScalingError;

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                Ok(Self(nonzero_usize($role, $operation, value)?))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, formatter)
            }
        }
    };
}

macro_rules! positive_u64_type {
    ($name:ident, $doc:literal, $role:literal, $operation:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(u64);

        impl TryFrom<u64> for $name {
            type Error = ScalingError;

            fn try_from(value: u64) -> Result<Self, Self::Error> {
                Ok(Self(nonzero_u64($role, $operation, value)?))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, formatter)
            }
        }
    };
}

positive_usize_type!(
    ModelWidth,
    "Hidden width used by one tiny experiment.",
    "model width must be greater than zero",
    "ModelWidth::try_from"
);
positive_usize_type!(
    LayerCount,
    "Number of repeated model blocks in one tiny experiment.",
    "layer count must be greater than zero",
    "LayerCount::try_from"
);
positive_usize_type!(
    RecordCount,
    "Number of metric records used by a scaling fit.",
    "record count must be greater than zero",
    "RecordCount::try_from"
);
positive_u64_type!(
    TokenCount,
    "Number of training tokens consumed by one run.",
    "token count must be greater than zero",
    "TokenCount::try_from"
);
positive_u64_type!(
    TrainingStep,
    "Number of optimizer steps in one run.",
    "training step count must be greater than zero",
    "TrainingStep::try_from"
);
positive_u64_type!(
    ParameterCount,
    "Approximate trainable parameter count.",
    "parameter count must be greater than zero",
    "ParameterCount::try_from"
);
positive_u64_type!(
    ComputeMultiplier,
    "Multiplier that turns parameter-token products into training FLOPs.",
    "compute multiplier must be greater than zero",
    "ComputeMultiplier::try_from"
);

impl ModelWidth {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl LayerCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TokenCount {
    fn as_u64(self) -> u64 {
        self.0
    }
}

impl ParameterCount {
    fn as_u64(self) -> u64 {
        self.0
    }
}

impl ComputeMultiplier {
    fn as_u64(self) -> u64 {
        self.0
    }
}

/// Stable label for one run in a tiny scaling study.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(String);

impl RunId {
    fn from_owned(value: String) -> Result<Self, ScalingError> {
        if value.trim().is_empty() {
            return Err(ScalingError::empty_input(
                "RunId::try_from",
                "run id cannot be empty",
            ));
        }
        Ok(Self(value))
    }
}

impl TryFrom<&str> for RunId {
    type Error = ScalingError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for RunId {
    type Error = ScalingError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Parameter-token product before applying the training-FLOP multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParameterTokenProduct(u128);

impl ParameterTokenProduct {
    fn from_raw(operation: &'static str, value: u128) -> Result<Self, ScalingError> {
        Ok(Self(nonzero_u128(
            "parameter-token product must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u128(self) -> u128 {
        self.0
    }
}

impl Mul<TokenCount> for ParameterCount {
    type Output = Result<ParameterTokenProduct, ScalingError>;

    fn mul(self, tokens: TokenCount) -> Self::Output {
        let product = u128::from(self.as_u64())
            .checked_mul(u128::from(tokens.as_u64()))
            .ok_or(ScalingError::overflow(
                "ParameterCount::mul",
                "parameter-token product exceeded u128",
            ))?;

        ParameterTokenProduct::from_raw("ParameterCount::mul", product)
    }
}

/// Estimated floating-point work for one training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeBudgetFlops(u128);

impl ComputeBudgetFlops {
    fn from_raw(operation: &'static str, value: u128) -> Result<Self, ScalingError> {
        Ok(Self(nonzero_u128(
            "compute budget must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_f64(self) -> f64 {
        self.0 as f64
    }

    fn as_u128(self) -> u128 {
        self.0
    }
}

impl TryFrom<u128> for ComputeBudgetFlops {
    type Error = ScalingError;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Self::from_raw("ComputeBudgetFlops::try_from", value)
    }
}

impl fmt::Display for ComputeBudgetFlops {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} FLOPs", self.0)
    }
}

impl Add for ComputeBudgetFlops {
    type Output = Result<ComputeBudgetFlops, ScalingError>;

    fn add(self, right: ComputeBudgetFlops) -> Self::Output {
        let total = self.0.checked_add(right.0).ok_or(ScalingError::overflow(
            "ComputeBudgetFlops::add",
            "compute budget exceeded u128",
        ))?;
        ComputeBudgetFlops::from_raw("ComputeBudgetFlops::add", total)
    }
}

impl Mul<ComputeMultiplier> for ParameterTokenProduct {
    type Output = Result<ComputeBudgetFlops, ScalingError>;

    fn mul(self, multiplier: ComputeMultiplier) -> Self::Output {
        let flops = self
            .as_u128()
            .checked_mul(u128::from(multiplier.as_u64()))
            .ok_or(ScalingError::overflow(
                "ParameterTokenProduct::mul",
                "compute budget exceeded u128",
            ))?;
        ComputeBudgetFlops::from_raw("ParameterTokenProduct::mul", flops)
    }
}

/// Absolute difference between two compute budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeDelta(u128);

impl ComputeDelta {
    fn between(left: ComputeBudgetFlops, right: ComputeBudgetFlops) -> Self {
        Self(left.as_u128().abs_diff(right.as_u128()))
    }
}

impl fmt::Display for ComputeDelta {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} FLOPs", self.0)
    }
}

impl ComputeMultiplier {
    /// Conventional rough estimate for dense Transformer training FLOPs.
    pub fn dense_transformer_training() -> Self {
        Self(6)
    }
}

/// Positive validation loss for one completed run.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ValidationLoss(f64);

impl ValidationLoss {
    fn from_raw(value: f64) -> Result<Self, ScalingError> {
        let value = finite("validation loss", value)?;
        if value <= 0.0 {
            return Err(ScalingError::out_of_range("validation loss", "> 0", value));
        }
        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }

    fn ln(self) -> f64 {
        self.0.ln()
    }
}

impl TryFrom<f64> for ValidationLoss {
    type Error = ScalingError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for ValidationLoss {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.6}", self.0)
    }
}

/// Signed difference between observed and predicted validation loss.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LossDelta(f64);

impl LossDelta {
    fn from_raw(value: f64) -> Result<Self, ScalingError> {
        Ok(Self(finite("loss delta", value)?))
    }
}

impl Sub<ValidationLoss> for ValidationLoss {
    type Output = Result<LossDelta, ScalingError>;

    fn sub(self, predicted: ValidationLoss) -> Self::Output {
        LossDelta::from_raw(self.as_f64() - predicted.as_f64())
    }
}

impl fmt::Display for LossDelta {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:+.6}", self.0)
    }
}

/// Multiplicative coefficient in `loss = coefficient * compute^exponent`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ScalingCoefficient(f64);

impl ScalingCoefficient {
    fn from_raw(value: f64) -> Result<Self, ScalingError> {
        let value = finite("scaling coefficient", value)?;
        if value <= 0.0 {
            return Err(ScalingError::out_of_range(
                "scaling coefficient",
                "> 0",
                value,
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for ScalingCoefficient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.6}", self.0)
    }
}

/// Exponent in `loss = coefficient * compute^exponent`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ScalingExponent(f64);

impl ScalingExponent {
    fn from_raw(value: f64) -> Result<Self, ScalingError> {
        Ok(Self(finite("scaling exponent", value)?))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl fmt::Display for ScalingExponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.6}", self.0)
    }
}

/// Configuration for one tiny scaling run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentConfig {
    run_id: RunId,
    model_width: ModelWidth,
    layer_count: LayerCount,
    training_tokens: TokenCount,
}

impl ExperimentConfig {
    /// Creates a typed experiment configuration.
    pub fn new(
        run_id: RunId,
        model_width: ModelWidth,
        layer_count: LayerCount,
        training_tokens: TokenCount,
    ) -> Self {
        Self {
            run_id,
            model_width,
            layer_count,
            training_tokens,
        }
    }

    /// Returns the run label.
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// Estimates a tiny dense Transformer-style parameter count.
    pub fn parameter_count(&self) -> Result<ParameterCount, ScalingError> {
        let width = checked_u64_from_usize(
            "ExperimentConfig::parameter_count",
            self.model_width.as_usize(),
        )?;
        let layers = checked_u64_from_usize(
            "ExperimentConfig::parameter_count",
            self.layer_count.as_usize(),
        )?;
        let width_squared = width.checked_mul(width).ok_or(ScalingError::overflow(
            "ExperimentConfig::parameter_count",
            "width squared exceeded u64",
        ))?;
        let per_layer = width_squared.checked_mul(12).ok_or(ScalingError::overflow(
            "ExperimentConfig::parameter_count",
            "per-layer parameter estimate exceeded u64",
        ))?;
        let total = per_layer.checked_mul(layers).ok_or(ScalingError::overflow(
            "ExperimentConfig::parameter_count",
            "parameter estimate exceeded u64",
        ))?;

        ParameterCount::try_from(total)
    }

    /// Plans a run by attaching a step count and derived resource estimates.
    pub fn plan_run(&self, training_steps: TrainingStep) -> Result<TrainingRun, ScalingError> {
        TrainingRun::from_config(self.clone(), training_steps)
    }
}

/// Completed or planned run with derived resource estimates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingRun {
    config: ExperimentConfig,
    training_steps: TrainingStep,
    parameter_count: ParameterCount,
    compute_budget: ComputeBudgetFlops,
}

impl TrainingRun {
    fn from_config(
        config: ExperimentConfig,
        training_steps: TrainingStep,
    ) -> Result<Self, ScalingError> {
        let parameter_count = config.parameter_count()?;
        let compute_budget = (parameter_count * config.training_tokens)?
            * ComputeMultiplier::dense_transformer_training();

        Ok(Self {
            config,
            training_steps,
            parameter_count,
            compute_budget: compute_budget?,
        })
    }

    /// Returns the run label.
    pub fn run_id(&self) -> &RunId {
        self.config.run_id()
    }

    /// Returns the original typed config.
    pub fn config(&self) -> &ExperimentConfig {
        &self.config
    }

    /// Returns the optimizer step count.
    pub fn training_steps(&self) -> TrainingStep {
        self.training_steps
    }

    /// Returns the parameter estimate.
    pub fn parameter_count(&self) -> ParameterCount {
        self.parameter_count
    }

    /// Returns the compute estimate.
    pub fn compute_budget(&self) -> ComputeBudgetFlops {
        self.compute_budget
    }
}

/// Metric record that keeps the measured loss attached to the run that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricRecord {
    run: TrainingRun,
    validation_loss: ValidationLoss,
}

impl MetricRecord {
    /// Creates a complete metric record.
    pub fn from_run(run: TrainingRun, validation_loss: ValidationLoss) -> Self {
        Self {
            run,
            validation_loss,
        }
    }

    /// Returns the producing run.
    pub fn run(&self) -> &TrainingRun {
        &self.run
    }

    /// Returns the measured validation loss.
    pub fn validation_loss(&self) -> ValidationLoss {
        self.validation_loss
    }
}

/// Non-empty collection of metric records for fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricRecords(Vec<MetricRecord>);

impl MetricRecords {
    /// Creates a checked non-empty metric record set.
    pub fn from_records(
        records: impl IntoIterator<Item = MetricRecord>,
    ) -> Result<Self, ScalingError> {
        let records = records.into_iter().collect::<Vec<_>>();
        if records.is_empty() {
            return Err(ScalingError::empty_input(
                "MetricRecords::from_records",
                "metric records cannot be empty",
            ));
        }
        Ok(Self(records))
    }

    /// Iterates over metric records.
    pub fn records(&self) -> impl ExactSizeIterator<Item = &MetricRecord> + '_ {
        self.0.iter()
    }

    /// Fits `loss = coefficient * compute^exponent` in log-log space.
    pub fn fit_power_law(&self) -> Result<ScalingFit, ScalingError> {
        ScalingFit::from_records(self)
    }
}

/// Tiny power-law fit over compute and validation loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingFit {
    coefficient: ScalingCoefficient,
    exponent: ScalingExponent,
    records_used: RecordCount,
}

impl ScalingFit {
    fn from_records(records: &MetricRecords) -> Result<Self, ScalingError> {
        if records.records().len() < 2 {
            return Err(ScalingError::degenerate_fit(
                "ScalingFit::from_records",
                "at least two records are required",
            ));
        }

        let mut count = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;

        for record in records.records() {
            let x = finite("log compute", record.run().compute_budget().as_f64().ln())?;
            let y = finite("log validation loss", record.validation_loss().ln())?;
            count += 1.0;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denominator = count * sum_xx - sum_x * sum_x;
        if denominator.abs() <= f64::EPSILON {
            return Err(ScalingError::degenerate_fit(
                "ScalingFit::from_records",
                "compute budgets must contain at least two distinct values",
            ));
        }

        let exponent = (count * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - exponent * sum_x) / count;
        let coefficient = intercept.exp();

        Ok(Self {
            coefficient: ScalingCoefficient::from_raw(coefficient)?,
            exponent: ScalingExponent::from_raw(exponent)?,
            records_used: RecordCount::try_from(records.records().len())?,
        })
    }

    /// Returns the fitted coefficient.
    pub fn coefficient(&self) -> ScalingCoefficient {
        self.coefficient
    }

    /// Returns the fitted exponent.
    pub fn exponent(&self) -> ScalingExponent {
        self.exponent
    }

    /// Returns the number of records used.
    pub fn records_used(&self) -> RecordCount {
        self.records_used
    }

    /// Predicts validation loss for a compute budget.
    pub fn predict_loss(
        &self,
        compute_budget: ComputeBudgetFlops,
    ) -> Result<ValidationLoss, ScalingError> {
        let predicted = self.coefficient.0 * compute_budget.as_f64().powf(self.exponent.as_f64());
        ValidationLoss::from_raw(predicted)
    }

    /// Compares an observed loss against the fitted forecast.
    pub fn forecast_error(
        &self,
        compute_budget: ComputeBudgetFlops,
        observed_loss: ValidationLoss,
    ) -> Result<LossDelta, ScalingError> {
        observed_loss - self.predict_loss(compute_budget)?
    }

    /// Scores a planned run with this fit.
    pub fn score_candidate(&self, run: TrainingRun) -> Result<ScalingCandidate, ScalingError> {
        ScalingCandidate::from_fit(*self, run)
    }

    /// Packages the fit with a human limitation note.
    pub fn report_with(&self, limitation: LimitationNote) -> ScalingReport {
        ScalingReport {
            fit: *self,
            limitation,
        }
    }
}

/// A planned run with a predicted validation loss.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalingCandidate {
    run: TrainingRun,
    predicted_loss: ValidationLoss,
}

impl ScalingCandidate {
    fn from_fit(fit: ScalingFit, run: TrainingRun) -> Result<Self, ScalingError> {
        let predicted_loss = fit.predict_loss(run.compute_budget())?;
        Ok(Self {
            run,
            predicted_loss,
        })
    }

    /// Returns the planned run.
    pub fn run(&self) -> &TrainingRun {
        &self.run
    }

    /// Returns the predicted validation loss.
    pub fn predicted_loss(&self) -> ValidationLoss {
        self.predicted_loss
    }

    /// Compares this candidate with a baseline candidate.
    pub fn compare_with_baseline(
        &self,
        baseline: &ScalingCandidate,
    ) -> Result<ScalingTradeoff, ScalingError> {
        ScalingTradeoff::between(baseline.clone(), self.clone())
    }
}

/// Direction of predicted-loss change for a candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictedLossTradeoff {
    /// Candidate predicts lower loss than the baseline.
    CandidateLower(LossDelta),
    /// Candidate predicts higher loss than the baseline.
    CandidateHigher(LossDelta),
    /// Candidate and baseline predict the same loss.
    SameLoss,
}

impl fmt::Display for PredictedLossTradeoff {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateLower(delta) => write!(formatter, "candidate lower loss ({delta})"),
            Self::CandidateHigher(delta) => write!(formatter, "candidate higher loss ({delta})"),
            Self::SameLoss => formatter.write_str("same predicted loss"),
        }
    }
}

/// Direction of compute-budget change for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTradeoff {
    /// Candidate uses less compute than the baseline.
    CandidateUsesLess(ComputeDelta),
    /// Candidate uses more compute than the baseline.
    CandidateUsesMore(ComputeDelta),
    /// Candidate and baseline use the same compute budget.
    SameCompute,
}

impl fmt::Display for ComputeTradeoff {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateUsesLess(delta) => {
                write!(formatter, "candidate uses less compute ({delta})")
            }
            Self::CandidateUsesMore(delta) => {
                write!(formatter, "candidate uses more compute ({delta})")
            }
            Self::SameCompute => formatter.write_str("same compute budget"),
        }
    }
}

/// Loss-first recommendation for a toy scaling decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingRecommendation {
    /// Prefer the candidate under a loss-first rule.
    PreferCandidate,
    /// Prefer the baseline under a loss-first rule.
    PreferBaseline,
    /// The fitted evidence cannot separate the two candidates.
    Inconclusive,
}

impl fmt::Display for ScalingRecommendation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreferCandidate => formatter.write_str("prefer candidate"),
            Self::PreferBaseline => formatter.write_str("prefer baseline"),
            Self::Inconclusive => formatter.write_str("inconclusive"),
        }
    }
}

/// Typed comparison between two scaling choices.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalingTradeoff {
    baseline: ScalingCandidate,
    candidate: ScalingCandidate,
    loss_tradeoff: PredictedLossTradeoff,
    compute_tradeoff: ComputeTradeoff,
    recommendation: ScalingRecommendation,
}

impl ScalingTradeoff {
    fn between(
        baseline: ScalingCandidate,
        candidate: ScalingCandidate,
    ) -> Result<Self, ScalingError> {
        let loss_delta = (candidate.predicted_loss() - baseline.predicted_loss())?;
        let loss_tradeoff =
            if candidate.predicted_loss().as_f64() < baseline.predicted_loss().as_f64() {
                PredictedLossTradeoff::CandidateLower(loss_delta)
            } else if candidate.predicted_loss().as_f64() > baseline.predicted_loss().as_f64() {
                PredictedLossTradeoff::CandidateHigher(loss_delta)
            } else {
                PredictedLossTradeoff::SameLoss
            };
        let compute_delta = ComputeDelta::between(
            candidate.run().compute_budget(),
            baseline.run().compute_budget(),
        );
        let compute_tradeoff = if candidate.run().compute_budget() < baseline.run().compute_budget()
        {
            ComputeTradeoff::CandidateUsesLess(compute_delta)
        } else if candidate.run().compute_budget() > baseline.run().compute_budget() {
            ComputeTradeoff::CandidateUsesMore(compute_delta)
        } else {
            ComputeTradeoff::SameCompute
        };
        let recommendation = match (loss_tradeoff, compute_tradeoff) {
            (PredictedLossTradeoff::CandidateLower(_), _) => ScalingRecommendation::PreferCandidate,
            (PredictedLossTradeoff::CandidateHigher(_), _) => ScalingRecommendation::PreferBaseline,
            (PredictedLossTradeoff::SameLoss, ComputeTradeoff::CandidateUsesLess(_)) => {
                ScalingRecommendation::PreferCandidate
            }
            (PredictedLossTradeoff::SameLoss, ComputeTradeoff::CandidateUsesMore(_)) => {
                ScalingRecommendation::PreferBaseline
            }
            (PredictedLossTradeoff::SameLoss, ComputeTradeoff::SameCompute) => {
                ScalingRecommendation::Inconclusive
            }
        };

        Ok(Self {
            baseline,
            candidate,
            loss_tradeoff,
            compute_tradeoff,
            recommendation,
        })
    }

    /// Returns the baseline candidate.
    pub fn baseline(&self) -> &ScalingCandidate {
        &self.baseline
    }

    /// Returns the compared candidate.
    pub fn candidate(&self) -> &ScalingCandidate {
        &self.candidate
    }

    /// Returns the predicted-loss direction.
    pub fn loss_tradeoff(&self) -> PredictedLossTradeoff {
        self.loss_tradeoff
    }

    /// Returns the compute-budget direction.
    pub fn compute_tradeoff(&self) -> ComputeTradeoff {
        self.compute_tradeoff
    }

    /// Returns the loss-first recommendation.
    pub fn recommendation(&self) -> ScalingRecommendation {
        self.recommendation
    }
}

impl fmt::Display for ScalingTradeoff {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} vs {} | {} | {} | {}",
            self.candidate.run().run_id(),
            self.baseline.run().run_id(),
            self.loss_tradeoff,
            self.compute_tradeoff,
            self.recommendation
        )
    }
}

impl fmt::Display for ScalingFit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "loss = {} * compute^{} using {} records",
            self.coefficient, self.exponent, self.records_used
        )
    }
}

/// Non-empty note that states what a tiny scaling result cannot prove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitationNote(String);

impl LimitationNote {
    fn from_owned(value: String) -> Result<Self, ScalingError> {
        if value.trim().is_empty() {
            return Err(ScalingError::empty_input(
                "LimitationNote::try_from",
                "limitation note cannot be empty",
            ));
        }
        Ok(Self(value))
    }
}

impl TryFrom<&str> for LimitationNote {
    type Error = ScalingError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for LimitationNote {
    type Error = ScalingError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for LimitationNote {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A learner-facing scaling report with an explicit limitation.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalingReport {
    fit: ScalingFit,
    limitation: LimitationNote,
}

impl ScalingReport {
    /// Returns the fitted curve.
    pub fn fit(&self) -> ScalingFit {
        self.fit
    }

    /// Returns the limitation note.
    pub fn limitation(&self) -> &LimitationNote {
        &self.limitation
    }
}

impl fmt::Display for ScalingReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}\nlimitation: {}", self.fit, self.limitation)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExperimentConfig, LayerCount, LimitationNote, MetricRecord, MetricRecords, ModelWidth,
        PredictedLossTradeoff, RunId, ScalingError, ScalingRecommendation, TokenCount,
        TrainingStep, ValidationLoss,
    };

    fn planned_record(
        run_id: RunId,
        width: ModelWidth,
        loss: ValidationLoss,
    ) -> Result<MetricRecord, ScalingError> {
        let config = ExperimentConfig::new(
            run_id,
            width,
            LayerCount::try_from(1)?,
            TokenCount::try_from(1_u64)?,
        );
        let run = config.plan_run(TrainingStep::try_from(1_u64)?)?;
        Ok(MetricRecord::from_run(run, loss))
    }

    #[test]
    fn experiment_config_estimates_parameters_and_compute() -> Result<(), ScalingError> {
        let config = ExperimentConfig::new(
            RunId::try_from("tiny")?,
            ModelWidth::try_from(4)?,
            LayerCount::try_from(2)?,
            TokenCount::try_from(10_u64)?,
        );
        let run = config.plan_run(TrainingStep::try_from(5_u64)?)?;

        assert_eq!(run.parameter_count().to_string(), "384");
        assert_eq!(run.compute_budget().to_string(), "23040 FLOPs");
        Ok(())
    }

    #[test]
    fn metric_records_reject_empty_sets() {
        let error = MetricRecords::from_records([]).err();
        assert!(matches!(error, Some(ScalingError::EmptyInput { .. })));
    }

    #[test]
    fn metric_record_preserves_the_producing_config() -> Result<(), ScalingError> {
        let record = planned_record(
            RunId::try_from("run-a")?,
            ModelWidth::try_from(2)?,
            ValidationLoss::try_from(1.0)?,
        )?;

        assert_eq!(record.run().run_id().to_string(), "run-a");
        assert_eq!(record.validation_loss().to_string(), "1.000000");
        Ok(())
    }

    #[test]
    fn power_law_fit_recovers_synthetic_exponent() -> Result<(), ScalingError> {
        let records = MetricRecords::from_records([
            planned_record(
                RunId::try_from("w1")?,
                ModelWidth::try_from(1)?,
                ValidationLoss::try_from(10.0 / 72.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w2")?,
                ModelWidth::try_from(2)?,
                ValidationLoss::try_from(10.0 / 288.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w4")?,
                ModelWidth::try_from(4)?,
                ValidationLoss::try_from(10.0 / 1152.0_f64.sqrt())?,
            )?,
        ])?;
        let fit = records.fit_power_law()?;

        assert_eq!(fit.exponent().to_string(), "-0.500000");
        assert_eq!(fit.records_used().to_string(), "3");
        Ok(())
    }

    #[test]
    fn forecast_error_reports_observed_minus_predicted() -> Result<(), ScalingError> {
        let records = MetricRecords::from_records([
            planned_record(
                RunId::try_from("w1")?,
                ModelWidth::try_from(1)?,
                ValidationLoss::try_from(10.0 / 72.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w2")?,
                ModelWidth::try_from(2)?,
                ValidationLoss::try_from(10.0 / 288.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w4")?,
                ModelWidth::try_from(4)?,
                ValidationLoss::try_from(10.0 / 1152.0_f64.sqrt())?,
            )?,
        ])?;
        let fit = records.fit_power_law()?;
        let future = ExperimentConfig::new(
            RunId::try_from("future")?,
            ModelWidth::try_from(8)?,
            LayerCount::try_from(1)?,
            TokenCount::try_from(1_u64)?,
        )
        .plan_run(TrainingStep::try_from(1_u64)?)?;
        let predicted = fit.predict_loss(future.compute_budget())?;
        let error = fit.forecast_error(future.compute_budget(), predicted)?;

        assert_eq!(error.to_string(), "+0.000000");
        Ok(())
    }

    #[test]
    fn report_requires_a_limitation_note() -> Result<(), ScalingError> {
        let records = MetricRecords::from_records([
            planned_record(
                RunId::try_from("w1")?,
                ModelWidth::try_from(1)?,
                ValidationLoss::try_from(10.0 / 72.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w2")?,
                ModelWidth::try_from(2)?,
                ValidationLoss::try_from(10.0 / 288.0_f64.sqrt())?,
            )?,
        ])?;
        let fit = records.fit_power_law()?;
        let report = fit.report_with(LimitationNote::try_from(
            "two tiny points show direction, not a deployment law",
        )?);

        assert!(report.to_string().contains("limitation:"));
        Ok(())
    }

    #[test]
    fn tradeoff_prefers_lower_predicted_loss() -> Result<(), ScalingError> {
        let records = MetricRecords::from_records([
            planned_record(
                RunId::try_from("w1")?,
                ModelWidth::try_from(1)?,
                ValidationLoss::try_from(10.0 / 72.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w2")?,
                ModelWidth::try_from(2)?,
                ValidationLoss::try_from(10.0 / 288.0_f64.sqrt())?,
            )?,
            planned_record(
                RunId::try_from("w4")?,
                ModelWidth::try_from(4)?,
                ValidationLoss::try_from(10.0 / 1152.0_f64.sqrt())?,
            )?,
        ])?;
        let fit = records.fit_power_law()?;
        let baseline = fit.score_candidate(
            ExperimentConfig::new(
                RunId::try_from("baseline")?,
                ModelWidth::try_from(4)?,
                LayerCount::try_from(1)?,
                TokenCount::try_from(1_u64)?,
            )
            .plan_run(TrainingStep::try_from(1_u64)?)?,
        )?;
        let candidate = fit.score_candidate(
            ExperimentConfig::new(
                RunId::try_from("candidate")?,
                ModelWidth::try_from(8)?,
                LayerCount::try_from(1)?,
                TokenCount::try_from(1_u64)?,
            )
            .plan_run(TrainingStep::try_from(1_u64)?)?,
        )?;
        let tradeoff = candidate.compare_with_baseline(&baseline)?;

        assert_eq!(
            tradeoff.recommendation(),
            ScalingRecommendation::PreferCandidate
        );
        assert!(matches!(
            tradeoff.loss_tradeoff(),
            PredictedLossTradeoff::CandidateLower(_)
        ));
        assert!(tradeoff.to_string().contains("candidate"));
        Ok(())
    }
}
