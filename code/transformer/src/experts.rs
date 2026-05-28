//! Typed expert routing for attention-alternative lessons.
//!
//! Mixture-of-experts starts with a routing decision: this token should be sent
//! to that expert. This module keeps the first version deliberately small. It
//! teaches the typed boundary before teaching distributed expert execution.

use std::{fmt, ops::Mul};

use crate::{
    error::ModelError,
    math::{DenseVector, ModelScalar, VectorIndex},
    transformer::FeedForward,
    types::{TokenEmbedding, TokenIndex},
};

/// Number of experts available to a router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpertCount(usize);

impl ExpertCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ExpertCount {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(ModelError::empty_input(
                "ExpertCount::try_from",
                "expert count must be greater than zero",
            ));
        }

        Ok(Self(value))
    }
}

impl fmt::Display for ExpertCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Zero-based index of one expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpertIndex(usize);

impl ExpertIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ExpertIndex {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for ExpertIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Maximum number of tokens assigned to one expert in a tiny routing batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpertCapacity(usize);

impl ExpertCapacity {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ExpertCapacity {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(ModelError::empty_input(
                "ExpertCapacity::try_from",
                "expert capacity must be greater than zero",
            ));
        }

        Ok(Self(value))
    }
}

impl fmt::Display for ExpertCapacity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of routed tokens assigned to one expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpertLoad(usize);

impl ExpertLoad {
    fn zero() -> Self {
        Self(0)
    }

    fn increment(self) -> Result<Self, ModelError> {
        let value = self.0.checked_add(1).ok_or(ModelError::numerical_issue(
            "ExpertLoad::increment",
            "expert load exceeded usize",
        ))?;

        Ok(Self(value))
    }
}

impl fmt::Display for ExpertLoad {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Whether one expert's load fits the requested capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLoadStatus {
    /// The expert has no more routed tokens than the configured capacity.
    WithinCapacity,
    /// The expert has more routed tokens than the configured capacity.
    OverCapacity,
}

impl fmt::Display for ExpertLoadStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WithinCapacity => formatter.write_str("within-capacity"),
            Self::OverCapacity => formatter.write_str("over-capacity"),
        }
    }
}

/// One finite router score for one expert.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ExpertScore(ModelScalar);

impl ExpertScore {
    /// Creates an expert score from an already checked model scalar.
    pub fn new(value: ModelScalar) -> Self {
        Self(value)
    }
}

impl From<ModelScalar> for ExpertScore {
    fn from(value: ModelScalar) -> Self {
        Self::new(value)
    }
}

impl TryFrom<f32> for ExpertScore {
    type Error = ModelError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Ok(Self(ModelScalar::try_from(value)?))
    }
}

impl fmt::Display for ExpertScore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Router scores for all experts for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertScores(DenseVector);

impl ExpertScores {
    /// Creates a checked score vector for one token.
    pub fn new(scores: impl IntoIterator<Item = ExpertScore>) -> Result<Self, ModelError> {
        Ok(Self(DenseVector::new(
            scores.into_iter().map(|score| score.0),
        )?))
    }

    /// Returns the number of expert scores.
    pub fn expert_count(&self) -> ExpertCount {
        ExpertCount(self.0.len().as_usize())
    }

    /// Iterates over the checked scores.
    pub fn scores(&self) -> impl ExactSizeIterator<Item = ExpertScore> + '_ {
        self.0.values().map(ExpertScore)
    }

    /// Reads the score assigned to one expert.
    pub fn score(&self, index: ExpertIndex) -> Result<ExpertScore, ModelError> {
        Ok(ExpertScore(
            self.0.component(VectorIndex::try_from(index.as_usize())?)?,
        ))
    }

    /// Returns the highest-scoring expert, keeping the first expert on ties.
    pub fn top_choice(&self) -> Result<ExpertChoice, ModelError> {
        let mut best_index = ExpertIndex::try_from(0)?;
        let mut best_score = self.score(best_index)?;

        for (position, score) in self.scores().enumerate() {
            if score > best_score {
                best_index = ExpertIndex::try_from(position)?;
                best_score = score;
            }
        }

        Ok(ExpertChoice::new(best_index, best_score))
    }
}

/// The expert selected by a router for one token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertChoice {
    expert_index: ExpertIndex,
    score: ExpertScore,
}

impl ExpertChoice {
    /// Creates a typed expert choice.
    pub fn new(expert_index: ExpertIndex, score: ExpertScore) -> Self {
        Self {
            expert_index,
            score,
        }
    }

    /// Returns the selected expert.
    pub fn expert_index(self) -> ExpertIndex {
        self.expert_index
    }

    /// Returns the score that selected this expert.
    pub fn score(self) -> ExpertScore {
        self.score
    }
}

/// A token-to-expert routing decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRoute {
    token_index: TokenIndex,
    choice: ExpertChoice,
}

impl ExpertRoute {
    /// Creates a token routing decision.
    pub fn new(token_index: TokenIndex, choice: ExpertChoice) -> Self {
        Self {
            token_index,
            choice,
        }
    }

    /// Returns the token being routed.
    pub fn token_index(self) -> TokenIndex {
        self.token_index
    }

    /// Returns the selected expert for that token.
    pub fn choice(self) -> ExpertChoice {
        self.choice
    }
}

/// A top-1 expert router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopExpertRouter {
    expert_count: ExpertCount,
}

impl TopExpertRouter {
    /// Creates a top-1 router for a fixed expert set.
    pub fn new(expert_count: ExpertCount) -> Self {
        Self { expert_count }
    }

    /// Returns the expert count this router expects.
    pub fn expert_count(self) -> ExpertCount {
        self.expert_count
    }

    /// Selects the top expert for one token's router scores.
    pub fn choose(&self, scores: &ExpertScores) -> Result<ExpertChoice, ModelError> {
        if scores.expert_count() != self.expert_count {
            return Err(ModelError::dimension_mismatch(
                "TopExpertRouter::choose",
                "router",
                vec![self.expert_count.as_usize()],
                "expert scores",
                vec![scores.expert_count().as_usize()],
                "router scores must provide exactly one score per expert",
            ));
        }

        scores.top_choice()
    }

    /// Routes one token to the highest-scoring expert.
    pub fn route(
        &self,
        token_index: TokenIndex,
        scores: &ExpertScores,
    ) -> Result<ExpertRoute, ModelError> {
        Ok(ExpertRoute::new(token_index, self.choose(scores)?))
    }
}

impl<'b> Mul<&'b ExpertScores> for &TopExpertRouter {
    type Output = Result<ExpertChoice, ModelError>;

    fn mul(self, right: &'b ExpertScores) -> Self::Output {
        self.choose(right)
    }
}

/// One position-wise feed-forward expert.
#[derive(Debug, Clone)]
pub struct FeedForwardExpert {
    feed_forward: FeedForward,
}

impl FeedForwardExpert {
    /// Creates an expert from a checked feed-forward map.
    pub fn new(feed_forward: FeedForward) -> Self {
        Self { feed_forward }
    }

    /// Applies the expert to one token.
    pub fn forward_token(&self, token: &TokenEmbedding) -> Result<TokenEmbedding, ModelError> {
        self.feed_forward.forward_token(token)
    }
}

/// A non-empty bank of feed-forward experts.
#[derive(Debug, Clone)]
pub struct ExpertBank {
    experts: Vec<FeedForwardExpert>,
}

impl ExpertBank {
    /// Creates a checked expert bank.
    pub fn new(experts: impl IntoIterator<Item = FeedForwardExpert>) -> Result<Self, ModelError> {
        let experts = experts.into_iter().collect::<Vec<_>>();
        if experts.is_empty() {
            return Err(ModelError::empty_input(
                "ExpertBank::new",
                "expert bank must contain at least one expert",
            ));
        }

        Ok(Self { experts })
    }

    /// Returns the number of experts in the bank.
    pub fn expert_count(&self) -> ExpertCount {
        ExpertCount(self.experts.len())
    }

    /// Applies the routed expert to one token.
    pub fn forward_routed_token(
        &self,
        token: &TokenEmbedding,
        route: ExpertRoute,
    ) -> Result<TokenEmbedding, ModelError> {
        let expected = self.expert_count();
        let index = route.choice().expert_index();
        let expert = self
            .experts
            .get(index.as_usize())
            .ok_or(ModelError::dimension_mismatch(
                "ExpertBank::forward_routed_token",
                "expert bank",
                vec![expected.as_usize()],
                "route",
                vec![index.as_usize() + 1],
                "route must select an expert that exists in the bank",
            ))?;

        expert.forward_token(token)
    }

    /// Counts routed tokens per expert for a small teaching batch.
    pub fn load_report(
        &self,
        capacity: ExpertCapacity,
        routes: impl IntoIterator<Item = ExpertRoute>,
    ) -> Result<ExpertLoadReport, ModelError> {
        let expected = self.expert_count();
        let mut loads = vec![ExpertLoad::zero(); expected.as_usize()];

        for route in routes {
            let index = route.choice().expert_index();
            let load = loads
                .get_mut(index.as_usize())
                .ok_or(ModelError::dimension_mismatch(
                    "ExpertBank::load_report",
                    "expert bank",
                    vec![expected.as_usize()],
                    "route",
                    vec![index.as_usize() + 1],
                    "every route must select an expert that exists in the bank",
                ))?;
            *load = load.increment()?;
        }

        Ok(ExpertLoadReport { capacity, loads })
    }
}

/// Per-expert load evidence for a routed teaching batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLoadReport {
    capacity: ExpertCapacity,
    loads: Vec<ExpertLoad>,
}

impl ExpertLoadReport {
    /// Returns the capacity used to classify the loads.
    pub fn capacity(&self) -> ExpertCapacity {
        self.capacity
    }

    /// Iterates over per-expert loads in expert-index order.
    pub fn loads(&self) -> impl ExactSizeIterator<Item = ExpertLoad> + '_ {
        self.loads.iter().copied()
    }

    /// Returns one expert's load.
    pub fn load_for(&self, expert_index: ExpertIndex) -> Result<ExpertLoad, ModelError> {
        self.loads
            .get(expert_index.as_usize())
            .copied()
            .ok_or(ModelError::dimension_mismatch(
                "ExpertLoadReport::load_for",
                "load report",
                vec![self.loads.len()],
                "expert index",
                vec![expert_index.as_usize() + 1],
                "expert index must exist in the load report",
            ))
    }

    /// Classifies one expert's load against the report capacity.
    pub fn status_for(&self, expert_index: ExpertIndex) -> Result<ExpertLoadStatus, ModelError> {
        if self.load_for(expert_index)? <= ExpertLoad(self.capacity.as_usize()) {
            return Ok(ExpertLoadStatus::WithinCapacity);
        }

        Ok(ExpertLoadStatus::OverCapacity)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExpertBank, ExpertCapacity, ExpertCount, ExpertScore, ExpertScores, FeedForwardExpert,
        TopExpertRouter,
    };
    use crate::{
        DenseMatrix, DenseVector, FeedForward, FeedForwardLayer1, FeedForwardLayer2,
        FeedForwardProjection1, FeedForwardProjection2, ModelError, ModelScalar, ProjectionBias,
        TokenEmbedding, TokenIndex, VectorLength,
    };

    fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
    }

    fn diagonal_expert(scale: ModelScalar) -> Result<FeedForwardExpert, ModelError> {
        let zero = ModelScalar::try_from(0.0)?;
        let layer1 = FeedForwardLayer1::new(
            FeedForwardProjection1::from_matrix(DenseMatrix::from_rows([
                [scale, zero],
                [zero, scale],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?;
        let layer2 = FeedForwardLayer2::new(
            FeedForwardProjection2::from_matrix(DenseMatrix::from_rows([
                [scale, zero],
                [zero, scale],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?;

        Ok(FeedForwardExpert::new(FeedForward::new(layer1, layer2)?))
    }

    #[test]
    fn top_router_selects_highest_scoring_expert() -> Result<(), ModelError> {
        let router = TopExpertRouter::new(ExpertCount::try_from(3)?);
        let scores = ExpertScores::new([
            ExpertScore::try_from(0.2)?,
            ExpertScore::try_from(1.5)?,
            ExpertScore::try_from(0.7)?,
        ])?;

        let choice = (&router * &scores)?;

        assert_eq!(choice.expert_index().to_string(), "1");
        assert_eq!(choice.score().to_string(), "1.5");
        Ok(())
    }

    #[test]
    fn top_router_keeps_first_expert_when_scores_tie() -> Result<(), ModelError> {
        let router = TopExpertRouter::new(ExpertCount::try_from(2)?);
        let scores = ExpertScores::new([ExpertScore::try_from(1.0)?, ExpertScore::try_from(1.0)?])?;

        let route = router.route(TokenIndex::try_from(0)?, &scores)?;

        assert_eq!(route.token_index().to_string(), "0");
        assert_eq!(route.choice().expert_index().to_string(), "0");
        Ok(())
    }

    #[test]
    fn router_rejects_score_count_mismatch() -> Result<(), ModelError> {
        let router = TopExpertRouter::new(ExpertCount::try_from(3)?);
        let scores = ExpertScores::new([ExpertScore::try_from(0.2)?, ExpertScore::try_from(1.5)?])?;

        assert!(matches!(
            router.choose(&scores),
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn expert_count_rejects_zero() {
        assert!(matches!(
            ExpertCount::try_from(0),
            Err(ModelError::EmptyInput { .. })
        ));
    }

    #[test]
    fn expert_bank_applies_the_routed_expert() -> Result<(), ModelError> {
        let bank = ExpertBank::new([
            diagonal_expert(ModelScalar::try_from(1.0)?)?,
            diagonal_expert(ModelScalar::try_from(2.0)?)?,
        ])?;
        let router = TopExpertRouter::new(bank.expert_count());
        let scores = ExpertScores::new([ExpertScore::try_from(0.2)?, ExpertScore::try_from(1.5)?])?;
        let token = TokenEmbedding::from_vector(DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.5)?,
        ])?);

        let route = router.route(TokenIndex::try_from(0)?, &scores)?;
        let output = bank.forward_routed_token(&token, route)?;

        let values = output
            .values()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        assert_eq!(values, ["4", "2"]);
        Ok(())
    }

    #[test]
    fn expert_bank_rejects_missing_routed_expert() -> Result<(), ModelError> {
        let bank = ExpertBank::new([diagonal_expert(ModelScalar::try_from(1.0)?)?])?;
        let router = TopExpertRouter::new(ExpertCount::try_from(2)?);
        let scores = ExpertScores::new([ExpertScore::try_from(0.2)?, ExpertScore::try_from(1.5)?])?;
        let token = TokenEmbedding::from_vector(DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.5)?,
        ])?);

        let route = router.route(TokenIndex::try_from(0)?, &scores)?;

        assert!(matches!(
            bank.forward_routed_token(&token, route),
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn expert_bank_reports_capacity_status() -> Result<(), ModelError> {
        let bank = ExpertBank::new([
            diagonal_expert(ModelScalar::try_from(1.0)?)?,
            diagonal_expert(ModelScalar::try_from(2.0)?)?,
        ])?;
        let router = TopExpertRouter::new(bank.expert_count());
        let scores = ExpertScores::new([ExpertScore::try_from(0.2)?, ExpertScore::try_from(1.5)?])?;

        let first = router.route(TokenIndex::try_from(0)?, &scores)?;
        let second = router.route(TokenIndex::try_from(1)?, &scores)?;
        let report = bank.load_report(ExpertCapacity::try_from(1)?, [first, second])?;

        assert_eq!(
            report.load_for(first.choice().expert_index())?.to_string(),
            "2"
        );
        assert_eq!(
            report
                .status_for(first.choice().expert_index())?
                .to_string(),
            "over-capacity"
        );
        Ok(())
    }

    #[test]
    fn expert_capacity_rejects_zero() {
        assert!(matches!(
            ExpertCapacity::try_from(0),
            Err(ModelError::EmptyInput { .. })
        ));
    }
}
