//! Typed inference primitives for the CS336 Rust equivalent track.
//!
//! This crate keeps inference concrete and inspectable:
//!
//! ```text
//! PromptTokens + DecodeRequest -> DecodeTrace
//! ContextTokens + TokenId -> ContextTokens
//! KvCache + KvCacheEntry -> KvCache
//! LatencyMillis + LatencyMillis -> LatencyMillis
//! ```
//!
//! Raw learner literals enter through `TryFrom` adapters. The public teaching
//! path then speaks in semantic values: token IDs, sampling controls, context
//! windows, decode steps, KV-cache entries, and latency budgets.

pub mod error;

use std::{
    cmp::Ordering,
    collections::BTreeSet,
    fmt,
    ops::{Add, Div, Mul},
};

use error::InferenceError;

pub use error::InferenceError as Error;

fn validate_nonempty(
    value: &str,
    operation: &'static str,
    details: &'static str,
) -> Result<(), InferenceError> {
    if value.trim().is_empty() {
        return Err(InferenceError::empty_input(operation, details));
    }

    Ok(())
}

fn nonzero_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, InferenceError> {
    if value == 0 {
        return Err(InferenceError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u64(
    role: &'static str,
    operation: &'static str,
    value: u64,
) -> Result<u64, InferenceError> {
    if value == 0 {
        return Err(InferenceError::empty_input(operation, role));
    }

    Ok(value)
}

fn finite(role: &'static str, value: f64) -> Result<f64, InferenceError> {
    if !value.is_finite() {
        return Err(InferenceError::non_finite_value(role, value));
    }

    Ok(value)
}

fn checked_usize_add(
    operation: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, InferenceError> {
    left.checked_add(right).ok_or(InferenceError::overflow(
        operation,
        "usize addition overflowed",
    ))
}

fn checked_u64_add(operation: &'static str, left: u64, right: u64) -> Result<u64, InferenceError> {
    left.checked_add(right).ok_or(InferenceError::overflow(
        operation,
        "u64 addition overflowed",
    ))
}

fn checked_u64_mul(operation: &'static str, left: u64, right: u64) -> Result<u64, InferenceError> {
    left.checked_mul(right).ok_or(InferenceError::overflow(
        operation,
        "u64 multiplication overflowed",
    ))
}

macro_rules! nonempty_text_type {
    ($name:ident, $doc:literal, $operation:literal, $details:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            fn from_owned(value: String) -> Result<Self, InferenceError> {
                validate_nonempty(&value, $operation, $details)?;
                Ok(Self(value))
            }

            fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<&str> for $name {
            type Error = InferenceError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::from_owned(value.to_owned())
            }
        }

        impl TryFrom<String> for $name {
            type Error = InferenceError;

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
    PromptText,
    "Human-facing prompt text before tokenization.",
    "PromptText::try_from",
    "prompt text cannot be empty"
);
nonempty_text_type!(
    TokenText,
    "Human-facing text attached to one toy vocabulary token.",
    "TokenText::try_from",
    "token text cannot be empty"
);
nonempty_text_type!(
    GeneratedText,
    "Human-facing decoded text produced by a generation trace.",
    "GeneratedText::try_from",
    "generated text cannot be empty"
);

/// Raw token index before it is checked against a vocabulary size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenIndex(usize);

impl TokenIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TokenIndex {
    type Error = InferenceError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for TokenIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Size of a fixed toy vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VocabularySize(usize);

impl VocabularySize {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for VocabularySize {
    type Error = InferenceError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_usize(
            "vocabulary size must be greater than zero",
            "VocabularySize::try_from",
            value,
        )?))
    }
}

impl fmt::Display for VocabularySize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Token identity after validation against a vocabulary size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(TokenIndex);

impl TokenId {
    /// Creates a token identity after checking it is inside the vocabulary.
    pub fn new(index: TokenIndex, vocabulary_size: VocabularySize) -> Result<Self, InferenceError> {
        if index.as_usize() >= vocabulary_size.as_usize() {
            return Err(InferenceError::count_out_of_range(
                "token id",
                "0..vocabulary_size",
                index.as_usize(),
            ));
        }

        Ok(Self(index))
    }

    fn as_usize(self) -> usize {
        self.0.as_usize()
    }

    fn fits_in(self, vocabulary_size: VocabularySize) -> TokenVocabularyFit {
        if self.as_usize() < vocabulary_size.as_usize() {
            TokenVocabularyFit::Fits
        } else {
            TokenVocabularyFit::OutOfVocabulary
        }
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "token#{}", self.as_usize())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenVocabularyFit {
    Fits,
    OutOfVocabulary,
}

/// Non-empty count of prompt or context tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SequenceTokenCount(usize);

impl SequenceTokenCount {
    fn from_raw(operation: &'static str, value: usize) -> Result<Self, InferenceError> {
        Ok(Self(nonzero_usize(
            "sequence token count must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for SequenceTokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Count of generated tokens. This may be zero before decoding starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GeneratedTokenCount(usize);

impl GeneratedTokenCount {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }

    fn is_less_than(self, right: MaxNewTokens) -> DecodeLoopState {
        if self.as_usize() < right.as_usize() {
            DecodeLoopState::Continue
        } else {
            DecodeLoopState::Finished
        }
    }
}

impl fmt::Display for GeneratedTokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeLoopState {
    Continue,
    Finished,
}

/// Non-zero maximum number of new tokens to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxNewTokens(usize);

impl MaxNewTokens {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for MaxNewTokens {
    type Error = InferenceError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_usize(
            "max new tokens must be greater than zero",
            "MaxNewTokens::try_from",
            value,
        )?))
    }
}

impl fmt::Display for MaxNewTokens {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Non-zero number of token slots in the inference context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContextWindow(usize);

impl ContextWindow {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ContextWindow {
    type Error = InferenceError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_usize(
            "context window must be greater than zero",
            "ContextWindow::try_from",
            value,
        )?))
    }
}

impl fmt::Display for ContextWindow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} tokens", self.0)
    }
}

/// Total token budget implied by a prompt and generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenBudget(usize);

impl TokenBudget {
    fn from_raw(operation: &'static str, value: usize) -> Result<Self, InferenceError> {
        Ok(Self(nonzero_usize(
            "token budget must be greater than zero",
            operation,
            value,
        )?))
    }

    fn fits_in(self, context_window: ContextWindow) -> ContextFit {
        if self.0 <= context_window.as_usize() {
            ContextFit::Fits
        } else {
            ContextFit::TooLong
        }
    }
}

impl Add<MaxNewTokens> for SequenceTokenCount {
    type Output = Result<TokenBudget, InferenceError>;

    fn add(self, right: MaxNewTokens) -> Self::Output {
        let total =
            checked_usize_add("SequenceTokenCount::add", self.as_usize(), right.as_usize())?;
        TokenBudget::from_raw("SequenceTokenCount::add", total)
    }
}

impl fmt::Display for TokenBudget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} tokens", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextFit {
    Fits,
    TooLong,
}

/// Prompt tokens after tokenization and vocabulary validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTokens {
    tokens: Vec<TokenId>,
}

impl PromptTokens {
    /// Builds non-empty prompt tokens from already validated token IDs.
    pub fn from_tokens(tokens: impl IntoIterator<Item = TokenId>) -> Result<Self, InferenceError> {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        SequenceTokenCount::from_raw("PromptTokens::from_tokens", tokens.len())?;
        Ok(Self { tokens })
    }

    /// Returns the number of tokens in the prompt.
    pub fn count(&self) -> SequenceTokenCount {
        SequenceTokenCount(self.tokens.len())
    }

    /// Iterates over prompt token IDs.
    pub fn tokens(&self) -> impl ExactSizeIterator<Item = TokenId> + '_ {
        self.tokens.iter().copied()
    }

    fn all_fit_in(&self, vocabulary_size: VocabularySize) -> TokenVocabularyFit {
        if self
            .tokens
            .iter()
            .all(|token| token.fits_in(vocabulary_size) == TokenVocabularyFit::Fits)
        {
            TokenVocabularyFit::Fits
        } else {
            TokenVocabularyFit::OutOfVocabulary
        }
    }
}

/// Generated tokens accumulated during autoregressive decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedTokens {
    tokens: Vec<TokenId>,
}

impl GeneratedTokens {
    /// Creates an empty generated-token sequence before decoding starts.
    pub fn empty() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Returns the number of generated tokens.
    pub fn count(&self) -> GeneratedTokenCount {
        GeneratedTokenCount::from_raw(self.tokens.len())
    }

    /// Iterates over generated token IDs.
    pub fn tokens(&self) -> impl ExactSizeIterator<Item = TokenId> + '_ {
        self.tokens.iter().copied()
    }
}

impl Add<TokenId> for GeneratedTokens {
    type Output = Result<GeneratedTokens, InferenceError>;

    fn add(mut self, token: TokenId) -> Self::Output {
        self.tokens.push(token);
        Ok(self)
    }
}

/// Active context tokens used to predict the next token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextTokens {
    tokens: Vec<TokenId>,
    window: ContextWindow,
}

impl ContextTokens {
    fn from_prompt(prompt: &PromptTokens, window: ContextWindow) -> Result<Self, InferenceError> {
        let context = Self {
            tokens: prompt.tokens().collect(),
            window,
        };

        if context.tokens.len() > window.as_usize() {
            return Err(InferenceError::context_overflow(
                "ContextTokens::from_prompt",
                "prompt tokens exceed the context window",
            ));
        }

        Ok(context)
    }

    /// Returns the number of active context tokens.
    pub fn count(&self) -> SequenceTokenCount {
        SequenceTokenCount(self.tokens.len())
    }

    /// Returns the most recent token in the active context.
    pub fn last_token(&self) -> Result<TokenId, InferenceError> {
        self.tokens
            .last()
            .copied()
            .ok_or(InferenceError::empty_input(
                "ContextTokens::last_token",
                "context tokens cannot be empty",
            ))
    }
}

impl Add<TokenId> for ContextTokens {
    type Output = Result<ContextTokens, InferenceError>;

    fn add(mut self, token: TokenId) -> Self::Output {
        if self.tokens.len() >= self.window.as_usize() {
            return Err(InferenceError::context_overflow(
                "ContextTokens::add",
                "adding the next token would exceed the context window",
            ));
        }

        self.tokens.push(token);
        Ok(self)
    }
}

/// Non-zero decode step. Step one is the first generated token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DecodeStep(usize);

impl DecodeStep {
    fn from_generated_count(count: GeneratedTokenCount) -> Result<Self, InferenceError> {
        Ok(Self(nonzero_usize(
            "decode step must be greater than zero",
            "DecodeStep::from_generated_count",
            checked_usize_add("DecodeStep::from_generated_count", count.as_usize(), 1)?,
        )?))
    }
}

impl fmt::Display for DecodeStep {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Non-zero temperature used to scale logits before deterministic top-k choice.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Temperature(f64);

impl Temperature {
    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Temperature {
    type Error = InferenceError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let value = finite("temperature", value)?;
        if value <= 0.0 {
            return Err(InferenceError::out_of_range("temperature", "> 0", value));
        }

        Ok(Self(value))
    }
}

impl fmt::Display for Temperature {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.3}", self.0)
    }
}

/// Non-zero top-k candidate limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TopK(usize);

impl TopK {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TopK {
    type Error = InferenceError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_usize(
            "top-k must be greater than zero",
            "TopK::try_from",
            value,
        )?))
    }
}

impl fmt::Display for TopK {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Finite model score for one candidate next token.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Logit(f64);

impl Logit {
    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Logit {
    type Error = InferenceError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Ok(Self(finite("logit", value)?))
    }
}

impl fmt::Display for Logit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.3}", self.0)
    }
}

/// Logit after temperature scaling.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AdjustedLogit(f64);

impl AdjustedLogit {
    fn from_raw(value: f64) -> Result<Self, InferenceError> {
        Ok(Self(finite("adjusted logit", value)?))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl Div<Temperature> for Logit {
    type Output = Result<AdjustedLogit, InferenceError>;

    fn div(self, right: Temperature) -> Self::Output {
        AdjustedLogit::from_raw(self.as_f64() / right.as_f64())
    }
}

impl fmt::Display for AdjustedLogit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.3}", self.0)
    }
}

/// One candidate next token and its model score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedToken {
    token_id: TokenId,
    logit: Logit,
}

impl RankedToken {
    /// Creates one scored next-token candidate.
    pub fn new(token_id: TokenId, logit: Logit) -> Self {
        Self { token_id, logit }
    }

    /// Returns the token identity.
    pub fn token_id(&self) -> TokenId {
        self.token_id
    }

    /// Returns the token logit.
    pub fn logit(&self) -> Logit {
        self.logit
    }
}

/// Sorted next-token candidates for a single decode step.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenRankings {
    candidates: Vec<RankedToken>,
}

impl TokenRankings {
    /// Creates non-empty next-token rankings from scored candidates.
    pub fn from_candidates(
        candidates: impl IntoIterator<Item = RankedToken>,
    ) -> Result<Self, InferenceError> {
        let mut candidates = candidates.into_iter().collect::<Vec<_>>();
        SequenceTokenCount::from_raw("TokenRankings::from_candidates", candidates.len())?;
        candidates.sort_by(compare_ranked_tokens);
        Ok(Self { candidates })
    }

    /// Iterates over ranked token candidates.
    pub fn candidates(&self) -> impl ExactSizeIterator<Item = &RankedToken> + '_ {
        self.candidates.iter()
    }

    fn best(&self) -> Result<TokenId, InferenceError> {
        self.candidates
            .first()
            .map(RankedToken::token_id)
            .ok_or(InferenceError::empty_input(
                "TokenRankings::best",
                "rankings cannot be empty",
            ))
    }

    fn all_fit_in(&self, vocabulary_size: VocabularySize) -> TokenVocabularyFit {
        if self.candidates.iter().all(|candidate| {
            candidate.token_id().fits_in(vocabulary_size) == TokenVocabularyFit::Fits
        }) {
            TokenVocabularyFit::Fits
        } else {
            TokenVocabularyFit::OutOfVocabulary
        }
    }
}

fn compare_ranked_tokens(left: &RankedToken, right: &RankedToken) -> Ordering {
    match right.logit().as_f64().partial_cmp(&left.logit().as_f64()) {
        Some(Ordering::Equal) => left.token_id().as_usize().cmp(&right.token_id().as_usize()),
        Some(ordering) => ordering,
        None => Ordering::Equal,
    }
}

/// Sampling strategy for selecting one token from model rankings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMode {
    /// Choose the highest-ranked token.
    Greedy,
    /// Restrict attention to top-k candidates and scale logits by temperature.
    TopK {
        /// Candidate limit.
        k: TopK,
        /// Positive logit temperature.
        temperature: Temperature,
    },
}

impl SamplingMode {
    /// Selects one token from sorted model rankings.
    pub fn select(&self, rankings: &TokenRankings) -> Result<TokenId, InferenceError> {
        match self {
            Self::Greedy => rankings.best(),
            Self::TopK { k, temperature } => {
                let mut best: Option<(TokenId, AdjustedLogit)> = None;

                for candidate in rankings.candidates().take(k.as_usize()) {
                    let adjusted = (candidate.logit() / *temperature)?;
                    best = match best {
                        None => Some((candidate.token_id(), adjusted)),
                        Some((current_token, current_score)) => {
                            if adjusted.as_f64() > current_score.as_f64() {
                                Some((candidate.token_id(), adjusted))
                            } else {
                                Some((current_token, current_score))
                            }
                        }
                    };
                }

                best.map(|(token, _score)| token)
                    .ok_or(InferenceError::empty_input(
                        "SamplingMode::select",
                        "top-k candidate set cannot be empty",
                    ))
            }
        }
    }
}

impl fmt::Display for SamplingMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greedy => formatter.write_str("greedy"),
            Self::TopK { k, temperature } => write!(formatter, "top-{k} @ temp {temperature}"),
        }
    }
}

/// One transition rule in a tiny deterministic next-token model.
#[derive(Debug, Clone, PartialEq)]
pub struct NextTokenRule {
    previous: TokenId,
    candidates: TokenRankings,
}

impl NextTokenRule {
    /// Creates a rule that maps the previous token to candidate next tokens.
    pub fn new(previous: TokenId, candidates: TokenRankings) -> Self {
        Self {
            previous,
            candidates,
        }
    }
}

/// Provider-agnostic contract for a model that can rank next tokens.
pub trait NextTokenModel {
    /// Returns the vocabulary size used by the model.
    fn vocabulary_size(&self) -> VocabularySize;

    /// Produces next-token candidates from the active context.
    fn next_token_candidates(
        &self,
        context: &ContextTokens,
    ) -> Result<TokenRankings, InferenceError>;
}

/// Tiny deterministic next-token model for inference pedagogy.
#[derive(Debug, Clone, PartialEq)]
pub struct ToyNextTokenModel {
    vocabulary_size: VocabularySize,
    fallback: TokenRankings,
    rules: Vec<NextTokenRule>,
}

impl ToyNextTokenModel {
    /// Creates a toy model with explicit fallback and transition rules.
    pub fn new(
        vocabulary_size: VocabularySize,
        fallback: TokenRankings,
        rules: impl IntoIterator<Item = NextTokenRule>,
    ) -> Result<Self, InferenceError> {
        if fallback.all_fit_in(vocabulary_size) == TokenVocabularyFit::OutOfVocabulary {
            return Err(InferenceError::count_out_of_range(
                "fallback token id",
                "0..vocabulary_size",
                vocabulary_size.as_usize(),
            ));
        }

        let rules = rules.into_iter().collect::<Vec<_>>();
        let mut seen_previous = BTreeSet::new();

        for rule in &rules {
            if rule.previous.fits_in(vocabulary_size) == TokenVocabularyFit::OutOfVocabulary {
                return Err(InferenceError::count_out_of_range(
                    "rule previous token id",
                    "0..vocabulary_size",
                    rule.previous.as_usize(),
                ));
            }
            if rule.candidates.all_fit_in(vocabulary_size) == TokenVocabularyFit::OutOfVocabulary {
                return Err(InferenceError::count_out_of_range(
                    "rule candidate token id",
                    "0..vocabulary_size",
                    vocabulary_size.as_usize(),
                ));
            }
            if !seen_previous.insert(rule.previous) {
                return Err(InferenceError::duplicate_value(
                    "ToyNextTokenModel::new",
                    "each previous token may have at most one transition rule",
                ));
            }
        }

        Ok(Self {
            vocabulary_size,
            fallback,
            rules,
        })
    }
}

impl NextTokenModel for ToyNextTokenModel {
    fn vocabulary_size(&self) -> VocabularySize {
        self.vocabulary_size
    }

    fn next_token_candidates(
        &self,
        context: &ContextTokens,
    ) -> Result<TokenRankings, InferenceError> {
        let previous = context.last_token()?;

        for rule in &self.rules {
            if rule.previous == previous {
                return Ok(rule.candidates.clone());
            }
        }

        Ok(self.fallback.clone())
    }
}

/// Token-to-text entry in a toy vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyEntry {
    token_id: TokenId,
    text: TokenText,
}

impl VocabularyEntry {
    /// Creates one vocabulary entry.
    pub fn new(token_id: TokenId, text: TokenText) -> Self {
        Self { token_id, text }
    }
}

/// Toy vocabulary used to decode generated token IDs for examples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToyVocabulary {
    size: VocabularySize,
    entries: Vec<VocabularyEntry>,
}

impl ToyVocabulary {
    /// Creates a toy vocabulary and rejects duplicate token IDs.
    pub fn new(
        size: VocabularySize,
        entries: impl IntoIterator<Item = VocabularyEntry>,
    ) -> Result<Self, InferenceError> {
        let entries = entries.into_iter().collect::<Vec<_>>();
        SequenceTokenCount::from_raw("ToyVocabulary::new", entries.len())?;

        let mut seen = BTreeSet::new();
        for entry in &entries {
            if entry.token_id.fits_in(size) == TokenVocabularyFit::OutOfVocabulary {
                return Err(InferenceError::count_out_of_range(
                    "vocabulary entry token id",
                    "0..vocabulary_size",
                    entry.token_id.as_usize(),
                ));
            }
            if !seen.insert(entry.token_id) {
                return Err(InferenceError::duplicate_value(
                    "ToyVocabulary::new",
                    "each token id may appear only once in the vocabulary",
                ));
            }
        }

        Ok(Self { size, entries })
    }

    /// Returns the vocabulary size.
    pub fn size(&self) -> VocabularySize {
        self.size
    }

    /// Decodes generated tokens into learner-facing text.
    pub fn decode(&self, tokens: &GeneratedTokens) -> Result<GeneratedText, InferenceError> {
        let mut pieces = Vec::new();
        for token in tokens.tokens() {
            pieces.push(self.text_for(token)?);
        }

        let decoded = pieces
            .iter()
            .map(TokenText::as_str)
            .collect::<Vec<_>>()
            .join(" ");

        GeneratedText::try_from(decoded)
    }

    fn text_for(&self, token_id: TokenId) -> Result<TokenText, InferenceError> {
        for entry in &self.entries {
            if entry.token_id == token_id {
                return Ok(entry.text.clone());
            }
        }

        Err(InferenceError::missing_value(
            "ToyVocabulary::text_for",
            "token id is not present in the toy vocabulary",
        ))
    }
}

/// Cache role for one token entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEntryRole {
    /// Entry created from the prompt during prefill.
    PromptPrefix,
    /// Entry created from an autoregressive generation step.
    GeneratedToken,
}

impl fmt::Display for CacheEntryRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PromptPrefix => formatter.write_str("prompt-prefix"),
            Self::GeneratedToken => formatter.write_str("generated-token"),
        }
    }
}

/// Zero-based slot in a toy KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CachePosition(usize);

impl CachePosition {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for CachePosition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cache[{}]", self.0)
    }
}

/// Count of entries currently stored in the toy KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheEntryCount(usize);

impl CacheEntryCount {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for CacheEntryCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// One semantic KV-cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheEntry {
    position: CachePosition,
    token_id: TokenId,
    role: CacheEntryRole,
    decode_step: Option<DecodeStep>,
}

impl KvCacheEntry {
    fn prompt(position: CachePosition, token_id: TokenId) -> Self {
        Self {
            position,
            token_id,
            role: CacheEntryRole::PromptPrefix,
            decode_step: None,
        }
    }

    fn generated(position: CachePosition, step: DecodeStep, token_id: TokenId) -> Self {
        Self {
            position,
            token_id,
            role: CacheEntryRole::GeneratedToken,
            decode_step: Some(step),
        }
    }

    /// Returns the cache slot.
    pub fn position(&self) -> CachePosition {
        self.position
    }

    /// Returns the cached token identity.
    pub fn token_id(&self) -> TokenId {
        self.token_id
    }

    /// Returns the cache role.
    pub fn role(&self) -> CacheEntryRole {
        self.role
    }

    /// Returns the decode step for generated-token entries.
    pub fn decode_step(&self) -> Option<DecodeStep> {
        self.decode_step
    }
}

/// Toy KV cache that records prompt and generated tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCache {
    entries: Vec<KvCacheEntry>,
}

impl KvCache {
    fn from_prompt(prompt: &PromptTokens) -> Self {
        let entries = prompt
            .tokens()
            .enumerate()
            .map(|(position, token_id)| {
                KvCacheEntry::prompt(CachePosition::from_raw(position), token_id)
            })
            .collect::<Vec<_>>();

        Self { entries }
    }

    /// Returns all cache entries.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = &KvCacheEntry> + '_ {
        self.entries.iter()
    }

    /// Returns the total number of cache entries.
    pub fn entry_count(&self) -> CacheEntryCount {
        CacheEntryCount::from_raw(self.entries.len())
    }

    /// Returns the number of generated-token cache entries.
    pub fn generated_entry_count(&self) -> GeneratedTokenCount {
        let count = self
            .entries
            .iter()
            .filter(|entry| entry.role() == CacheEntryRole::GeneratedToken)
            .count();
        GeneratedTokenCount::from_raw(count)
    }

    fn next_position(&self) -> CachePosition {
        CachePosition::from_raw(self.entries.len())
    }
}

impl Add<KvCacheEntry> for KvCache {
    type Output = Result<KvCache, InferenceError>;

    fn add(mut self, entry: KvCacheEntry) -> Self::Output {
        if entry.position().as_usize() != self.entries.len() {
            return Err(InferenceError::count_out_of_range(
                "KV-cache entry position",
                "next cache position",
                entry.position().as_usize(),
            ));
        }

        self.entries.push(entry);
        Ok(self)
    }
}

/// Validated request for one deterministic decode trace.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeRequest {
    prompt: PromptTokens,
    context_window: ContextWindow,
    max_new_tokens: MaxNewTokens,
    sampling_mode: SamplingMode,
}

impl DecodeRequest {
    /// Creates a request and rejects generation that cannot fit in context.
    pub fn new(
        prompt: PromptTokens,
        context_window: ContextWindow,
        max_new_tokens: MaxNewTokens,
        sampling_mode: SamplingMode,
    ) -> Result<Self, InferenceError> {
        let token_budget = (prompt.count() + max_new_tokens)?;
        if token_budget.fits_in(context_window) == ContextFit::TooLong {
            return Err(InferenceError::context_overflow(
                "DecodeRequest::new",
                "prompt tokens plus requested generation exceed the context window",
            ));
        }

        Ok(Self {
            prompt,
            context_window,
            max_new_tokens,
            sampling_mode,
        })
    }

    /// Returns the prompt tokens.
    pub fn prompt(&self) -> &PromptTokens {
        &self.prompt
    }

    /// Returns the context window.
    pub fn context_window(&self) -> ContextWindow {
        self.context_window
    }

    /// Returns the maximum generation length.
    pub fn max_new_tokens(&self) -> MaxNewTokens {
        self.max_new_tokens
    }

    /// Returns the sampling strategy.
    pub fn sampling_mode(&self) -> SamplingMode {
        self.sampling_mode
    }
}

/// One generated-token event in a decode trace.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeStepRecord {
    step: DecodeStep,
    token_id: TokenId,
    sampling_mode: SamplingMode,
    cache_entry: KvCacheEntry,
}

impl DecodeStepRecord {
    fn new(
        step: DecodeStep,
        token_id: TokenId,
        sampling_mode: SamplingMode,
        cache_entry: KvCacheEntry,
    ) -> Self {
        Self {
            step,
            token_id,
            sampling_mode,
            cache_entry,
        }
    }

    /// Returns the decode step.
    pub fn step(&self) -> DecodeStep {
        self.step
    }

    /// Returns the selected token.
    pub fn token_id(&self) -> TokenId {
        self.token_id
    }

    /// Returns the sampling mode used at this step.
    pub fn sampling_mode(&self) -> SamplingMode {
        self.sampling_mode
    }

    /// Returns the KV-cache entry created by this step.
    pub fn cache_entry(&self) -> &KvCacheEntry {
        &self.cache_entry
    }
}

/// Full deterministic generation trace.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTrace {
    request: DecodeRequest,
    generated: GeneratedTokens,
    cache: KvCache,
    records: Vec<DecodeStepRecord>,
}

impl DecodeTrace {
    fn new(
        request: DecodeRequest,
        generated: GeneratedTokens,
        cache: KvCache,
        records: Vec<DecodeStepRecord>,
    ) -> Result<Self, InferenceError> {
        SequenceTokenCount::from_raw("DecodeTrace::new", records.len())?;
        Ok(Self {
            request,
            generated,
            cache,
            records,
        })
    }

    /// Returns the original decode request.
    pub fn request(&self) -> &DecodeRequest {
        &self.request
    }

    /// Returns generated tokens.
    pub fn generated_tokens(&self) -> &GeneratedTokens {
        &self.generated
    }

    /// Returns the cache after decoding.
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    /// Iterates over step records.
    pub fn records(&self) -> impl ExactSizeIterator<Item = &DecodeStepRecord> + '_ {
        self.records.iter()
    }
}

/// Runs one deterministic autoregressive decode.
pub fn decode<M>(model: &M, request: DecodeRequest) -> Result<DecodeTrace, InferenceError>
where
    M: NextTokenModel,
{
    if request.prompt().all_fit_in(model.vocabulary_size()) == TokenVocabularyFit::OutOfVocabulary {
        return Err(InferenceError::count_out_of_range(
            "prompt token id",
            "0..model vocabulary size",
            model.vocabulary_size().as_usize(),
        ));
    }

    let mut context = ContextTokens::from_prompt(request.prompt(), request.context_window())?;
    let mut cache = KvCache::from_prompt(request.prompt());
    let mut generated = GeneratedTokens::empty();
    let mut records = Vec::new();

    while generated.count().is_less_than(request.max_new_tokens()) == DecodeLoopState::Continue {
        let step = DecodeStep::from_generated_count(generated.count())?;
        let rankings = model.next_token_candidates(&context)?;
        let selected = request.sampling_mode().select(&rankings)?;

        if selected.fits_in(model.vocabulary_size()) == TokenVocabularyFit::OutOfVocabulary {
            return Err(InferenceError::count_out_of_range(
                "selected token id",
                "0..model vocabulary size",
                selected.as_usize(),
            ));
        }

        let cache_entry = KvCacheEntry::generated(cache.next_position(), step, selected);
        cache = (cache + cache_entry.clone())?;
        generated = (generated + selected)?;
        context = (context + selected)?;
        records.push(DecodeStepRecord::new(
            step,
            selected,
            request.sampling_mode(),
            cache_entry,
        ));
    }

    DecodeTrace::new(request, generated, cache, records)
}

/// Positive latency duration in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LatencyMillis(u64);

impl LatencyMillis {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, InferenceError> {
        Ok(Self(nonzero_u64(
            "latency milliseconds must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for LatencyMillis {
    type Error = InferenceError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_raw("LatencyMillis::try_from", value)
    }
}

impl Add for LatencyMillis {
    type Output = Result<LatencyMillis, InferenceError>;

    fn add(self, right: LatencyMillis) -> Self::Output {
        LatencyMillis::from_raw(
            "LatencyMillis::add",
            checked_u64_add("LatencyMillis::add", self.as_u64(), right.as_u64())?,
        )
    }
}

impl Mul<GeneratedTokenCount> for LatencyMillis {
    type Output = Result<LatencyMillis, InferenceError>;

    fn mul(self, right: GeneratedTokenCount) -> Self::Output {
        LatencyMillis::from_raw(
            "LatencyMillis::mul",
            checked_u64_mul(
                "LatencyMillis::mul",
                self.as_u64(),
                u64::try_from(right.as_usize()).map_err(|_| {
                    InferenceError::overflow("LatencyMillis::mul", "token count exceeded u64")
                })?,
            )?,
        )
    }
}

impl fmt::Display for LatencyMillis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ms", self.0)
    }
}

/// Latency budget for one generated response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyBudget {
    limit: LatencyMillis,
}

impl LatencyBudget {
    /// Creates a latency budget from a positive duration.
    pub fn new(limit: LatencyMillis) -> Self {
        Self { limit }
    }

    /// Returns the budget limit.
    pub fn limit(&self) -> LatencyMillis {
        self.limit
    }
}

/// Whether a trace met its latency budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyStatus {
    /// The estimated latency is no greater than the budget.
    WithinBudget,
    /// The estimated latency is greater than the budget.
    ExceedsBudget,
}

impl fmt::Display for LatencyStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WithinBudget => formatter.write_str("within-budget"),
            Self::ExceedsBudget => formatter.write_str("exceeds-budget"),
        }
    }
}

/// Latency estimate for one decode trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyReport {
    prefill: LatencyMillis,
    per_token: LatencyMillis,
    generated: GeneratedTokenCount,
    total: LatencyMillis,
    budget: LatencyBudget,
    status: LatencyStatus,
}

impl LatencyReport {
    /// Estimates latency from prefill cost, per-token generation cost, and a budget.
    pub fn from_trace(
        trace: &DecodeTrace,
        prefill: LatencyMillis,
        per_token: LatencyMillis,
        budget: LatencyBudget,
    ) -> Result<Self, InferenceError> {
        let generated = trace.generated_tokens().count();
        let generation_latency = (per_token * generated)?;
        let total = (prefill + generation_latency)?;
        let status = if total.as_u64() <= budget.limit().as_u64() {
            LatencyStatus::WithinBudget
        } else {
            LatencyStatus::ExceedsBudget
        };

        Ok(Self {
            prefill,
            per_token,
            generated,
            total,
            budget,
            status,
        })
    }

    /// Returns estimated total latency.
    pub fn total(&self) -> LatencyMillis {
        self.total
    }

    /// Returns the budget.
    pub fn budget(&self) -> LatencyBudget {
        self.budget
    }

    /// Returns whether the trace met the budget.
    pub fn status(&self) -> LatencyStatus {
        self.status
    }

    /// Returns the prefill latency component.
    pub fn prefill(&self) -> LatencyMillis {
        self.prefill
    }

    /// Returns the per-token latency component.
    pub fn per_token(&self) -> LatencyMillis {
        self.per_token
    }

    /// Returns the generated-token count used in the estimate.
    pub fn generated(&self) -> GeneratedTokenCount {
        self.generated
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ContextWindow, DecodeRequest, GeneratedTokenCount, LatencyBudget, LatencyMillis, Logit,
        MaxNewTokens, NextTokenRule, PromptTokens, RankedToken, SamplingMode, Temperature, TokenId,
        TokenIndex, TokenRankings, TopK, ToyNextTokenModel, VocabularySize, decode,
    };
    use crate::{Error, LatencyReport};

    fn token(index: TokenIndex, vocabulary_size: VocabularySize) -> Result<TokenId, Error> {
        TokenId::new(index, vocabulary_size)
    }

    fn demo_model() -> Result<ToyNextTokenModel, Error> {
        let vocabulary_size = VocabularySize::try_from(4)?;
        let start = token(TokenIndex::try_from(0)?, vocabulary_size)?;
        let typed = token(TokenIndex::try_from(1)?, vocabulary_size)?;
        let rust = token(TokenIndex::try_from(2)?, vocabulary_size)?;
        let end = token(TokenIndex::try_from(3)?, vocabulary_size)?;

        ToyNextTokenModel::new(
            vocabulary_size,
            TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(1.0)?)])?,
            [
                NextTokenRule::new(
                    start,
                    TokenRankings::from_candidates([
                        RankedToken::new(typed, Logit::try_from(3.0)?),
                        RankedToken::new(rust, Logit::try_from(2.0)?),
                    ])?,
                ),
                NextTokenRule::new(
                    typed,
                    TokenRankings::from_candidates([RankedToken::new(
                        rust,
                        Logit::try_from(4.0)?,
                    )])?,
                ),
                NextTokenRule::new(
                    rust,
                    TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(5.0)?)])?,
                ),
            ],
        )
    }

    #[test]
    fn token_id_rejects_index_outside_vocabulary() -> Result<(), Error> {
        let vocabulary_size = VocabularySize::try_from(2)?;
        let rejected = TokenId::new(TokenIndex::try_from(2)?, vocabulary_size);

        assert!(matches!(rejected, Err(Error::CountOutOfRange { .. })));
        Ok(())
    }

    #[test]
    fn request_rejects_zero_generation_length() -> Result<(), Error> {
        let vocabulary_size = VocabularySize::try_from(2)?;
        let prompt =
            PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
        let rejected = MaxNewTokens::try_from(0).and_then(|max_new| {
            DecodeRequest::new(
                prompt,
                ContextWindow::try_from(2)?,
                max_new,
                SamplingMode::Greedy,
            )
        });

        assert!(matches!(rejected, Err(Error::EmptyInput { .. })));
        Ok(())
    }

    #[test]
    fn request_rejects_context_overflow() -> Result<(), Error> {
        let vocabulary_size = VocabularySize::try_from(4)?;
        let prompt = PromptTokens::from_tokens([
            token(TokenIndex::try_from(0)?, vocabulary_size)?,
            token(TokenIndex::try_from(1)?, vocabulary_size)?,
        ])?;
        let rejected = DecodeRequest::new(
            prompt,
            ContextWindow::try_from(2)?,
            MaxNewTokens::try_from(1)?,
            SamplingMode::Greedy,
        );

        assert!(matches!(rejected, Err(Error::ContextOverflow { .. })));
        Ok(())
    }

    #[test]
    fn greedy_decode_produces_deterministic_trace() -> Result<(), Error> {
        let model = demo_model()?;
        let vocabulary_size = VocabularySize::try_from(4)?;
        let prompt =
            PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
        let trace = decode(
            &model,
            DecodeRequest::new(
                prompt,
                ContextWindow::try_from(4)?,
                MaxNewTokens::try_from(3)?,
                SamplingMode::Greedy,
            )?,
        )?;

        let generated = trace.generated_tokens().tokens().collect::<Vec<_>>();
        assert_eq!(
            generated,
            [
                token(TokenIndex::try_from(1)?, vocabulary_size)?,
                token(TokenIndex::try_from(2)?, vocabulary_size)?,
                token(TokenIndex::try_from(3)?, vocabulary_size)?,
            ]
        );
        Ok(())
    }

    #[test]
    fn cache_records_one_generated_entry_per_decode_step() -> Result<(), Error> {
        let model = demo_model()?;
        let vocabulary_size = VocabularySize::try_from(4)?;
        let prompt =
            PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
        let trace = decode(
            &model,
            DecodeRequest::new(
                prompt,
                ContextWindow::try_from(4)?,
                MaxNewTokens::try_from(2)?,
                SamplingMode::Greedy,
            )?,
        )?;

        assert_eq!(
            trace.cache().generated_entry_count(),
            GeneratedTokenCount::from_raw(2)
        );
        for record in trace.records() {
            assert_eq!(record.cache_entry().token_id(), record.token_id());
        }
        Ok(())
    }

    #[test]
    fn latency_report_uses_typed_addition_and_multiplication() -> Result<(), Error> {
        let model = demo_model()?;
        let vocabulary_size = VocabularySize::try_from(4)?;
        let prompt =
            PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
        let trace = decode(
            &model,
            DecodeRequest::new(
                prompt,
                ContextWindow::try_from(4)?,
                MaxNewTokens::try_from(2)?,
                SamplingMode::TopK {
                    k: TopK::try_from(2)?,
                    temperature: Temperature::try_from(0.7)?,
                },
            )?,
        )?;

        let report = LatencyReport::from_trace(
            &trace,
            LatencyMillis::try_from(10)?,
            LatencyMillis::try_from(5)?,
            LatencyBudget::new(LatencyMillis::try_from(30)?),
        )?;

        assert_eq!(report.total(), LatencyMillis::try_from(20)?);
        Ok(())
    }
}
