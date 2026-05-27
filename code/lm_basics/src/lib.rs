//! Typed language-modeling basics for the CS336 Rust equivalent track.
//!
//! This crate intentionally starts with a tiny bigram model. The goal is not
//! scale; the goal is to make the first language-modeling path visible:
//!
//! ```text
//! RawText -> Tokens -> TokenIds -> Batch -> Logits -> Loss -> Update
//! ReviewedRawText -> PublicLanguageModelingExample
//! ```
//!
//! Raw learner data enters through explicit `TryFrom` adapters. Once a value is
//! inside the crate's domain model, public APIs use semantic types instead of
//! unlabelled primitives. Public examples add one more typed gate: reviewed text
//! must be classified as public before it can become learner-facing material.

pub mod error;

use std::{collections::BTreeMap, fmt, ops::Mul};

use error::LmBasicsError;

pub use error::LmBasicsError as Error;

fn finite(role: &'static str, value: f64) -> Result<f64, LmBasicsError> {
    if !value.is_finite() {
        return Err(LmBasicsError::non_finite_value(role, value));
    }

    Ok(value)
}

/// Raw learner-provided text before tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawText(String);

impl RawText {
    fn from_owned(value: String) -> Result<Self, LmBasicsError> {
        if value.trim().is_empty() {
            return Err(LmBasicsError::empty_input(
                "RawText::try_from",
                "text cannot be empty",
            ));
        }

        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for RawText {
    type Error = LmBasicsError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for RawText {
    type Error = LmBasicsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

/// One tokenizer output unit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(String);

impl Token {
    fn from_owned(value: String) -> Result<Self, LmBasicsError> {
        if value.trim().is_empty() {
            return Err(LmBasicsError::empty_input(
                "Token::try_from",
                "token cannot be empty",
            ));
        }

        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for Token {
    type Error = LmBasicsError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for Token {
    type Error = LmBasicsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A count of tokens in a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenCount(usize);

impl TokenCount {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TokenCount {
    type Error = LmBasicsError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self::from_raw(value))
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A non-empty sequence of textual tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenTextSequence(Vec<Token>);

impl TokenTextSequence {
    /// Creates a non-empty token text sequence.
    pub fn from_tokens(tokens: impl IntoIterator<Item = Token>) -> Result<Self, LmBasicsError> {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        if tokens.is_empty() {
            return Err(LmBasicsError::empty_input(
                "TokenTextSequence::from_tokens",
                "token sequence cannot be empty",
            ));
        }

        Ok(Self(tokens))
    }

    /// Returns token count as a semantic value.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_raw(self.len_value())
    }

    /// Iterates over token values.
    pub fn tokens(&self) -> impl ExactSizeIterator<Item = &Token> + '_ {
        self.0.iter()
    }

    fn len_value(&self) -> usize {
        self.0.len()
    }
}

/// A minimal whitespace tokenizer for tiny examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Lowercases and splits text on Unicode whitespace.
    pub fn tokenize(&self, text: &RawText) -> Result<TokenTextSequence, LmBasicsError> {
        let tokens = text
            .as_str()
            .split_whitespace()
            .map(|part| Token::try_from(part.to_lowercase()))
            .collect::<Result<Vec<_>, _>>()?;

        TokenTextSequence::from_tokens(tokens)
    }
}

/// Number of entries in a vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabularySize(usize);

impl VocabularySize {
    fn from_raw(value: usize) -> Result<Self, LmBasicsError> {
        if value == 0 {
            return Err(LmBasicsError::empty_input(
                "VocabularySize::try_from",
                "vocabulary size must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for VocabularySize {
    type Error = LmBasicsError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl TryFrom<TokenCount> for VocabularySize {
    type Error = LmBasicsError;

    fn try_from(value: TokenCount) -> Result<Self, Self::Error> {
        Self::from_raw(value.as_usize())
    }
}

impl fmt::Display for VocabularySize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A raw token index after it has been labelled with meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenIndex(usize);

impl TokenIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TokenIndex {
    type Error = LmBasicsError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for TokenIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A checked token identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(usize);

impl TokenId {
    /// Creates a token id known to be inside the vocabulary.
    pub fn new(index: TokenIndex, vocab_size: VocabularySize) -> Result<Self, LmBasicsError> {
        if index.as_usize() >= vocab_size.as_usize() {
            return Err(LmBasicsError::invalid_token_id(
                "TokenId::new",
                index.as_usize(),
                vocab_size.as_usize(),
            ));
        }

        Ok(Self(index.as_usize()))
    }

    /// Returns the semantic token index.
    pub fn index(self) -> TokenIndex {
        TokenIndex(self.0)
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A token position after it has been labelled with meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionIndex(usize);

impl PositionIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for PositionIndex {
    type Error = LmBasicsError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for PositionIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A token position inside a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position(usize);

impl Position {
    /// Creates a checked position.
    pub fn new(index: PositionIndex, sequence_len: TokenCount) -> Result<Self, LmBasicsError> {
        if index.as_usize() >= sequence_len.as_usize() {
            return Err(LmBasicsError::dimension_mismatch(
                "Position::new",
                "position",
                vec![index.as_usize()],
                "sequence length",
                vec![sequence_len.as_usize()],
                "position must be less than sequence length",
            ));
        }

        Ok(Self(index.as_usize()))
    }

    /// Returns the semantic position index.
    pub fn index(self) -> PositionIndex {
        PositionIndex(self.0)
    }
}

/// A positive context length for language-modeling windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLength(usize);

impl ContextLength {
    /// Creates a context length from a non-empty token count.
    pub fn from_token_count(value: TokenCount) -> Result<Self, LmBasicsError> {
        if value.as_usize() == 0 {
            return Err(LmBasicsError::empty_input(
                "ContextLength::from_token_count",
                "context length must be greater than zero",
            ));
        }

        Ok(Self(value.as_usize()))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for ContextLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Vocabulary with stable token-to-id mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vocabulary {
    tokens: Vec<Token>,
    ids_by_token: BTreeMap<Token, TokenId>,
    size: VocabularySize,
}

impl Vocabulary {
    /// Builds a vocabulary in first-seen token order.
    pub fn from_tokens(tokens: &TokenTextSequence) -> Result<Self, LmBasicsError> {
        let mut unique_tokens = Vec::new();
        let mut ids_by_text = BTreeMap::new();

        for token in tokens.tokens() {
            if !ids_by_text.contains_key(token) {
                let id = TokenIndex::try_from(unique_tokens.len())?;
                unique_tokens.push(token.clone());
                ids_by_text.insert(token.clone(), id);
            }
        }

        let size = VocabularySize::try_from(unique_tokens.len())?;
        let mut ids_by_token = BTreeMap::new();
        for (token, raw_id) in ids_by_text {
            ids_by_token.insert(token, TokenId::new(raw_id, size)?);
        }

        Ok(Self {
            tokens: unique_tokens,
            ids_by_token,
            size,
        })
    }

    /// Returns vocabulary size.
    pub fn size(&self) -> VocabularySize {
        self.size
    }

    /// Returns the token text for an id.
    pub fn token(&self, id: TokenId) -> &Token {
        &self.tokens[id.index().as_usize()]
    }

    /// Encodes one textual token.
    pub fn encode_token(&self, token: &Token) -> Result<TokenId, LmBasicsError> {
        self.ids_by_token.get(token).copied().ok_or_else(|| {
            LmBasicsError::unknown_token("Vocabulary::encode_token", token.as_str().to_owned())
        })
    }

    /// Encodes a token text sequence into token ids.
    pub fn encode(&self, tokens: &TokenTextSequence) -> Result<TokenIdSequence, LmBasicsError> {
        let ids = tokens
            .tokens()
            .map(|token| self.encode_token(token))
            .collect::<Result<Vec<_>, _>>()?;

        TokenIdSequence::from_ids(ids, self.size)
    }
}

impl Mul<&TokenTextSequence> for &Vocabulary {
    type Output = Result<TokenIdSequence, LmBasicsError>;

    fn mul(self, tokens: &TokenTextSequence) -> Self::Output {
        self.encode(tokens)
    }
}

/// Publication class attached to raw text before learner-facing release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextVisibility {
    /// Safe to include in learner-facing public language-modeling examples.
    Public,
    /// Useful for restricted study, but not public learner-facing material.
    ResearchRestricted,
    /// Must stay out of public learner-facing material.
    Private,
}

impl fmt::Display for TextVisibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Public => "public",
            Self::ResearchRestricted => "research-restricted",
            Self::Private => "private",
        };
        formatter.write_str(label)
    }
}

/// Typed decision at the language-modeling public-example boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicTextDecision {
    /// The text can appear in a public learner-facing example.
    Publishable,
    /// The text must stay out of public learner-facing examples.
    Blocked,
}

/// Raw text plus explicit public-release review evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewedRawText {
    text: RawText,
    visibility: TextVisibility,
}

impl ReviewedRawText {
    /// Creates reviewed raw text.
    pub fn new(text: RawText, visibility: TextVisibility) -> Self {
        Self { text, visibility }
    }

    /// Returns the reviewed raw text.
    pub fn text(&self) -> &RawText {
        &self.text
    }

    /// Returns the publication class.
    pub fn visibility(&self) -> TextVisibility {
        self.visibility
    }

    /// Classifies whether this text can enter public learner-facing material.
    pub fn release_decision(&self) -> PublicTextDecision {
        match self.visibility {
            TextVisibility::Public => PublicTextDecision::Publishable,
            TextVisibility::ResearchRestricted | TextVisibility::Private => {
                PublicTextDecision::Blocked
            }
        }
    }

    fn into_text(self) -> RawText {
        self.text
    }
}

/// A non-empty sequence of checked token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIdSequence {
    ids: Vec<TokenId>,
    vocab_size: VocabularySize,
}

impl TokenIdSequence {
    /// Creates a sequence after checking every id belongs to the same vocabulary.
    pub fn from_ids(
        ids: impl IntoIterator<Item = TokenId>,
        vocab_size: VocabularySize,
    ) -> Result<Self, LmBasicsError> {
        let ids = ids.into_iter().collect::<Vec<_>>();
        if ids.is_empty() {
            return Err(LmBasicsError::empty_input(
                "TokenIdSequence::from_ids",
                "id sequence cannot be empty",
            ));
        }

        for id in &ids {
            TokenId::new(id.index(), vocab_size)?;
        }

        Ok(Self { ids, vocab_size })
    }

    /// Returns token id count as a semantic value.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_raw(self.len_value())
    }

    /// Iterates over all token ids.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = &TokenId> + '_ {
        self.ids.iter()
    }

    /// Returns the vocabulary size shared by the ids.
    pub fn vocab_size(&self) -> VocabularySize {
        self.vocab_size
    }

    fn len_value(&self) -> usize {
        self.ids.len()
    }
}

/// Input and target ids for next-token language modeling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextTokenBatch {
    inputs: TokenIdSequence,
    targets: TokenIdSequence,
    context_length: ContextLength,
}

impl NextTokenBatch {
    /// Creates a checked next-token batch.
    pub fn new(inputs: TokenIdSequence, targets: TokenIdSequence) -> Result<Self, LmBasicsError> {
        if inputs.vocab_size() != targets.vocab_size() {
            return Err(LmBasicsError::dimension_mismatch(
                "NextTokenBatch::new",
                "input vocabulary size",
                vec![inputs.vocab_size().as_usize()],
                "target vocabulary size",
                vec![targets.vocab_size().as_usize()],
                "inputs and targets must use the same vocabulary",
            ));
        }

        if inputs.len() != targets.len() {
            return Err(LmBasicsError::dimension_mismatch(
                "NextTokenBatch::new",
                "input ids",
                vec![inputs.len().as_usize()],
                "target ids",
                vec![targets.len().as_usize()],
                "each input token needs exactly one target token",
            ));
        }

        let context_length = ContextLength::from_token_count(inputs.len())?;
        Ok(Self {
            inputs,
            targets,
            context_length,
        })
    }

    /// Creates adjacent next-token training pairs from one id sequence.
    pub fn from_sequence(sequence: &TokenIdSequence) -> Result<Self, LmBasicsError> {
        if sequence.len_value() < 2 {
            return Err(LmBasicsError::dimension_mismatch(
                "NextTokenBatch::from_sequence",
                "sequence length",
                vec![sequence.len_value()],
                "minimum length",
                vec![2],
                "next-token training needs at least two ids",
            ));
        }

        let sequence_len = sequence.len_value();
        let inputs = TokenIdSequence::from_ids(
            sequence.ids().copied().take(sequence_len - 1),
            sequence.vocab_size(),
        )?;
        let targets =
            TokenIdSequence::from_ids(sequence.ids().copied().skip(1), sequence.vocab_size())?;

        Self::new(inputs, targets)
    }

    /// Returns input ids.
    pub fn inputs(&self) -> &TokenIdSequence {
        &self.inputs
    }

    /// Returns target ids.
    pub fn targets(&self) -> &TokenIdSequence {
        &self.targets
    }

    /// Returns context length.
    pub fn context_length(&self) -> ContextLength {
        self.context_length
    }
}

/// One finite logit value before softmax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Logit(f64);

impl Logit {
    fn from_raw(value: f64) -> Result<Self, LmBasicsError> {
        Ok(Self(finite("logit", value)?))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Logit {
    type Error = LmBasicsError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for Logit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A non-empty logit row over the vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitRow(Vec<Logit>);

impl LogitRow {
    /// Creates logits for every token in the vocabulary.
    pub fn from_logits(
        logits: impl IntoIterator<Item = Logit>,
        vocab_size: VocabularySize,
    ) -> Result<Self, LmBasicsError> {
        let logits = logits.into_iter().collect::<Vec<_>>();
        if logits.len() != vocab_size.as_usize() {
            return Err(LmBasicsError::dimension_mismatch(
                "LogitRow::from_logits",
                "logits",
                vec![logits.len()],
                "vocabulary",
                vec![vocab_size.as_usize()],
                "a language model needs one logit for each vocabulary entry",
            ));
        }

        Ok(Self(logits))
    }

    /// Iterates over logits.
    pub fn logits(&self) -> impl ExactSizeIterator<Item = &Logit> + '_ {
        self.0.iter()
    }

    fn as_mut_slice(&mut self) -> &mut [Logit] {
        &mut self.0
    }
}

/// A non-negative loss value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Loss(f64);

impl Loss {
    fn from_raw(value: f64) -> Result<Self, LmBasicsError> {
        let value = finite("loss", value)?;
        if value < 0.0 {
            return Err(LmBasicsError::out_of_range("loss", "0..", value));
        }

        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }

    /// Compares two losses with a semantic tolerance.
    pub fn compare_with_tolerance(
        self,
        expected: Loss,
        tolerance: LossTolerance,
    ) -> LossComparison {
        if (self.as_f64() - expected.as_f64()).abs() <= tolerance.as_f64() {
            LossComparison::Close
        } else {
            LossComparison::Different
        }
    }
}

impl TryFrom<f64> for Loss {
    type Error = LmBasicsError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for Loss {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A finite non-negative tolerance for comparing learner-visible loss values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LossTolerance(f64);

impl LossTolerance {
    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for LossTolerance {
    type Error = LmBasicsError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let value = finite("loss tolerance", value)?;
        if value < 0.0 {
            return Err(LmBasicsError::out_of_range("loss tolerance", "0..", value));
        }

        Ok(Self(value))
    }
}

/// Result of comparing two losses with a tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossComparison {
    /// The losses are within tolerance.
    Close,
    /// The losses differ by more than the tolerance.
    Different,
}

/// A positive optimizer step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRate(f64);

impl LearningRate {
    fn from_raw(value: f64) -> Result<Self, LmBasicsError> {
        let value = finite("learning rate", value)?;
        if value <= 0.0 {
            return Err(LmBasicsError::out_of_range("learning rate", "> 0", value));
        }

        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for LearningRate {
    type Error = LmBasicsError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for LearningRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Learner-visible trace for one training update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingStepTrace {
    loss_before: Loss,
    loss_after: Loss,
}

impl TrainingStepTrace {
    /// Loss before the update.
    pub fn loss_before(self) -> Loss {
        self.loss_before
    }

    /// Loss after the update on the same batch.
    pub fn loss_after(self) -> Loss {
        self.loss_after
    }
}

/// A tiny trainable bigram language model.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyBigramLanguageModel {
    vocab_size: VocabularySize,
    logits_by_input: Vec<LogitRow>,
}

impl TinyBigramLanguageModel {
    /// Creates a model from a square `vocab_size x vocab_size` logit table.
    pub fn from_logit_rows(
        vocab_size: VocabularySize,
        logits_by_input: impl IntoIterator<Item = LogitRow>,
    ) -> Result<Self, LmBasicsError> {
        let logits_by_input = logits_by_input.into_iter().collect::<Vec<_>>();
        if logits_by_input.len() != vocab_size.as_usize() {
            return Err(LmBasicsError::dimension_mismatch(
                "TinyBigramLanguageModel::from_logit_rows",
                "logit rows",
                vec![logits_by_input.len()],
                "vocabulary",
                vec![vocab_size.as_usize()],
                "there must be one row for each possible input token",
            ));
        }

        for row in &logits_by_input {
            LogitRow::from_logits(row.logits().copied(), vocab_size)?;
        }

        Ok(Self {
            vocab_size,
            logits_by_input,
        })
    }

    /// Creates a model with zero logits, which starts as a uniform predictor.
    pub fn uniform(vocab_size: VocabularySize) -> Result<Self, LmBasicsError> {
        let row = LogitRow::from_logits(
            (0..vocab_size.as_usize())
                .map(|_| Logit::from_raw(0.0))
                .collect::<Result<Vec<_>, _>>()?,
            vocab_size,
        )?;
        Self::from_logit_rows(vocab_size, (0..vocab_size.as_usize()).map(|_| row.clone()))
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> VocabularySize {
        self.vocab_size
    }

    /// Returns logits for one input token.
    pub fn logits_for(&self, input: TokenId) -> Result<LogitRow, LmBasicsError> {
        TokenId::new(input.index(), self.vocab_size)?;
        Ok(self.logits_by_input[input.index().as_usize()].clone())
    }

    /// Computes cross-entropy loss for one next-token pair.
    pub fn loss_for_pair(&self, input: TokenId, target: TokenId) -> Result<Loss, LmBasicsError> {
        TokenId::new(input.index(), self.vocab_size)?;
        TokenId::new(target.index(), self.vocab_size)?;

        let logits = self.logits_for(input)?;
        let probabilities = softmax(&logits)?;
        Loss::from_raw(-probabilities[target.index().as_usize()].ln())
    }

    /// Computes mean loss over a batch.
    pub fn average_loss(&self, batch: &NextTokenBatch) -> Result<Loss, LmBasicsError> {
        if batch.inputs().vocab_size() != self.vocab_size {
            return Err(LmBasicsError::dimension_mismatch(
                "TinyBigramLanguageModel::average_loss",
                "model vocabulary",
                vec![self.vocab_size.as_usize()],
                "batch vocabulary",
                vec![batch.inputs().vocab_size().as_usize()],
                "batch ids must belong to the model vocabulary",
            ));
        }

        let total = batch
            .inputs()
            .ids()
            .zip(batch.targets().ids())
            .map(|(input, target)| self.loss_for_pair(*input, *target).map(Loss::as_f64))
            .sum::<Result<f64, _>>()?;

        Loss::from_raw(total / batch.context_length().as_usize() as f64)
    }

    /// Runs one gradient step over a batch and returns loss before/after.
    pub fn train_one_step(
        &mut self,
        batch: &NextTokenBatch,
        learning_rate: LearningRate,
    ) -> Result<TrainingStepTrace, LmBasicsError> {
        let loss_before = self.average_loss(batch)?;

        for (input, target) in batch.inputs().ids().zip(batch.targets().ids()) {
            self.update_pair(*input, *target, learning_rate)?;
        }

        let loss_after = self.average_loss(batch)?;
        Ok(TrainingStepTrace {
            loss_before,
            loss_after,
        })
    }

    fn update_pair(
        &mut self,
        input: TokenId,
        target: TokenId,
        learning_rate: LearningRate,
    ) -> Result<(), LmBasicsError> {
        let logits = self.logits_for(input)?;
        let probabilities = softmax(&logits)?;
        let row = &mut self.logits_by_input[input.index().as_usize()];

        for (index, logit) in row.as_mut_slice().iter_mut().enumerate() {
            let target_signal = if index == target.index().as_usize() {
                1.0
            } else {
                0.0
            };
            let gradient = probabilities[index] - target_signal;
            *logit = Logit::from_raw(logit.as_f64() - learning_rate.as_f64() * gradient)?;
        }

        Ok(())
    }
}

impl Mul<&NextTokenBatch> for &TinyBigramLanguageModel {
    type Output = Result<Loss, LmBasicsError>;

    fn mul(self, batch: &NextTokenBatch) -> Self::Output {
        self.average_loss(batch)
    }
}

/// First language-modeling path checked for learner-facing public release.
#[derive(Debug, Clone, PartialEq)]
pub struct PublicLanguageModelingExample {
    text: RawText,
    tokens: TokenTextSequence,
    vocabulary: Vocabulary,
    ids: TokenIdSequence,
    batch: NextTokenBatch,
}

impl PublicLanguageModelingExample {
    /// Builds a public example only from reviewed public text.
    pub fn from_reviewed_text(
        reviewed: ReviewedRawText,
        tokenizer: WhitespaceTokenizer,
    ) -> Result<Self, LmBasicsError> {
        if reviewed.release_decision() == PublicTextDecision::Blocked {
            return Err(LmBasicsError::invalid_public_example(
                "PublicLanguageModelingExample::from_reviewed_text",
                "public language-modeling examples cannot include restricted or private text",
            ));
        }

        let text = reviewed.into_text();
        let tokens = tokenizer.tokenize(&text)?;
        let vocabulary = Vocabulary::from_tokens(&tokens)?;
        let ids = (&vocabulary * &tokens)?;
        let batch = NextTokenBatch::from_sequence(&ids)?;

        Ok(Self {
            text,
            tokens,
            vocabulary,
            ids,
            batch,
        })
    }

    /// Returns the checked public text.
    pub fn text(&self) -> &RawText {
        &self.text
    }

    /// Returns the tokenized public text.
    pub fn tokens(&self) -> &TokenTextSequence {
        &self.tokens
    }

    /// Returns the public-example vocabulary.
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Returns the checked token ids.
    pub fn ids(&self) -> &TokenIdSequence {
        &self.ids
    }

    /// Returns the next-token batch derived from the public text.
    pub fn batch(&self) -> &NextTokenBatch {
        &self.batch
    }
}

fn softmax(logits: &LogitRow) -> Result<Vec<f64>, LmBasicsError> {
    let max = logits
        .logits()
        .map(|logit| logit.as_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_values = logits
        .logits()
        .map(|logit| (logit.as_f64() - max).exp())
        .collect::<Vec<_>>();
    let sum: f64 = exp_values.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return Err(LmBasicsError::non_finite_value("softmax normalizer", sum));
    }

    Ok(exp_values.into_iter().map(|value| value / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::{
        ContextLength, LearningRate, LmBasicsError, Loss, LossComparison, LossTolerance,
        NextTokenBatch, PublicLanguageModelingExample, RawText, ReviewedRawText, TextVisibility,
        TinyBigramLanguageModel, Token, TokenCount, TokenId, TokenIdSequence, TokenIndex,
        Vocabulary, VocabularySize, WhitespaceTokenizer,
    };

    fn tiny_vocab() -> Result<Vocabulary, LmBasicsError> {
        let text = RawText::try_from("red blue red")?;
        let tokens = WhitespaceTokenizer.tokenize(&text)?;
        Vocabulary::from_tokens(&tokens)
    }

    fn public_reviewed_text() -> Result<ReviewedRawText, LmBasicsError> {
        Ok(ReviewedRawText::new(
            RawText::try_from("red blue red")?,
            TextVisibility::Public,
        ))
    }

    #[test]
    fn tokenizer_and_vocabulary_preserve_first_seen_ids() -> Result<(), LmBasicsError> {
        let text = RawText::try_from("Red blue red")?;
        let tokens = WhitespaceTokenizer.tokenize(&text)?;
        let vocabulary = Vocabulary::from_tokens(&tokens)?;
        let ids = (&vocabulary * &tokens)?;
        let ids = ids.ids().copied().collect::<Vec<_>>();

        assert_eq!(tokens.tokens().next(), Some(&Token::try_from("red")?));
        assert_eq!(vocabulary.token(ids[0]), &Token::try_from("red")?);
        assert_eq!(vocabulary.token(ids[1]), &Token::try_from("blue")?);
        assert_eq!(ids[0], ids[2]);
        Ok(())
    }

    #[test]
    fn vocabulary_times_tokens_keeps_encoding_as_a_typed_map() -> Result<(), LmBasicsError> {
        let text = RawText::try_from("red blue red")?;
        let tokens = WhitespaceTokenizer.tokenize(&text)?;
        let vocabulary = Vocabulary::from_tokens(&tokens)?;

        let encoded_with_method = vocabulary.encode(&tokens)?;
        let encoded_with_operator = (&vocabulary * &tokens)?;

        assert_eq!(encoded_with_operator, encoded_with_method);
        Ok(())
    }

    #[test]
    fn token_id_rejects_values_outside_vocabulary() -> Result<(), LmBasicsError> {
        let error = TokenId::new(TokenIndex::try_from(2)?, VocabularySize::try_from(2)?);

        assert!(matches!(error, Err(LmBasicsError::InvalidTokenId { .. })));
        Ok(())
    }

    #[test]
    fn vocabulary_rejects_unknown_token() -> Result<(), LmBasicsError> {
        let vocabulary = tiny_vocab()?;
        let unknown = Token::try_from("green")?;
        let error = vocabulary.encode_token(&unknown);

        assert!(matches!(error, Err(LmBasicsError::UnknownToken { .. })));
        Ok(())
    }

    #[test]
    fn next_token_batch_rejects_mismatched_lengths() -> Result<(), LmBasicsError> {
        let size = VocabularySize::try_from(2)?;
        let inputs =
            TokenIdSequence::from_ids([TokenId::new(TokenIndex::try_from(0)?, size)?], size)?;
        let targets = TokenIdSequence::from_ids(
            [
                TokenId::new(TokenIndex::try_from(1)?, size)?,
                TokenId::new(TokenIndex::try_from(0)?, size)?,
            ],
            size,
        )?;

        let error = NextTokenBatch::new(inputs, targets);

        assert!(matches!(
            error,
            Err(LmBasicsError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn uniform_bigram_loss_is_log_vocabulary_size() -> Result<(), LmBasicsError> {
        let vocabulary = tiny_vocab()?;
        let red = vocabulary.encode_token(&Token::try_from("red")?)?;
        let blue = vocabulary.encode_token(&Token::try_from("blue")?)?;
        let model = TinyBigramLanguageModel::uniform(vocabulary.size())?;

        let loss = model.loss_for_pair(red, blue)?;

        assert_eq!(
            loss.compare_with_tolerance(
                Loss::try_from(2.0_f64.ln())?,
                LossTolerance::try_from(1e-6)?,
            ),
            LossComparison::Close
        );
        Ok(())
    }

    #[test]
    fn bigram_model_times_batch_returns_average_loss() -> Result<(), LmBasicsError> {
        let text = RawText::try_from("red blue red")?;
        let tokens = WhitespaceTokenizer.tokenize(&text)?;
        let vocabulary = Vocabulary::from_tokens(&tokens)?;
        let ids = (&vocabulary * &tokens)?;
        let batch = NextTokenBatch::from_sequence(&ids)?;
        let model = TinyBigramLanguageModel::uniform(vocabulary.size())?;

        let loss_from_method = model.average_loss(&batch)?;
        let loss_from_operator = (&model * &batch)?;

        assert_eq!(
            loss_from_operator
                .compare_with_tolerance(loss_from_method, LossTolerance::try_from(1e-12)?,),
            LossComparison::Close
        );
        Ok(())
    }

    #[test]
    fn one_training_step_lowers_loss_for_same_batch() -> Result<(), LmBasicsError> {
        let text = RawText::try_from("red blue red")?;
        let tokens = WhitespaceTokenizer.tokenize(&text)?;
        let vocabulary = Vocabulary::from_tokens(&tokens)?;
        let ids = (&vocabulary * &tokens)?;
        let batch = NextTokenBatch::from_sequence(&ids)?;
        let mut model = TinyBigramLanguageModel::uniform(vocabulary.size())?;

        let trace = model.train_one_step(&batch, LearningRate::try_from(0.5)?)?;

        assert!(trace.loss_after() < trace.loss_before());
        Ok(())
    }

    #[test]
    fn public_language_modeling_example_accepts_public_text() -> Result<(), LmBasicsError> {
        let example = PublicLanguageModelingExample::from_reviewed_text(
            public_reviewed_text()?,
            WhitespaceTokenizer,
        )?;

        assert_eq!(example.tokens().len(), TokenCount::try_from(3)?);
        assert_eq!(example.vocabulary().size(), VocabularySize::try_from(2)?);
        assert_eq!(
            example.batch().context_length(),
            ContextLength::from_token_count(TokenCount::try_from(2)?)?
        );
        Ok(())
    }

    #[test]
    fn public_language_modeling_example_blocks_restricted_and_private_text()
    -> Result<(), LmBasicsError> {
        let restricted = PublicLanguageModelingExample::from_reviewed_text(
            ReviewedRawText::new(
                RawText::try_from("red blue red")?,
                TextVisibility::ResearchRestricted,
            ),
            WhitespaceTokenizer,
        );
        let private = PublicLanguageModelingExample::from_reviewed_text(
            ReviewedRawText::new(RawText::try_from("red blue red")?, TextVisibility::Private),
            WhitespaceTokenizer,
        );

        assert!(matches!(
            restricted.err(),
            Some(LmBasicsError::InvalidPublicExample { .. })
        ));
        assert!(matches!(
            private.err(),
            Some(LmBasicsError::InvalidPublicExample { .. })
        ));
        Ok(())
    }
}
