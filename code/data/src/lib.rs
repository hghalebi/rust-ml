//! Typed data-pipeline primitives for the CS336 Rust equivalent track.
//!
//! This crate teaches data preparation as a sequence of checked maps:
//!
//! ```text
//! RawDocument -> NormalizedDocument -> FilterDecision -> CorpusShard
//! ```
//!
//! Raw learner strings enter only through `TryFrom` adapters. The public
//! pipeline then talks about semantic values such as [`DocumentId`],
//! [`SourceName`], [`RawText`], [`DedupKey`], [`FilterReason`], and
//! [`MixtureWeight`].

pub mod error;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Add,
};

use error::DataError;

pub use error::DataError as Error;

fn validate_nonempty(
    value: &str,
    operation: &'static str,
    details: &'static str,
) -> Result<(), DataError> {
    if value.trim().is_empty() {
        return Err(DataError::empty_input(operation, details));
    }

    Ok(())
}

fn finite(role: &'static str, value: f64) -> Result<f64, DataError> {
    if !value.is_finite() {
        return Err(DataError::non_finite_value(role, value));
    }

    Ok(value)
}

fn positive_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, DataError> {
    if value == 0 {
        return Err(DataError::empty_input(operation, role));
    }

    Ok(value)
}

/// Stable document identity inside one corpus build.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocumentId(String);

impl DocumentId {
    fn from_owned(value: String) -> Result<Self, DataError> {
        validate_nonempty(
            &value,
            "DocumentId::try_from",
            "document id cannot be empty",
        )?;
        Ok(Self(value))
    }
}

impl TryFrom<&str> for DocumentId {
    type Error = DataError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for DocumentId {
    type Error = DataError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Public source label such as a dataset slice or local file group.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceName(String);

impl SourceName {
    fn from_owned(value: String) -> Result<Self, DataError> {
        validate_nonempty(
            &value,
            "SourceName::try_from",
            "source name cannot be empty",
        )?;
        Ok(Self(value))
    }
}

impl TryFrom<&str> for SourceName {
    type Error = DataError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for SourceName {
    type Error = DataError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for SourceName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Raw document text after boundary validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawText(String);

impl RawText {
    fn from_owned(value: String) -> Result<Self, DataError> {
        validate_nonempty(&value, "RawText::try_from", "raw text cannot be empty")?;
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for RawText {
    type Error = DataError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for RawText {
    type Error = DataError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for RawText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One raw document with source metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawDocument {
    id: DocumentId,
    source: SourceName,
    text: RawText,
}

impl RawDocument {
    /// Creates a raw document from already validated boundary values.
    pub fn new(id: DocumentId, source: SourceName, text: RawText) -> Self {
        Self { id, source, text }
    }

    /// Returns the document identity.
    pub fn id(&self) -> &DocumentId {
        &self.id
    }

    /// Returns the source name.
    pub fn source(&self) -> &SourceName {
        &self.source
    }
}

/// Text after deterministic normalization rules.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NormalizedText(String);

impl NormalizedText {
    fn from_owned(value: String) -> Result<Self, DataError> {
        validate_nonempty(
            &value,
            "NormalizedText::from_owned",
            "normalized text cannot be empty",
        )?;
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NormalizedText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Number of whitespace-separated tokens after normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenCount(usize);

impl TokenCount {
    fn from_raw(value: usize) -> Result<Self, DataError> {
        Ok(Self(positive_usize(
            "token count must be greater than zero",
            "TokenCount::from_raw",
            value,
        )?))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Minimum token count accepted by the filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MinTokenCount(usize);

impl MinTokenCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for MinTokenCount {
    type Error = DataError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(positive_usize(
            "minimum token count must be greater than zero",
            "MinTokenCount::try_from",
            value,
        )?))
    }
}

impl fmt::Display for MinTokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Deterministic duplicate key for normalized text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DedupKey(u64);

impl DedupKey {
    fn from_text(text: &NormalizedText) -> Self {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in text.as_str().bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self(hash)
    }
}

impl fmt::Display for DedupKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016x}", self.0)
    }
}

/// Deterministic normalization rule set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationRules;

impl NormalizationRules {
    /// Normalizes case and whitespace while preserving document metadata.
    pub fn normalize(&self, document: &RawDocument) -> Result<NormalizedDocument, DataError> {
        let normalized = document
            .text
            .as_str()
            .split_whitespace()
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
            .join(" ");
        let normalized_text = NormalizedText::from_owned(normalized)?;
        let token_count =
            TokenCount::from_raw(normalized_text.as_str().split_whitespace().count())?;
        let dedup_key = DedupKey::from_text(&normalized_text);

        Ok(NormalizedDocument {
            id: document.id.clone(),
            source: document.source.clone(),
            text: normalized_text,
            token_count,
            dedup_key,
        })
    }
}

/// One normalized document ready for filtering and deduplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedDocument {
    id: DocumentId,
    source: SourceName,
    text: NormalizedText,
    token_count: TokenCount,
    dedup_key: DedupKey,
}

impl NormalizedDocument {
    /// Returns the document identity.
    pub fn id(&self) -> &DocumentId {
        &self.id
    }

    /// Returns the source name.
    pub fn source(&self) -> &SourceName {
        &self.source
    }

    /// Returns normalized text for learner display.
    pub fn text(&self) -> &NormalizedText {
        &self.text
    }

    /// Returns normalized token count.
    pub fn token_count(&self) -> TokenCount {
        self.token_count
    }

    /// Returns deterministic duplicate key.
    pub fn dedup_key(&self) -> DedupKey {
        self.dedup_key
    }
}

/// Reason a normalized document was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterReason {
    /// The document had too few normalized tokens.
    TooFewTokens {
        /// Observed normalized token count.
        observed: TokenCount,
        /// Required minimum token count.
        minimum: MinTokenCount,
    },
    /// Another document with the same dedup key was already accepted.
    Duplicate {
        /// Previously accepted document.
        existing: DocumentId,
        /// Duplicate key shared by both documents.
        dedup_key: DedupKey,
    },
}

impl fmt::Display for FilterReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewTokens { observed, minimum } => write!(
                formatter,
                "too few tokens: observed {}, minimum {}",
                observed, minimum
            ),
            Self::Duplicate {
                existing,
                dedup_key,
            } => write!(
                formatter,
                "duplicate of {} with key {}",
                existing, dedup_key
            ),
        }
    }
}

/// Accepted normalized document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedDocument(NormalizedDocument);

impl AcceptedDocument {
    /// Returns the accepted normalized document.
    pub fn document(&self) -> &NormalizedDocument {
        &self.0
    }
}

/// Rejected normalized document plus explicit reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedDocument {
    document: NormalizedDocument,
    reason: FilterReason,
}

impl RejectedDocument {
    /// Returns the rejected document.
    pub fn document(&self) -> &NormalizedDocument {
        &self.document
    }

    /// Returns the rejection reason.
    pub fn reason(&self) -> &FilterReason {
        &self.reason
    }
}

/// Filter result for a normalized document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterDecision {
    /// Document is kept.
    Accepted(AcceptedDocument),
    /// Document is rejected with a durable reason.
    Rejected(RejectedDocument),
}

impl FilterDecision {
    /// Returns the source associated with this decision.
    pub fn source(&self) -> &SourceName {
        match self {
            Self::Accepted(document) => document.document().source(),
            Self::Rejected(document) => document.document().source(),
        }
    }
}

/// Tracks accepted deduplication keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupIndex(BTreeMap<DedupKey, DocumentId>);

impl DedupIndex {
    /// Creates an empty deduplication index.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn record_or_find_duplicate(
        &mut self,
        document: &NormalizedDocument,
    ) -> Option<(DocumentId, DedupKey)> {
        if let Some(existing) = self.0.get(&document.dedup_key()) {
            return Some((existing.clone(), document.dedup_key()));
        }
        self.0.insert(document.dedup_key(), document.id().clone());
        None
    }
}

impl Default for DedupIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Corpus filter with typed length and dedup rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocumentFilter {
    minimum_tokens: MinTokenCount,
}

impl DocumentFilter {
    /// Creates a document filter.
    pub fn new(minimum_tokens: MinTokenCount) -> Self {
        Self { minimum_tokens }
    }

    /// Applies length filtering first, then duplicate detection.
    pub fn decide(
        &self,
        document: NormalizedDocument,
        dedup_index: &mut DedupIndex,
    ) -> FilterDecision {
        if document.token_count().as_usize() < self.minimum_tokens.as_usize() {
            return FilterDecision::Rejected(RejectedDocument {
                reason: FilterReason::TooFewTokens {
                    observed: document.token_count(),
                    minimum: self.minimum_tokens,
                },
                document,
            });
        }

        if let Some((existing, dedup_key)) = dedup_index.record_or_find_duplicate(&document) {
            return FilterDecision::Rejected(RejectedDocument {
                reason: FilterReason::Duplicate {
                    existing,
                    dedup_key,
                },
                document,
            });
        }

        FilterDecision::Accepted(AcceptedDocument(document))
    }
}

/// Number of accepted documents in a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AcceptedCount(usize);

impl AcceptedCount {
    fn from_raw(value: usize) -> Result<Self, DataError> {
        Ok(Self(positive_usize(
            "accepted document count must be greater than zero",
            "AcceptedCount::from_raw",
            value,
        )?))
    }
}

impl fmt::Display for AcceptedCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Accepted documents grouped as one training shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusShard {
    documents: Vec<AcceptedDocument>,
    sources: BTreeSet<SourceName>,
}

impl CorpusShard {
    /// Builds a shard from filter decisions while preserving rejection records outside the shard.
    pub fn from_decisions(
        decisions: impl IntoIterator<Item = FilterDecision>,
    ) -> Result<Self, DataError> {
        let mut documents = Vec::new();
        let mut sources = BTreeSet::new();

        for decision in decisions {
            if let FilterDecision::Accepted(document) = decision {
                sources.insert(document.document().source().clone());
                documents.push(document);
            }
        }

        if documents.is_empty() {
            return Err(DataError::empty_input(
                "CorpusShard::from_decisions",
                "corpus shard must contain at least one accepted document",
            ));
        }

        Ok(Self { documents, sources })
    }

    /// Returns accepted document count.
    pub fn accepted_count(&self) -> Result<AcceptedCount, DataError> {
        AcceptedCount::from_raw(self.documents.len())
    }

    /// Iterates over accepted documents.
    pub fn documents(&self) -> impl ExactSizeIterator<Item = &AcceptedDocument> + '_ {
        self.documents.iter()
    }

    /// Iterates over sources represented in the shard.
    pub fn sources(&self) -> impl ExactSizeIterator<Item = &SourceName> + '_ {
        self.sources.iter()
    }
}

/// Non-negative weight for a source in a training mixture.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MixtureWeight(f64);

impl MixtureWeight {
    fn from_raw(value: f64) -> Result<Self, DataError> {
        let value = finite("mixture weight", value)?;
        if value < 0.0 {
            return Err(DataError::out_of_range("mixture weight", ">= 0", value));
        }
        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for MixtureWeight {
    type Error = DataError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl Add for MixtureWeight {
    type Output = Result<MixtureWeight, DataError>;

    fn add(self, right: MixtureWeight) -> Self::Output {
        MixtureWeight::from_raw(self.as_f64() + right.as_f64())
    }
}

impl fmt::Display for MixtureWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.4}", self.0)
    }
}

/// Weight assigned to one source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceWeight {
    source: SourceName,
    weight: MixtureWeight,
}

impl SourceWeight {
    /// Creates a source weight.
    pub fn new(source: SourceName, weight: MixtureWeight) -> Self {
        Self { source, weight }
    }

    /// Returns the source name.
    pub fn source(&self) -> &SourceName {
        &self.source
    }

    /// Returns the source weight.
    pub fn weight(&self) -> MixtureWeight {
        self.weight
    }
}

/// Total positive weight for a sampling mixture.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TotalMixtureWeight(MixtureWeight);

impl fmt::Display for TotalMixtureWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Source sampling mixture with an explicit positive total.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingMixture {
    source_weights: Vec<SourceWeight>,
    total_weight: TotalMixtureWeight,
}

impl SamplingMixture {
    /// Creates a checked source mixture.
    pub fn from_weights(
        source_weights: impl IntoIterator<Item = SourceWeight>,
    ) -> Result<Self, DataError> {
        let source_weights = source_weights.into_iter().collect::<Vec<_>>();
        if source_weights.is_empty() {
            return Err(DataError::empty_input(
                "SamplingMixture::from_weights",
                "sampling mixture must contain at least one source",
            ));
        }

        let mut total = MixtureWeight::try_from(0.0)?;
        for source_weight in &source_weights {
            total = (total + source_weight.weight())?;
        }

        if total.as_f64() <= 0.0 {
            return Err(DataError::out_of_range(
                "total mixture weight",
                "> 0",
                total.as_f64(),
            ));
        }

        Ok(Self {
            source_weights,
            total_weight: TotalMixtureWeight(total),
        })
    }

    /// Iterates over source weights.
    pub fn source_weights(&self) -> impl ExactSizeIterator<Item = &SourceWeight> + '_ {
        self.source_weights.iter()
    }

    /// Returns the positive total weight.
    pub fn total_weight(&self) -> TotalMixtureWeight {
        self.total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CorpusShard, DataError, DedupIndex, DocumentFilter, DocumentId, FilterDecision,
        FilterReason, MinTokenCount, MixtureWeight, NormalizationRules, RawDocument, RawText,
        SamplingMixture, SourceName, SourceWeight,
    };

    fn document(id: DocumentId, source: SourceName, text: RawText) -> RawDocument {
        RawDocument::new(id, source, text)
    }

    #[test]
    fn normalization_is_deterministic() -> Result<(), DataError> {
        let rules = NormalizationRules;
        let source = SourceName::try_from("public-notes")?;
        let first = rules.normalize(&document(
            DocumentId::try_from("doc-a")?,
            source.clone(),
            RawText::try_from("  Rust   ML\nMaps  ")?,
        ))?;
        let second = rules.normalize(&document(
            DocumentId::try_from("doc-a")?,
            source,
            RawText::try_from("Rust ML Maps")?,
        ))?;

        assert_eq!(first.text().to_string(), "rust ml maps");
        assert_eq!(first.text(), second.text());
        assert_eq!(first.dedup_key(), second.dedup_key());
        Ok(())
    }

    #[test]
    fn filter_reasons_are_recorded() -> Result<(), DataError> {
        let rules = NormalizationRules;
        let document = rules.normalize(&document(
            DocumentId::try_from("doc-short")?,
            SourceName::try_from("public-notes")?,
            RawText::try_from("tiny")?,
        ))?;
        let filter = DocumentFilter::new(MinTokenCount::try_from(2)?);
        let mut dedup_index = DedupIndex::new();

        let decision = filter.decide(document, &mut dedup_index);

        assert!(matches!(
            decision,
            FilterDecision::Rejected(rejected)
                if matches!(rejected.reason(), FilterReason::TooFewTokens { .. })
        ));
        Ok(())
    }

    #[test]
    fn duplicate_documents_share_a_key_and_get_rejected() -> Result<(), DataError> {
        let rules = NormalizationRules;
        let source = SourceName::try_from("public-notes")?;
        let first = rules.normalize(&document(
            DocumentId::try_from("doc-a")?,
            source.clone(),
            RawText::try_from("Same   PUBLIC text")?,
        ))?;
        let second = rules.normalize(&document(
            DocumentId::try_from("doc-b")?,
            source,
            RawText::try_from("same public text")?,
        ))?;
        let first_key = first.dedup_key();
        let second_key = second.dedup_key();
        let filter = DocumentFilter::new(MinTokenCount::try_from(2)?);
        let mut dedup_index = DedupIndex::new();

        let first_decision = filter.decide(first, &mut dedup_index);
        let second_decision = filter.decide(second, &mut dedup_index);

        assert_eq!(first_key, second_key);
        assert!(matches!(first_decision, FilterDecision::Accepted(_)));
        assert!(matches!(
            second_decision,
            FilterDecision::Rejected(rejected)
                if matches!(rejected.reason(), FilterReason::Duplicate { .. })
        ));
        Ok(())
    }

    #[test]
    fn corpus_shard_keeps_only_accepted_documents() -> Result<(), DataError> {
        let rules = NormalizationRules;
        let filter = DocumentFilter::new(MinTokenCount::try_from(2)?);
        let mut dedup_index = DedupIndex::new();
        let source = SourceName::try_from("public-notes")?;
        let decisions = [
            filter.decide(
                rules.normalize(&document(
                    DocumentId::try_from("doc-a")?,
                    source.clone(),
                    RawText::try_from("first public document")?,
                ))?,
                &mut dedup_index,
            ),
            filter.decide(
                rules.normalize(&document(
                    DocumentId::try_from("doc-b")?,
                    source,
                    RawText::try_from("short")?,
                ))?,
                &mut dedup_index,
            ),
        ];

        let shard = CorpusShard::from_decisions(decisions)?;

        assert_eq!(shard.accepted_count()?.to_string(), "1");
        assert_eq!(shard.sources().len(), 1);
        Ok(())
    }

    #[test]
    fn mixture_weights_must_have_positive_total() -> Result<(), DataError> {
        let empty_total = SamplingMixture::from_weights([
            SourceWeight::new(
                SourceName::try_from("source-a")?,
                MixtureWeight::try_from(0.0)?,
            ),
            SourceWeight::new(
                SourceName::try_from("source-b")?,
                MixtureWeight::try_from(0.0)?,
            ),
        ])
        .err();

        assert!(matches!(empty_total, Some(DataError::OutOfRange { .. })));

        let mixture = SamplingMixture::from_weights([
            SourceWeight::new(
                SourceName::try_from("source-a")?,
                MixtureWeight::try_from(0.25)?,
            ),
            SourceWeight::new(
                SourceName::try_from("source-b")?,
                MixtureWeight::try_from(0.75)?,
            ),
        ])?;

        assert_eq!(mixture.total_weight().to_string(), "1.0000");
        Ok(())
    }
}
