//! A small, typed attention teaching crate.
//!
//! The crate focuses on one attention head before the full Transformer:
//! a token asks a query, compares it with keys, turns scores into weights,
//! then mixes value vectors.
//!
//! Raw learner literals enter through explicit `TryFrom` adapters. Public
//! teaching APIs then move through semantic values such as [`TokenComponent`],
//! [`QueryComponent`], [`KeyComponent`], [`ValueComponent`], [`AttentionScore`],
//! [`AttentionWeight`], [`VectorWidth`], and [`TokenIndex`].

pub mod error;

use std::{
    fmt,
    ops::{Add, Mul},
};

use error::AttentionError;

pub use error::AttentionError as Error;

fn finite(role: &'static str, value: f64) -> Result<f64, AttentionError> {
    if !value.is_finite() {
        return Err(AttentionError::non_finite_value(role, value));
    }

    Ok(value)
}

macro_rules! finite_scalar {
    ($name:ident, $doc:literal, $role:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            fn from_raw(value: f64) -> Result<Self, AttentionError> {
                Ok(Self(finite($role, value)?))
            }

            fn as_f64(self) -> f64 {
                self.0
            }
        }

        impl TryFrom<f64> for $name {
            type Error = AttentionError;

            fn try_from(value: f64) -> Result<Self, Self::Error> {
                Self::from_raw(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, formatter)
            }
        }
    };
}

finite_scalar!(
    TokenComponent,
    "One coordinate of a token embedding.",
    "token component"
);
finite_scalar!(
    QueryComponent,
    "One coordinate of a query vector.",
    "query component"
);
finite_scalar!(
    KeyComponent,
    "One coordinate of a key vector.",
    "key component"
);
finite_scalar!(
    ValueComponent,
    "One coordinate of a value vector.",
    "value component"
);
finite_scalar!(
    ProjectionWeight,
    "One learned projection matrix entry.",
    "projection weight"
);
finite_scalar!(
    ProjectionBiasValue,
    "One learned projection bias entry.",
    "projection bias"
);
finite_scalar!(
    ProjectionProduct,
    "One token component multiplied by one projection weight.",
    "projection product"
);
finite_scalar!(
    ProjectionOutput,
    "One projection output coordinate before role conversion.",
    "projection output"
);
finite_scalar!(
    ScoreContribution,
    "One query component multiplied by one key component.",
    "score contribution"
);
finite_scalar!(
    AttentionScore,
    "A scaled query-key compatibility score.",
    "attention score"
);
/// A normalized softmax attention weight.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AttentionWeight(f64);

impl AttentionWeight {
    fn as_f64(self) -> f64 {
        self.0
    }

    #[cfg(test)]
    fn is_close_to(self, expected: Self, tolerance: Self) -> bool {
        (self.as_f64() - expected.as_f64()).abs() <= tolerance.as_f64().abs()
    }
}

finite_scalar!(
    WeightedValueComponent,
    "One value component scaled by one attention weight.",
    "weighted value component"
);
finite_scalar!(
    AttentionOutputComponent,
    "One coordinate of the mixed attention output.",
    "attention output component"
);

impl ProjectionOutput {
    fn zero() -> Result<Self, AttentionError> {
        Self::from_raw(0.0)
    }
}

impl AttentionScore {
    fn zero() -> Result<Self, AttentionError> {
        Self::from_raw(0.0)
    }

    #[cfg(test)]
    fn is_close_to(self, expected: Self, tolerance: Self) -> bool {
        (self.as_f64() - expected.as_f64()).abs() <= tolerance.as_f64().abs()
    }

    fn scaled_by(self, width: VectorWidth) -> Result<Self, AttentionError> {
        Self::from_raw(self.as_f64() / (width.as_usize() as f64).sqrt())
    }
}

impl AttentionOutputComponent {
    fn zero() -> Result<Self, AttentionError> {
        Self::from_raw(0.0)
    }
}

impl TryFrom<f64> for AttentionWeight {
    type Error = AttentionError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let value = finite("attention weight", value)?;
        if !(0.0..=1.0).contains(&value) {
            return Err(AttentionError::out_of_range(
                "attention weight",
                "0..=1",
                value,
            ));
        }

        Ok(Self(value))
    }
}

impl fmt::Display for AttentionWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl From<ProjectionOutput> for QueryComponent {
    fn from(value: ProjectionOutput) -> Self {
        Self(value.as_f64())
    }
}

impl From<ProjectionOutput> for KeyComponent {
    fn from(value: ProjectionOutput) -> Self {
        Self(value.as_f64())
    }
}

impl From<ProjectionOutput> for ValueComponent {
    fn from(value: ProjectionOutput) -> Self {
        Self(value.as_f64())
    }
}

impl Mul<ProjectionWeight> for TokenComponent {
    type Output = Result<ProjectionProduct, AttentionError>;

    fn mul(self, right: ProjectionWeight) -> Self::Output {
        ProjectionProduct::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Add<ProjectionProduct> for ProjectionOutput {
    type Output = Result<ProjectionOutput, AttentionError>;

    fn add(self, right: ProjectionProduct) -> Self::Output {
        ProjectionOutput::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Add<ProjectionBiasValue> for ProjectionOutput {
    type Output = Result<ProjectionOutput, AttentionError>;

    fn add(self, right: ProjectionBiasValue) -> Self::Output {
        ProjectionOutput::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Mul<KeyComponent> for QueryComponent {
    type Output = Result<ScoreContribution, AttentionError>;

    fn mul(self, right: KeyComponent) -> Self::Output {
        ScoreContribution::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Add<ScoreContribution> for AttentionScore {
    type Output = Result<AttentionScore, AttentionError>;

    fn add(self, right: ScoreContribution) -> Self::Output {
        AttentionScore::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Mul<ValueComponent> for AttentionWeight {
    type Output = Result<WeightedValueComponent, AttentionError>;

    fn mul(self, right: ValueComponent) -> Self::Output {
        WeightedValueComponent::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Add<WeightedValueComponent> for AttentionOutputComponent {
    type Output = Result<AttentionOutputComponent, AttentionError>;

    fn add(self, right: WeightedValueComponent) -> Self::Output {
        AttentionOutputComponent::from_raw(self.as_f64() + right.as_f64())
    }
}

/// Width of a vector space in the attention example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VectorWidth(usize);

impl VectorWidth {
    fn from_raw(value: usize) -> Result<Self, AttentionError> {
        if value == 0 {
            return Err(AttentionError::empty_input(
                "VectorWidth::try_from",
                "width must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for VectorWidth {
    type Error = AttentionError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for VectorWidth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of tokens, scores, weights, or value vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenCount(usize);

impl TokenCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Index of a query token in a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenIndex(usize);

impl TokenIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TokenIndex {
    type Error = AttentionError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for TokenIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of projection rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowCount(usize);

impl RowCount {
    fn from_raw(value: usize) -> Result<Self, AttentionError> {
        if value == 0 {
            return Err(AttentionError::empty_input(
                "RowCount::from_raw",
                "row count must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for RowCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Shape of a projection matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    rows: RowCount,
    cols: VectorWidth,
}

impl MatrixShape {
    fn new(rows: RowCount, cols: VectorWidth) -> Self {
        Self { rows, cols }
    }

    /// Returns row count.
    pub fn rows(self) -> RowCount {
        self.rows
    }

    /// Returns column count.
    pub fn cols(self) -> VectorWidth {
        self.cols
    }
}

impl fmt::Display for MatrixShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}x{}", self.rows, self.cols)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DenseVector<T> {
    values: Vec<T>,
}

impl<T> DenseVector<T> {
    fn from_values(
        operation: &'static str,
        values: impl IntoIterator<Item = T>,
    ) -> Result<Self, AttentionError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(AttentionError::empty_input(
                operation,
                "vector cannot be empty",
            ));
        }

        Ok(Self { values })
    }

    fn width(&self) -> VectorWidth {
        VectorWidth(self.values.len())
    }

    fn values(&self) -> impl ExactSizeIterator<Item = &T> + '_ {
        self.values.iter()
    }
}

macro_rules! vector_role {
    ($name:ident, $component:ident, $operation:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(DenseVector<$component>);

        impl $name {
            /// Creates the typed vector after checking it is non-empty and finite.
            pub fn from_values(
                values: impl IntoIterator<Item = $component>,
            ) -> Result<Self, AttentionError> {
                Ok(Self(DenseVector::from_values($operation, values)?))
            }

            /// Returns the vector width.
            pub fn width(&self) -> VectorWidth {
                self.0.width()
            }

            /// Iterates over values for printing and hand checks.
            pub fn values(&self) -> impl ExactSizeIterator<Item = &$component> + '_ {
                self.0.values()
            }
        }
    };
}

vector_role!(
    TokenEmbedding,
    TokenComponent,
    "TokenEmbedding::from_values",
    "A token representation before attention."
);
vector_role!(
    Query,
    QueryComponent,
    "Query::from_values",
    "A token's question about what information it needs."
);
vector_role!(
    Key,
    KeyComponent,
    "Key::from_values",
    "A token's match label for other queries."
);
vector_role!(
    Value,
    ValueComponent,
    "Value::from_values",
    "The information a token can contribute."
);
vector_role!(
    ProjectionBias,
    ProjectionBiasValue,
    "ProjectionBias::from_values",
    "A bias vector used by a query/key/value projection."
);
vector_role!(
    AttentionOutput,
    AttentionOutputComponent,
    "AttentionOutput::from_values",
    "The value mixture produced for one query token."
);

/// A non-empty same-width sequence of token embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSequence {
    tokens: Vec<TokenEmbedding>,
    token_width: VectorWidth,
}

impl TokenSequence {
    /// Creates a token sequence and checks every token has the same width.
    pub fn from_tokens(
        tokens: impl IntoIterator<Item = TokenEmbedding>,
    ) -> Result<Self, AttentionError> {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        if tokens.is_empty() {
            return Err(AttentionError::empty_input(
                "TokenSequence::from_tokens",
                "sequence cannot be empty",
            ));
        }

        let token_width = tokens[0].width();
        for token in &tokens {
            if token.width() != token_width {
                return Err(AttentionError::dimension_mismatch(
                    "TokenSequence::from_tokens",
                    "first token",
                    vec![token_width.as_usize()],
                    "later token",
                    vec![token.width().as_usize()],
                    "all tokens in a sequence must share the same width; a later token had a different width",
                ));
            }
        }

        Ok(Self {
            tokens,
            token_width,
        })
    }

    /// Returns the token count.
    pub fn len(&self) -> TokenCount {
        TokenCount(self.tokens.len())
    }

    /// Returns the shared token width.
    pub fn token_width(&self) -> VectorWidth {
        self.token_width
    }

    /// Iterates over all tokens.
    pub fn tokens(&self) -> impl ExactSizeIterator<Item = &TokenEmbedding> + '_ {
        self.tokens.iter()
    }

    fn get(&self, index: TokenIndex) -> Result<&TokenEmbedding, AttentionError> {
        self.tokens
            .get(index.as_usize())
            .ok_or(AttentionError::invalid_token_index(
                "TokenSequence::get",
                index.as_usize(),
                self.tokens.len(),
            ))
    }
}

/// Raw attention scores before softmax.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionScores(Vec<AttentionScore>);

impl AttentionScores {
    /// Creates non-empty finite attention scores.
    pub fn from_scores(
        values: impl IntoIterator<Item = AttentionScore>,
    ) -> Result<Self, AttentionError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(AttentionError::empty_input(
                "AttentionScores::from_scores",
                "scores cannot be empty",
            ));
        }

        Ok(Self(values))
    }

    /// Iterates over score values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &AttentionScore> + '_ {
        self.0.iter()
    }

    /// Returns score count.
    pub fn len(&self) -> TokenCount {
        TokenCount(self.0.len())
    }
}

/// Normalized attention weights after softmax.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionWeights(Vec<AttentionWeight>);

impl AttentionWeights {
    /// Creates non-empty, finite, non-negative weights that sum to one.
    pub fn from_weights(
        values: impl IntoIterator<Item = AttentionWeight>,
    ) -> Result<Self, AttentionError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(AttentionError::empty_input(
                "AttentionWeights::from_weights",
                "weights cannot be empty",
            ));
        }

        let sum: f64 = values.iter().map(|value| value.as_f64()).sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(AttentionError::numerical_issue(
                "AttentionWeights::from_weights",
                "attention weights must sum to one",
            ));
        }

        Ok(Self(values))
    }

    /// Iterates over weight values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &AttentionWeight> + '_ {
        self.0.iter()
    }

    /// Returns weight count.
    pub fn len(&self) -> TokenCount {
        TokenCount(self.0.len())
    }

    #[cfg(test)]
    fn sum_is_close_to(&self, expected: AttentionWeight, tolerance: AttentionWeight) -> bool {
        let initial = match AttentionWeight::try_from(0.0) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let sum = self.values().copied().try_fold(initial, |sum, value| {
            AttentionWeight::try_from(sum.as_f64() + value.as_f64())
        });

        matches!(sum, Ok(actual) if actual.is_close_to(expected, tolerance))
    }
}

/// A non-empty same-width sequence of value vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSequence {
    values: Vec<Value>,
    value_width: VectorWidth,
}

impl ValueSequence {
    /// Creates a checked sequence of value vectors.
    pub fn from_values(values: impl IntoIterator<Item = Value>) -> Result<Self, AttentionError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(AttentionError::empty_input(
                "ValueSequence::from_values",
                "values cannot be empty",
            ));
        }

        let value_width = values[0].width();
        for value in &values {
            if value.width() != value_width {
                return Err(AttentionError::dimension_mismatch(
                    "ValueSequence::from_values",
                    "first value",
                    vec![value_width.as_usize()],
                    "next value",
                    vec![value.width().as_usize()],
                    "all value vectors must have the same width",
                ));
            }
        }

        Ok(Self {
            values,
            value_width,
        })
    }

    /// Returns value vector count.
    pub fn len(&self) -> TokenCount {
        TokenCount(self.values.len())
    }

    /// Returns shared value width.
    pub fn value_width(&self) -> VectorWidth {
        self.value_width
    }

    /// Iterates over values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &Value> + '_ {
        self.values.iter()
    }
}

impl<'b> Mul<&'b Key> for &Query {
    type Output = Result<AttentionScore, AttentionError>;

    fn mul(self, right: &'b Key) -> Self::Output {
        if self.width() != right.width() {
            return Err(AttentionError::dimension_mismatch(
                "Query::mul_key",
                "query",
                vec![self.width().as_usize()],
                "key",
                vec![right.width().as_usize()],
                "dot product requires equal vector widths",
            ));
        }

        let mut score = AttentionScore::zero()?;
        for (query_component, key_component) in self.values().zip(right.values()) {
            let contribution = (*query_component * *key_component)?;
            score = (score + contribution)?;
        }

        score.scaled_by(self.width())
    }
}

impl<'b> Mul<&'b ValueSequence> for &AttentionWeights {
    type Output = Result<AttentionOutput, AttentionError>;

    fn mul(self, right: &'b ValueSequence) -> Self::Output {
        if self.len() != right.len() {
            return Err(AttentionError::dimension_mismatch(
                "AttentionWeights::mul_value_sequence",
                "attention weights",
                vec![self.len().as_usize()],
                "value vectors",
                vec![right.len().as_usize()],
                "one weight is required for each value vector",
            ));
        }

        let mut output = (0..right.value_width().as_usize())
            .map(|_| AttentionOutputComponent::zero())
            .collect::<Result<Vec<_>, _>>()?;

        for (weight, value) in self.values().zip(right.values()) {
            for (index, value_component) in value.values().enumerate() {
                let contribution = (*weight * *value_component)?;
                output[index] = (output[index] + contribution)?;
            }
        }

        AttentionOutput::from_values(output)
    }
}

/// Outputs produced by running an attention head over a full sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOutputSequence(Vec<AttentionOutput>);

impl AttentionOutputSequence {
    fn from_outputs(
        outputs: impl IntoIterator<Item = AttentionOutput>,
    ) -> Result<Self, AttentionError> {
        let outputs = outputs.into_iter().collect::<Vec<_>>();
        if outputs.is_empty() {
            return Err(AttentionError::empty_input(
                "AttentionOutputSequence::from_outputs",
                "outputs cannot be empty",
            ));
        }

        Ok(Self(outputs))
    }

    /// Iterates over attention outputs.
    pub fn outputs(&self) -> impl ExactSizeIterator<Item = &AttentionOutput> + '_ {
        self.0.iter()
    }
}

/// One row of a projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionRow(DenseVector<ProjectionWeight>);

impl ProjectionRow {
    /// Creates one non-empty finite projection row.
    pub fn from_values(
        values: impl IntoIterator<Item = ProjectionWeight>,
    ) -> Result<Self, AttentionError> {
        Ok(Self(DenseVector::from_values(
            "ProjectionRow::from_values",
            values,
        )?))
    }

    fn width(&self) -> VectorWidth {
        self.0.width()
    }

    fn values(&self) -> impl ExactSizeIterator<Item = &ProjectionWeight> + '_ {
        self.0.values()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ProjectionMatrix {
    rows: Vec<ProjectionRow>,
    shape: MatrixShape,
}

impl ProjectionMatrix {
    fn from_rows(
        operation: &'static str,
        rows: impl IntoIterator<Item = ProjectionRow>,
    ) -> Result<Self, AttentionError> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        if rows.is_empty() {
            return Err(AttentionError::empty_input(
                operation,
                "matrix cannot be empty",
            ));
        }

        let cols = rows[0].width();
        for row in &rows {
            if row.width() != cols {
                return Err(AttentionError::invalid_matrix_data(
                    operation,
                    "all rows must have the same length",
                ));
            }
        }

        let shape = MatrixShape::new(RowCount::from_raw(rows.len())?, cols);
        Ok(Self { rows, shape })
    }

    fn identity(width: VectorWidth) -> Result<Self, AttentionError> {
        let mut rows = Vec::with_capacity(width.as_usize());
        for row_index in 0..width.as_usize() {
            let row = (0..width.as_usize())
                .map(|col_index| {
                    ProjectionWeight::try_from(if row_index == col_index { 1.0 } else { 0.0 })
                })
                .collect::<Result<Vec<_>, _>>()?;
            rows.push(ProjectionRow::from_values(row)?);
        }

        Self::from_rows("ProjectionMatrix::identity", rows)
    }

    fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn project(
        &self,
        token: &TokenEmbedding,
    ) -> Result<DenseVector<ProjectionOutput>, AttentionError> {
        if self.shape.cols() != token.width() {
            return Err(AttentionError::dimension_mismatch(
                "ProjectionMatrix::project",
                "projection matrix",
                vec![self.shape.rows().as_usize(), self.shape.cols().as_usize()],
                "token embedding",
                vec![token.width().as_usize()],
                "matrix columns must equal incoming token width",
            ));
        }

        let token_values = token.values().copied().collect::<Vec<_>>();
        let mut output = Vec::with_capacity(self.shape.rows().as_usize());
        for row in &self.rows {
            let mut sum = ProjectionOutput::zero()?;
            for (component, weight) in token_values.iter().zip(row.values()) {
                let product = (*component * *weight)?;
                sum = (sum + product)?;
            }
            output.push(sum);
        }

        DenseVector::from_values("ProjectionMatrix::project", output)
    }
}

macro_rules! matrix_role {
    ($name:ident, $doc:literal, $operation:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(ProjectionMatrix);

        impl $name {
            /// Creates a typed projection matrix from rows.
            pub fn from_rows(
                rows: impl IntoIterator<Item = ProjectionRow>,
            ) -> Result<Self, AttentionError> {
                Ok(Self(ProjectionMatrix::from_rows($operation, rows)?))
            }

            /// Creates an identity projection matrix.
            pub fn identity(width: VectorWidth) -> Result<Self, AttentionError> {
                Ok(Self(ProjectionMatrix::identity(width)?))
            }

            /// Returns matrix shape.
            pub fn shape(&self) -> MatrixShape {
                self.0.shape()
            }
        }
    };
}

matrix_role!(
    QueryProjection,
    "Projection matrix from token embeddings to queries.",
    "QueryProjection::from_rows"
);
matrix_role!(
    KeyProjection,
    "Projection matrix from token embeddings to keys.",
    "KeyProjection::from_rows"
);
matrix_role!(
    ValueProjection,
    "Projection matrix from token embeddings to values.",
    "ValueProjection::from_rows"
);

fn validate_projection(
    operation: &'static str,
    rows: RowCount,
    bias: &ProjectionBias,
) -> Result<(), AttentionError> {
    if rows.as_usize() != bias.width().as_usize() {
        return Err(AttentionError::invalid_projection(
            operation,
            "projection matrix rows must match bias width",
        ));
    }

    Ok(())
}

fn add_projection_bias(
    operation: &'static str,
    projected: DenseVector<ProjectionOutput>,
    bias: &ProjectionBias,
) -> Result<DenseVector<ProjectionOutput>, AttentionError> {
    if projected.width() != bias.width() {
        return Err(AttentionError::dimension_mismatch(
            operation,
            "projection output",
            vec![projected.width().as_usize()],
            "projection bias",
            vec![bias.width().as_usize()],
            "projection output and bias must have the same width",
        ));
    }

    DenseVector::from_values(
        operation,
        projected
            .values()
            .zip(bias.values())
            .map(|(left, right)| *left + *right)
            .collect::<Result<Vec<_>, _>>()?,
    )
}

/// A typed projection from token embeddings into query space.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryLayer {
    matrix: QueryProjection,
    bias: ProjectionBias,
}

impl QueryLayer {
    /// Creates a query projection layer.
    pub fn new(matrix: QueryProjection, bias: ProjectionBias) -> Result<Self, AttentionError> {
        validate_projection("QueryLayer::new", matrix.shape().rows(), &bias)?;
        Ok(Self { matrix, bias })
    }

    /// Creates an identity query projection.
    pub fn identity(width: VectorWidth) -> Result<Self, AttentionError> {
        Self::new(
            QueryProjection::identity(width)?,
            ProjectionBias::from_values(
                (0..width.as_usize())
                    .map(|_| ProjectionBiasValue::try_from(0.0))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
        )
    }

    /// Returns expected token width.
    pub fn input_dim(&self) -> VectorWidth {
        self.matrix.shape().cols()
    }

    /// Returns query width.
    pub fn output_dim(&self) -> VectorWidth {
        VectorWidth(self.matrix.shape().rows().as_usize())
    }

    /// Projects one token into a query.
    pub fn forward(&self, token: &TokenEmbedding) -> Result<Query, AttentionError> {
        let projected = self.matrix.0.project(token)?;
        let with_bias = add_projection_bias("QueryLayer::forward", projected, &self.bias)?;
        Query::from_values(with_bias.values().copied().map(QueryComponent::from))
    }
}

/// A typed projection from token embeddings into key space.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyLayer {
    matrix: KeyProjection,
    bias: ProjectionBias,
}

impl KeyLayer {
    /// Creates a key projection layer.
    pub fn new(matrix: KeyProjection, bias: ProjectionBias) -> Result<Self, AttentionError> {
        validate_projection("KeyLayer::new", matrix.shape().rows(), &bias)?;
        Ok(Self { matrix, bias })
    }

    /// Creates an identity key projection.
    pub fn identity(width: VectorWidth) -> Result<Self, AttentionError> {
        Self::new(
            KeyProjection::identity(width)?,
            ProjectionBias::from_values(
                (0..width.as_usize())
                    .map(|_| ProjectionBiasValue::try_from(0.0))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
        )
    }

    /// Returns expected token width.
    pub fn input_dim(&self) -> VectorWidth {
        self.matrix.shape().cols()
    }

    /// Returns key width.
    pub fn output_dim(&self) -> VectorWidth {
        VectorWidth(self.matrix.shape().rows().as_usize())
    }

    /// Projects one token into a key.
    pub fn forward(&self, token: &TokenEmbedding) -> Result<Key, AttentionError> {
        let projected = self.matrix.0.project(token)?;
        let with_bias = add_projection_bias("KeyLayer::forward", projected, &self.bias)?;
        Key::from_values(with_bias.values().copied().map(KeyComponent::from))
    }
}

/// A typed projection from token embeddings into value space.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueLayer {
    matrix: ValueProjection,
    bias: ProjectionBias,
}

impl ValueLayer {
    /// Creates a value projection layer.
    pub fn new(matrix: ValueProjection, bias: ProjectionBias) -> Result<Self, AttentionError> {
        validate_projection("ValueLayer::new", matrix.shape().rows(), &bias)?;
        Ok(Self { matrix, bias })
    }

    /// Creates an identity value projection.
    pub fn identity(width: VectorWidth) -> Result<Self, AttentionError> {
        Self::new(
            ValueProjection::identity(width)?,
            ProjectionBias::from_values(
                (0..width.as_usize())
                    .map(|_| ProjectionBiasValue::try_from(0.0))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
        )
    }

    /// Returns expected token width.
    pub fn input_dim(&self) -> VectorWidth {
        self.matrix.shape().cols()
    }

    /// Returns value width.
    pub fn output_dim(&self) -> VectorWidth {
        VectorWidth(self.matrix.shape().rows().as_usize())
    }

    /// Projects one token into a value.
    pub fn forward(&self, token: &TokenEmbedding) -> Result<Value, AttentionError> {
        let projected = self.matrix.0.project(token)?;
        let with_bias = add_projection_bias("ValueLayer::forward", projected, &self.bias)?;
        Value::from_values(with_bias.values().copied().map(ValueComponent::from))
    }
}

/// Computes the scaled dot-product score between one query and one key.
pub fn scaled_attention_score(query: &Query, key: &Key) -> Result<AttentionScore, AttentionError> {
    query * key
}

/// Converts scores into normalized focus weights.
pub fn softmax(scores: &AttentionScores) -> Result<AttentionWeights, AttentionError> {
    let max = scores
        .values()
        .map(|score| score.as_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_values = scores
        .values()
        .map(|score| (score.as_f64() - max).exp())
        .collect::<Vec<_>>();
    let sum: f64 = exp_values.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return Err(AttentionError::numerical_issue(
            "softmax",
            "normalization sum must be positive and finite",
        ));
    }

    AttentionWeights::from_weights(
        exp_values
            .into_iter()
            .map(|value| AttentionWeight::try_from(value / sum))
            .collect::<Result<Vec<_>, _>>()?,
    )
}

/// Mixes values according to attention weights.
pub fn weighted_sum(
    weights: &AttentionWeights,
    values: &ValueSequence,
) -> Result<AttentionOutput, AttentionError> {
    weights * values
}

/// Learner-visible intermediate values for one query token.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionTrace {
    query_index: TokenIndex,
    scores: AttentionScores,
    weights: AttentionWeights,
    output: AttentionOutput,
}

impl AttentionTrace {
    /// Returns the token index that asked the query.
    pub fn query_index(&self) -> TokenIndex {
        self.query_index
    }

    /// Raw scores against every key.
    pub fn scores(&self) -> &AttentionScores {
        &self.scores
    }

    /// Normalized attention weights.
    pub fn weights(&self) -> &AttentionWeights {
        &self.weights
    }

    /// Mixed value vector.
    pub fn output(&self) -> &AttentionOutput {
        &self.output
    }
}

/// One scaled dot-product attention head.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionHead {
    query_layer: QueryLayer,
    key_layer: KeyLayer,
    value_layer: ValueLayer,
}

impl AttentionHead {
    /// Creates an attention head with compatible query, key, and value projections.
    pub fn new(
        query_layer: QueryLayer,
        key_layer: KeyLayer,
        value_layer: ValueLayer,
    ) -> Result<Self, AttentionError> {
        if query_layer.input_dim() != key_layer.input_dim()
            || query_layer.input_dim() != value_layer.input_dim()
        {
            return Err(AttentionError::invalid_projection(
                "AttentionHead::new",
                "query, key, and value layers must accept the same token width",
            ));
        }

        if query_layer.output_dim() != key_layer.output_dim() {
            return Err(AttentionError::invalid_projection(
                "AttentionHead::new",
                "query and key layers must produce the same score width",
            ));
        }

        Ok(Self {
            query_layer,
            key_layer,
            value_layer,
        })
    }

    /// Creates an identity attention head for tiny hand-checkable examples.
    pub fn identity(width: VectorWidth) -> Result<Self, AttentionError> {
        Self::new(
            QueryLayer::identity(width)?,
            KeyLayer::identity(width)?,
            ValueLayer::identity(width)?,
        )
    }

    /// Returns expected token width.
    pub fn input_dim(&self) -> VectorWidth {
        self.query_layer.input_dim()
    }

    /// Returns query/key width used for scores.
    pub fn score_dim(&self) -> VectorWidth {
        self.query_layer.output_dim()
    }

    /// Returns value width of the output.
    pub fn value_dim(&self) -> VectorWidth {
        self.value_layer.output_dim()
    }

    /// Returns an attention trace for one query token.
    pub fn trace_token(
        &self,
        sequence: &TokenSequence,
        query_index: TokenIndex,
    ) -> Result<AttentionTrace, AttentionError> {
        if sequence.token_width() != self.input_dim() {
            return Err(AttentionError::dimension_mismatch(
                "AttentionHead::trace_token",
                "attention head input width",
                vec![self.input_dim().as_usize()],
                "token sequence width",
                vec![sequence.token_width().as_usize()],
                "the attention head must accept this sequence's token width",
            ));
        }

        sequence.get(query_index)?;

        let queries = sequence
            .tokens()
            .map(|token| self.query_layer.forward(token))
            .collect::<Result<Vec<_>, _>>()?;
        let keys = sequence
            .tokens()
            .map(|token| self.key_layer.forward(token))
            .collect::<Result<Vec<_>, _>>()?;
        let values = ValueSequence::from_values(
            sequence
                .tokens()
                .map(|token| self.value_layer.forward(token))
                .collect::<Result<Vec<_>, _>>()?,
        )?;

        let scores = AttentionScores::from_scores(
            keys.iter()
                .map(|key| &queries[query_index.as_usize()] * key)
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let weights = softmax(&scores)?;
        let output = (&weights * &values)?;

        Ok(AttentionTrace {
            query_index,
            scores,
            weights,
            output,
        })
    }

    /// Runs the attention head for every token in the sequence.
    pub fn forward(
        &self,
        sequence: &TokenSequence,
    ) -> Result<AttentionOutputSequence, AttentionError> {
        AttentionOutputSequence::from_outputs(
            (0..sequence.len().as_usize())
                .map(|query_index| {
                    self.trace_token(sequence, TokenIndex::try_from(query_index)?)
                        .map(|trace| trace.output)
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionError, AttentionHead, AttentionOutputComponent, AttentionScore, AttentionScores,
        AttentionWeight, AttentionWeights, Key, KeyComponent, KeyLayer, KeyProjection,
        ProjectionBias, ProjectionBiasValue, ProjectionRow, ProjectionWeight, Query,
        QueryComponent, QueryLayer, QueryProjection, TokenComponent, TokenEmbedding, TokenIndex,
        TokenSequence, Value, ValueComponent, ValueLayer, ValueProjection, ValueSequence,
        VectorWidth, softmax,
    };

    fn assert_close_score(
        left: AttentionScore,
        right: AttentionScore,
    ) -> Result<(), AttentionError> {
        assert!(left.is_close_to(right, AttentionScore::try_from(1e-6)?));
        Ok(())
    }

    fn token(
        left: TokenComponent,
        right: TokenComponent,
    ) -> Result<TokenEmbedding, AttentionError> {
        TokenEmbedding::from_values([left, right])
    }

    fn query(left: QueryComponent, right: QueryComponent) -> Result<Query, AttentionError> {
        Query::from_values([left, right])
    }

    fn key(left: KeyComponent, right: KeyComponent) -> Result<Key, AttentionError> {
        Key::from_values([left, right])
    }

    fn value(left: ValueComponent, right: ValueComponent) -> Result<Value, AttentionError> {
        Value::from_values([left, right])
    }

    fn projection_row(values: [ProjectionWeight; 2]) -> Result<ProjectionRow, AttentionError> {
        ProjectionRow::from_values(values)
    }

    fn projection_bias(value: ProjectionBiasValue) -> Result<ProjectionBias, AttentionError> {
        ProjectionBias::from_values([value])
    }

    fn sequence() -> Result<TokenSequence, AttentionError> {
        TokenSequence::from_tokens([
            token(
                TokenComponent::try_from(1.0)?,
                TokenComponent::try_from(0.0)?,
            )?,
            token(
                TokenComponent::try_from(0.0)?,
                TokenComponent::try_from(1.0)?,
            )?,
        ])
    }

    #[test]
    fn scaled_score_matches_hand_calculation() -> Result<(), AttentionError> {
        let query = query(
            QueryComponent::try_from(1.0)?,
            QueryComponent::try_from(1.0)?,
        )?;
        let key = key(KeyComponent::try_from(1.0)?, KeyComponent::try_from(0.0)?)?;
        let score = (&query * &key)?;

        assert_close_score(score, AttentionScore::try_from(1.0 / 2.0_f64.sqrt())?)?;
        Ok(())
    }

    #[test]
    fn softmax_weights_sum_to_one() -> Result<(), AttentionError> {
        let weights = softmax(&AttentionScores::from_scores([
            AttentionScore::try_from(2.0)?,
            AttentionScore::try_from(1.0)?,
            AttentionScore::try_from(0.0)?,
        ])?)?;
        assert!(weights.sum_is_close_to(
            AttentionWeight::try_from(1.0)?,
            AttentionWeight::try_from(1e-6)?
        ));
        assert!(
            weights
                .values()
                .copied()
                .zip(weights.values().copied().skip(1))
                .all(|(left, right)| left > right)
        );
        Ok(())
    }

    #[test]
    fn weighted_sum_matches_manual_mixture() -> Result<(), AttentionError> {
        let weights = AttentionWeights::from_weights([
            AttentionWeight::try_from(0.75)?,
            AttentionWeight::try_from(0.25)?,
        ])?;
        let values = ValueSequence::from_values([
            value(
                ValueComponent::try_from(2.0)?,
                ValueComponent::try_from(0.0)?,
            )?,
            value(
                ValueComponent::try_from(0.0)?,
                ValueComponent::try_from(4.0)?,
            )?,
        ])?;

        let output = (&weights * &values)?;
        let output_values = output.values().copied().collect::<Vec<_>>();

        assert_eq!(
            output_values,
            vec![
                AttentionOutputComponent::try_from(1.5)?,
                AttentionOutputComponent::try_from(1.0)?
            ]
        );
        Ok(())
    }

    #[test]
    fn token_sequence_rejects_width_mismatch() -> Result<(), AttentionError> {
        let error = TokenSequence::from_tokens([
            token(
                TokenComponent::try_from(1.0)?,
                TokenComponent::try_from(0.0)?,
            )?,
            TokenEmbedding::from_values([
                TokenComponent::try_from(1.0)?,
                TokenComponent::try_from(0.0)?,
                TokenComponent::try_from(0.0)?,
            ])?,
        ]);

        assert!(matches!(
            error,
            Err(AttentionError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn identity_attention_prefers_matching_token() -> Result<(), AttentionError> {
        let head = AttentionHead::identity(VectorWidth::try_from(2)?)?;
        let trace = head.trace_token(&sequence()?, TokenIndex::try_from(0)?)?;
        assert!(
            trace
                .weights()
                .values()
                .copied()
                .zip(trace.weights().values().copied().skip(1))
                .all(|(left, right)| left > right)
        );
        assert!(
            trace
                .output()
                .values()
                .copied()
                .zip(trace.output().values().copied().skip(1))
                .all(|(left, right)| left > right)
        );
        Ok(())
    }

    #[test]
    fn attention_head_rejects_projection_width_mismatch() -> Result<(), AttentionError> {
        let query = QueryLayer::new(
            QueryProjection::from_rows([projection_row([
                ProjectionWeight::try_from(1.0)?,
                ProjectionWeight::try_from(0.0)?,
            ])?])?,
            projection_bias(ProjectionBiasValue::try_from(0.0)?)?,
        )?;
        let key = KeyLayer::new(
            KeyProjection::from_rows([ProjectionRow::from_values([
                ProjectionWeight::try_from(1.0)?,
                ProjectionWeight::try_from(0.0)?,
                ProjectionWeight::try_from(0.0)?,
            ])?])?,
            projection_bias(ProjectionBiasValue::try_from(0.0)?)?,
        )?;
        let value = ValueLayer::new(
            ValueProjection::from_rows([projection_row([
                ProjectionWeight::try_from(1.0)?,
                ProjectionWeight::try_from(0.0)?,
            ])?])?,
            projection_bias(ProjectionBiasValue::try_from(0.0)?)?,
        )?;

        let error = AttentionHead::new(query, key, value);

        assert!(matches!(
            error,
            Err(AttentionError::InvalidProjection { .. })
        ));
        Ok(())
    }

    #[test]
    fn trace_rejects_out_of_bounds_query_index() -> Result<(), AttentionError> {
        let head = AttentionHead::identity(VectorWidth::try_from(2)?)?;
        let error = head.trace_token(&sequence()?, TokenIndex::try_from(2)?);

        assert!(matches!(
            error,
            Err(AttentionError::InvalidTokenIndex { .. })
        ));
        Ok(())
    }
}
