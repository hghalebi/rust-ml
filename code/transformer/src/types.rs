//! Semantic model types layered on top of the dense math primitives.

use std::{
    fmt,
    ops::{Add, Mul},
};

use crate::error::ModelError;
use crate::math::{DenseMatrix, DenseVector, MatrixShape, ModelScalar, ScalarValues, VectorLength};

macro_rules! vector_role {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(DenseVector);

        impl $name {
            /// Creates the semantic role from an already checked vector.
            pub fn from_vector(vector: DenseVector) -> Self {
                Self(vector)
            }

            /// Returns the wrapped vector width.
            pub fn len(&self) -> VectorLength {
                self.0.len()
            }

            /// Returns a printable checked scalar view.
            pub fn as_slice(&self) -> ScalarValues<'_> {
                self.0.as_slice()
            }

            /// Iterates over scalar values.
            pub fn values(&self) -> impl ExactSizeIterator<Item = ModelScalar> + '_ {
                self.0.values()
            }

            pub(crate) fn vector(&self) -> &DenseVector {
                &self.0
            }
        }

        impl From<DenseVector> for $name {
            fn from(value: DenseVector) -> Self {
                Self::from_vector(value)
            }
        }
    };
}

vector_role!(TokenEmbedding, "A token representation inside the model.");
vector_role!(
    PositionEncoding,
    "A positional signal added to a token embedding."
);
vector_role!(Query, "A query vector used to score relevance.");
vector_role!(Key, "A key vector used as a match target for queries.");
vector_role!(
    Value,
    "A value vector that carries the information to be mixed."
);
vector_role!(
    AttentionOutput,
    "The output of one attention head for one token."
);
vector_role!(
    ConcatenatedHeadOutput,
    "The concatenation of several head outputs before output projection."
);
vector_role!(
    HiddenPreActivation,
    "A feed-forward hidden value before ReLU is applied."
);
vector_role!(
    HiddenActivation,
    "The hidden activation inside the position-wise feed-forward network."
);
vector_role!(ProjectionBias, "A bias vector used by a projection layer.");
vector_role!(
    NormScale,
    "The learned scale parameter in layer normalization."
);
vector_role!(
    NormShift,
    "The learned shift parameter in layer normalization."
);

fn add_dense_vectors(
    operation: &'static str,
    left_label: &'static str,
    left: &DenseVector,
    right_label: &'static str,
    right: &DenseVector,
    hint: &'static str,
) -> Result<DenseVector, ModelError> {
    if left.len() != right.len() {
        return Err(ModelError::dimension_mismatch(
            operation,
            left_label,
            vec![left.len().as_usize()],
            right_label,
            vec![right.len().as_usize()],
            hint,
        ));
    }

    DenseVector::new(
        left.values()
            .zip(right.values())
            .map(|(left, right)| left + right)
            .collect::<Result<Vec<_>, _>>()?,
    )
}

fn project_vector(
    operation: &'static str,
    projection: &DenseMatrix,
    input_label: &'static str,
    input: &DenseVector,
) -> Result<DenseVector, ModelError> {
    if projection.cols() != input.len() {
        return Err(ModelError::dimension_mismatch(
            operation,
            "projection",
            vec![projection.rows().as_usize(), projection.cols().as_usize()],
            input_label,
            vec![input.len().as_usize()],
            "projection input width must match vector width",
        ));
    }

    projection * input
}

fn add_projection_bias(
    operation: &'static str,
    role_label: &'static str,
    role: &DenseVector,
    bias: &ProjectionBias,
) -> Result<DenseVector, ModelError> {
    add_dense_vectors(
        operation,
        role_label,
        role,
        "projection bias",
        bias.vector(),
        "projection bias addition requires matching widths",
    )
}

impl<'b> Add<&'b PositionEncoding> for &TokenEmbedding {
    type Output = Result<TokenEmbedding, ModelError>;

    fn add(self, right: &'b PositionEncoding) -> Self::Output {
        Ok(TokenEmbedding::from(add_dense_vectors(
            "TokenEmbedding::add_position_encoding",
            "token embedding",
            self.vector(),
            "position encoding",
            right.vector(),
            "positional encoding addition requires matching model widths",
        )?))
    }
}

impl<'b> Add<&'b TokenEmbedding> for &TokenEmbedding {
    type Output = Result<TokenEmbedding, ModelError>;

    fn add(self, right: &'b TokenEmbedding) -> Self::Output {
        Ok(TokenEmbedding::from(add_dense_vectors(
            "TokenEmbedding::add",
            "left token embedding",
            self.vector(),
            "right token embedding",
            right.vector(),
            "residual token addition requires matching model widths",
        )?))
    }
}

macro_rules! impl_projection_mul {
    ($projection:ident, $input:ident, $output:ident, $operation:literal, $input_label:literal) => {
        impl<'b> Mul<&'b $input> for &$projection {
            type Output = Result<$output, ModelError>;

            fn mul(self, right: &'b $input) -> Self::Output {
                Ok($output::from_vector(project_vector(
                    $operation,
                    self.matrix(),
                    $input_label,
                    right.vector(),
                )?))
            }
        }
    };
}

macro_rules! impl_projection_bias_add {
    ($role:ident, $operation:literal, $role_label:literal) => {
        impl<'b> Add<&'b ProjectionBias> for &$role {
            type Output = Result<$role, ModelError>;

            fn add(self, right: &'b ProjectionBias) -> Self::Output {
                Ok($role::from_vector(add_projection_bias(
                    $operation,
                    $role_label,
                    self.vector(),
                    right,
                )?))
            }
        }
    };
}

macro_rules! projection_role {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(DenseMatrix);

        impl $name {
            /// Creates the semantic projection role from a checked matrix.
            pub fn from_matrix(matrix: DenseMatrix) -> Self {
                Self(matrix)
            }

            /// Returns matrix shape.
            pub fn shape(&self) -> MatrixShape {
                self.0.shape()
            }

            /// Returns input width.
            pub fn input_dim(&self) -> VectorLength {
                self.0.cols()
            }

            /// Returns output width.
            pub fn output_dim(&self) -> VectorLength {
                self.0.rows()
            }

            pub(crate) fn matrix(&self) -> &DenseMatrix {
                &self.0
            }
        }

        impl From<DenseMatrix> for $name {
            fn from(value: DenseMatrix) -> Self {
                Self::from_matrix(value)
            }
        }
    };
}

projection_role!(QueryProjection, "A typed query projection matrix.");
projection_role!(KeyProjection, "A typed key projection matrix.");
projection_role!(ValueProjection, "A typed value projection matrix.");
projection_role!(OutputProjection, "A typed output projection matrix.");
projection_role!(
    FeedForwardProjection1,
    "The first feed-forward projection matrix."
);
projection_role!(
    FeedForwardProjection2,
    "The second feed-forward projection matrix."
);

impl_projection_mul!(
    QueryProjection,
    TokenEmbedding,
    Query,
    "QueryProjection::mul_token",
    "token embedding"
);
impl_projection_mul!(
    KeyProjection,
    TokenEmbedding,
    Key,
    "KeyProjection::mul_token",
    "token embedding"
);
impl_projection_mul!(
    ValueProjection,
    TokenEmbedding,
    Value,
    "ValueProjection::mul_token",
    "token embedding"
);
impl_projection_mul!(
    OutputProjection,
    ConcatenatedHeadOutput,
    TokenEmbedding,
    "OutputProjection::mul_concatenated_head_output",
    "concatenated head output"
);
impl_projection_mul!(
    FeedForwardProjection1,
    TokenEmbedding,
    HiddenPreActivation,
    "FeedForwardProjection1::mul_token",
    "token embedding"
);
impl_projection_mul!(
    FeedForwardProjection2,
    HiddenActivation,
    TokenEmbedding,
    "FeedForwardProjection2::mul_hidden_activation",
    "hidden activation"
);

impl_projection_bias_add!(Query, "Query::add_projection_bias", "query");
impl_projection_bias_add!(Key, "Key::add_projection_bias", "key");
impl_projection_bias_add!(Value, "Value::add_projection_bias", "value");
impl_projection_bias_add!(
    TokenEmbedding,
    "TokenEmbedding::add_projection_bias",
    "token embedding"
);
impl_projection_bias_add!(
    HiddenPreActivation,
    "HiddenPreActivation::add_projection_bias",
    "hidden pre-activation"
);

impl HiddenPreActivation {
    /// Applies ReLU and returns the semantic hidden activation.
    pub fn relu(&self) -> Result<HiddenActivation, ModelError> {
        Ok(HiddenActivation::from_vector(
            self.vector()
                .map_components(|value| ModelScalar::try_from(value.as_f32().max(0.0)))?,
        ))
    }
}

/// A scaled query-key compatibility score.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AttentionScore(ModelScalar);

impl AttentionScore {
    pub(crate) fn from_raw(value: f32) -> Result<Self, ModelError> {
        Ok(Self(ModelScalar::try_from(value)?))
    }

    pub(crate) fn as_f32(self) -> f32 {
        self.0.as_f32()
    }

    #[cfg(test)]
    pub(crate) fn ensure_close_to(
        self,
        expected: AttentionScore,
        tolerance: AttentionScore,
    ) -> Result<(), ModelError> {
        self.0.ensure_close_to(expected.0, tolerance.0)
    }
}

impl TryFrom<f32> for AttentionScore {
    type Error = ModelError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for AttentionScore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl<'b> Mul<&'b Key> for &Query {
    type Output = Result<AttentionScore, ModelError>;

    fn mul(self, right: &'b Key) -> Self::Output {
        if self.len() != right.len() {
            return Err(ModelError::dimension_mismatch(
                "Query::mul_key",
                "query",
                vec![self.len().as_usize()],
                "key",
                vec![right.len().as_usize()],
                "query and key must have the same dimension",
            ));
        }

        let dot = (self.vector() * right.vector())?;
        AttentionScore::from_raw(dot.as_f32() / (self.len().as_usize() as f32).sqrt())
    }
}

/// A normalized attention weight.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AttentionWeight(ModelScalar);

impl AttentionWeight {
    pub(crate) fn as_f32(self) -> f32 {
        self.0.as_f32()
    }
}

impl TryFrom<f32> for AttentionWeight {
    type Error = ModelError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        let scalar = ModelScalar::try_from(value)?;
        if !(0.0..=1.0).contains(&scalar.as_f32()) {
            return Err(ModelError::out_of_range("attention weight", "0..=1", value));
        }
        Ok(Self(scalar))
    }
}

impl fmt::Display for AttentionWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Raw attention scores before softmax normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionScores(Vec<AttentionScore>);

impl AttentionScores {
    /// Creates a non-empty score list.
    pub fn from_scores(
        scores: impl IntoIterator<Item = AttentionScore>,
    ) -> Result<Self, ModelError> {
        let scores = scores.into_iter().collect::<Vec<_>>();
        if scores.is_empty() {
            return Err(ModelError::empty_input(
                "AttentionScores::from_scores",
                "scores cannot be empty",
            ));
        }
        Ok(Self(scores))
    }

    /// Iterates over scores.
    pub fn values(&self) -> impl ExactSizeIterator<Item = AttentionScore> + '_ {
        self.0.iter().copied()
    }

    /// Returns score count.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_known_nonzero(self.0.len())
    }
}

/// Attention weights after softmax normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionWeights(Vec<AttentionWeight>);

impl AttentionWeights {
    /// Creates non-empty attention weights that sum to one.
    pub fn from_weights(
        weights: impl IntoIterator<Item = AttentionWeight>,
    ) -> Result<Self, ModelError> {
        let weights = weights.into_iter().collect::<Vec<_>>();
        if weights.is_empty() {
            return Err(ModelError::empty_input(
                "AttentionWeights::from_weights",
                "weights cannot be empty",
            ));
        }

        let sum: f32 = weights.iter().map(|weight| weight.as_f32()).sum();
        if (sum - 1.0).abs() > 1e-5 {
            return Err(ModelError::numerical_issue(
                "AttentionWeights::from_weights",
                "attention weights must sum to one",
            ));
        }

        Ok(Self(weights))
    }

    /// Iterates over weights.
    pub fn values(&self) -> impl ExactSizeIterator<Item = AttentionWeight> + '_ {
        self.0.iter().copied()
    }

    /// Returns weight count.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_known_nonzero(self.0.len())
    }
}

/// Number of tokens, scores, weights, or head outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenCount(usize);

impl TokenCount {
    pub(crate) fn from_known_nonzero(value: usize) -> Self {
        Self(value)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TokenCount {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(ModelError::empty_input(
                "TokenCount::try_from",
                "token count must be greater than zero",
            ));
        }

        Ok(Self(value))
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Zero-based token index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenIndex(usize);

impl TokenIndex {
    pub(crate) fn from_raw(value: usize) -> Self {
        Self(value)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for TokenIndex {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for TokenIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A non-empty same-width sequence of value vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSequence {
    values: Vec<Value>,
    value_width: VectorLength,
}

impl ValueSequence {
    /// Creates a checked sequence of values.
    pub fn from_values(values: impl IntoIterator<Item = Value>) -> Result<Self, ModelError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(ModelError::empty_input(
                "ValueSequence::from_values",
                "values cannot be empty",
            ));
        }

        let value_width = values[0].len();
        for (index, value) in values.iter().enumerate() {
            if value.len() != value_width {
                return Err(ModelError::inconsistent_token_dimensions(
                    "ValueSequence::from_values",
                    index,
                    value_width.as_usize(),
                    value.len().as_usize(),
                ));
            }
        }

        Ok(Self {
            values,
            value_width,
        })
    }

    /// Returns value count.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_known_nonzero(self.values.len())
    }

    /// Returns shared value width.
    pub fn value_width(&self) -> VectorLength {
        self.value_width
    }

    /// Iterates over values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &Value> + '_ {
        self.values.iter()
    }
}

impl<'b> Mul<&'b ValueSequence> for &AttentionWeights {
    type Output = Result<AttentionOutput, ModelError>;

    fn mul(self, right: &'b ValueSequence) -> Self::Output {
        if self.len() != right.len() {
            return Err(ModelError::dimension_mismatch(
                "AttentionWeights::mul_value_sequence",
                "attention weights",
                vec![self.len().as_usize()],
                "values",
                vec![right.len().as_usize()],
                "the number of weights must equal the number of values",
            ));
        }

        let mut out = vec![ModelScalar::try_from(0.0)?; right.value_width().as_usize()];

        for (weight, value) in self.values().zip(right.values()) {
            let weight = ModelScalar::try_from(weight.as_f32())?;
            for (slot, component) in out.iter_mut().zip(value.values()) {
                *slot = (*slot + (weight * component)?)?;
            }
        }

        Ok(AttentionOutput::from_vector(DenseVector::new(out)?))
    }
}

/// Outputs from running one attention head across a sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOutputSequence(Vec<AttentionOutput>);

impl AttentionOutputSequence {
    /// Creates a non-empty output sequence.
    pub fn from_outputs(
        outputs: impl IntoIterator<Item = AttentionOutput>,
    ) -> Result<Self, ModelError> {
        let outputs = outputs.into_iter().collect::<Vec<_>>();
        if outputs.is_empty() {
            return Err(ModelError::empty_input(
                "AttentionOutputSequence::from_outputs",
                "outputs cannot be empty",
            ));
        }
        Ok(Self(outputs))
    }

    /// Returns output count.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_known_nonzero(self.0.len())
    }

    /// Iterates over outputs.
    pub fn outputs(&self) -> impl ExactSizeIterator<Item = &AttentionOutput> + '_ {
        self.0.iter()
    }

    /// Returns one output by token index.
    pub fn output(&self, index: TokenIndex) -> Result<&AttentionOutput, ModelError> {
        self.0
            .get(index.as_usize())
            .ok_or(ModelError::invalid_token_index(
                "AttentionOutputSequence::output",
                index.as_usize(),
                self.0.len(),
            ))
    }
}

/// A sequence of same-width token embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSequence {
    tokens: Vec<TokenEmbedding>,
    d_model: VectorLength,
}

impl TokenSequence {
    /// Creates a sequence from token embeddings and checks a shared model width.
    pub fn new(tokens: impl IntoIterator<Item = TokenEmbedding>) -> Result<Self, ModelError> {
        Self::from_tokens(tokens)
    }

    /// Creates a sequence from token embeddings and checks a shared model width.
    pub fn from_tokens(
        tokens: impl IntoIterator<Item = TokenEmbedding>,
    ) -> Result<Self, ModelError> {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        if tokens.is_empty() {
            return Err(ModelError::empty_input(
                "TokenSequence::from_tokens",
                "sequence cannot be empty",
            ));
        }

        let d_model = tokens[0].len();
        for (token_index, token) in tokens.iter().enumerate() {
            if token.len() != d_model {
                return Err(ModelError::inconsistent_token_dimensions(
                    "TokenSequence::from_tokens",
                    token_index,
                    d_model.as_usize(),
                    token.len().as_usize(),
                ));
            }
        }

        Ok(Self { tokens, d_model })
    }

    /// Returns the token count.
    pub fn len(&self) -> TokenCount {
        TokenCount::from_known_nonzero(self.tokens.len())
    }

    /// Returns the shared token width.
    pub fn d_model(&self) -> VectorLength {
        self.d_model
    }

    /// Iterates over all tokens.
    pub fn tokens(&self) -> impl ExactSizeIterator<Item = &TokenEmbedding> + '_ {
        self.tokens.iter()
    }

    /// Returns one token by index.
    pub fn token(&self, index: TokenIndex) -> Result<&TokenEmbedding, ModelError> {
        self.tokens
            .get(index.as_usize())
            .ok_or(ModelError::invalid_token_index(
                "TokenSequence::token",
                index.as_usize(),
                self.tokens.len(),
            ))
    }

    /// Maps a fallible token transformation over the whole sequence.
    pub fn map_tokens<F>(&self, f: F) -> Result<TokenSequence, ModelError>
    where
        F: Fn(&TokenEmbedding) -> Result<TokenEmbedding, ModelError>,
    {
        let tokens = self.tokens.iter().map(f).collect::<Result<Vec<_>, _>>()?;
        TokenSequence::from_tokens(tokens)
    }
}

impl<'b> Add<&'b TokenSequence> for &TokenSequence {
    type Output = Result<TokenSequence, ModelError>;

    fn add(self, right: &'b TokenSequence) -> Self::Output {
        if self.len() != right.len() {
            return Err(ModelError::dimension_mismatch(
                "TokenSequence::add",
                "left sequence length",
                vec![self.len().as_usize()],
                "right sequence length",
                vec![right.len().as_usize()],
                "residual addition requires the same number of tokens",
            ));
        }

        if self.d_model() != right.d_model() {
            return Err(ModelError::dimension_mismatch(
                "TokenSequence::add",
                "left d_model",
                vec![self.d_model().as_usize()],
                "right d_model",
                vec![right.d_model().as_usize()],
                "residual addition requires matching token widths",
            ));
        }

        let tokens = self
            .tokens()
            .zip(right.tokens())
            .map(|(left, right)| left + right)
            .collect::<Result<Vec<_>, _>>()?;

        TokenSequence::from_tokens(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionScore, FeedForwardProjection1, Key, PositionEncoding, ProjectionBias, Query,
        QueryProjection, TokenEmbedding, TokenIndex, TokenSequence,
    };
    use crate::error::ModelError;
    use crate::math::{DenseMatrix, DenseVector, ModelScalar};

    fn vector(values: impl IntoIterator<Item = ModelScalar>) -> Result<DenseVector, ModelError> {
        DenseVector::new(values)
    }

    fn token(values: impl IntoIterator<Item = ModelScalar>) -> Result<TokenEmbedding, ModelError> {
        Ok(TokenEmbedding::from_vector(vector(values)?))
    }

    fn bias(values: impl IntoIterator<Item = ModelScalar>) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias::from_vector(vector(values)?))
    }

    fn assert_scalars_eq(
        actual: impl ExactSizeIterator<Item = ModelScalar>,
        expected: impl IntoIterator<Item = ModelScalar>,
    ) {
        let actual = actual.map(|value| value.to_string()).collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    fn assert_token_values(
        token: &TokenEmbedding,
        expected: impl IntoIterator<Item = ModelScalar>,
    ) {
        assert_scalars_eq(token.values(), expected);
    }

    #[test]
    fn token_sequence_rejects_empty_sequences() {
        assert!(matches!(
            TokenSequence::from_tokens([]),
            Err(ModelError::EmptyInput { .. })
        ));
    }

    #[test]
    fn token_sequence_rejects_inconsistent_widths() -> Result<(), ModelError> {
        let token_a = TokenEmbedding::from_vector(vector([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
        ])?);
        let token_b = TokenEmbedding::from_vector(vector([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?);

        assert!(matches!(
            TokenSequence::from_tokens([token_a, token_b]),
            Err(ModelError::InconsistentTokenDimensions { .. })
        ));
        Ok(())
    }

    #[test]
    fn token_embedding_plus_position_encoding_uses_typed_addition() -> Result<(), ModelError> {
        let token = token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
        let position = PositionEncoding::from_vector(vector([
            ModelScalar::try_from(0.5)?,
            ModelScalar::try_from(-0.5)?,
        ])?);

        let sum = (&token + &position)?;

        assert_token_values(
            &sum,
            [ModelScalar::try_from(1.5)?, ModelScalar::try_from(1.5)?],
        );
        Ok(())
    }

    #[test]
    fn projection_and_bias_operators_keep_query_path_readable() -> Result<(), ModelError> {
        let projection = QueryProjection::from_matrix(DenseMatrix::from_rows([
            [ModelScalar::try_from(2.0)?, ModelScalar::try_from(0.0)?],
            [ModelScalar::try_from(0.0)?, ModelScalar::try_from(3.0)?],
        ])?);
        let token = token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
        let bias = bias([ModelScalar::try_from(0.5)?, ModelScalar::try_from(-1.0)?])?;

        let projected = (&projection * &token)?;
        let query = (&projected + &bias)?;

        assert_scalars_eq(
            query.values(),
            [ModelScalar::try_from(2.5)?, ModelScalar::try_from(5.0)?],
        );
        Ok(())
    }

    #[test]
    fn query_times_key_operator_returns_scaled_attention_score() -> Result<(), ModelError> {
        let query = Query::from_vector(vector([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(1.0)?,
        ])?);
        let key = Key::from_vector(vector([
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(0.0)?,
        ])?);

        let score = (&query * &key)?;

        score.ensure_close_to(
            AttentionScore::try_from(2.0 / 2.0_f32.sqrt())?,
            AttentionScore::try_from(0.000001)?,
        )?;
        Ok(())
    }

    #[test]
    fn feed_forward_projection_operator_preserves_preactivation_role() -> Result<(), ModelError> {
        let projection = FeedForwardProjection1::from_matrix(DenseMatrix::from_rows([
            [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?],
            [ModelScalar::try_from(0.0)?, ModelScalar::try_from(-2.0)?],
        ])?);
        let token = token([ModelScalar::try_from(3.0)?, ModelScalar::try_from(2.0)?])?;
        let bias = bias([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?;

        let pre_activation = (&projection * &token)?;
        let shifted = (&pre_activation + &bias)?;
        let activation = shifted.relu()?;

        assert_scalars_eq(
            activation.values(),
            [ModelScalar::try_from(3.0)?, ModelScalar::try_from(0.0)?],
        );
        Ok(())
    }

    #[test]
    fn token_sequence_plus_token_sequence_adds_residuals() -> Result<(), ModelError> {
        let left = TokenSequence::from_tokens([
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?,
            token([ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?])?,
        ])?;
        let right = TokenSequence::from_tokens([
            token([ModelScalar::try_from(0.5)?, ModelScalar::try_from(0.5)?])?,
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(-1.0)?])?,
        ])?;

        let sum = (&left + &right)?;

        assert_token_values(
            sum.token(TokenIndex::try_from(0)?)?,
            [ModelScalar::try_from(1.5)?, ModelScalar::try_from(2.5)?],
        );
        assert_token_values(
            sum.token(TokenIndex::try_from(1)?)?,
            [ModelScalar::try_from(4.0)?, ModelScalar::try_from(3.0)?],
        );
        Ok(())
    }

    #[test]
    fn token_sequence_plus_token_sequence_reports_mismatched_lengths() -> Result<(), ModelError> {
        let left = TokenSequence::from_tokens([token([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
        ])?])?;
        let right = TokenSequence::from_tokens([
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?,
            token([ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?])?,
        ])?;

        let error = &left + &right;

        assert!(matches!(error, Err(ModelError::DimensionMismatch { .. })));
        Ok(())
    }
}
