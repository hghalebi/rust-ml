//! Attention-specific building blocks.

use crate::error::ModelError;
use crate::math::{
    ColumnIndex, DenseMatrix, DenseVector, MatrixShape, ModelScalar, RowIndex, VectorIndex,
    VectorLength,
};
use crate::types::{
    AttentionOutput, AttentionOutputSequence, AttentionScore, AttentionScores, AttentionWeights,
    ConcatenatedHeadOutput, Key, KeyProjection, OutputProjection, ProjectionBias, Query,
    QueryProjection, TokenEmbedding, TokenIndex, TokenSequence, Value, ValueProjection,
    ValueSequence,
};

fn validate_projection(
    operation: &'static str,
    weight: &DenseMatrix,
    bias: &ProjectionBias,
) -> Result<(), ModelError> {
    if weight.rows() != bias.len() {
        return Err(ModelError::invalid_projection(
            operation,
            "weight output dimension must match bias length",
        ));
    }

    Ok(())
}

/// A typed linear map from token embeddings to queries.
#[derive(Debug, Clone)]
pub struct QueryLayer {
    weight: QueryProjection,
    bias: ProjectionBias,
}

impl QueryLayer {
    /// Creates a query projection layer.
    pub fn new(weight: QueryProjection, bias: ProjectionBias) -> Result<Self, ModelError> {
        validate_projection("QueryLayer::new", weight.matrix(), &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> VectorLength {
        self.weight.input_dim()
    }

    /// Returns the produced query width.
    pub fn output_dim(&self) -> VectorLength {
        self.weight.output_dim()
    }

    /// Projects one token embedding into query space.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<Query, ModelError> {
        let projected = (&self.weight * x)?;
        &projected + &self.bias
    }
}

/// A typed linear map from token embeddings to keys.
#[derive(Debug, Clone)]
pub struct KeyLayer {
    weight: KeyProjection,
    bias: ProjectionBias,
}

impl KeyLayer {
    /// Creates a key projection layer.
    pub fn new(weight: KeyProjection, bias: ProjectionBias) -> Result<Self, ModelError> {
        validate_projection("KeyLayer::new", weight.matrix(), &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> VectorLength {
        self.weight.input_dim()
    }

    /// Returns the produced key width.
    pub fn output_dim(&self) -> VectorLength {
        self.weight.output_dim()
    }

    /// Projects one token embedding into key space.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<Key, ModelError> {
        let projected = (&self.weight * x)?;
        &projected + &self.bias
    }
}

/// A typed linear map from token embeddings to values.
#[derive(Debug, Clone)]
pub struct ValueLayer {
    weight: ValueProjection,
    bias: ProjectionBias,
}

impl ValueLayer {
    /// Creates a value projection layer.
    pub fn new(weight: ValueProjection, bias: ProjectionBias) -> Result<Self, ModelError> {
        validate_projection("ValueLayer::new", weight.matrix(), &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> VectorLength {
        self.weight.input_dim()
    }

    /// Returns the produced value width.
    pub fn output_dim(&self) -> VectorLength {
        self.weight.output_dim()
    }

    /// Projects one token embedding into value space.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<Value, ModelError> {
        let projected = (&self.weight * x)?;
        &projected + &self.bias
    }
}

/// A typed linear map from concatenated head outputs back into model space.
#[derive(Debug, Clone)]
pub struct OutputLayer {
    weight: OutputProjection,
    bias: ProjectionBias,
}

impl OutputLayer {
    /// Creates an output projection layer.
    pub fn new(weight: OutputProjection, bias: ProjectionBias) -> Result<Self, ModelError> {
        validate_projection("OutputLayer::new", weight.matrix(), &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected concatenated head width.
    pub fn input_dim(&self) -> VectorLength {
        self.weight.input_dim()
    }

    /// Returns the projected model width.
    pub fn output_dim(&self) -> VectorLength {
        self.weight.output_dim()
    }

    /// Projects concatenated head output back into model space.
    pub fn forward(&self, x: &ConcatenatedHeadOutput) -> Result<TokenEmbedding, ModelError> {
        let projected = (&self.weight * x)?;
        &projected + &self.bias
    }
}

/// Computes the scaled attention score for one query-key pair.
pub fn scaled_attention_score(query: &Query, key: &Key) -> Result<AttentionScore, ModelError> {
    query * key
}

/// Normalizes attention scores into attention weights.
pub fn softmax(scores: &AttentionScores) -> Result<AttentionWeights, ModelError> {
    let max = scores
        .values()
        .map(|score| score.as_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = scores
        .values()
        .map(|score| (score.as_f32() - max).exp())
        .collect();
    let sum: f32 = exp_values.iter().sum();

    if !sum.is_finite() || sum == 0.0 {
        return Err(ModelError::numerical_issue(
            "softmax",
            "softmax normalization sum was zero or non-finite",
        ));
    }

    AttentionWeights::from_weights(
        exp_values
            .into_iter()
            .map(|value| crate::types::AttentionWeight::try_from(value / sum))
            .collect::<Result<Vec<_>, _>>()?,
    )
}

/// Applies attention weights to value vectors.
pub fn weighted_sum(
    weights: &AttentionWeights,
    values: &ValueSequence,
) -> Result<AttentionOutput, ModelError> {
    weights * values
}

/// Concatenates several head outputs into one larger vector.
pub fn concat_attention_outputs(
    outputs: impl IntoIterator<Item = AttentionOutput>,
) -> Result<ConcatenatedHeadOutput, ModelError> {
    let outputs = outputs.into_iter().collect::<Vec<_>>();
    if outputs.is_empty() {
        return Err(ModelError::empty_input(
            "concat_attention_outputs",
            "there must be at least one head output",
        ));
    }

    let mut data = Vec::new();
    for output in &outputs {
        data.extend(output.vector().values());
    }

    Ok(ConcatenatedHeadOutput::from_vector(DenseVector::new(data)?))
}

/// One scaled dot-product attention head.
#[derive(Debug, Clone)]
pub struct AttentionHead {
    query_layer: QueryLayer,
    key_layer: KeyLayer,
    value_layer: ValueLayer,
}

impl AttentionHead {
    /// Creates one attention head with compatible projections.
    pub fn new(
        query_layer: QueryLayer,
        key_layer: KeyLayer,
        value_layer: ValueLayer,
    ) -> Result<Self, ModelError> {
        if query_layer.input_dim() != key_layer.input_dim()
            || query_layer.input_dim() != value_layer.input_dim()
        {
            return Err(ModelError::invalid_head_configuration(
                "AttentionHead::new",
                "query, key, and value layers must accept the same token width",
            ));
        }

        if query_layer.output_dim() != key_layer.output_dim() {
            return Err(ModelError::invalid_head_configuration(
                "AttentionHead::new",
                "query and key layers must produce the same head dimension",
            ));
        }

        Ok(Self {
            query_layer,
            key_layer,
            value_layer,
        })
    }

    /// Returns the expected token width.
    pub fn input_dim(&self) -> VectorLength {
        self.query_layer.input_dim()
    }

    /// Returns the query/key head dimension.
    pub fn score_dim(&self) -> VectorLength {
        self.query_layer.output_dim()
    }

    /// Returns the output width contributed by this head.
    pub fn value_dim(&self) -> VectorLength {
        self.value_layer.output_dim()
    }

    /// Runs scaled dot-product attention across the whole sequence.
    pub fn forward(&self, seq: &TokenSequence) -> Result<AttentionOutputSequence, ModelError> {
        let queries: Vec<Query> = seq
            .tokens()
            .map(|token| self.query_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let keys: Vec<Key> = seq
            .tokens()
            .map(|token| self.key_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let values: Vec<Value> = seq
            .tokens()
            .map(|token| self.value_layer.forward(token))
            .collect::<Result<_, _>>()?;
        let values = ValueSequence::from_values(values)?;

        let mut outputs = Vec::with_capacity(seq.len().as_usize());

        for query in &queries {
            let scores = AttentionScores::from_scores(
                keys.iter()
                    .map(|key| query * key)
                    .collect::<Result<Vec<_>, _>>()?,
            )?;

            let weights = softmax(&scores)?;
            let output = (&weights * &values)?;
            outputs.push(output);
        }

        AttentionOutputSequence::from_outputs(outputs)
    }
}

/// A simplified linear-attention head that rewrites the attention math through summaries.
#[derive(Debug, Clone)]
pub struct LinearAttentionHead {
    query_layer: QueryLayer,
    key_layer: KeyLayer,
    value_layer: ValueLayer,
}

impl LinearAttentionHead {
    /// Creates one simplified linear-attention head.
    pub fn new(
        query_layer: QueryLayer,
        key_layer: KeyLayer,
        value_layer: ValueLayer,
    ) -> Result<Self, ModelError> {
        AttentionHead::new(query_layer.clone(), key_layer.clone(), value_layer.clone())?;
        Ok(Self {
            query_layer,
            key_layer,
            value_layer,
        })
    }

    fn phi(vector: &DenseVector) -> Result<DenseVector, ModelError> {
        vector.map_components(|value| ModelScalar::try_from(value.as_f32().max(0.0) + 1e-6))
    }

    /// Runs the simplified linear-attention forward pass across the whole sequence.
    pub fn forward(&self, seq: &TokenSequence) -> Result<AttentionOutputSequence, ModelError> {
        let queries: Vec<Query> = seq
            .tokens()
            .map(|token| {
                self.query_layer
                    .forward(token)
                    .and_then(|query| Ok(Query::from_vector(Self::phi(query.vector())?)))
            })
            .collect::<Result<_, _>>()?;

        let keys: Vec<Key> = seq
            .tokens()
            .map(|token| {
                self.key_layer
                    .forward(token)
                    .and_then(|key| Ok(Key::from_vector(Self::phi(key.vector())?)))
            })
            .collect::<Result<_, _>>()?;

        let values: Vec<Value> = seq
            .tokens()
            .map(|token| self.value_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let key_dim = keys[0].len();
        let value_dim = values[0].len();

        let mut summary = DenseMatrix::zeros(MatrixShape::new(key_dim, value_dim))?;
        let mut normalizer = DenseVector::zeros(key_dim)?;

        for (key, value) in keys.iter().zip(values.iter()) {
            for row in 0..key_dim.as_usize() {
                let vector_row = VectorIndex::try_from(row)?;
                let matrix_row = RowIndex::try_from(row)?;
                let key_component = key.vector().component(vector_row)?;
                let updated_normalizer = (normalizer.component(vector_row)? + key_component)?;
                normalizer.set_component(vector_row, updated_normalizer)?;

                for col in 0..value_dim.as_usize() {
                    let vector_col = VectorIndex::try_from(col)?;
                    let matrix_col = ColumnIndex::try_from(col)?;
                    let product = (key_component * value.vector().component(vector_col)?)?;
                    let updated = (summary.component(matrix_row, matrix_col)? + product)?;
                    summary.set_component(matrix_row, matrix_col, updated)?;
                }
            }
        }

        let mut outputs = Vec::with_capacity(seq.len().as_usize());

        for query in &queries {
            let mut numerator = Vec::with_capacity(value_dim.as_usize());

            for col in 0..value_dim.as_usize() {
                let matrix_col = ColumnIndex::try_from(col)?;
                let mut sum = ModelScalar::try_from(0.0)?;
                for row in 0..key_dim.as_usize() {
                    let vector_row = VectorIndex::try_from(row)?;
                    let matrix_row = RowIndex::try_from(row)?;
                    let product = (query.vector().component(vector_row)?
                        * summary.component(matrix_row, matrix_col)?)?;
                    sum = (sum + product)?;
                }
                numerator.push(sum);
            }

            let denominator = query.vector().dot(&normalizer)?.as_f32().max(1e-6);
            let output =
                DenseVector::new(numerator)?.scale(ModelScalar::try_from(1.0 / denominator)?)?;
            outputs.push(AttentionOutput::from_vector(output));
        }

        AttentionOutputSequence::from_outputs(outputs)
    }
}

/// Multi-head attention using several scaled dot-product heads in parallel.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    output_layer: OutputLayer,
}

impl MultiHeadAttention {
    /// Creates a multi-head attention module with a valid output projection.
    pub fn new(
        heads: impl IntoIterator<Item = AttentionHead>,
        output_layer: OutputLayer,
    ) -> Result<Self, ModelError> {
        let heads = heads.into_iter().collect::<Vec<_>>();
        if heads.is_empty() {
            return Err(ModelError::invalid_head_configuration(
                "MultiHeadAttention::new",
                "multi-head attention needs at least one head",
            ));
        }

        let input_dim = heads[0].input_dim();
        if heads.iter().any(|head| head.input_dim() != input_dim) {
            return Err(ModelError::invalid_head_configuration(
                "MultiHeadAttention::new",
                "all heads must accept the same token width",
            ));
        }

        let concatenated_dim: usize = heads.iter().map(|head| head.value_dim().as_usize()).sum();
        if output_layer.input_dim().as_usize() != concatenated_dim {
            return Err(ModelError::invalid_head_configuration(
                "MultiHeadAttention::new",
                "output layer input dimension must match concatenated head width",
            ));
        }

        Ok(Self {
            heads,
            output_layer,
        })
    }

    /// Returns the model width expected by each head.
    pub fn input_dim(&self) -> VectorLength {
        self.heads[0].input_dim()
    }

    /// Returns the projected model width.
    pub fn output_dim(&self) -> VectorLength {
        self.output_layer.output_dim()
    }

    /// Runs multi-head attention over a token sequence.
    pub fn forward(&self, seq: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let per_head_outputs: Vec<AttentionOutputSequence> = self
            .heads
            .iter()
            .map(|head| head.forward(seq))
            .collect::<Result<_, _>>()?;

        let mut tokens = Vec::with_capacity(seq.len().as_usize());

        for token_index in 0..seq.len().as_usize() {
            let token_index = TokenIndex::try_from(token_index)?;
            let outputs_for_token: Vec<AttentionOutput> = per_head_outputs
                .iter()
                .map(|head_outputs| head_outputs.output(token_index).cloned())
                .collect::<Result<_, _>>()?;

            let concatenated = concat_attention_outputs(outputs_for_token)?;
            let projected = self.output_layer.forward(&concatenated)?;
            tokens.push(projected);
        }

        TokenSequence::from_tokens(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionHead, KeyLayer, LinearAttentionHead, MultiHeadAttention, OutputLayer,
        ProjectionBias, QueryLayer, ValueLayer, concat_attention_outputs, scaled_attention_score,
        softmax,
    };
    use crate::error::ModelError;
    use crate::math::{DenseMatrix, DenseVector, ModelScalar, VectorLength};
    use crate::types::{
        AttentionOutput, AttentionScore, AttentionScores, AttentionWeight, AttentionWeights, Key,
        KeyProjection, OutputProjection, Query, QueryProjection, TokenCount, TokenEmbedding,
        TokenIndex, TokenSequence, Value, ValueProjection, ValueSequence,
    };

    fn vector(values: impl IntoIterator<Item = ModelScalar>) -> Result<DenseVector, ModelError> {
        DenseVector::new(values)
    }

    fn bias(values: impl IntoIterator<Item = ModelScalar>) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias::from_vector(vector(values)?))
    }

    fn matrix<R, C>(rows: R) -> Result<DenseMatrix, ModelError>
    where
        R: IntoIterator<Item = C>,
        C: IntoIterator<Item = ModelScalar>,
    {
        DenseMatrix::from_rows(rows)
    }

    fn query_projection<R, C>(rows: R) -> Result<QueryProjection, ModelError>
    where
        R: IntoIterator<Item = C>,
        C: IntoIterator<Item = ModelScalar>,
    {
        Ok(QueryProjection::from_matrix(matrix(rows)?))
    }

    fn output_projection<R, C>(rows: R) -> Result<OutputProjection, ModelError>
    where
        R: IntoIterator<Item = C>,
        C: IntoIterator<Item = ModelScalar>,
    {
        Ok(OutputProjection::from_matrix(matrix(rows)?))
    }

    fn token(values: impl IntoIterator<Item = ModelScalar>) -> Result<TokenEmbedding, ModelError> {
        Ok(TokenEmbedding::from_vector(vector(values)?))
    }

    fn query(values: impl IntoIterator<Item = ModelScalar>) -> Result<Query, ModelError> {
        Ok(Query::from_vector(vector(values)?))
    }

    fn key(values: impl IntoIterator<Item = ModelScalar>) -> Result<Key, ModelError> {
        Ok(Key::from_vector(vector(values)?))
    }

    fn value(values: impl IntoIterator<Item = ModelScalar>) -> Result<Value, ModelError> {
        Ok(Value::from_vector(vector(values)?))
    }

    fn attention_output(
        values: impl IntoIterator<Item = ModelScalar>,
    ) -> Result<AttentionOutput, ModelError> {
        Ok(AttentionOutput::from_vector(vector(values)?))
    }

    fn identity_matrix(dim: VectorLength) -> Result<DenseMatrix, ModelError> {
        DenseMatrix::identity(dim)
    }

    fn zero_bias(dim: VectorLength) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias::from_vector(DenseVector::zeros(dim)?))
    }

    fn identity_query_layer(dim: VectorLength) -> Result<QueryLayer, ModelError> {
        QueryLayer::new(
            QueryProjection::from_matrix(identity_matrix(dim)?),
            zero_bias(dim)?,
        )
    }

    fn identity_key_layer(dim: VectorLength) -> Result<KeyLayer, ModelError> {
        KeyLayer::new(
            KeyProjection::from_matrix(identity_matrix(dim)?),
            zero_bias(dim)?,
        )
    }

    fn identity_value_layer(dim: VectorLength) -> Result<ValueLayer, ModelError> {
        ValueLayer::new(
            ValueProjection::from_matrix(identity_matrix(dim)?),
            zero_bias(dim)?,
        )
    }

    fn assert_vector_values(vector: &DenseVector, expected: impl IntoIterator<Item = ModelScalar>) {
        let actual = vector
            .values()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    fn assert_vector_values_equal(left: &DenseVector, right: &DenseVector) {
        let left = left
            .values()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let right = right
            .values()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        assert_eq!(left, right);
    }

    fn ensure_vector_values_close(
        vector: &DenseVector,
        expected: impl IntoIterator<Item = ModelScalar>,
        tolerance: ModelScalar,
    ) -> Result<(), ModelError> {
        let expected = expected.into_iter().collect::<Vec<_>>();
        if vector.values().len() != expected.len() {
            return Err(ModelError::dimension_mismatch(
                "ensure_vector_values_close",
                "actual vector",
                vec![vector.values().len()],
                "expected vector",
                vec![expected.len()],
                "close vector comparison requires equal lengths",
            ));
        }

        for (actual, expected) in vector.values().zip(expected) {
            actual.ensure_close_to(expected, tolerance)?;
        }

        Ok(())
    }

    fn assert_attention_weights(
        weights: &AttentionWeights,
        expected: impl IntoIterator<Item = AttentionWeight>,
    ) {
        let actual = weights
            .values()
            .map(|weight| weight.to_string())
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|weight| weight.to_string())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn query_layer_projects_tokens() -> Result<(), ModelError> {
        let layer = QueryLayer::new(
            query_projection([
                [ModelScalar::try_from(2.0)?, ModelScalar::try_from(0.0)?],
                [ModelScalar::try_from(0.0)?, ModelScalar::try_from(3.0)?],
            ])?,
            bias([ModelScalar::try_from(0.5)?, ModelScalar::try_from(-1.0)?])?,
        )?;
        let token = token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;

        let query = layer.forward(&token)?;
        assert_vector_values(
            query.vector(),
            [ModelScalar::try_from(2.5)?, ModelScalar::try_from(5.0)?],
        );
        Ok(())
    }

    #[test]
    fn scaled_attention_score_reports_mismatch() -> Result<(), ModelError> {
        let query = query([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
        let key = key([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;

        assert!(matches!(
            scaled_attention_score(&query, &key),
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn softmax_is_stable_for_large_values() -> Result<(), ModelError> {
        let weights = softmax(&AttentionScores::from_scores([
            AttentionScore::try_from(1000.0)?,
            AttentionScore::try_from(1000.0)?,
        ])?)?;
        assert_attention_weights(
            &weights,
            [
                AttentionWeight::try_from(0.5)?,
                AttentionWeight::try_from(0.5)?,
            ],
        );
        Ok(())
    }

    #[test]
    fn weighted_sum_matches_manual_computation() -> Result<(), ModelError> {
        let weights = AttentionWeights::from_weights([
            AttentionWeight::try_from(0.25)?,
            AttentionWeight::try_from(0.75)?,
        ])?;
        let values = ValueSequence::from_values([
            value([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
            value([ModelScalar::try_from(0.0)?, ModelScalar::try_from(2.0)?])?,
        ])?;

        let output = (&weights * &values)?;
        assert_vector_values(
            output.vector(),
            [ModelScalar::try_from(0.25)?, ModelScalar::try_from(1.5)?],
        );
        Ok(())
    }

    #[test]
    fn weighted_sum_reports_mismatch() -> Result<(), ModelError> {
        let weights = AttentionWeights::from_weights([AttentionWeight::try_from(1.0)?])?;
        let values = ValueSequence::from_values([
            value([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
            value([ModelScalar::try_from(0.0)?, ModelScalar::try_from(2.0)?])?,
        ])?;

        assert!(matches!(
            &weights * &values,
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn attention_head_single_token_reduces_to_value_projection() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let head = AttentionHead::new(
            identity_query_layer(dim)?,
            identity_key_layer(dim)?,
            identity_value_layer(dim)?,
        )?;
        let sequence = TokenSequence::new([token([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
        ])?])?;

        let outputs = head.forward(&sequence)?;
        assert_vector_values(
            outputs.output(TokenIndex::try_from(0)?)?.vector(),
            [ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?],
        );
        Ok(())
    }

    #[test]
    fn concat_attention_outputs_joins_vectors_in_order() -> Result<(), ModelError> {
        let outputs = [
            attention_output([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?,
            attention_output([ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?])?,
        ];

        let concatenated = concat_attention_outputs(outputs)?;
        assert_vector_values(
            concatenated.vector(),
            [
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(2.0)?,
                ModelScalar::try_from(3.0)?,
                ModelScalar::try_from(4.0)?,
            ],
        );
        Ok(())
    }

    #[test]
    fn multi_head_attention_rejects_empty_heads() -> Result<(), ModelError> {
        let output_layer = OutputLayer::new(
            output_projection([[ModelScalar::try_from(1.0)?]])?,
            bias([ModelScalar::try_from(0.0)?])?,
        )?;

        assert!(matches!(
            MultiHeadAttention::new(vec![], output_layer),
            Err(ModelError::InvalidHeadConfiguration { .. })
        ));
        Ok(())
    }

    #[test]
    fn multi_head_attention_validates_output_projection_width() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let head = AttentionHead::new(
            identity_query_layer(dim)?,
            identity_key_layer(dim)?,
            identity_value_layer(dim)?,
        )?;
        let output_layer = OutputLayer::new(
            output_projection([[
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(0.0)?,
            ]])?,
            bias([ModelScalar::try_from(0.0)?])?,
        )?;

        assert!(matches!(
            MultiHeadAttention::new(vec![head], output_layer),
            Err(ModelError::InvalidHeadConfiguration { .. })
        ));
        Ok(())
    }

    #[test]
    fn multi_head_attention_preserves_sequence_length() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let head_a = AttentionHead::new(
            identity_query_layer(dim)?,
            identity_key_layer(dim)?,
            identity_value_layer(dim)?,
        )?;
        let head_b = AttentionHead::new(
            identity_query_layer(dim)?,
            identity_key_layer(dim)?,
            identity_value_layer(dim)?,
        )?;
        let output_layer = OutputLayer::new(
            output_projection([
                [
                    ModelScalar::try_from(1.0)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.0)?,
                ],
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(1.0)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.0)?,
                ],
            ])?,
            bias([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?,
        )?;
        let mha = MultiHeadAttention::new(vec![head_a, head_b], output_layer)?;
        let sequence = TokenSequence::new([
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
            token([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?,
        ])?;

        let output = mha.forward(&sequence)?;
        assert_eq!(output.len(), TokenCount::try_from(2)?);
        assert_eq!(output.d_model(), VectorLength::try_from(2)?);
        Ok(())
    }

    #[test]
    fn linear_attention_single_token_reduces_to_value_projection() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let head = LinearAttentionHead::new(
            identity_query_layer(dim)?,
            identity_key_layer(dim)?,
            identity_value_layer(dim)?,
        )?;
        let sequence = TokenSequence::new([token([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
        ])?])?;

        let outputs = head.forward(&sequence)?;
        let output = outputs.output(TokenIndex::try_from(0)?)?.vector();
        ensure_vector_values_close(
            output,
            [ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?],
            ModelScalar::try_from(0.00001)?,
        )?;
        Ok(())
    }

    #[test]
    fn linear_attention_is_permutation_equivariant_without_positions() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let head = LinearAttentionHead::new(
            identity_query_layer(dim)?,
            identity_key_layer(dim)?,
            identity_value_layer(dim)?,
        )?;
        let original = TokenSequence::new([
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
            token([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?,
        ])?;
        let permuted = TokenSequence::new([
            token([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?,
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
        ])?;

        let original_outputs = head.forward(&original)?;
        let permuted_outputs = head.forward(&permuted)?;

        assert_vector_values_equal(
            original_outputs.output(TokenIndex::try_from(0)?)?.vector(),
            permuted_outputs.output(TokenIndex::try_from(1)?)?.vector(),
        );
        assert_vector_values_equal(
            original_outputs.output(TokenIndex::try_from(1)?)?.vector(),
            permuted_outputs.output(TokenIndex::try_from(0)?)?.vector(),
        );
        Ok(())
    }
}
