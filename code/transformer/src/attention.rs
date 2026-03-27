//! Attention-specific building blocks.

use crate::error::ModelError;
use crate::math::{DenseMatrix, DenseVector};
use crate::types::{
    AttentionOutput, AttentionScores, AttentionWeights, ConcatenatedHeadOutput, Key, KeyProjection,
    OutputProjection, ProjectionBias, Query, QueryProjection, TokenEmbedding, TokenSequence, Value,
    ValueProjection,
};

fn validate_projection(
    operation: &'static str,
    weight: &DenseMatrix,
    bias: &ProjectionBias,
) -> Result<(), ModelError> {
    if weight.rows() != bias.len() {
        return Err(ModelError::InvalidProjection {
            operation,
            details: "weight output dimension must match bias length",
        });
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
        validate_projection("QueryLayer::new", &weight.0, &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> usize {
        self.weight.0.cols()
    }

    /// Returns the produced query width.
    pub fn output_dim(&self) -> usize {
        self.weight.0.rows()
    }

    /// Projects one token embedding into query space.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<Query, ModelError> {
        let wx = self.weight.0.mul_vec(&x.0)?;
        let y = wx.add(&self.bias.0)?;
        Ok(Query(y))
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
        validate_projection("KeyLayer::new", &weight.0, &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> usize {
        self.weight.0.cols()
    }

    /// Returns the produced key width.
    pub fn output_dim(&self) -> usize {
        self.weight.0.rows()
    }

    /// Projects one token embedding into key space.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<Key, ModelError> {
        let wx = self.weight.0.mul_vec(&x.0)?;
        let y = wx.add(&self.bias.0)?;
        Ok(Key(y))
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
        validate_projection("ValueLayer::new", &weight.0, &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> usize {
        self.weight.0.cols()
    }

    /// Returns the produced value width.
    pub fn output_dim(&self) -> usize {
        self.weight.0.rows()
    }

    /// Projects one token embedding into value space.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<Value, ModelError> {
        let wx = self.weight.0.mul_vec(&x.0)?;
        let y = wx.add(&self.bias.0)?;
        Ok(Value(y))
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
        validate_projection("OutputLayer::new", &weight.0, &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected concatenated head width.
    pub fn input_dim(&self) -> usize {
        self.weight.0.cols()
    }

    /// Returns the projected model width.
    pub fn output_dim(&self) -> usize {
        self.weight.0.rows()
    }

    /// Projects concatenated head output back into model space.
    pub fn forward(&self, x: &ConcatenatedHeadOutput) -> Result<TokenEmbedding, ModelError> {
        let wx = self.weight.0.mul_vec(&x.0)?;
        let y = wx.add(&self.bias.0)?;
        Ok(TokenEmbedding(y))
    }
}

/// Computes the scaled attention score for one query-key pair.
pub fn scaled_attention_score(query: &Query, key: &Key) -> Result<f32, ModelError> {
    if query.len() != key.len() {
        return Err(ModelError::DimensionMismatch {
            operation: "scaled_attention_score",
            left_label: "query",
            left_shape: vec![query.len()],
            right_label: "key",
            right_shape: vec![key.len()],
            hint: "query and key must have the same dimension",
        });
    }

    let dot = query.0.dot(&key.0)?;
    Ok(dot / (query.len() as f32).sqrt())
}

/// Normalizes attention scores into attention weights.
pub fn softmax(scores: &AttentionScores) -> Result<AttentionWeights, ModelError> {
    if scores.0.is_empty() {
        return Err(ModelError::EmptyInput {
            operation: "softmax",
            details: "scores cannot be empty",
        });
    }

    let max = scores.0.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = scores.0.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();

    if !sum.is_finite() || sum == 0.0 {
        return Err(ModelError::NumericalIssue {
            operation: "softmax",
            details: "softmax normalization sum was zero or non-finite",
        });
    }

    Ok(AttentionWeights(
        exp_values.into_iter().map(|value| value / sum).collect(),
    ))
}

/// Applies attention weights to value vectors.
pub fn weighted_sum(
    weights: &AttentionWeights,
    values: &[Value],
) -> Result<AttentionOutput, ModelError> {
    if values.is_empty() {
        return Err(ModelError::EmptyInput {
            operation: "weighted_sum",
            details: "values cannot be empty",
        });
    }

    if weights.0.len() != values.len() {
        return Err(ModelError::DimensionMismatch {
            operation: "weighted_sum",
            left_label: "attention weights",
            left_shape: vec![weights.0.len()],
            right_label: "values",
            right_shape: vec![values.len()],
            hint: "the number of weights must equal the number of values",
        });
    }

    let value_dim = values[0].len();

    for value in values {
        if value.len() != value_dim {
            return Err(ModelError::DimensionMismatch {
                operation: "weighted_sum",
                left_label: "first value",
                left_shape: vec![value_dim],
                right_label: "next value",
                right_shape: vec![value.len()],
                hint: "all value vectors must have the same dimension",
            });
        }
    }

    let mut out = vec![0.0; value_dim];

    for (weight, value) in weights.0.iter().zip(values.iter()) {
        for (slot, component) in out.iter_mut().zip(value.0.as_slice().iter()) {
            *slot += weight * component;
        }
    }

    Ok(AttentionOutput(DenseVector::new(out)?))
}

/// Concatenates several head outputs into one larger vector.
pub fn concat_attention_outputs(
    outputs: &[AttentionOutput],
) -> Result<ConcatenatedHeadOutput, ModelError> {
    if outputs.is_empty() {
        return Err(ModelError::EmptyInput {
            operation: "concat_attention_outputs",
            details: "there must be at least one head output",
        });
    }

    let mut data = Vec::new();
    for output in outputs {
        data.extend_from_slice(output.as_slice());
    }

    Ok(ConcatenatedHeadOutput(DenseVector::new(data)?))
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
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "AttentionHead::new",
                details: "query, key, and value layers must accept the same token width",
            });
        }

        if query_layer.output_dim() != key_layer.output_dim() {
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "AttentionHead::new",
                details: "query and key layers must produce the same head dimension",
            });
        }

        Ok(Self {
            query_layer,
            key_layer,
            value_layer,
        })
    }

    /// Returns the expected token width.
    pub fn input_dim(&self) -> usize {
        self.query_layer.input_dim()
    }

    /// Returns the query/key head dimension.
    pub fn score_dim(&self) -> usize {
        self.query_layer.output_dim()
    }

    /// Returns the output width contributed by this head.
    pub fn value_dim(&self) -> usize {
        self.value_layer.output_dim()
    }

    /// Runs scaled dot-product attention across the whole sequence.
    pub fn forward(&self, seq: &TokenSequence) -> Result<Vec<AttentionOutput>, ModelError> {
        let queries: Vec<Query> = seq
            .tokens()
            .iter()
            .map(|token| self.query_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let keys: Vec<Key> = seq
            .tokens()
            .iter()
            .map(|token| self.key_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let values: Vec<Value> = seq
            .tokens()
            .iter()
            .map(|token| self.value_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let mut outputs = Vec::with_capacity(seq.len());

        for query in &queries {
            let scores = AttentionScores(
                keys.iter()
                    .map(|key| scaled_attention_score(query, key))
                    .collect::<Result<Vec<_>, _>>()?,
            );

            let weights = softmax(&scores)?;
            let output = weighted_sum(&weights, &values)?;
            outputs.push(output);
        }

        Ok(outputs)
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

    fn phi(vector: &DenseVector) -> DenseVector {
        vector.map(|value| value.max(0.0) + 1e-6)
    }

    /// Runs the simplified linear-attention forward pass across the whole sequence.
    pub fn forward(&self, seq: &TokenSequence) -> Result<Vec<AttentionOutput>, ModelError> {
        let queries: Vec<Query> = seq
            .tokens()
            .iter()
            .map(|token| {
                self.query_layer
                    .forward(token)
                    .map(|query| Query(Self::phi(&query.0)))
            })
            .collect::<Result<_, _>>()?;

        let keys: Vec<Key> = seq
            .tokens()
            .iter()
            .map(|token| {
                self.key_layer
                    .forward(token)
                    .map(|key| Key(Self::phi(&key.0)))
            })
            .collect::<Result<_, _>>()?;

        let values: Vec<Value> = seq
            .tokens()
            .iter()
            .map(|token| self.value_layer.forward(token))
            .collect::<Result<_, _>>()?;

        let key_dim = keys[0].len();
        let value_dim = values[0].len();

        let mut summary = DenseMatrix::zeros(key_dim, value_dim)?;
        let mut normalizer = DenseVector::zeros(key_dim)?;

        for (key, value) in keys.iter().zip(values.iter()) {
            for row in 0..key_dim {
                normalizer.set(row, normalizer.get(row) + key.0.get(row));

                for col in 0..value_dim {
                    let updated = summary.get(row, col) + key.0.get(row) * value.0.get(col);
                    summary.set(row, col, updated);
                }
            }
        }

        let mut outputs = Vec::with_capacity(seq.len());

        for query in &queries {
            let mut numerator = vec![0.0; value_dim];

            for (col, slot) in numerator.iter_mut().enumerate() {
                let mut sum = 0.0;
                for row in 0..key_dim {
                    sum += query.0.get(row) * summary.get(row, col);
                }
                *slot = sum;
            }

            let denominator = query.0.dot(&normalizer)?.max(1e-6);
            let output = DenseVector::new(numerator)?.scale(1.0 / denominator);
            outputs.push(AttentionOutput(output));
        }

        Ok(outputs)
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
    pub fn new(heads: Vec<AttentionHead>, output_layer: OutputLayer) -> Result<Self, ModelError> {
        if heads.is_empty() {
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "MultiHeadAttention::new",
                details: "multi-head attention needs at least one head",
            });
        }

        let input_dim = heads[0].input_dim();
        if heads.iter().any(|head| head.input_dim() != input_dim) {
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "MultiHeadAttention::new",
                details: "all heads must accept the same token width",
            });
        }

        let concatenated_dim: usize = heads.iter().map(AttentionHead::value_dim).sum();
        if output_layer.input_dim() != concatenated_dim {
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "MultiHeadAttention::new",
                details: "output layer input dimension must match concatenated head width",
            });
        }

        Ok(Self {
            heads,
            output_layer,
        })
    }

    /// Returns the model width expected by each head.
    pub fn input_dim(&self) -> usize {
        self.heads[0].input_dim()
    }

    /// Returns the projected model width.
    pub fn output_dim(&self) -> usize {
        self.output_layer.output_dim()
    }

    /// Runs multi-head attention over a token sequence.
    pub fn forward(&self, seq: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let per_head_outputs: Vec<Vec<AttentionOutput>> = self
            .heads
            .iter()
            .map(|head| head.forward(seq))
            .collect::<Result<_, _>>()?;

        let mut tokens = Vec::with_capacity(seq.len());

        for token_index in 0..seq.len() {
            let outputs_for_token: Vec<AttentionOutput> = per_head_outputs
                .iter()
                .map(|head_outputs| head_outputs[token_index].clone())
                .collect();

            let concatenated = concat_attention_outputs(&outputs_for_token)?;
            let projected = self.output_layer.forward(&concatenated)?;
            tokens.push(projected);
        }

        TokenSequence::new(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionHead, KeyLayer, LinearAttentionHead, MultiHeadAttention, OutputLayer,
        ProjectionBias, QueryLayer, ValueLayer, concat_attention_outputs, scaled_attention_score,
        softmax, weighted_sum,
    };
    use crate::error::ModelError;
    use crate::math::{DenseMatrix, DenseVector};
    use crate::types::{
        AttentionOutput, AttentionScores, KeyProjection, OutputProjection, QueryProjection,
        TokenEmbedding, TokenSequence, Value, ValueProjection,
    };

    fn bias(values: &[f32]) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias(DenseVector::new(values.to_vec())?))
    }

    fn identity_query_layer(dim: usize) -> Result<QueryLayer, ModelError> {
        let rows = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| if row == col { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        QueryLayer::new(
            QueryProjection(DenseMatrix::from_rows(rows)?),
            bias(&vec![0.0; dim])?,
        )
    }

    fn identity_key_layer(dim: usize) -> Result<KeyLayer, ModelError> {
        let rows = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| if row == col { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        KeyLayer::new(
            KeyProjection(DenseMatrix::from_rows(rows)?),
            bias(&vec![0.0; dim])?,
        )
    }

    fn identity_value_layer(dim: usize) -> Result<ValueLayer, ModelError> {
        let rows = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| if row == col { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        ValueLayer::new(
            ValueProjection(DenseMatrix::from_rows(rows)?),
            bias(&vec![0.0; dim])?,
        )
    }

    #[test]
    fn query_layer_projects_tokens() -> Result<(), ModelError> {
        let layer = QueryLayer::new(
            QueryProjection(DenseMatrix::from_rows(vec![
                vec![2.0, 0.0],
                vec![0.0, 3.0],
            ])?),
            bias(&[0.5, -1.0])?,
        )?;
        let token = TokenEmbedding(DenseVector::new(vec![1.0, 2.0])?);

        let query = layer.forward(&token)?;
        assert_eq!(query.as_slice(), &[2.5, 5.0]);
        Ok(())
    }

    #[test]
    fn scaled_attention_score_reports_mismatch() -> Result<(), ModelError> {
        let query = crate::types::Query(DenseVector::new(vec![1.0, 2.0])?);
        let key = crate::types::Key(DenseVector::new(vec![1.0, 2.0, 3.0])?);

        let error = scaled_attention_score(&query, &key).expect_err("mismatched score should fail");
        assert!(matches!(error, ModelError::DimensionMismatch { .. }));
        Ok(())
    }

    #[test]
    fn softmax_is_stable_for_large_values() -> Result<(), ModelError> {
        let weights = softmax(&AttentionScores(vec![1000.0, 1000.0]))?;
        let sum: f32 = weights.0.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights.0.iter().all(|weight| weight.is_finite()));
        Ok(())
    }

    #[test]
    fn weighted_sum_matches_manual_computation() -> Result<(), ModelError> {
        let weights = crate::types::AttentionWeights(vec![0.25, 0.75]);
        let values = vec![
            Value(DenseVector::new(vec![1.0, 0.0])?),
            Value(DenseVector::new(vec![0.0, 2.0])?),
        ];

        let output = weighted_sum(&weights, &values)?;
        assert_eq!(output.as_slice(), &[0.25, 1.5]);
        Ok(())
    }

    #[test]
    fn weighted_sum_reports_mismatch() -> Result<(), ModelError> {
        let weights = crate::types::AttentionWeights(vec![1.0]);
        let values = vec![
            Value(DenseVector::new(vec![1.0, 0.0])?),
            Value(DenseVector::new(vec![0.0, 2.0])?),
        ];

        let error =
            weighted_sum(&weights, &values).expect_err("mismatched weighted sum should fail");
        assert!(matches!(error, ModelError::DimensionMismatch { .. }));
        Ok(())
    }

    #[test]
    fn attention_head_single_token_reduces_to_value_projection() -> Result<(), ModelError> {
        let head = AttentionHead::new(
            identity_query_layer(2)?,
            identity_key_layer(2)?,
            identity_value_layer(2)?,
        )?;
        let sequence = TokenSequence::new(vec![TokenEmbedding(DenseVector::new(vec![1.0, 2.0])?)])?;

        let outputs = head.forward(&sequence)?;
        assert_eq!(outputs[0].as_slice(), &[1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn concat_attention_outputs_joins_vectors_in_order() -> Result<(), ModelError> {
        let outputs = vec![
            AttentionOutput(DenseVector::new(vec![1.0, 2.0])?),
            AttentionOutput(DenseVector::new(vec![3.0, 4.0])?),
        ];

        let concatenated = concat_attention_outputs(&outputs)?;
        assert_eq!(concatenated.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn multi_head_attention_rejects_empty_heads() -> Result<(), ModelError> {
        let output_layer = OutputLayer::new(
            OutputProjection(DenseMatrix::from_rows(vec![vec![1.0]])?),
            bias(&[0.0])?,
        )?;

        let error =
            MultiHeadAttention::new(vec![], output_layer).expect_err("empty head list should fail");
        assert!(matches!(error, ModelError::InvalidHeadConfiguration { .. }));
        Ok(())
    }

    #[test]
    fn multi_head_attention_validates_output_projection_width() -> Result<(), ModelError> {
        let head = AttentionHead::new(
            identity_query_layer(2)?,
            identity_key_layer(2)?,
            identity_value_layer(2)?,
        )?;
        let output_layer = OutputLayer::new(
            OutputProjection(DenseMatrix::from_rows(vec![vec![1.0, 0.0, 0.0]])?),
            bias(&[0.0])?,
        )?;

        let error = MultiHeadAttention::new(vec![head], output_layer)
            .expect_err("bad output width should fail");
        assert!(matches!(error, ModelError::InvalidHeadConfiguration { .. }));
        Ok(())
    }

    #[test]
    fn multi_head_attention_preserves_sequence_length() -> Result<(), ModelError> {
        let head_a = AttentionHead::new(
            identity_query_layer(2)?,
            identity_key_layer(2)?,
            identity_value_layer(2)?,
        )?;
        let head_b = AttentionHead::new(
            identity_query_layer(2)?,
            identity_key_layer(2)?,
            identity_value_layer(2)?,
        )?;
        let output_layer = OutputLayer::new(
            OutputProjection(DenseMatrix::from_rows(vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ])?),
            bias(&[0.0, 0.0])?,
        )?;
        let mha = MultiHeadAttention::new(vec![head_a, head_b], output_layer)?;
        let sequence = TokenSequence::new(vec![
            TokenEmbedding(DenseVector::new(vec![1.0, 0.0])?),
            TokenEmbedding(DenseVector::new(vec![0.0, 1.0])?),
        ])?;

        let output = mha.forward(&sequence)?;
        assert_eq!(output.len(), 2);
        assert_eq!(output.d_model(), 2);
        Ok(())
    }

    #[test]
    fn linear_attention_single_token_reduces_to_value_projection() -> Result<(), ModelError> {
        let head = LinearAttentionHead::new(
            identity_query_layer(2)?,
            identity_key_layer(2)?,
            identity_value_layer(2)?,
        )?;
        let sequence = TokenSequence::new(vec![TokenEmbedding(DenseVector::new(vec![1.0, 2.0])?)])?;

        let outputs = head.forward(&sequence)?;
        assert!((outputs[0].as_slice()[0] - 1.0).abs() < 1e-5);
        assert!((outputs[0].as_slice()[1] - 2.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn linear_attention_is_permutation_equivariant_without_positions() -> Result<(), ModelError> {
        let head = LinearAttentionHead::new(
            identity_query_layer(2)?,
            identity_key_layer(2)?,
            identity_value_layer(2)?,
        )?;
        let original = TokenSequence::new(vec![
            TokenEmbedding(DenseVector::new(vec![1.0, 0.0])?),
            TokenEmbedding(DenseVector::new(vec![0.0, 1.0])?),
        ])?;
        let permuted = TokenSequence::new(vec![
            TokenEmbedding(DenseVector::new(vec![0.0, 1.0])?),
            TokenEmbedding(DenseVector::new(vec![1.0, 0.0])?),
        ])?;

        let original_outputs = head.forward(&original)?;
        let permuted_outputs = head.forward(&permuted)?;

        assert_eq!(
            original_outputs[0].as_slice(),
            permuted_outputs[1].as_slice()
        );
        assert_eq!(
            original_outputs[1].as_slice(),
            permuted_outputs[0].as_slice()
        );
        Ok(())
    }
}
