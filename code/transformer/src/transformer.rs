//! Encoder-side Transformer building blocks.

use crate::attention::MultiHeadAttention;
use crate::error::ModelError;
use crate::math::{DenseMatrix, DenseVector, ModelScalar, VectorIndex, VectorLength};
use crate::types::{
    FeedForwardProjection1, FeedForwardProjection2, HiddenActivation, NormScale, NormShift,
    PositionEncoding, ProjectionBias, TokenEmbedding, TokenIndex, TokenSequence,
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

/// A table of sinusoidal positional encodings.
#[derive(Debug, Clone)]
pub struct PositionalEncodingTable {
    d_model: VectorLength,
}

impl PositionalEncodingTable {
    /// Creates a positional encoding table for a given model width.
    pub fn new(d_model: VectorLength) -> Self {
        Self { d_model }
    }

    /// Returns the model width.
    pub fn d_model(&self) -> VectorLength {
        self.d_model
    }

    /// Computes the positional encoding for one position.
    pub fn encoding_for(&self, pos: TokenIndex) -> Result<PositionEncoding, ModelError> {
        let d_model = self.d_model.as_usize();

        let values = (0..d_model)
            .map(|index| {
                let exponent = (2 * (index / 2)) as f32 / d_model as f32;
                let angle = pos.as_usize() as f32 / 10000_f32.powf(exponent);
                let value = if index % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };

                ModelScalar::try_from(value)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(PositionEncoding::from_vector(DenseVector::new(values)?))
    }

    /// Adds positional encodings to a whole token sequence.
    pub fn add_to_sequence(&self, seq: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let tokens = seq
            .tokens()
            .enumerate()
            .map(|(pos, token)| {
                let encoding = self.encoding_for(TokenIndex::from_raw(pos))?;
                token + &encoding
            })
            .collect::<Result<Vec<_>, ModelError>>()?;

        TokenSequence::from_tokens(tokens)
    }
}

/// Adds two token embeddings elementwise.
pub fn add_token_embeddings(
    left: &TokenEmbedding,
    right: &TokenEmbedding,
) -> Result<TokenEmbedding, ModelError> {
    left + right
}

/// Adds two sequences token-by-token for residual connections.
pub fn add_sequences(
    left: &TokenSequence,
    right: &TokenSequence,
) -> Result<TokenSequence, ModelError> {
    left + right
}

/// Layer normalization with learnable scale and shift.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: NormScale,
    beta: NormShift,
    eps: ModelScalar,
}

impl LayerNorm {
    /// Creates layer-norm parameters for a given model width.
    pub fn new(d_model: VectorLength) -> Result<Self, ModelError> {
        Ok(Self {
            gamma: NormScale::from_vector(DenseVector::ones(d_model)?),
            beta: NormShift::from_vector(DenseVector::zeros(d_model)?),
            eps: ModelScalar::try_from(1e-5)?,
        })
    }

    /// Returns the width expected by this normalization layer.
    pub fn dimension(&self) -> VectorLength {
        self.gamma.len()
    }

    /// Normalizes one token embedding.
    pub fn forward_token(&self, token: &TokenEmbedding) -> Result<TokenEmbedding, ModelError> {
        if token.len() != self.dimension() || token.len() != self.beta.len() {
            return Err(ModelError::dimension_mismatch(
                "LayerNorm::forward_token",
                "token",
                vec![token.len().as_usize()],
                "gamma/beta",
                vec![self.dimension().as_usize()],
                "layer norm parameters must match token width",
            ));
        }

        let mean = token.vector().mean();
        let variance = token.vector().variance();
        let denom = (variance.as_f32() + self.eps.as_f32()).sqrt();

        if !denom.is_finite() || denom == 0.0 {
            return Err(ModelError::numerical_issue(
                "LayerNorm::forward_token",
                "normalization denominator was zero or non-finite",
            ));
        }

        let normalized = DenseVector::new(
            (0..token.len().as_usize())
                .map(|index| {
                    let index = VectorIndex::try_from(index)?;
                    let z = (token.vector().component(index)?.as_f32() - mean.as_f32()) / denom;
                    let scaled = self.gamma.vector().component(index)?.as_f32() * z
                        + self.beta.vector().component(index)?.as_f32();
                    ModelScalar::try_from(scaled)
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;

        Ok(TokenEmbedding::from_vector(normalized))
    }

    /// Normalizes a whole token sequence.
    pub fn forward_sequence(&self, seq: &TokenSequence) -> Result<TokenSequence, ModelError> {
        seq.map_tokens(|token| self.forward_token(token))
    }
}

/// The first projection in the position-wise feed-forward network.
#[derive(Debug, Clone)]
pub struct FeedForwardLayer1 {
    weight: FeedForwardProjection1,
    bias: ProjectionBias,
}

impl FeedForwardLayer1 {
    /// Creates the first feed-forward layer.
    pub fn new(weight: FeedForwardProjection1, bias: ProjectionBias) -> Result<Self, ModelError> {
        validate_projection("FeedForwardLayer1::new", weight.matrix(), &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> VectorLength {
        self.weight.input_dim()
    }

    /// Returns the hidden width.
    pub fn output_dim(&self) -> VectorLength {
        self.weight.output_dim()
    }

    /// Projects one token into hidden space and applies ReLU.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<HiddenActivation, ModelError> {
        let projected = (&self.weight * x)?;
        let shifted = (&projected + &self.bias)?;
        shifted.relu()
    }
}

/// The second projection in the position-wise feed-forward network.
#[derive(Debug, Clone)]
pub struct FeedForwardLayer2 {
    weight: FeedForwardProjection2,
    bias: ProjectionBias,
}

impl FeedForwardLayer2 {
    /// Creates the second feed-forward layer.
    pub fn new(weight: FeedForwardProjection2, bias: ProjectionBias) -> Result<Self, ModelError> {
        validate_projection("FeedForwardLayer2::new", weight.matrix(), &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected hidden width.
    pub fn input_dim(&self) -> VectorLength {
        self.weight.input_dim()
    }

    /// Returns the projected token width.
    pub fn output_dim(&self) -> VectorLength {
        self.weight.output_dim()
    }

    /// Projects one hidden activation back into token space.
    pub fn forward(&self, x: &HiddenActivation) -> Result<TokenEmbedding, ModelError> {
        let projected = (&self.weight * x)?;
        &projected + &self.bias
    }
}

/// A position-wise feed-forward network.
#[derive(Debug, Clone)]
pub struct FeedForward {
    layer1: FeedForwardLayer1,
    layer2: FeedForwardLayer2,
}

impl FeedForward {
    /// Creates a feed-forward network with compatible hidden dimensions.
    pub fn new(layer1: FeedForwardLayer1, layer2: FeedForwardLayer2) -> Result<Self, ModelError> {
        if layer1.output_dim() != layer2.input_dim() {
            return Err(ModelError::invalid_projection(
                "FeedForward::new",
                "layer1 output dimension must match layer2 input dimension",
            ));
        }

        Ok(Self { layer1, layer2 })
    }

    /// Returns the expected token width.
    pub fn input_dim(&self) -> VectorLength {
        self.layer1.input_dim()
    }

    /// Returns the projected token width.
    pub fn output_dim(&self) -> VectorLength {
        self.layer2.output_dim()
    }

    /// Applies the feed-forward network to one token.
    pub fn forward_token(&self, token: &TokenEmbedding) -> Result<TokenEmbedding, ModelError> {
        let hidden = self.layer1.forward(token)?;
        self.layer2.forward(&hidden)
    }

    /// Applies the feed-forward network independently to every token.
    pub fn forward_sequence(&self, seq: &TokenSequence) -> Result<TokenSequence, ModelError> {
        seq.map_tokens(|token| self.forward_token(token))
    }
}

/// One Transformer encoder block.
#[derive(Debug, Clone)]
pub struct TransformerEncoderBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    feed_forward: FeedForward,
    norm2: LayerNorm,
}

impl TransformerEncoderBlock {
    /// Creates an encoder block with compatible sublayer widths.
    pub fn new(
        attention: MultiHeadAttention,
        norm1: LayerNorm,
        feed_forward: FeedForward,
        norm2: LayerNorm,
    ) -> Result<Self, ModelError> {
        let d_model = attention.output_dim();

        if attention.input_dim() != d_model {
            return Err(ModelError::invalid_head_configuration(
                "TransformerEncoderBlock::new",
                "attention must project back to the same token width for residual addition",
            ));
        }

        if norm1.dimension() != d_model
            || norm2.dimension() != d_model
            || feed_forward.input_dim() != d_model
            || feed_forward.output_dim() != d_model
        {
            return Err(ModelError::invalid_head_configuration(
                "TransformerEncoderBlock::new",
                "attention, norms, and feed-forward network must share the same model width",
            ));
        }

        Ok(Self {
            attention,
            norm1,
            feed_forward,
            norm2,
        })
    }

    /// Returns the block model width.
    pub fn d_model(&self) -> VectorLength {
        self.attention.output_dim()
    }

    /// Runs one encoder block forward pass.
    pub fn forward(&self, x: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let attention_out = self.attention.forward(x)?;
        let after_attention_residual = (x + &attention_out)?;
        let after_attention_norm = self.norm1.forward_sequence(&after_attention_residual)?;

        let ff_out = self.feed_forward.forward_sequence(&after_attention_norm)?;
        let after_ff_residual = (&after_attention_norm + &ff_out)?;
        self.norm2.forward_sequence(&after_ff_residual)
    }
}

/// A stack of encoder blocks.
#[derive(Debug, Clone)]
pub struct Encoder {
    blocks: Vec<TransformerEncoderBlock>,
}

impl Encoder {
    /// Creates an encoder from a non-empty list of compatible blocks.
    pub fn new(
        blocks: impl IntoIterator<Item = TransformerEncoderBlock>,
    ) -> Result<Self, ModelError> {
        let blocks = blocks.into_iter().collect::<Vec<_>>();
        if blocks.is_empty() {
            return Err(ModelError::empty_input(
                "Encoder::new",
                "encoder must contain at least one block",
            ));
        }

        let d_model = blocks[0].d_model();
        if blocks.iter().any(|block| block.d_model() != d_model) {
            return Err(ModelError::invalid_head_configuration(
                "Encoder::new",
                "all encoder blocks must share the same model width",
            ));
        }

        Ok(Self { blocks })
    }

    /// Runs the sequence through every encoder block in order.
    pub fn forward(&self, x: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let mut current = x.clone();
        for block in &self.blocks {
            current = block.forward(&current)?;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Encoder, FeedForward, FeedForwardLayer1, FeedForwardLayer2, LayerNorm,
        PositionalEncodingTable, TransformerEncoderBlock, add_sequences, add_token_embeddings,
    };
    use crate::attention::{
        AttentionHead, KeyLayer, MultiHeadAttention, OutputLayer, QueryLayer, ValueLayer,
    };
    use crate::error::ModelError;
    use crate::math::{DenseMatrix, DenseVector, ModelScalar, VectorLength};
    use crate::types::{
        FeedForwardProjection1, FeedForwardProjection2, KeyProjection, OutputProjection,
        ProjectionBias, QueryProjection, TokenCount, TokenEmbedding, TokenIndex, TokenSequence,
        ValueProjection,
    };

    fn vector(values: impl IntoIterator<Item = ModelScalar>) -> Result<DenseVector, ModelError> {
        DenseVector::new(values)
    }

    fn token(values: impl IntoIterator<Item = ModelScalar>) -> Result<TokenEmbedding, ModelError> {
        Ok(TokenEmbedding::from_vector(vector(values)?))
    }

    fn identity_matrix(dim: VectorLength) -> Result<DenseMatrix, ModelError> {
        DenseMatrix::identity(dim)
    }

    fn zero_bias(dim: VectorLength) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias::from_vector(DenseVector::zeros(dim)?))
    }

    fn identity_head(dim: VectorLength) -> Result<AttentionHead, ModelError> {
        AttentionHead::new(
            QueryLayer::new(
                QueryProjection::from_matrix(identity_matrix(dim)?),
                zero_bias(dim)?,
            )?,
            KeyLayer::new(
                KeyProjection::from_matrix(identity_matrix(dim)?),
                zero_bias(dim)?,
            )?,
            ValueLayer::new(
                ValueProjection::from_matrix(identity_matrix(dim)?),
                zero_bias(dim)?,
            )?,
        )
    }

    fn identity_mha(dim: VectorLength) -> Result<MultiHeadAttention, ModelError> {
        MultiHeadAttention::new(
            vec![identity_head(dim)?],
            OutputLayer::new(
                OutputProjection::from_matrix(identity_matrix(dim)?),
                zero_bias(dim)?,
            )?,
        )
    }

    fn simple_feed_forward(dim: VectorLength) -> Result<FeedForward, ModelError> {
        FeedForward::new(
            FeedForwardLayer1::new(
                FeedForwardProjection1::from_matrix(identity_matrix(dim)?),
                zero_bias(dim)?,
            )?,
            FeedForwardLayer2::new(
                FeedForwardProjection2::from_matrix(identity_matrix(dim)?),
                zero_bias(dim)?,
            )?,
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

    fn ensure_vector_values_finite(vector: &DenseVector) -> Result<(), ModelError> {
        for value in vector.values() {
            value.ensure_finite()?;
        }

        Ok(())
    }

    #[test]
    fn positional_encoding_matches_expected_pattern_at_position_zero() -> Result<(), ModelError> {
        let table = PositionalEncodingTable::new(VectorLength::try_from(4)?);
        let encoding = table.encoding_for(TokenIndex::try_from(0)?)?;

        assert_vector_values(
            encoding.vector(),
            [
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(1.0)?,
            ],
        );
        Ok(())
    }
    #[test]
    fn positional_encoding_adds_signal_without_changing_shape() -> Result<(), ModelError> {
        let table = PositionalEncodingTable::new(VectorLength::try_from(4)?);
        let sequence = TokenSequence::new(vec![
            token([
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(0.0)?,
            ])?,
            token([
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(1.0)?,
            ])?,
        ])?;

        let output = table.add_to_sequence(&sequence)?;
        assert_eq!(output.len(), TokenCount::try_from(2)?);
        assert_eq!(output.d_model(), VectorLength::try_from(4)?);
        Ok(())
    }

    #[test]
    fn add_token_embeddings_is_elementwise() -> Result<(), ModelError> {
        let left = token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
        let right = token([ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?])?;

        let sum = add_token_embeddings(&left, &right)?;
        assert_vector_values(
            sum.vector(),
            [ModelScalar::try_from(4.0)?, ModelScalar::try_from(6.0)?],
        );
        Ok(())
    }

    #[test]
    fn add_sequences_reports_mismatched_lengths() -> Result<(), ModelError> {
        let left = TokenSequence::new([token([ModelScalar::try_from(1.0)?])?])?;
        let right = TokenSequence::new([
            token([ModelScalar::try_from(1.0)?])?,
            token([ModelScalar::try_from(2.0)?])?,
        ])?;

        assert!(matches!(
            add_sequences(&left, &right),
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn layer_norm_centers_and_scales_non_constant_tokens() -> Result<(), ModelError> {
        let norm = LayerNorm::new(VectorLength::try_from(3)?)?;
        let token = token([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;

        let normalized = norm.forward_token(&token)?;
        normalized
            .vector()
            .mean()
            .ensure_close_to(ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.00001)?)?;
        normalized
            .vector()
            .variance()
            .ensure_close_to(ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0002)?)?;
        Ok(())
    }

    #[test]
    fn layer_norm_handles_constant_tokens_with_finite_output() -> Result<(), ModelError> {
        let norm = LayerNorm::new(VectorLength::try_from(3)?)?;
        let token = token([
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(2.0)?,
        ])?;

        let normalized = norm.forward_token(&token)?;
        assert_vector_values(
            normalized.vector(),
            [
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(0.0)?,
            ],
        );
        Ok(())
    }

    #[test]
    fn feed_forward_preserves_sequence_shape() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let feed_forward = simple_feed_forward(dim)?;
        let sequence = TokenSequence::new(vec![
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(-2.0)?])?,
            token([ModelScalar::try_from(0.5)?, ModelScalar::try_from(3.0)?])?,
        ])?;

        let output = feed_forward.forward_sequence(&sequence)?;
        assert_eq!(output.len(), TokenCount::try_from(2)?);
        assert_eq!(output.d_model(), VectorLength::try_from(2)?);
        Ok(())
    }

    #[test]
    fn transformer_block_rejects_width_mismatch() -> Result<(), ModelError> {
        let attention = identity_mha(VectorLength::try_from(2)?)?;
        let norm1 = LayerNorm::new(VectorLength::try_from(2)?)?;
        let feed_forward = simple_feed_forward(VectorLength::try_from(2)?)?;
        let norm2 = LayerNorm::new(VectorLength::try_from(3)?)?;

        assert!(matches!(
            TransformerEncoderBlock::new(attention, norm1, feed_forward, norm2),
            Err(ModelError::InvalidHeadConfiguration { .. })
        ));
        Ok(())
    }

    #[test]
    fn transformer_block_output_is_finite() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let block = TransformerEncoderBlock::new(
            identity_mha(dim)?,
            LayerNorm::new(dim)?,
            simple_feed_forward(dim)?,
            LayerNorm::new(dim)?,
        )?;
        let sequence = TokenSequence::new(vec![
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
            token([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?,
        ])?;

        let output = block.forward(&sequence)?;
        for token in output.tokens() {
            ensure_vector_values_finite(token.vector())?;
        }
        Ok(())
    }

    #[test]
    fn encoder_rejects_empty_block_lists() {
        assert!(matches!(
            Encoder::new(vec![]),
            Err(ModelError::EmptyInput { .. })
        ));
    }

    #[test]
    fn encoder_runs_multiple_blocks() -> Result<(), ModelError> {
        let dim = VectorLength::try_from(2)?;
        let block = TransformerEncoderBlock::new(
            identity_mha(dim)?,
            LayerNorm::new(dim)?,
            simple_feed_forward(dim)?,
            LayerNorm::new(dim)?,
        )?;
        let encoder = Encoder::new(vec![block.clone(), block])?;
        let sequence = TokenSequence::new(vec![
            token([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?,
            token([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?,
        ])?;

        let output = encoder.forward(&sequence)?;
        assert_eq!(output.len(), TokenCount::try_from(2)?);
        assert_eq!(output.d_model(), VectorLength::try_from(2)?);
        Ok(())
    }
}
