//! Encoder-side Transformer building blocks.

use crate::attention::MultiHeadAttention;
use crate::error::ModelError;
use crate::math::{DenseMatrix, DenseVector};
use crate::types::{
    FeedForwardProjection1, FeedForwardProjection2, HiddenActivation, NormScale, NormShift,
    PositionEncoding, ProjectionBias, TokenEmbedding, TokenSequence,
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

/// A table of sinusoidal positional encodings.
#[derive(Debug, Clone)]
pub struct PositionalEncodingTable {
    d_model: usize,
}

impl PositionalEncodingTable {
    /// Creates a positional encoding table for a given model width.
    pub fn new(d_model: usize) -> Result<Self, ModelError> {
        if d_model == 0 {
            return Err(ModelError::EmptyInput {
                operation: "PositionalEncodingTable::new",
                details: "d_model must be greater than zero",
            });
        }

        Ok(Self { d_model })
    }

    /// Returns the model width.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Computes the positional encoding for one position.
    pub fn encoding_for(&self, pos: usize) -> Result<PositionEncoding, ModelError> {
        let mut values = vec![0.0; self.d_model];

        for (index, slot) in values.iter_mut().enumerate() {
            let exponent = (2 * (index / 2)) as f32 / self.d_model as f32;
            let angle = pos as f32 / 10000_f32.powf(exponent);
            *slot = if index % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }

        Ok(PositionEncoding(DenseVector::new(values)?))
    }

    /// Adds positional encodings to a whole token sequence.
    pub fn add_to_sequence(&self, seq: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let tokens = seq
            .tokens()
            .iter()
            .enumerate()
            .map(|(pos, token)| {
                let encoding = self.encoding_for(pos)?;
                let combined = token.0.add(&encoding.0)?;
                Ok(TokenEmbedding(combined))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;

        TokenSequence::new(tokens)
    }
}

/// Adds two token embeddings elementwise.
pub fn add_token_embeddings(
    left: &TokenEmbedding,
    right: &TokenEmbedding,
) -> Result<TokenEmbedding, ModelError> {
    let sum = left.0.add(&right.0)?;
    Ok(TokenEmbedding(sum))
}

/// Adds two sequences token-by-token for residual connections.
pub fn add_sequences(
    left: &TokenSequence,
    right: &TokenSequence,
) -> Result<TokenSequence, ModelError> {
    if left.len() != right.len() {
        return Err(ModelError::DimensionMismatch {
            operation: "add_sequences",
            left_label: "left sequence length",
            left_shape: vec![left.len()],
            right_label: "right sequence length",
            right_shape: vec![right.len()],
            hint: "residual addition requires the same number of tokens",
        });
    }

    if left.d_model() != right.d_model() {
        return Err(ModelError::DimensionMismatch {
            operation: "add_sequences",
            left_label: "left d_model",
            left_shape: vec![left.d_model()],
            right_label: "right d_model",
            right_shape: vec![right.d_model()],
            hint: "residual addition requires matching token widths",
        });
    }

    let tokens = left
        .tokens()
        .iter()
        .zip(right.tokens().iter())
        .map(|(a, b)| add_token_embeddings(a, b))
        .collect::<Result<Vec<_>, _>>()?;

    TokenSequence::new(tokens)
}

/// Layer normalization with learnable scale and shift.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: NormScale,
    beta: NormShift,
    eps: f32,
}

impl LayerNorm {
    /// Creates layer-norm parameters for a given model width.
    pub fn new(d_model: usize) -> Result<Self, ModelError> {
        if d_model == 0 {
            return Err(ModelError::EmptyInput {
                operation: "LayerNorm::new",
                details: "d_model must be greater than zero",
            });
        }

        Ok(Self {
            gamma: NormScale(DenseVector::ones(d_model)?),
            beta: NormShift(DenseVector::zeros(d_model)?),
            eps: 1e-5,
        })
    }

    /// Returns the width expected by this normalization layer.
    pub fn dimension(&self) -> usize {
        self.gamma.len()
    }

    /// Normalizes one token embedding.
    pub fn forward_token(&self, token: &TokenEmbedding) -> Result<TokenEmbedding, ModelError> {
        if token.len() != self.dimension() || token.len() != self.beta.len() {
            return Err(ModelError::DimensionMismatch {
                operation: "LayerNorm::forward_token",
                left_label: "token",
                left_shape: vec![token.len()],
                right_label: "gamma/beta",
                right_shape: vec![self.dimension()],
                hint: "layer norm parameters must match token width",
            });
        }

        let mean = token.0.mean();
        let variance = token.0.variance();
        let denom = (variance + self.eps).sqrt();

        if !denom.is_finite() || denom == 0.0 {
            return Err(ModelError::NumericalIssue {
                operation: "LayerNorm::forward_token",
                details: "normalization denominator was zero or non-finite",
            });
        }

        let normalized = DenseVector::new(
            (0..token.len())
                .map(|index| {
                    let z = (token.0.get(index) - mean) / denom;
                    self.gamma.0.get(index) * z + self.beta.0.get(index)
                })
                .collect(),
        )?;

        Ok(TokenEmbedding(normalized))
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
        validate_projection("FeedForwardLayer1::new", &weight.0, &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected input width.
    pub fn input_dim(&self) -> usize {
        self.weight.0.cols()
    }

    /// Returns the hidden width.
    pub fn output_dim(&self) -> usize {
        self.weight.0.rows()
    }

    /// Projects one token into hidden space and applies ReLU.
    pub fn forward(&self, x: &TokenEmbedding) -> Result<HiddenActivation, ModelError> {
        let wx = self.weight.0.mul_vec(&x.0)?;
        let y = wx.add(&self.bias.0)?;
        Ok(HiddenActivation(y.map(|value| value.max(0.0))))
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
        validate_projection("FeedForwardLayer2::new", &weight.0, &bias)?;
        Ok(Self { weight, bias })
    }

    /// Returns the expected hidden width.
    pub fn input_dim(&self) -> usize {
        self.weight.0.cols()
    }

    /// Returns the projected token width.
    pub fn output_dim(&self) -> usize {
        self.weight.0.rows()
    }

    /// Projects one hidden activation back into token space.
    pub fn forward(&self, x: &HiddenActivation) -> Result<TokenEmbedding, ModelError> {
        let wx = self.weight.0.mul_vec(&x.0)?;
        let y = wx.add(&self.bias.0)?;
        Ok(TokenEmbedding(y))
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
            return Err(ModelError::InvalidProjection {
                operation: "FeedForward::new",
                details: "layer1 output dimension must match layer2 input dimension",
            });
        }

        Ok(Self { layer1, layer2 })
    }

    /// Returns the expected token width.
    pub fn input_dim(&self) -> usize {
        self.layer1.input_dim()
    }

    /// Returns the projected token width.
    pub fn output_dim(&self) -> usize {
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
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "TransformerEncoderBlock::new",
                details: "attention must project back to the same token width for residual addition",
            });
        }

        if norm1.dimension() != d_model
            || norm2.dimension() != d_model
            || feed_forward.input_dim() != d_model
            || feed_forward.output_dim() != d_model
        {
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "TransformerEncoderBlock::new",
                details: "attention, norms, and feed-forward network must share the same model width",
            });
        }

        Ok(Self {
            attention,
            norm1,
            feed_forward,
            norm2,
        })
    }

    /// Returns the block model width.
    pub fn d_model(&self) -> usize {
        self.attention.output_dim()
    }

    /// Runs one encoder block forward pass.
    pub fn forward(&self, x: &TokenSequence) -> Result<TokenSequence, ModelError> {
        let attention_out = self.attention.forward(x)?;
        let after_attention_residual = add_sequences(x, &attention_out)?;
        let after_attention_norm = self.norm1.forward_sequence(&after_attention_residual)?;

        let ff_out = self.feed_forward.forward_sequence(&after_attention_norm)?;
        let after_ff_residual = add_sequences(&after_attention_norm, &ff_out)?;
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
    pub fn new(blocks: Vec<TransformerEncoderBlock>) -> Result<Self, ModelError> {
        if blocks.is_empty() {
            return Err(ModelError::EmptyInput {
                operation: "Encoder::new",
                details: "encoder must contain at least one block",
            });
        }

        let d_model = blocks[0].d_model();
        if blocks.iter().any(|block| block.d_model() != d_model) {
            return Err(ModelError::InvalidHeadConfiguration {
                operation: "Encoder::new",
                details: "all encoder blocks must share the same model width",
            });
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
    use crate::math::{DenseMatrix, DenseVector};
    use crate::types::{
        FeedForwardProjection1, FeedForwardProjection2, KeyProjection, OutputProjection,
        ProjectionBias, QueryProjection, TokenEmbedding, TokenSequence, ValueProjection,
    };

    fn vector(values: &[f32]) -> Result<DenseVector, ModelError> {
        DenseVector::new(values.to_vec())
    }

    fn bias(values: &[f32]) -> Result<ProjectionBias, ModelError> {
        Ok(ProjectionBias(vector(values)?))
    }

    fn identity_matrix(dim: usize) -> Result<DenseMatrix, ModelError> {
        DenseMatrix::from_rows(
            (0..dim)
                .map(|row| {
                    (0..dim)
                        .map(|col| if row == col { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>()
                })
                .collect(),
        )
    }

    fn identity_head(dim: usize) -> Result<AttentionHead, ModelError> {
        AttentionHead::new(
            QueryLayer::new(
                QueryProjection(identity_matrix(dim)?),
                bias(&vec![0.0; dim])?,
            )?,
            KeyLayer::new(KeyProjection(identity_matrix(dim)?), bias(&vec![0.0; dim])?)?,
            ValueLayer::new(
                ValueProjection(identity_matrix(dim)?),
                bias(&vec![0.0; dim])?,
            )?,
        )
    }

    fn identity_mha(dim: usize) -> Result<MultiHeadAttention, ModelError> {
        MultiHeadAttention::new(
            vec![identity_head(dim)?],
            OutputLayer::new(
                OutputProjection(identity_matrix(dim)?),
                bias(&vec![0.0; dim])?,
            )?,
        )
    }

    fn simple_feed_forward(dim: usize) -> Result<FeedForward, ModelError> {
        FeedForward::new(
            FeedForwardLayer1::new(
                FeedForwardProjection1(identity_matrix(dim)?),
                bias(&vec![0.0; dim])?,
            )?,
            FeedForwardLayer2::new(
                FeedForwardProjection2(identity_matrix(dim)?),
                bias(&vec![0.0; dim])?,
            )?,
        )
    }

    #[test]
    fn positional_encoding_matches_expected_pattern_at_position_zero() -> Result<(), ModelError> {
        let table = PositionalEncodingTable::new(4)?;
        let encoding = table.encoding_for(0)?;

        assert_eq!(encoding.as_slice(), &[0.0, 1.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn positional_encoding_adds_signal_without_changing_shape() -> Result<(), ModelError> {
        let table = PositionalEncodingTable::new(4)?;
        let sequence = TokenSequence::new(vec![
            TokenEmbedding(vector(&[1.0, 0.0, 1.0, 0.0])?),
            TokenEmbedding(vector(&[0.0, 1.0, 0.0, 1.0])?),
        ])?;

        let output = table.add_to_sequence(&sequence)?;
        assert_eq!(output.len(), 2);
        assert_eq!(output.d_model(), 4);
        Ok(())
    }

    #[test]
    fn add_token_embeddings_is_elementwise() -> Result<(), ModelError> {
        let left = TokenEmbedding(vector(&[1.0, 2.0])?);
        let right = TokenEmbedding(vector(&[3.0, 4.0])?);

        let sum = add_token_embeddings(&left, &right)?;
        assert_eq!(sum.as_slice(), &[4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn add_sequences_reports_mismatched_lengths() -> Result<(), ModelError> {
        let left = TokenSequence::new(vec![TokenEmbedding(vector(&[1.0])?)])?;
        let right = TokenSequence::new(vec![
            TokenEmbedding(vector(&[1.0])?),
            TokenEmbedding(vector(&[2.0])?),
        ])?;

        let error =
            add_sequences(&left, &right).expect_err("mismatched sequence lengths should fail");
        assert!(matches!(error, ModelError::DimensionMismatch { .. }));
        Ok(())
    }

    #[test]
    fn layer_norm_centers_and_scales_non_constant_tokens() -> Result<(), ModelError> {
        let norm = LayerNorm::new(3)?;
        let token = TokenEmbedding(vector(&[1.0, 2.0, 3.0])?);

        let normalized = norm.forward_token(&token)?;
        let mean = normalized.0.mean();
        let variance = normalized.0.variance();

        assert!(mean.abs() < 1e-5);
        assert!((variance - 1.0).abs() < 2e-4);
        Ok(())
    }

    #[test]
    fn layer_norm_handles_constant_tokens_with_finite_output() -> Result<(), ModelError> {
        let norm = LayerNorm::new(3)?;
        let token = TokenEmbedding(vector(&[2.0, 2.0, 2.0])?);

        let normalized = norm.forward_token(&token)?;
        assert!(normalized.as_slice().iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn feed_forward_preserves_sequence_shape() -> Result<(), ModelError> {
        let feed_forward = simple_feed_forward(2)?;
        let sequence = TokenSequence::new(vec![
            TokenEmbedding(vector(&[1.0, -2.0])?),
            TokenEmbedding(vector(&[0.5, 3.0])?),
        ])?;

        let output = feed_forward.forward_sequence(&sequence)?;
        assert_eq!(output.len(), 2);
        assert_eq!(output.d_model(), 2);
        Ok(())
    }

    #[test]
    fn transformer_block_rejects_width_mismatch() -> Result<(), ModelError> {
        let attention = identity_mha(2)?;
        let norm1 = LayerNorm::new(2)?;
        let feed_forward = simple_feed_forward(2)?;
        let norm2 = LayerNorm::new(3)?;

        let error = TransformerEncoderBlock::new(attention, norm1, feed_forward, norm2)
            .expect_err("mismatched block widths should fail");
        assert!(matches!(error, ModelError::InvalidHeadConfiguration { .. }));
        Ok(())
    }

    #[test]
    fn transformer_block_output_is_finite() -> Result<(), ModelError> {
        let block = TransformerEncoderBlock::new(
            identity_mha(2)?,
            LayerNorm::new(2)?,
            simple_feed_forward(2)?,
            LayerNorm::new(2)?,
        )?;
        let sequence = TokenSequence::new(vec![
            TokenEmbedding(vector(&[1.0, 0.0])?),
            TokenEmbedding(vector(&[0.0, 1.0])?),
        ])?;

        let output = block.forward(&sequence)?;
        assert!(
            output
                .tokens()
                .iter()
                .flat_map(|token| token.as_slice().iter())
                .all(|value| value.is_finite())
        );
        Ok(())
    }

    #[test]
    fn encoder_rejects_empty_block_lists() {
        let error = Encoder::new(vec![]).expect_err("empty encoder should fail");
        assert!(matches!(error, ModelError::EmptyInput { .. }));
    }

    #[test]
    fn encoder_runs_multiple_blocks() -> Result<(), ModelError> {
        let block = TransformerEncoderBlock::new(
            identity_mha(2)?,
            LayerNorm::new(2)?,
            simple_feed_forward(2)?,
            LayerNorm::new(2)?,
        )?;
        let encoder = Encoder::new(vec![block.clone(), block])?;
        let sequence = TokenSequence::new(vec![
            TokenEmbedding(vector(&[1.0, 0.0])?),
            TokenEmbedding(vector(&[0.0, 1.0])?),
        ])?;

        let output = encoder.forward(&sequence)?;
        assert_eq!(output.len(), 2);
        assert_eq!(output.d_model(), 2);
        Ok(())
    }
}
