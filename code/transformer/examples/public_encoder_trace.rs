use rust_ml_transformer::{
    AttentionHead, DenseMatrix, DenseVector, Encoder, EncoderTraceVisibility, FeedForward,
    FeedForwardLayer1, FeedForwardLayer2, FeedForwardProjection1, FeedForwardProjection2, KeyLayer,
    KeyProjection, LayerNorm, ModelError, ModelScalar, MultiHeadAttention, OutputLayer,
    OutputProjection, ProjectionBias, PublicEncoderTrace, QueryLayer, QueryProjection,
    ReviewedEncoderTrace, TokenEmbedding, TokenSequence, TransformerEncoderBlock, ValueLayer,
    ValueProjection, VectorLength,
};

fn identity_matrix(width: VectorLength) -> Result<DenseMatrix, ModelError> {
    DenseMatrix::identity(width)
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn identity_head(width: VectorLength) -> Result<AttentionHead, ModelError> {
    AttentionHead::new(
        QueryLayer::new(
            QueryProjection::from_matrix(identity_matrix(width)?),
            zero_bias(width)?,
        )?,
        KeyLayer::new(
            KeyProjection::from_matrix(identity_matrix(width)?),
            zero_bias(width)?,
        )?,
        ValueLayer::new(
            ValueProjection::from_matrix(identity_matrix(width)?),
            zero_bias(width)?,
        )?,
    )
}

fn encoder_block(width: VectorLength) -> Result<TransformerEncoderBlock, ModelError> {
    let attention = MultiHeadAttention::new(
        [identity_head(width)?],
        OutputLayer::new(
            OutputProjection::from_matrix(identity_matrix(width)?),
            zero_bias(width)?,
        )?,
    )?;
    let feed_forward = FeedForward::new(
        FeedForwardLayer1::new(
            FeedForwardProjection1::from_matrix(identity_matrix(width)?),
            zero_bias(width)?,
        )?,
        FeedForwardLayer2::new(
            FeedForwardProjection2::from_matrix(identity_matrix(width)?),
            zero_bias(width)?,
        )?,
    )?;

    TransformerEncoderBlock::new(
        attention,
        LayerNorm::new(width)?,
        feed_forward,
        LayerNorm::new(width)?,
    )
}

fn public_input() -> Result<TokenSequence, ModelError> {
    TokenSequence::new([
        TokenEmbedding::from_vector(DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.0)?,
        ])?),
        TokenEmbedding::from_vector(DenseVector::new([
            ModelScalar::try_from(0.0)?,
            ModelScalar::try_from(1.0)?,
        ])?),
    ])
}

fn reviewed_trace(visibility: EncoderTraceVisibility) -> Result<ReviewedEncoderTrace, ModelError> {
    let width = VectorLength::try_from(2)?;
    let encoder = Encoder::new([encoder_block(width)?])?;
    let trace = encoder.trace_forward(&public_input()?)?;

    Ok(ReviewedEncoderTrace::new(trace, visibility))
}

fn main() -> Result<(), ModelError> {
    let public_trace =
        PublicEncoderTrace::from_reviewed_trace(reviewed_trace(EncoderTraceVisibility::Public)?)?;

    println!("public encoder blocks = {}", public_trace.block_count());
    println!("public token count    = {}", public_trace.token_count());
    println!("public model width    = {}", public_trace.d_model());

    let private_trace =
        PublicEncoderTrace::from_reviewed_trace(reviewed_trace(EncoderTraceVisibility::Private)?);

    if let Err(error) = private_trace {
        println!("blocked from public Transformer trace: {error}");
    }

    Ok(())
}
