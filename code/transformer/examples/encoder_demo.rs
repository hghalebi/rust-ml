use rust_ml_transformer::{
    AttentionHead, DenseMatrix, DenseVector, FeedForward, FeedForwardLayer1, FeedForwardLayer2,
    FeedForwardProjection1, FeedForwardProjection2, KeyLayer, KeyProjection, LayerNorm, ModelError,
    ModelScalar, MultiHeadAttention, OutputLayer, OutputProjection, PositionalEncodingTable,
    ProjectionBias, QueryLayer, QueryProjection, TokenEmbedding, TokenSequence,
    TransformerEncoderBlock, ValueLayer, ValueProjection, VectorLength,
};

fn vector(values: impl IntoIterator<Item = ModelScalar>) -> Result<DenseVector, ModelError> {
    DenseVector::new(values)
}

fn matrix(
    rows: impl IntoIterator<Item = impl IntoIterator<Item = ModelScalar>>,
) -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows(rows)
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn head_one() -> Result<AttentionHead, ModelError> {
    AttentionHead::new(
        QueryLayer::new(
            QueryProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                ],
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.4)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?,
        KeyLayer::new(
            KeyProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.2)?,
                ],
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?,
        ValueLayer::new(
            ValueProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.0)?,
                ],
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.1)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?,
    )
}

fn head_two() -> Result<AttentionHead, ModelError> {
    AttentionHead::new(
        QueryLayer::new(
            QueryProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.0)?,
                ],
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.4)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?,
        KeyLayer::new(
            KeyProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.3)?,
                ],
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.4)?,
                    ModelScalar::try_from(0.2)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?,
        ValueLayer::new(
            ValueProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.3)?,
                ],
                [
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.0)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(2)?)?,
        )?,
    )
}

fn main() -> Result<(), ModelError> {
    let input = TokenSequence::new(vec![
        TokenEmbedding::from_vector(vector([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.0)?,
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.0)?,
        ])?),
        TokenEmbedding::from_vector(vector([
            ModelScalar::try_from(0.0)?,
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.0)?,
            ModelScalar::try_from(1.0)?,
        ])?),
        TokenEmbedding::from_vector(vector([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(0.0)?,
            ModelScalar::try_from(0.0)?,
        ])?),
    ])?;

    let positions = PositionalEncodingTable::new(VectorLength::try_from(4)?);
    let with_position = positions.add_to_sequence(&input)?;

    let attention = MultiHeadAttention::new(
        vec![head_one()?, head_two()?],
        OutputLayer::new(
            OutputProjection::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.1)?,
                ],
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                ],
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.2)?,
                ],
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(4)?)?,
        )?,
    )?;

    let feed_forward = FeedForward::new(
        FeedForwardLayer1::new(
            FeedForwardProjection1::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                ],
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.0)?,
                ],
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                ],
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                ],
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.1)?,
                ],
                [
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.1)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(6)?)?,
        )?,
        FeedForwardLayer2::new(
            FeedForwardProjection2::from_matrix(matrix([
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                ],
                [
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.0)?,
                ],
                [
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.3)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                ],
                [
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.0)?,
                    ModelScalar::try_from(0.2)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.1)?,
                    ModelScalar::try_from(0.3)?,
                ],
            ])?),
            zero_bias(VectorLength::try_from(4)?)?,
        )?,
    )?;

    let block = TransformerEncoderBlock::new(
        attention,
        LayerNorm::new(VectorLength::try_from(4)?)?,
        feed_forward,
        LayerNorm::new(VectorLength::try_from(4)?)?,
    )?;

    let output = block.forward(&with_position)?;

    for (index, token) in output.tokens().enumerate() {
        println!("token {index}: {:?}", token.scalar_values());
    }

    Ok(())
}
