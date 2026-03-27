use rust_ml_transformer::{
    AttentionHead, DenseMatrix, DenseVector, FeedForward, FeedForwardLayer1, FeedForwardLayer2,
    FeedForwardProjection1, FeedForwardProjection2, KeyLayer, KeyProjection, LayerNorm, ModelError,
    MultiHeadAttention, OutputLayer, OutputProjection, PositionalEncodingTable, ProjectionBias,
    QueryLayer, QueryProjection, TokenEmbedding, TokenSequence, TransformerEncoderBlock,
    ValueLayer, ValueProjection,
};

fn matrix(rows: Vec<Vec<f32>>) -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows(rows)
}

fn bias(values: Vec<f32>) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias(DenseVector::new(values)?))
}

fn head_one() -> Result<AttentionHead, ModelError> {
    AttentionHead::new(
        QueryLayer::new(
            QueryProjection(matrix(vec![
                vec![0.2, 0.1, 0.0, 0.3],
                vec![0.0, 0.4, 0.1, 0.2],
            ])?),
            bias(vec![0.0, 0.0])?,
        )?,
        KeyLayer::new(
            KeyProjection(matrix(vec![
                vec![0.1, 0.0, 0.3, 0.2],
                vec![0.2, 0.2, 0.1, 0.0],
            ])?),
            bias(vec![0.0, 0.0])?,
        )?,
        ValueLayer::new(
            ValueProjection(matrix(vec![
                vec![0.3, 0.1, 0.2, 0.0],
                vec![0.0, 0.2, 0.3, 0.1],
            ])?),
            bias(vec![0.0, 0.0])?,
        )?,
    )
}

fn head_two() -> Result<AttentionHead, ModelError> {
    AttentionHead::new(
        QueryLayer::new(
            QueryProjection(matrix(vec![
                vec![0.1, 0.3, 0.2, 0.0],
                vec![0.2, 0.1, 0.0, 0.4],
            ])?),
            bias(vec![0.0, 0.0])?,
        )?,
        KeyLayer::new(
            KeyProjection(matrix(vec![
                vec![0.2, 0.1, 0.1, 0.3],
                vec![0.1, 0.0, 0.4, 0.2],
            ])?),
            bias(vec![0.0, 0.0])?,
        )?,
        ValueLayer::new(
            ValueProjection(matrix(vec![
                vec![0.0, 0.2, 0.1, 0.3],
                vec![0.3, 0.1, 0.2, 0.0],
            ])?),
            bias(vec![0.0, 0.0])?,
        )?,
    )
}

fn main() -> Result<(), ModelError> {
    let input = TokenSequence::new(vec![
        TokenEmbedding(DenseVector::new(vec![1.0, 0.0, 1.0, 0.0])?),
        TokenEmbedding(DenseVector::new(vec![0.0, 1.0, 0.0, 1.0])?),
        TokenEmbedding(DenseVector::new(vec![1.0, 1.0, 0.0, 0.0])?),
    ])?;

    let positions = PositionalEncodingTable::new(4)?;
    let with_position = positions.add_to_sequence(&input)?;

    let attention = MultiHeadAttention::new(
        vec![head_one()?, head_two()?],
        OutputLayer::new(
            OutputProjection(matrix(vec![
                vec![0.2, 0.1, 0.0, 0.1],
                vec![0.0, 0.3, 0.2, 0.1],
                vec![0.1, 0.0, 0.3, 0.2],
                vec![0.2, 0.1, 0.1, 0.2],
            ])?),
            bias(vec![0.0, 0.0, 0.0, 0.0])?,
        )?,
    )?;

    let feed_forward = FeedForward::new(
        FeedForwardLayer1::new(
            FeedForwardProjection1(matrix(vec![
                vec![0.2, 0.1, 0.0, 0.3],
                vec![0.1, 0.2, 0.3, 0.0],
                vec![0.0, 0.3, 0.1, 0.2],
                vec![0.2, 0.0, 0.2, 0.1],
                vec![0.1, 0.1, 0.1, 0.1],
                vec![0.3, 0.2, 0.0, 0.1],
            ])?),
            bias(vec![0.0; 6])?,
        )?,
        FeedForwardLayer2::new(
            FeedForwardProjection2(matrix(vec![
                vec![0.2, 0.1, 0.0, 0.3, 0.1, 0.2],
                vec![0.1, 0.2, 0.3, 0.0, 0.1, 0.0],
                vec![0.0, 0.3, 0.1, 0.2, 0.2, 0.1],
                vec![0.2, 0.0, 0.2, 0.1, 0.1, 0.3],
            ])?),
            bias(vec![0.0; 4])?,
        )?,
    )?;

    let block = TransformerEncoderBlock::new(
        attention,
        LayerNorm::new(4)?,
        feed_forward,
        LayerNorm::new(4)?,
    )?;

    let output = block.forward(&with_position)?;

    for (index, token) in output.tokens().iter().enumerate() {
        println!("token {index}: {:?}", token.as_slice());
    }

    Ok(())
}
