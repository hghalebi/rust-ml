use rust_ml_transformer::{
    DenseMatrix, DenseVector, ExpertBank, ExpertCount, ExpertScore, ExpertScores, FeedForward,
    FeedForwardExpert, FeedForwardLayer1, FeedForwardLayer2, FeedForwardProjection1,
    FeedForwardProjection2, ModelError, ModelScalar, ProjectionBias, TokenEmbedding, TokenIndex,
    TopExpertRouter, VectorLength,
};

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn diagonal_expert(scale: ModelScalar) -> Result<FeedForwardExpert, ModelError> {
    let zero = ModelScalar::try_from(0.0)?;
    let layer1 = FeedForwardLayer1::new(
        FeedForwardProjection1::from_matrix(DenseMatrix::from_rows([
            [scale, zero],
            [zero, scale],
        ])?),
        zero_bias(VectorLength::try_from(2)?)?,
    )?;
    let layer2 = FeedForwardLayer2::new(
        FeedForwardProjection2::from_matrix(DenseMatrix::from_rows([
            [scale, zero],
            [zero, scale],
        ])?),
        zero_bias(VectorLength::try_from(2)?)?,
    )?;

    Ok(FeedForwardExpert::new(FeedForward::new(layer1, layer2)?))
}

fn main() -> Result<(), ModelError> {
    let expert_bank = ExpertBank::new([
        diagonal_expert(ModelScalar::try_from(1.0)?)?,
        diagonal_expert(ModelScalar::try_from(2.0)?)?,
        diagonal_expert(ModelScalar::try_from(0.5)?)?,
    ])?;
    let router = TopExpertRouter::new(ExpertCount::try_from(3)?);
    let token_index = TokenIndex::try_from(0)?;
    let token = TokenEmbedding::from_vector(DenseVector::new([
        ModelScalar::try_from(1.0)?,
        ModelScalar::try_from(0.5)?,
    ])?);
    let scores = ExpertScores::new([
        ExpertScore::try_from(0.1)?,
        ExpertScore::try_from(2.4)?,
        ExpertScore::try_from(0.8)?,
    ])?;

    let route = router.route(token_index, &scores)?;
    println!("token: {}", route.token_index());
    println!("expert: {}", route.choice().expert_index());
    println!("router score: {}", route.choice().score());
    println!(
        "expert output: {:?}",
        expert_bank
            .forward_routed_token(&token, route)?
            .scalar_values()
    );

    Ok(())
}
