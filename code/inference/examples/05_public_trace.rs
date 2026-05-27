use rust_ml_inference::{
    ContextWindow, DecodeRequest, Logit, MaxNewTokens, NextTokenRule, PromptTokens,
    PublicDecodeTrace, RankedToken, ReviewedDecodeTrace, SamplingMode, TokenId, TokenIndex,
    TokenRankings, TokenText, ToyNextTokenModel, ToyVocabulary, TraceVisibility, VocabularyEntry,
    VocabularySize, decode,
};

fn token(
    index: TokenIndex,
    vocabulary_size: VocabularySize,
) -> Result<TokenId, rust_ml_inference::Error> {
    TokenId::new(index, vocabulary_size)
}

fn teaching_model()
-> Result<(ToyVocabulary, ToyNextTokenModel, VocabularySize), rust_ml_inference::Error> {
    let vocabulary_size = VocabularySize::try_from(4)?;
    let start = token(TokenIndex::try_from(0)?, vocabulary_size)?;
    let public = token(TokenIndex::try_from(1)?, vocabulary_size)?;
    let trace = token(TokenIndex::try_from(2)?, vocabulary_size)?;
    let end = token(TokenIndex::try_from(3)?, vocabulary_size)?;

    let vocabulary = ToyVocabulary::new(
        vocabulary_size,
        [
            VocabularyEntry::new(start, TokenText::try_from("<prompt>")?),
            VocabularyEntry::new(public, TokenText::try_from("public")?),
            VocabularyEntry::new(trace, TokenText::try_from("trace")?),
            VocabularyEntry::new(end, TokenText::try_from("<end>")?),
        ],
    )?;

    let model = ToyNextTokenModel::new(
        vocabulary_size,
        TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(1.0)?)])?,
        [
            NextTokenRule::new(
                start,
                TokenRankings::from_candidates([RankedToken::new(public, Logit::try_from(4.0)?)])?,
            ),
            NextTokenRule::new(
                public,
                TokenRankings::from_candidates([RankedToken::new(trace, Logit::try_from(4.0)?)])?,
            ),
        ],
    )?;

    Ok((vocabulary, model, vocabulary_size))
}

fn reviewed_trace(
    model: &ToyNextTokenModel,
    vocabulary_size: VocabularySize,
    visibility: TraceVisibility,
) -> Result<ReviewedDecodeTrace, rust_ml_inference::Error> {
    let prompt = PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
    let trace = decode(
        model,
        DecodeRequest::new(
            prompt,
            ContextWindow::try_from(3)?,
            MaxNewTokens::try_from(2)?,
            SamplingMode::Greedy,
        )?,
    )?;

    Ok(ReviewedDecodeTrace::new(trace, visibility))
}

fn main() -> Result<(), rust_ml_inference::Error> {
    let (vocabulary, model, vocabulary_size) = teaching_model()?;

    let public_trace = PublicDecodeTrace::from_reviewed_trace(reviewed_trace(
        &model,
        vocabulary_size,
        TraceVisibility::Public,
    )?)?;
    println!(
        "public generated = {}",
        vocabulary.decode(public_trace.generated_tokens())?
    );

    let blocked = PublicDecodeTrace::from_reviewed_trace(reviewed_trace(
        &model,
        vocabulary_size,
        TraceVisibility::ResearchRestricted,
    )?);

    match blocked {
        Ok(trace) => println!(
            "restricted trace published with {}",
            trace.trace().cache().entry_count()
        ),
        Err(error) => println!("blocked from public trace: {}", error),
    }

    Ok(())
}
