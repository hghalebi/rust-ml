use rust_ml_inference::{
    ContextWindow, DecodeRequest, LatencyBudget, LatencyMillis, LatencyReport, Logit, MaxNewTokens,
    NextTokenRule, PromptTokens, RankedToken, SamplingMode, TokenId, TokenIndex, TokenRankings,
    TokenText, ToyNextTokenModel, ToyVocabulary, VocabularyEntry, VocabularySize, decode,
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
    let fast = token(TokenIndex::try_from(1)?, vocabulary_size)?;
    let token_word = token(TokenIndex::try_from(2)?, vocabulary_size)?;
    let end = token(TokenIndex::try_from(3)?, vocabulary_size)?;

    let vocabulary = ToyVocabulary::new(
        vocabulary_size,
        [
            VocabularyEntry::new(start, TokenText::try_from("<prompt>")?),
            VocabularyEntry::new(fast, TokenText::try_from("fast")?),
            VocabularyEntry::new(token_word, TokenText::try_from("tokens")?),
            VocabularyEntry::new(end, TokenText::try_from("<end>")?),
        ],
    )?;

    let model = ToyNextTokenModel::new(
        vocabulary_size,
        TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(1.0)?)])?,
        [
            NextTokenRule::new(
                start,
                TokenRankings::from_candidates([RankedToken::new(fast, Logit::try_from(4.0)?)])?,
            ),
            NextTokenRule::new(
                fast,
                TokenRankings::from_candidates([RankedToken::new(
                    token_word,
                    Logit::try_from(4.0)?,
                )])?,
            ),
        ],
    )?;

    Ok((vocabulary, model, vocabulary_size))
}

fn main() -> Result<(), rust_ml_inference::Error> {
    let (vocabulary, model, vocabulary_size) = teaching_model()?;
    let prompt = PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
    let trace = decode(
        &model,
        DecodeRequest::new(
            prompt,
            ContextWindow::try_from(3)?,
            MaxNewTokens::try_from(2)?,
            SamplingMode::Greedy,
        )?,
    )?;
    let report = LatencyReport::from_trace(
        &trace,
        LatencyMillis::try_from(18)?,
        LatencyMillis::try_from(12)?,
        LatencyBudget::new(LatencyMillis::try_from(40)?),
    )?;

    println!(
        "generated = {}",
        vocabulary.decode(trace.generated_tokens())?
    );
    println!("total latency = {}", report.total());
    println!("budget status = {}", report.status());
    Ok(())
}
