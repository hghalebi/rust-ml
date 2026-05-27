use rust_ml_inference::{
    ContextWindow, DecodeRequest, Logit, MaxNewTokens, NextTokenRule, PromptTokens, RankedToken,
    SamplingMode, TokenId, TokenIndex, TokenRankings, TokenText, ToyNextTokenModel, ToyVocabulary,
    VocabularyEntry, VocabularySize, decode,
};

fn token(
    index: TokenIndex,
    vocabulary_size: VocabularySize,
) -> Result<TokenId, rust_ml_inference::Error> {
    TokenId::new(index, vocabulary_size)
}

fn teaching_model()
-> Result<(ToyVocabulary, ToyNextTokenModel, VocabularySize), rust_ml_inference::Error> {
    let vocabulary_size = VocabularySize::try_from(5)?;
    let start = token(TokenIndex::try_from(0)?, vocabulary_size)?;
    let typed = token(TokenIndex::try_from(1)?, vocabulary_size)?;
    let rust = token(TokenIndex::try_from(2)?, vocabulary_size)?;
    let maps = token(TokenIndex::try_from(3)?, vocabulary_size)?;
    let end = token(TokenIndex::try_from(4)?, vocabulary_size)?;

    let vocabulary = ToyVocabulary::new(
        vocabulary_size,
        [
            VocabularyEntry::new(start, TokenText::try_from("<prompt>")?),
            VocabularyEntry::new(typed, TokenText::try_from("typed")?),
            VocabularyEntry::new(rust, TokenText::try_from("Rust")?),
            VocabularyEntry::new(maps, TokenText::try_from("maps")?),
            VocabularyEntry::new(end, TokenText::try_from("<end>")?),
        ],
    )?;

    let model = ToyNextTokenModel::new(
        vocabulary_size,
        TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(1.0)?)])?,
        [
            NextTokenRule::new(
                start,
                TokenRankings::from_candidates([
                    RankedToken::new(typed, Logit::try_from(4.0)?),
                    RankedToken::new(rust, Logit::try_from(2.0)?),
                ])?,
            ),
            NextTokenRule::new(
                typed,
                TokenRankings::from_candidates([RankedToken::new(rust, Logit::try_from(4.0)?)])?,
            ),
            NextTokenRule::new(
                rust,
                TokenRankings::from_candidates([RankedToken::new(maps, Logit::try_from(4.0)?)])?,
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
            ContextWindow::try_from(4)?,
            MaxNewTokens::try_from(3)?,
            SamplingMode::Greedy,
        )?,
    )?;

    println!("mode = {}", trace.request().sampling_mode());
    println!(
        "generated = {}",
        vocabulary.decode(trace.generated_tokens())?
    );
    println!("cache entries = {}", trace.cache().entry_count());
    Ok(())
}
