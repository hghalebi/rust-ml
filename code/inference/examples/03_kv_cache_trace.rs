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
    let vocabulary_size = VocabularySize::try_from(4)?;
    let start = token(TokenIndex::try_from(0)?, vocabulary_size)?;
    let query = token(TokenIndex::try_from(1)?, vocabulary_size)?;
    let cache = token(TokenIndex::try_from(2)?, vocabulary_size)?;
    let end = token(TokenIndex::try_from(3)?, vocabulary_size)?;

    let vocabulary = ToyVocabulary::new(
        vocabulary_size,
        [
            VocabularyEntry::new(start, TokenText::try_from("<prompt>")?),
            VocabularyEntry::new(query, TokenText::try_from("query")?),
            VocabularyEntry::new(cache, TokenText::try_from("cache")?),
            VocabularyEntry::new(end, TokenText::try_from("<end>")?),
        ],
    )?;

    let model = ToyNextTokenModel::new(
        vocabulary_size,
        TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(1.0)?)])?,
        [
            NextTokenRule::new(
                start,
                TokenRankings::from_candidates([RankedToken::new(query, Logit::try_from(4.0)?)])?,
            ),
            NextTokenRule::new(
                query,
                TokenRankings::from_candidates([RankedToken::new(cache, Logit::try_from(4.0)?)])?,
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

    println!(
        "generated = {}",
        vocabulary.decode(trace.generated_tokens())?
    );
    for entry in trace.cache().entries() {
        println!("{} {} {}", entry.position(), entry.role(), entry.token_id());
    }
    Ok(())
}
