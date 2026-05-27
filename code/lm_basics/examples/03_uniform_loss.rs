use rust_ml_lm_basics::{
    NextTokenBatch, RawText, TinyBigramLanguageModel, Vocabulary, WhitespaceTokenizer,
};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let text = RawText::try_from("red blue red")?;
    let tokens = WhitespaceTokenizer.tokenize(&text)?;
    let vocabulary = Vocabulary::from_tokens(&tokens)?;
    let ids = (&vocabulary * &tokens)?;
    let batch = NextTokenBatch::from_sequence(&ids)?;
    let model = TinyBigramLanguageModel::uniform(vocabulary.size())?;

    let loss = (&model * &batch)?;

    println!("uniform batch loss = {loss:.4}");
    Ok(())
}
