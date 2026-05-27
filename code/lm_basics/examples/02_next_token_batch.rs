use rust_ml_lm_basics::{NextTokenBatch, RawText, Vocabulary, WhitespaceTokenizer};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let text = RawText::try_from("red blue red")?;
    let tokens = WhitespaceTokenizer.tokenize(&text)?;
    let vocabulary = Vocabulary::from_tokens(&tokens)?;
    let ids = (&vocabulary * &tokens)?;
    let batch = NextTokenBatch::from_sequence(&ids)?;

    let input_ids = batch
        .inputs()
        .ids()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let target_ids = batch
        .targets()
        .ids()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    println!("inputs  = [{input_ids}]");
    println!("targets = [{target_ids}]");
    println!("context length = {}", batch.context_length());

    Ok(())
}
