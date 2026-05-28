use rust_ml_lm_basics::{RawText, Vocabulary, WhitespaceTokenizer};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let text = RawText::try_from("red blue red")?;
    let tokens = WhitespaceTokenizer.tokenize(&text)?;
    let vocabulary = Vocabulary::from_tokens(&tokens)?;
    let ids = (&vocabulary * &tokens)?;
    let printable_tokens = tokens
        .tokens()
        .map(|token| token.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let printable_ids = ids
        .ids()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    println!("tokens = [{printable_tokens}]");
    println!("ids    = [{printable_ids}]");
    println!("vocab size = {}", vocabulary.size());

    Ok(())
}
