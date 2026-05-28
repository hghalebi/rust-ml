use rust_ml_lm_basics::{
    LearningRate, PublicLanguageModelingExample, RawText, ReviewedRawText, TextVisibility,
    TinyBigramLanguageModel, WhitespaceTokenizer,
};

fn public_text() -> Result<ReviewedRawText, rust_ml_lm_basics::Error> {
    Ok(ReviewedRawText::new(
        RawText::try_from("red blue red")?,
        TextVisibility::Public,
    ))
}

fn private_text() -> Result<ReviewedRawText, rust_ml_lm_basics::Error> {
    Ok(ReviewedRawText::new(
        RawText::try_from("red blue red")?,
        TextVisibility::Private,
    ))
}

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let example =
        PublicLanguageModelingExample::from_reviewed_text(public_text()?, WhitespaceTokenizer)?;
    let mut model = TinyBigramLanguageModel::uniform(example.vocabulary().size())?;
    let trace = model.train_one_step(example.batch(), LearningRate::try_from(0.5)?)?;

    println!("public token count = {}", example.tokens().len());
    println!("public vocabulary size = {}", example.vocabulary().size());
    println!("public loss before = {:.4}", trace.loss_before());
    println!("public loss after = {:.4}", trace.loss_after());

    let blocked =
        PublicLanguageModelingExample::from_reviewed_text(private_text()?, WhitespaceTokenizer);
    if let Err(error) = blocked {
        println!("blocked from public language-modeling example: {error}");
    }

    Ok(())
}
