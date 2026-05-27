use rust_ml_lm_basics::{
    LearningRate, NextTokenBatch, RawText, TinyBigramLanguageModel, Vocabulary, WhitespaceTokenizer,
};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let text = RawText::try_from("red blue red")?;
    let tokens = WhitespaceTokenizer.tokenize(&text)?;
    let vocabulary = Vocabulary::from_tokens(&tokens)?;
    let ids = (&vocabulary * &tokens)?;
    let batch = NextTokenBatch::from_sequence(&ids)?;
    let mut model = TinyBigramLanguageModel::uniform(vocabulary.size())?;

    let baseline = (&model * &batch)?;
    let trace = model.train_one_step(&batch, LearningRate::try_from(0.5)?)?;

    println!("baseline loss = {baseline:.4}");
    println!("loss before   = {:.4}", trace.loss_before());
    println!("loss after    = {:.4}", trace.loss_after());

    Ok(())
}
