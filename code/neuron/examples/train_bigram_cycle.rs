use rust_ml_neuron::{BigramDataset, LearningRate, TinyBigramModel, TokenId};

fn main() -> Result<(), rust_ml_neuron::BigramError> {
    let vocab = ["<pad>", "I", "love", "Rust", "."];
    let token_stream = vec![
        TokenId(1),
        TokenId(2),
        TokenId(3),
        TokenId(4),
        TokenId(1),
        TokenId(2),
        TokenId(3),
        TokenId(4),
        TokenId(1),
        TokenId(2),
        TokenId(3),
        TokenId(4),
    ];

    let dataset = BigramDataset::from_token_stream(&token_stream, vocab.len())?;
    let mut model = TinyBigramModel::new(vocab.len(), 8)?;

    let before = f64::from(model.average_loss(&dataset)?);
    let metrics = model.train_epochs(&dataset, LearningRate(0.1), 300)?;
    let after = f64::from(model.average_loss(&dataset)?);

    println!("average loss: before={before:.4}, after={after:.4}");

    for input_id in 1..vocab.len() {
        let predicted = model.predict_next(TokenId(input_id))?;
        let probabilities = model.probabilities_for_token(TokenId(input_id))?;
        println!(
            "{:>4} -> {:>4}   confidence {:.3}",
            vocab[input_id], vocab[predicted.0], probabilities[predicted.0]
        );
    }

    if let Some(last_epoch) = metrics.last() {
        println!(
            "last epoch={} average_loss={:.4}",
            last_epoch.epoch,
            f64::from(last_epoch.average_loss)
        );
    }

    Ok(())
}
