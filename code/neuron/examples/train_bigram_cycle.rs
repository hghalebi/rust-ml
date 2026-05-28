use std::{cmp::Ordering, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BigramExample {
    input: TokenId,
    target: TokenId,
}

#[derive(Debug)]
enum Error {
    EmptyVocabulary,
    ZeroModelWidth,
    TokenOutOfRange { token: usize, vocab_size: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVocabulary => write!(formatter, "vocabulary size must be greater than zero"),
            Self::ZeroModelWidth => write!(formatter, "model width must be greater than zero"),
            Self::TokenOutOfRange { token, vocab_size } => {
                write!(
                    formatter,
                    "token {token} is outside vocabulary size {vocab_size}"
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
struct BigramEpochMetrics {
    epoch: usize,
    average_loss: f64,
}

#[derive(Debug)]
struct TinyBigramModel {
    embedding: Vec<Vec<f64>>,
    lm_head: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

impl TinyBigramModel {
    fn new(vocab_size: usize, d_model: usize) -> Result<Self, Error> {
        if vocab_size == 0 {
            return Err(Error::EmptyVocabulary);
        }
        if d_model == 0 {
            return Err(Error::ZeroModelWidth);
        }

        Ok(Self {
            embedding: init_matrix(vocab_size, d_model, 0.2, 1),
            lm_head: init_matrix(d_model, vocab_size, 0.2, 2),
            bias: vec![0.0; vocab_size],
        })
    }

    fn logits_for_token(&self, input: TokenId) -> Result<Vec<f64>, Error> {
        if input.0 >= self.embedding.len() {
            return Err(Error::TokenOutOfRange {
                token: input.0,
                vocab_size: self.embedding.len(),
            });
        }

        let hidden = &self.embedding[input.0];
        let mut logits = self.bias.clone();

        for (token_id, logit) in logits.iter_mut().enumerate() {
            for (dimension, hidden_component) in hidden.iter().enumerate() {
                *logit += hidden_component * self.lm_head[dimension][token_id];
            }
        }

        Ok(logits)
    }

    fn probabilities_for_token(&self, input: TokenId) -> Result<Vec<f64>, Error> {
        let logits = self.logits_for_token(input)?;
        Ok(softmax_unchecked(&logits))
    }

    fn predict_next(&self, input: TokenId) -> Result<TokenId, Error> {
        let probabilities = self.probabilities_for_token(input)?;
        Ok(TokenId(argmax(&probabilities)))
    }

    fn average_loss(&self, dataset: &[BigramExample], vocab_size: usize) -> Result<f64, Error> {
        if vocab_size == 0 {
            return Err(Error::EmptyVocabulary);
        }

        let mut total = 0.0;
        for example in dataset {
            let logits = self.logits_for_token(example.input)?;
            let probabilities = softmax_unchecked(&logits);
            total += -probabilities[example.target.0].ln();
        }

        Ok(total / dataset.len() as f64)
    }

    fn train_one_example(
        &mut self,
        example: &BigramExample,
        learning_rate: f64,
    ) -> Result<(), Error> {
        if example.input.0 >= self.embedding.len() || example.target.0 >= self.bias.len() {
            return Err(Error::TokenOutOfRange {
                token: example.target.0,
                vocab_size: self.embedding.len(),
            });
        }

        let logits = self.logits_for_token(example.input)?;
        let d_model = self.embedding[example.input.0].len();

        let mut probabilities = softmax_unchecked(&logits);
        probabilities[example.target.0] -= 1.0;

        for (token_id, grad) in probabilities.iter().copied().enumerate() {
            for dimension in 0..d_model {
                let update = grad * self.embedding[example.input.0][dimension];
                self.lm_head[dimension][token_id] -= learning_rate * update;
            }
            self.bias[token_id] -= learning_rate * grad;
        }

        for dimension in 0..d_model {
            let mut embedding_grad = 0.0;
            for (probability, weight) in probabilities.iter().zip(self.lm_head[dimension].iter()) {
                embedding_grad += probability * weight;
            }
            self.embedding[example.input.0][dimension] -= learning_rate * embedding_grad;
        }

        Ok(())
    }

    fn train_epochs(
        &mut self,
        dataset: &[BigramExample],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<Vec<BigramEpochMetrics>, Error> {
        let mut metrics = Vec::with_capacity(epochs);
        for epoch in 1..=epochs {
            for example in dataset.iter() {
                self.train_one_example(example, learning_rate)?;
            }
            let average_loss = self
                .average_loss(dataset, self.embedding.len())
                .unwrap_or(0.0);
            metrics.push(BigramEpochMetrics {
                epoch,
                average_loss,
            });
        }
        Ok(metrics)
    }
}

fn init_matrix(rows: usize, cols: usize, scale: f64, seed: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; cols]; rows];
    for (row_index, row) in out.iter_mut().enumerate() {
        for (col_index, slot) in row.iter_mut().enumerate() {
            let linear_index = row_index * cols + col_index;
            let raw = (linear_index.wrapping_mul(37) + seed.wrapping_mul(101)) % 1000;
            let unit = raw as f64 / 1000.0;
            *slot = (unit - 0.5) * scale;
        }
    }
    out
}

fn softmax_unchecked(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();
    let sum: f64 = exponentials.iter().sum();
    exponentials.into_iter().map(|value| value / sum).collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Less))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn main() {
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

    let mut dataset = Vec::with_capacity(token_stream.len().saturating_sub(1));
    for window in token_stream.windows(2) {
        if let [input, target] = window {
            if target.0 >= vocab.len() || input.0 >= vocab.len() {
                println!("token stream contains invalid token id; skipping");
                return;
            }
            dataset.push(BigramExample {
                input: *input,
                target: *target,
            });
        }
    }

    if dataset.is_empty() {
        println!("dataset is empty; nothing to train");
        return;
    }

    let mut model = match TinyBigramModel::new(vocab.len(), 8) {
        Ok(model) => model,
        Err(error) => {
            println!("failed to create model: {error:?}");
            return;
        }
    };

    let before = model
        .average_loss(&dataset, vocab.len())
        .unwrap_or(f64::NAN);
    let metrics = match model.train_epochs(&dataset, 0.1, 300) {
        Ok(metrics) => metrics,
        Err(error) => {
            println!("training failed: {error:?}");
            return;
        }
    };
    let after = model
        .average_loss(&dataset, vocab.len())
        .unwrap_or(f64::NAN);

    println!("average loss: before={before:.4}, after={after:.4}");

    for input_id in 1..vocab.len() {
        let predicted = match model.predict_next(TokenId(input_id)) {
            Ok(token_id) => token_id,
            Err(error) => {
                println!("prediction failed for input {input_id}: {error:?}");
                continue;
            }
        };
        let probabilities = match model.probabilities_for_token(TokenId(input_id)) {
            Ok(probabilities) => probabilities,
            Err(error) => {
                println!("probabilities failed for input {input_id}: {error:?}");
                continue;
            }
        };
        println!(
            "{:>4} -> {:>4}   confidence {:.3}",
            vocab[input_id], vocab[predicted.0], probabilities[predicted.0]
        );
    }

    if let Some(last_epoch) = metrics.last() {
        println!(
            "last epoch={} average_loss={:.4}",
            last_epoch.epoch, last_epoch.average_loss
        );
    }
}
