//! A tiny next-token model that bridges scalar neuron learning to token
//! supervision.
//!
//! The model is intentionally minimal:
//! - input token id
//! - embedding lookup
//! - linear language-model head
//! - logits
//! - softmax cross-entropy loss
//!
//! This is still not a full Transformer. It exists to make the training
//! machinery around token targets concrete before attention is introduced.

use thiserror::Error;

use crate::{neuron::Loss, optimizer::LearningRate, token_targets::TokenId};

/// One one-step next-token example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BigramExample {
    /// The current token.
    pub input: TokenId,
    /// The token that should come next.
    pub target: TokenId,
}

impl BigramExample {
    /// Creates one next-token pair.
    pub const fn new(input: TokenId, target: TokenId) -> Self {
        Self { input, target }
    }
}

/// Errors returned by the tiny next-token training bridge.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BigramError {
    /// The vocabulary must contain at least one token.
    #[error("vocab size must be at least 1")]
    EmptyVocabulary,
    /// The model width must contain at least one hidden feature.
    #[error("model width must be at least 1")]
    ZeroModelWidth,
    /// At least two tokens are needed to create adjacent next-token pairs.
    #[error("token streams need at least two tokens to create next-token pairs")]
    TooShortTokenStream,
    /// Token ids must remain inside the configured vocabulary.
    #[error("token index {token} is out of range for vocab size {vocab_size}")]
    TokenOutOfRange { token: usize, vocab_size: usize },
}

/// A non-empty dataset of adjacent next-token pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigramDataset {
    examples: Vec<BigramExample>,
    vocab_size: usize,
}

impl BigramDataset {
    /// Builds a next-token dataset from adjacent token windows.
    pub fn from_token_stream(tokens: &[TokenId], vocab_size: usize) -> Result<Self, BigramError> {
        if vocab_size == 0 {
            return Err(BigramError::EmptyVocabulary);
        }

        if tokens.len() < 2 {
            return Err(BigramError::TooShortTokenStream);
        }

        for token in tokens {
            if token.0 >= vocab_size {
                return Err(BigramError::TokenOutOfRange {
                    token: token.0,
                    vocab_size,
                });
            }
        }

        let examples = tokens
            .windows(2)
            .map(|window| BigramExample::new(window[0], window[1]))
            .collect();

        Ok(Self {
            examples,
            vocab_size,
        })
    }

    /// Returns the number of training pairs.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Returns the configured vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Iterates over the next-token pairs.
    pub fn iter(&self) -> impl Iterator<Item = &BigramExample> {
        self.examples.iter()
    }
}

/// Summary metrics for one training epoch of the tiny next-token model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigramEpochMetrics {
    /// The 1-based epoch index.
    pub epoch: usize,
    /// The average cross-entropy loss across the dataset for that epoch.
    pub average_loss: Loss,
}

/// A tiny next-token model using embeddings and a linear output head.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyBigramModel {
    embedding: Vec<Vec<f64>>,
    lm_head: Vec<Vec<f64>>,
    bias: Vec<f64>,
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

fn cross_entropy_loss_unchecked(logits: &[f64], target: TokenId) -> f64 {
    let probabilities = softmax_unchecked(logits);
    -probabilities[target.0].ln()
}

fn cross_entropy_gradient_unchecked(logits: &[f64], target: TokenId) -> Vec<f64> {
    let mut probabilities = softmax_unchecked(logits);
    probabilities[target.0] -= 1.0;
    probabilities
}

fn argmax(values: &[f64]) -> usize {
    let mut best_index = 0;
    let mut best_value = values[0];

    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }

    best_index
}

impl TinyBigramModel {
    /// Creates a deterministic tiny next-token model.
    pub fn new(vocab_size: usize, d_model: usize) -> Result<Self, BigramError> {
        if vocab_size == 0 {
            return Err(BigramError::EmptyVocabulary);
        }

        if d_model == 0 {
            return Err(BigramError::ZeroModelWidth);
        }

        Ok(Self {
            embedding: init_matrix(vocab_size, d_model, 0.2, 1),
            lm_head: init_matrix(d_model, vocab_size, 0.2, 2),
            bias: vec![0.0; vocab_size],
        })
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.bias.len()
    }

    /// Returns the embedding width.
    pub fn d_model(&self) -> usize {
        self.embedding[0].len()
    }

    fn validate_token(&self, token: TokenId) -> Result<(), BigramError> {
        if token.0 >= self.vocab_size() {
            return Err(BigramError::TokenOutOfRange {
                token: token.0,
                vocab_size: self.vocab_size(),
            });
        }

        Ok(())
    }

    /// Computes the output logits for a single input token.
    pub fn logits_for_token(&self, input: TokenId) -> Result<Vec<f64>, BigramError> {
        self.validate_token(input)?;

        let hidden = &self.embedding[input.0];
        let vocab_size = self.vocab_size();
        let mut logits = self.bias.clone();

        for (token_id, logit) in logits.iter_mut().enumerate().take(vocab_size) {
            for (dim, hidden_component) in hidden.iter().enumerate() {
                *logit += hidden_component * self.lm_head[dim][token_id];
            }
        }

        Ok(logits)
    }

    /// Computes the output probabilities for a single input token.
    pub fn probabilities_for_token(&self, input: TokenId) -> Result<Vec<f64>, BigramError> {
        let logits = self.logits_for_token(input)?;
        Ok(softmax_unchecked(&logits))
    }

    /// Predicts the most likely next token.
    pub fn predict_next(&self, input: TokenId) -> Result<TokenId, BigramError> {
        let probabilities = self.probabilities_for_token(input)?;
        Ok(TokenId(argmax(&probabilities)))
    }

    /// Trains on one next-token example using plain SGD.
    pub fn train_one_example(
        &mut self,
        example: BigramExample,
        learning_rate: LearningRate,
    ) -> Result<Loss, BigramError> {
        self.validate_token(example.input)?;
        self.validate_token(example.target)?;

        let hidden = self.embedding[example.input.0].clone();
        let d_model = hidden.len();
        let vocab_size = self.vocab_size();

        let logits = self.logits_for_token(example.input)?;
        let loss = cross_entropy_loss_unchecked(&logits, example.target);
        let dlogits = cross_entropy_gradient_unchecked(&logits, example.target);

        // Compute all upstream gradients before mutating the head weights.
        let mut d_hidden = vec![0.0; d_model];
        for (dim, gradient_slot) in d_hidden.iter_mut().enumerate().take(d_model) {
            for (token_id, dlogit) in dlogits.iter().enumerate().take(vocab_size) {
                *gradient_slot += self.lm_head[dim][token_id] * dlogit;
            }
        }

        for (dim, hidden_component) in hidden.iter().enumerate().take(d_model) {
            for (token_id, dlogit) in dlogits.iter().enumerate().take(vocab_size) {
                let gradient = hidden_component * dlogit;
                self.lm_head[dim][token_id] -= learning_rate.0 * gradient;
            }
        }

        for (token_id, dlogit) in dlogits.iter().enumerate().take(vocab_size) {
            self.bias[token_id] -= learning_rate.0 * dlogit;
        }

        for (dim, gradient) in d_hidden.iter().enumerate().take(d_model) {
            self.embedding[example.input.0][dim] -= learning_rate.0 * gradient;
        }

        Ok(Loss(loss))
    }

    /// Computes the average cross-entropy loss across the dataset.
    pub fn average_loss(&self, dataset: &BigramDataset) -> Result<Loss, BigramError> {
        let mut total_loss = 0.0;

        for example in dataset.iter() {
            let logits = self.logits_for_token(example.input)?;
            total_loss += cross_entropy_loss_unchecked(&logits, example.target);
        }

        Ok(Loss(total_loss / dataset.len() as f64))
    }

    /// Trains for a fixed number of epochs and records the average loss.
    pub fn train_epochs(
        &mut self,
        dataset: &BigramDataset,
        learning_rate: LearningRate,
        epochs: usize,
    ) -> Result<Vec<BigramEpochMetrics>, BigramError> {
        let mut metrics = Vec::with_capacity(epochs);

        for epoch_index in 0..epochs {
            let mut total_loss = 0.0;

            for example in dataset.iter() {
                total_loss += f64::from(self.train_one_example(*example, learning_rate)?);
            }

            metrics.push(BigramEpochMetrics {
                epoch: epoch_index + 1,
                average_loss: Loss(total_loss / dataset.len() as f64),
            });
        }

        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle_tokens() -> Vec<TokenId> {
        vec![
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
        ]
    }

    #[test]
    fn dataset_from_token_stream_creates_adjacent_pairs() {
        let dataset = match BigramDataset::from_token_stream(&cycle_tokens(), 5) {
            Ok(dataset) => dataset,
            Err(error) => panic!("unexpected error: {error}"),
        };

        assert_eq!(dataset.len(), 11);

        let first = dataset.iter().next().copied();
        assert_eq!(first, Some(BigramExample::new(TokenId(1), TokenId(2))));
    }

    #[test]
    fn dataset_rejects_short_token_streams() {
        let result = BigramDataset::from_token_stream(&[TokenId(1)], 5);
        assert_eq!(result, Err(BigramError::TooShortTokenStream));
    }

    #[test]
    fn training_reduces_loss_and_learns_the_cycle() {
        let dataset = match BigramDataset::from_token_stream(&cycle_tokens(), 5) {
            Ok(dataset) => dataset,
            Err(error) => panic!("unexpected error: {error}"),
        };
        let mut model = match TinyBigramModel::new(5, 8) {
            Ok(model) => model,
            Err(error) => panic!("unexpected error: {error}"),
        };

        let before = match model.average_loss(&dataset) {
            Ok(loss) => f64::from(loss),
            Err(error) => panic!("unexpected error: {error}"),
        };
        let metrics = match model.train_epochs(&dataset, LearningRate(0.1), 300) {
            Ok(metrics) => metrics,
            Err(error) => panic!("unexpected error: {error}"),
        };
        let after = match model.average_loss(&dataset) {
            Ok(loss) => f64::from(loss),
            Err(error) => panic!("unexpected error: {error}"),
        };

        assert!(after < before, "before={before}, after={after}");
        assert_eq!(metrics.len(), 300);

        let expectations = [
            (TokenId(1), TokenId(2)),
            (TokenId(2), TokenId(3)),
            (TokenId(3), TokenId(4)),
            (TokenId(4), TokenId(1)),
        ];

        for (input, expected) in expectations {
            let predicted = match model.predict_next(input) {
                Ok(predicted) => predicted,
                Err(error) => panic!("unexpected error: {error}"),
            };
            assert_eq!(predicted, expected);
        }
    }
}
