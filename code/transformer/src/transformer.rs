//! Sequence and attention layers for the tiny Transformer crate.

use crate::math::{Matrix, Vector};
use crate::nn::{FeedForward, Linear, layer_norm, phi, softmax};
use crate::types::{ColumnCount, ColumnIndex, Dimension, RowCount, RowIndex, Scalar, TokenCount};

/// A sequence of token vectors that all share the same model dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    tokens: Vec<Vector>,
}

impl Sequence {
    /// Creates a new non-empty sequence with a consistent token dimension.
    pub fn new(tokens: Vec<Vector>) -> Self {
        assert!(!tokens.is_empty(), "sequence cannot be empty");
        let dimension = tokens[0].len();
        assert!(
            tokens.iter().all(|token| token.len() == dimension),
            "all tokens must have same dimension"
        );
        Self { tokens }
    }

    /// Returns the number of tokens in the sequence.
    pub fn len(&self) -> TokenCount {
        TokenCount::new(self.tokens.len())
    }

    /// Returns true when the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the shared token dimension.
    pub fn d_model(&self) -> Dimension {
        self.tokens[0].len()
    }

    /// Borrows the token vectors.
    pub fn tokens(&self) -> &[Vector] {
        &self.tokens
    }
}

/// Standard self-attention with dynamic vectors.
#[derive(Debug, Clone)]
pub struct SelfAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl SelfAttention {
    /// Creates a self-attention layer from three projections.
    pub fn new(w_q: Linear, w_k: Linear, w_v: Linear) -> Self {
        Self { w_q, w_k, w_v }
    }

    /// Applies self-attention to a full sequence.
    pub fn forward(&self, seq: &Sequence) -> Sequence {
        let queries: Vec<Vector> = seq.tokens().iter().map(|x| self.w_q.forward(x)).collect();
        let keys: Vec<Vector> = seq.tokens().iter().map(|x| self.w_k.forward(x)).collect();
        let values: Vec<Vector> = seq.tokens().iter().map(|x| self.w_v.forward(x)).collect();

        let scale = Scalar::from(queries[0].len().get() as f32).sqrt();
        let mut outputs = Vec::with_capacity(seq.len().get());

        for query in &queries {
            let scores: Vec<Scalar> = keys.iter().map(|key| query.dot(key) / scale).collect();
            let weights = softmax(&scores);

            let mut output = vec![Scalar::ZERO; values[0].len().get()];
            for (weight, value_vector) in weights.iter().zip(values.iter()) {
                for (feature_index, value) in value_vector.as_slice().iter().enumerate() {
                    output[feature_index] += *weight * *value;
                }
            }

            outputs.push(Vector::new(output));
        }

        Sequence::new(outputs)
    }
}

/// A simplified linear-attention layer using positive feature maps.
#[derive(Debug, Clone)]
pub struct LinearAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl LinearAttention {
    /// Creates a linear-attention layer from three projections.
    pub fn new(w_q: Linear, w_k: Linear, w_v: Linear) -> Self {
        Self { w_q, w_k, w_v }
    }

    /// Applies linear attention to a full sequence.
    pub fn forward(&self, seq: &Sequence) -> Sequence {
        let queries: Vec<Vector> = seq
            .tokens()
            .iter()
            .map(|x| phi(&self.w_q.forward(x)))
            .collect();
        let keys: Vec<Vector> = seq
            .tokens()
            .iter()
            .map(|x| phi(&self.w_k.forward(x)))
            .collect();
        let values: Vec<Vector> = seq.tokens().iter().map(|x| self.w_v.forward(x)).collect();

        let key_dimension = Dimension::new(queries[0].len().get());
        let value_dimension = Dimension::new(values[0].len().get());

        let mut summary = Matrix::zeros(
            RowCount::new(key_dimension.get()),
            ColumnCount::new(value_dimension.get()),
        );
        let mut normalizer = vec![Scalar::ZERO; key_dimension.get()];

        for (key, value) in keys.iter().zip(values.iter()) {
            for (feature_index, normalizer_slot) in normalizer.iter_mut().enumerate() {
                *normalizer_slot += key.as_slice()[feature_index];
                for value_index in 0..value_dimension.get() {
                    let row = RowIndex::new(feature_index);
                    let column = ColumnIndex::new(value_index);
                    let updated = summary.get(row, column)
                        + key.as_slice()[feature_index] * value.as_slice()[value_index];
                    summary.set(row, column, updated);
                }
            }
        }

        let mut outputs = Vec::with_capacity(seq.len().get());

        for query in &queries {
            let mut numerator = vec![Scalar::ZERO; value_dimension.get()];
            for (value_index, slot) in numerator.iter_mut().enumerate() {
                let mut sum = Scalar::ZERO;
                for feature_index in 0..key_dimension.get() {
                    sum += query.as_slice()[feature_index]
                        * summary.get(RowIndex::new(feature_index), ColumnIndex::new(value_index));
                }
                *slot = sum;
            }

            let denominator = query
                .as_slice()
                .iter()
                .zip(normalizer.iter())
                .map(|(a, b)| *a * *b)
                .sum::<Scalar>()
                .max(Scalar::from(1e-6));

            let output = numerator
                .into_iter()
                .map(|value| value / denominator)
                .collect();
            outputs.push(Vector::new(output));
        }

        Sequence::new(outputs)
    }
}

/// A minimal Transformer block using linear attention and a feed-forward network.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention: LinearAttention,
    ff: FeedForward,
}

impl TransformerBlock {
    /// Creates a Transformer block from attention and feed-forward sublayers.
    pub fn new(attention: LinearAttention, ff: FeedForward) -> Self {
        Self { attention, ff }
    }

    /// Applies attention, residual connections, layer norm, and feed-forward logic.
    pub fn forward(&self, seq: &Sequence) -> Sequence {
        let attention_out = self.attention.forward(seq);

        let after_attention: Vec<Vector> = seq
            .tokens()
            .iter()
            .zip(attention_out.tokens().iter())
            .map(|(x, attended)| layer_norm(&x.add(attended)))
            .collect();

        let contextualized = Sequence::new(after_attention);

        let after_feed_forward: Vec<Vector> = contextualized
            .tokens()
            .iter()
            .map(|x| {
                let feed_forward = self.ff.forward(x);
                layer_norm(&x.add(&feed_forward))
            })
            .collect();

        Sequence::new(after_feed_forward)
    }
}
