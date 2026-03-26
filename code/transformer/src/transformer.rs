//! Sequence and attention layers for the tiny Transformer crate.

use crate::math::{Matrix, Vector};
use crate::nn::{FeedForward, Linear, layer_norm, phi, softmax};

/// A sequence of token vectors that all share the same model dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    tokens: Vec<Vector>,
}

impl Sequence {
    /// Creates a new non-empty sequence with a consistent token dimension.
    pub fn new(tokens: Vec<Vector>) -> Self {
        assert!(!tokens.is_empty(), "sequence cannot be empty");
        let d = tokens[0].len();
        assert!(
            tokens.iter().all(|t| t.len() == d),
            "all tokens must have same dimension"
        );
        Self { tokens }
    }

    /// Returns the number of tokens in the sequence.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true when the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the shared token dimension.
    pub fn d_model(&self) -> usize {
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
        let qs: Vec<Vector> = seq.tokens().iter().map(|x| self.w_q.forward(x)).collect();
        let ks: Vec<Vector> = seq.tokens().iter().map(|x| self.w_k.forward(x)).collect();
        let vs: Vec<Vector> = seq.tokens().iter().map(|x| self.w_v.forward(x)).collect();

        let d_k = qs[0].len() as f32;
        let scale = d_k.sqrt();

        let mut outputs = Vec::with_capacity(seq.len());

        for q in &qs {
            let scores: Vec<f32> = ks.iter().map(|k| q.dot(k) / scale).collect();
            let weights = softmax(&scores);

            let mut out = vec![0.0; vs[0].len()];
            for (weight, v) in weights.iter().zip(vs.iter()) {
                for (j, value) in v.as_slice().iter().enumerate() {
                    out[j] += weight * value;
                }
            }

            outputs.push(Vector::new(out));
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
        let qs: Vec<Vector> = seq
            .tokens()
            .iter()
            .map(|x| phi(&self.w_q.forward(x)))
            .collect();
        let ks: Vec<Vector> = seq
            .tokens()
            .iter()
            .map(|x| phi(&self.w_k.forward(x)))
            .collect();
        let vs: Vec<Vector> = seq.tokens().iter().map(|x| self.w_v.forward(x)).collect();

        let d_k = qs[0].len();
        let d_v = vs[0].len();

        let mut s = Matrix::zeros(d_k, d_v);
        let mut z = vec![0.0; d_k];

        for (k, v) in ks.iter().zip(vs.iter()) {
            for (i, z_slot) in z.iter_mut().enumerate() {
                *z_slot += k.as_slice()[i];
                for j in 0..d_v {
                    let current = s.get(i, j);
                    s.set(i, j, current + k.as_slice()[i] * v.as_slice()[j]);
                }
            }
        }

        let mut outputs = Vec::with_capacity(seq.len());

        for q in &qs {
            let mut numerator = vec![0.0; d_v];
            for (j, slot) in numerator.iter_mut().enumerate() {
                let mut sum = 0.0;
                for i in 0..d_k {
                    sum += q.as_slice()[i] * s.get(i, j);
                }
                *slot = sum;
            }

            let denominator: f32 = q
                .as_slice()
                .iter()
                .zip(z.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                .max(1e-6);

            let out = numerator.into_iter().map(|x| x / denominator).collect();
            outputs.push(Vector::new(out));
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
        let attn_out = self.attention.forward(seq);

        let after_attn: Vec<Vector> = seq
            .tokens()
            .iter()
            .zip(attn_out.tokens().iter())
            .map(|(x, a)| layer_norm(&x.add(a)))
            .collect();

        let seq2 = Sequence::new(after_attn);

        let after_ff: Vec<Vector> = seq2
            .tokens()
            .iter()
            .map(|x| {
                let ff = self.ff.forward(x);
                layer_norm(&x.add(&ff))
            })
            .collect();

        Sequence::new(after_ff)
    }
}
