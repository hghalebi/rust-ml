//! Neural-network building blocks for the tiny Transformer crate.

use crate::math::{Matrix, MatrixMN, Vector, VectorN};

/// A dynamic linear layer computing `y = Wx + b`.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Matrix,
    bias: Vector,
}

impl Linear {
    /// Creates a linear layer with weight shape `[out_dim, in_dim]`.
    pub fn new(weight: Matrix, bias: Vector) -> Self {
        assert_eq!(weight.rows(), bias.len(), "linear: bias/output mismatch");
        Self { weight, bias }
    }

    /// Applies the linear layer to one vector.
    pub fn forward(&self, x: &Vector) -> Vector {
        self.weight.mul_vec(x).add(&self.bias)
    }
}

/// Applies ReLU element-wise to a vector.
pub fn relu(v: &Vector) -> Vector {
    v.map(|x| x.max(0.0))
}

/// Computes a numerically stable softmax over a slice.
pub fn softmax(xs: &[f32]) -> Vec<f32> {
    assert!(!xs.is_empty(), "softmax: input cannot be empty");
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Applies a positive feature map used by simple linear-attention variants.
pub fn phi(v: &Vector) -> Vector {
    let eps = 1e-6;
    v.map(|x| x.max(0.0) + eps)
}

/// Computes a simple layer normalization without learned affine terms.
pub fn layer_norm(x: &Vector) -> Vector {
    let n = x.len() as f32;
    let mean: f32 = x.as_slice().iter().sum::<f32>() / n;
    let var: f32 = x
        .as_slice()
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;

    let eps = 1e-5;
    Vector::new(
        x.as_slice()
            .iter()
            .map(|v| (v - mean) / (var + eps).sqrt())
            .collect(),
    )
}

/// A two-layer feed-forward network applied independently to each token.
#[derive(Debug, Clone)]
pub struct FeedForward {
    l1: Linear,
    l2: Linear,
}

impl FeedForward {
    /// Creates a feed-forward network from two linear layers.
    pub fn new(l1: Linear, l2: Linear) -> Self {
        Self { l1, l2 }
    }

    /// Applies the feed-forward network to one vector.
    pub fn forward(&self, x: &Vector) -> Vector {
        let h = relu(&self.l1.forward(x));
        self.l2.forward(&h)
    }
}

/// A compile-time-sized linear layer for shape-safe teaching examples.
#[derive(Debug, Clone, Copy)]
pub struct StaticLinear<const IN: usize, const OUT: usize> {
    /// Weight matrix with shape `[OUT, IN]`.
    pub weight: MatrixMN<OUT, IN>,
    /// Bias vector with shape `[OUT]`.
    pub bias: VectorN<OUT>,
}

impl<const IN: usize, const OUT: usize> StaticLinear<IN, OUT> {
    /// Applies the compile-time-sized linear layer.
    pub fn forward(&self, x: &VectorN<IN>) -> VectorN<OUT> {
        self.weight.mul_vec(x).add(&self.bias)
    }
}
