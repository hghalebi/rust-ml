//! Tiny typed Transformer building blocks for the rust-ml lessons.

pub mod math;
pub mod nn;
pub mod transformer;

pub use math::{Matrix, MatrixMN, Vector, VectorN};
pub use nn::{FeedForward, Linear, StaticLinear, layer_norm, phi, relu, softmax};
pub use transformer::{LinearAttention, SelfAttention, Sequence, TransformerBlock};

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "expected {a} ~= {b}, diff={diff}");
    }

    fn approx_vec_eq(actual: &Vector, expected: &[f32]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "vector length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );

        for (actual_value, expected_value) in actual.as_slice().iter().zip(expected.iter()) {
            approx_eq(*actual_value, *expected_value);
        }
    }

    fn identity_matrix(n: usize) -> Matrix {
        let mut matrix = Matrix::zeros(n, n);
        for i in 0..n {
            matrix.set(i, i, 1.0);
        }
        matrix
    }

    fn identity_linear(n: usize) -> Linear {
        Linear::new(identity_matrix(n), Vector::new(vec![0.0; n]))
    }

    fn sample_sequence() -> Sequence {
        Sequence::new(vec![
            Vector::new(vec![1.0, 0.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 0.0, 1.0]),
        ])
    }

    #[test]
    fn vector_dot_and_add_work() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);

        approx_eq(a.dot(&b), 32.0);
        assert_eq!(a.add(&b).as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn matrix_mul_vec_works() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let vector = Vector::new(vec![5.0, 6.0]);
        let out = matrix.mul_vec(&vector);

        assert_eq!(out.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn matrix_transpose_round_trips_shape() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose();

        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        approx_eq(transposed.get(0, 1), 4.0);
        approx_eq(transposed.get(2, 1), 6.0);
    }

    #[test]
    fn softmax_normalizes() {
        let weights = softmax(&[2.0, 1.0, 0.0]);
        let sum: f32 = weights.iter().sum();
        approx_eq(sum, 1.0);
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }

    #[test]
    fn softmax_stays_finite_for_large_magnitudes() {
        let weights = softmax(&[1_000.0, 0.0, -1_000.0]);

        let sum: f32 = weights.iter().sum();
        approx_eq(sum, 1.0);
        for weight in &weights {
            assert!(weight.is_finite());
            assert!(*weight >= 0.0);
        }
    }

    #[test]
    fn phi_produces_strictly_positive_values() {
        let mapped = phi(&Vector::new(vec![-3.0, 0.0, 2.5]));

        for value in mapped.as_slice() {
            assert!(*value > 0.0);
        }
    }

    #[test]
    fn layer_norm_recenters_constant_vectors() {
        let normalized = layer_norm(&Vector::new(vec![5.0, 5.0, 5.0, 5.0]));

        approx_vec_eq(&normalized, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn layer_norm_produces_zero_mean_output() {
        let normalized = layer_norm(&Vector::new(vec![1.0, 2.0, 3.0, 4.0]));
        let mean: f32 = normalized.as_slice().iter().sum::<f32>() / normalized.len() as f32;

        approx_eq(mean, 0.0);
    }

    #[test]
    fn self_attention_preserves_shape() {
        let seq = sample_sequence();
        let attention = SelfAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );

        let out = attention.forward(&seq);

        assert_eq!(out.len(), seq.len());
        assert_eq!(out.d_model(), seq.d_model());
    }

    #[test]
    fn self_attention_with_one_token_returns_that_token_value() {
        let seq = Sequence::new(vec![Vector::new(vec![1.0, -2.0, 3.0])]);
        let attention = SelfAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );

        let out = attention.forward(&seq);

        approx_vec_eq(&out.tokens()[0], seq.tokens()[0].as_slice());
    }

    #[test]
    fn linear_attention_preserves_shape_and_finite_values() {
        let seq = sample_sequence();
        let attention = LinearAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );

        let out = attention.forward(&seq);

        assert_eq!(out.len(), seq.len());
        assert_eq!(out.d_model(), seq.d_model());
        for token in out.tokens() {
            for value in token.as_slice() {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn linear_attention_with_one_token_returns_that_token_value() {
        let seq = Sequence::new(vec![Vector::new(vec![2.0, -1.0, 0.5])]);
        let attention = LinearAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );

        let out = attention.forward(&seq);

        approx_vec_eq(&out.tokens()[0], seq.tokens()[0].as_slice());
    }

    #[test]
    fn linear_attention_is_permutation_equivariant_without_positions() {
        let original = Sequence::new(vec![
            Vector::new(vec![1.0, 0.0, 2.0]),
            Vector::new(vec![0.5, 1.0, 0.0]),
            Vector::new(vec![2.0, 1.5, 1.0]),
        ]);
        let permuted = Sequence::new(vec![
            original.tokens()[2].clone(),
            original.tokens()[0].clone(),
            original.tokens()[1].clone(),
        ]);
        let attention = LinearAttention::new(
            identity_linear(original.d_model()),
            identity_linear(original.d_model()),
            identity_linear(original.d_model()),
        );

        let original_out = attention.forward(&original);
        let permuted_out = attention.forward(&permuted);

        approx_vec_eq(
            &permuted_out.tokens()[0],
            original_out.tokens()[2].as_slice(),
        );
        approx_vec_eq(
            &permuted_out.tokens()[1],
            original_out.tokens()[0].as_slice(),
        );
        approx_vec_eq(
            &permuted_out.tokens()[2],
            original_out.tokens()[1].as_slice(),
        );
    }

    #[test]
    fn transformer_block_preserves_shape() {
        let seq = sample_sequence();
        let attention = LinearAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );
        let ff = FeedForward::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );
        let block = TransformerBlock::new(attention, ff);

        let out = block.forward(&seq);

        assert_eq!(out.len(), seq.len());
        assert_eq!(out.d_model(), seq.d_model());
    }

    #[test]
    fn transformer_block_outputs_stay_finite() {
        let seq = sample_sequence();
        let attention = LinearAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );
        let ff = FeedForward::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );
        let block = TransformerBlock::new(attention, ff);

        let out = block.forward(&seq);

        for token in out.tokens() {
            for value in token.as_slice() {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn static_linear_respects_dimensions() {
        let layer = StaticLinear {
            weight: MatrixMN {
                data: [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            },
            bias: VectorN {
                data: [0.5, 0.5, 0.5],
            },
        };
        let x = VectorN { data: [2.0, 3.0] };
        let out = layer.forward(&x);

        approx_eq(out.data[0], 2.5);
        approx_eq(out.data[1], 3.5);
        approx_eq(out.data[2], 5.5);
    }

    #[test]
    #[should_panic(expected = "dot: dimension mismatch")]
    fn vector_dot_panics_on_dimension_mismatch() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);

        let _ = a.dot(&b);
    }

    #[test]
    #[should_panic(expected = "mul_vec: dimension mismatch")]
    fn matrix_mul_vec_panics_on_dimension_mismatch() {
        let matrix = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        let _ = matrix.mul_vec(&vector);
    }

    #[test]
    #[should_panic(expected = "sequence cannot be empty")]
    fn sequence_new_panics_on_empty_input() {
        let _ = Sequence::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "all tokens must have same dimension")]
    fn sequence_new_panics_on_inconsistent_dimensions() {
        let _ = Sequence::new(vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![1.0, 2.0, 3.0]),
        ]);
    }
}
