//! Tiny typed Transformer building blocks for the rust-ml lessons.

pub mod math;
pub mod nn;
pub mod transformer;
pub mod types;

pub use math::{Matrix, MatrixMN, Vector, VectorN};
pub use nn::{FeedForward, Linear, StaticLinear, layer_norm, phi, relu, softmax};
pub use transformer::{LinearAttention, SelfAttention, Sequence, TransformerBlock};
pub use types::{ColumnCount, ColumnIndex, Dimension, RowCount, RowIndex, Scalar, TokenCount};

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f32) -> Scalar {
        Scalar::from(value)
    }

    fn vector(values: &[f32]) -> Vector {
        Vector::from_f32s(values.to_vec())
    }

    fn approx_eq(actual: Scalar, expected: Scalar) {
        let diff = (actual - expected).abs();
        assert!(
            diff < scalar(1e-4),
            "expected {actual} ~= {expected}, diff={diff}"
        );
    }

    fn approx_vec_eq(actual: &Vector, expected: &[f32]) {
        assert_eq!(
            actual.len(),
            Dimension::new(expected.len()),
            "vector length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );

        for (actual_value, expected_value) in actual.as_slice().iter().zip(expected.iter()) {
            approx_eq(*actual_value, scalar(*expected_value));
        }
    }

    fn identity_matrix(size: Dimension) -> Matrix {
        let mut matrix = Matrix::zeros(RowCount::new(size.get()), ColumnCount::new(size.get()));
        for index in 0..size.get() {
            matrix.set(RowIndex::new(index), ColumnIndex::new(index), Scalar::ONE);
        }
        matrix
    }

    fn identity_linear(size: Dimension) -> Linear {
        Linear::new(
            identity_matrix(size),
            Vector::new(vec![Scalar::ZERO; size.get()]),
        )
    }

    fn sample_sequence() -> Sequence {
        Sequence::new(vec![
            vector(&[1.0, 0.0, 1.0, 0.0]),
            vector(&[0.0, 1.0, 1.0, 0.0]),
            vector(&[0.0, 1.0, 0.0, 1.0]),
        ])
    }

    #[test]
    fn vector_dot_and_add_work() {
        let a = vector(&[1.0, 2.0, 3.0]);
        let b = vector(&[4.0, 5.0, 6.0]);

        approx_eq(a.dot(&b), scalar(32.0));
        approx_vec_eq(&a.add(&b), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn matrix_mul_vec_works() {
        let matrix = Matrix::from_f32s(
            RowCount::new(2),
            ColumnCount::new(2),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let vector = vector(&[5.0, 6.0]);
        let out = matrix.mul_vec(&vector);

        approx_vec_eq(&out, &[17.0, 39.0]);
    }

    #[test]
    fn matrix_transpose_round_trips_shape() {
        let matrix = Matrix::from_f32s(
            RowCount::new(2),
            ColumnCount::new(3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let transposed = matrix.transpose();

        assert_eq!(transposed.rows(), RowCount::new(3));
        assert_eq!(transposed.cols(), ColumnCount::new(2));
        approx_eq(
            transposed.get(RowIndex::new(0), ColumnIndex::new(1)),
            scalar(4.0),
        );
        approx_eq(
            transposed.get(RowIndex::new(2), ColumnIndex::new(1)),
            scalar(6.0),
        );
    }

    #[test]
    fn softmax_normalizes() {
        let weights = softmax(&[scalar(2.0), scalar(1.0), scalar(0.0)]);
        let sum: Scalar = weights.iter().copied().sum();
        approx_eq(sum, Scalar::ONE);
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }

    #[test]
    fn softmax_stays_finite_for_large_magnitudes() {
        let weights = softmax(&[scalar(1_000.0), scalar(0.0), scalar(-1_000.0)]);

        let sum: Scalar = weights.iter().copied().sum();
        approx_eq(sum, Scalar::ONE);
        for weight in &weights {
            assert!(weight.is_finite());
            assert!(*weight >= Scalar::ZERO);
        }
    }

    #[test]
    fn phi_produces_strictly_positive_values() {
        let mapped = phi(&vector(&[-3.0, 0.0, 2.5]));

        for value in mapped.as_slice() {
            assert!(*value > Scalar::ZERO);
        }
    }

    #[test]
    fn layer_norm_recenters_constant_vectors() {
        let normalized = layer_norm(&vector(&[5.0, 5.0, 5.0, 5.0]));

        approx_vec_eq(&normalized, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn layer_norm_produces_zero_mean_output() {
        let normalized = layer_norm(&vector(&[1.0, 2.0, 3.0, 4.0]));
        let mean: Scalar = normalized.as_slice().iter().copied().sum::<Scalar>()
            / scalar(normalized.len().get() as f32);

        approx_eq(mean, Scalar::ZERO);
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
        let seq = Sequence::new(vec![vector(&[1.0, -2.0, 3.0])]);
        let attention = SelfAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );

        let out = attention.forward(&seq);

        approx_vec_eq(&out.tokens()[0], &[1.0, -2.0, 3.0]);
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
        let seq = Sequence::new(vec![vector(&[2.0, -1.0, 0.5])]);
        let attention = LinearAttention::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );

        let out = attention.forward(&seq);

        approx_vec_eq(&out.tokens()[0], &[2.0, -1.0, 0.5]);
    }

    #[test]
    fn linear_attention_is_permutation_equivariant_without_positions() {
        let original = Sequence::new(vec![
            vector(&[1.0, 0.0, 2.0]),
            vector(&[0.5, 1.0, 0.0]),
            vector(&[2.0, 1.5, 1.0]),
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
            original_out.tokens()[2]
                .as_slice()
                .iter()
                .map(|value| value.into_inner())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        approx_vec_eq(
            &permuted_out.tokens()[1],
            original_out.tokens()[0]
                .as_slice()
                .iter()
                .map(|value| value.into_inner())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        approx_vec_eq(
            &permuted_out.tokens()[2],
            original_out.tokens()[1]
                .as_slice()
                .iter()
                .map(|value| value.into_inner())
                .collect::<Vec<_>>()
                .as_slice(),
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
        let feed_forward = FeedForward::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );
        let block = TransformerBlock::new(attention, feed_forward);

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
        let feed_forward = FeedForward::new(
            identity_linear(seq.d_model()),
            identity_linear(seq.d_model()),
        );
        let block = TransformerBlock::new(attention, feed_forward);

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
                data: [
                    [scalar(1.0), scalar(0.0)],
                    [scalar(0.0), scalar(1.0)],
                    [scalar(1.0), scalar(1.0)],
                ],
            },
            bias: VectorN {
                data: [scalar(0.5), scalar(0.5), scalar(0.5)],
            },
        };
        let x = VectorN {
            data: [scalar(2.0), scalar(3.0)],
        };
        let out = layer.forward(&x);

        approx_eq(out.data[0], scalar(2.5));
        approx_eq(out.data[1], scalar(3.5));
        approx_eq(out.data[2], scalar(5.5));
    }

    #[test]
    #[should_panic(expected = "dot: dimension mismatch")]
    fn vector_dot_panics_on_dimension_mismatch() {
        let a = vector(&[1.0, 2.0]);
        let b = vector(&[1.0, 2.0, 3.0]);

        let _ = a.dot(&b);
    }

    #[test]
    #[should_panic(expected = "mul_vec: dimension mismatch")]
    fn matrix_mul_vec_panics_on_dimension_mismatch() {
        let matrix = Matrix::from_f32s(
            RowCount::new(2),
            ColumnCount::new(2),
            vec![1.0, 0.0, 0.0, 1.0],
        );
        let vector = vector(&[1.0, 2.0, 3.0]);

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
        let _ = Sequence::new(vec![vector(&[1.0, 2.0]), vector(&[1.0, 2.0, 3.0])]);
    }
}
