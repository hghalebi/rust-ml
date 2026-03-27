//! Low-level dense math types used by the teaching crate.

use crate::error::ModelError;

/// A dense one-dimensional vector of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector(Vec<f32>);

impl DenseVector {
    /// Creates a vector from owned data.
    pub fn new(data: Vec<f32>) -> Result<Self, ModelError> {
        if data.is_empty() {
            return Err(ModelError::EmptyInput {
                operation: "DenseVector::new",
                details: "vector cannot be empty",
            });
        }

        Ok(Self(data))
    }

    /// Creates a zero vector.
    pub fn zeros(len: usize) -> Result<Self, ModelError> {
        if len == 0 {
            return Err(ModelError::EmptyInput {
                operation: "DenseVector::zeros",
                details: "length must be greater than zero",
            });
        }

        Ok(Self(vec![0.0; len]))
    }

    /// Creates a vector filled with ones.
    pub fn ones(len: usize) -> Result<Self, ModelError> {
        if len == 0 {
            return Err(ModelError::EmptyInput {
                operation: "DenseVector::ones",
                details: "length must be greater than zero",
            });
        }

        Ok(Self(vec![1.0; len]))
    }

    /// Returns the vector length.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the vector as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Reads a single element.
    pub fn get(&self, index: usize) -> f32 {
        self.0[index]
    }

    /// Overwrites a single element.
    pub fn set(&mut self, index: usize, value: f32) {
        self.0[index] = value;
    }

    /// Computes the dot product between two vectors.
    pub fn dot(&self, other: &DenseVector) -> Result<f32, ModelError> {
        if self.len() != other.len() {
            return Err(ModelError::DimensionMismatch {
                operation: "DenseVector::dot",
                left_label: "left vector",
                left_shape: vec![self.len()],
                right_label: "right vector",
                right_shape: vec![other.len()],
                hint: "dot product requires equal vector lengths",
            });
        }

        Ok(self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum())
    }

    /// Adds two vectors elementwise.
    pub fn add(&self, other: &DenseVector) -> Result<DenseVector, ModelError> {
        if self.len() != other.len() {
            return Err(ModelError::DimensionMismatch {
                operation: "DenseVector::add",
                left_label: "left vector",
                left_shape: vec![self.len()],
                right_label: "right vector",
                right_shape: vec![other.len()],
                hint: "vector addition requires equal lengths",
            });
        }

        DenseVector::new(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a + b)
                .collect(),
        )
    }

    /// Scales a vector by a scalar.
    pub fn scale(&self, scalar: f32) -> DenseVector {
        DenseVector(self.0.iter().map(|value| value * scalar).collect())
    }

    /// Applies a scalar function to every element.
    pub fn map<F>(&self, f: F) -> DenseVector
    where
        F: Fn(f32) -> f32,
    {
        DenseVector(self.0.iter().copied().map(f).collect())
    }

    /// Computes the mean of the vector.
    pub fn mean(&self) -> f32 {
        self.0.iter().sum::<f32>() / self.len() as f32
    }

    /// Computes the population variance of the vector.
    pub fn variance(&self) -> f32 {
        let mean = self.mean();

        self.0
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / self.len() as f32
    }
}

/// A dense row-major matrix of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DenseMatrix {
    /// Creates a matrix from raw row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, ModelError> {
        if rows == 0 || cols == 0 {
            return Err(ModelError::EmptyInput {
                operation: "DenseMatrix::new",
                details: "rows and cols must be greater than zero",
            });
        }

        if rows * cols != data.len() {
            return Err(ModelError::InvalidMatrixData {
                operation: "DenseMatrix::new",
                rows,
                cols,
                data_len: data.len(),
            });
        }

        Ok(Self { rows, cols, data })
    }

    /// Creates a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, ModelError> {
        Self::new(rows, cols, vec![0.0; rows * cols])
    }

    /// Creates a matrix from nested row vectors.
    pub fn from_rows(rows_data: Vec<Vec<f32>>) -> Result<Self, ModelError> {
        if rows_data.is_empty() {
            return Err(ModelError::EmptyInput {
                operation: "DenseMatrix::from_rows",
                details: "matrix cannot be empty",
            });
        }

        let rows = rows_data.len();
        let cols = rows_data[0].len();

        if cols == 0 {
            return Err(ModelError::EmptyInput {
                operation: "DenseMatrix::from_rows",
                details: "matrix rows cannot be empty",
            });
        }

        for row in &rows_data {
            if row.len() != cols {
                return Err(ModelError::InvalidMatrixData {
                    operation: "DenseMatrix::from_rows",
                    rows,
                    cols,
                    data_len: rows_data.iter().map(Vec::len).sum(),
                });
            }
        }

        Ok(Self {
            rows,
            cols,
            data: rows_data.into_iter().flatten().collect(),
        })
    }

    /// Returns the row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the column count.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Reads one matrix element.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Overwrites one matrix element.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Multiplies the matrix by a vector.
    pub fn mul_vec(&self, vector: &DenseVector) -> Result<DenseVector, ModelError> {
        if self.cols != vector.len() {
            return Err(ModelError::DimensionMismatch {
                operation: "DenseMatrix::mul_vec",
                left_label: "matrix",
                left_shape: vec![self.rows, self.cols],
                right_label: "vector",
                right_shape: vec![vector.len()],
                hint: "matrix columns must equal vector length",
            });
        }

        let mut out = vec![0.0; self.rows];

        for (row, slot) in out.iter_mut().enumerate().take(self.rows) {
            let mut sum = 0.0;

            for col in 0..self.cols {
                sum += self.get(row, col) * vector.get(col);
            }

            *slot = sum;
        }

        DenseVector::new(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMatrix, DenseVector};
    use crate::error::ModelError;

    #[test]
    fn dense_vector_rejects_empty_input() {
        let error = DenseVector::new(vec![]).expect_err("empty vector should fail");
        assert!(matches!(error, ModelError::EmptyInput { .. }));
    }

    #[test]
    fn dense_vector_dot_product_matches_manual_computation() -> Result<(), ModelError> {
        let left = DenseVector::new(vec![1.0, 2.0, 3.0])?;
        let right = DenseVector::new(vec![4.0, 5.0, 6.0])?;

        assert!((left.dot(&right)? - 32.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn dense_vector_dot_product_reports_shape_mismatch() -> Result<(), ModelError> {
        let left = DenseVector::new(vec![1.0, 2.0])?;
        let right = DenseVector::new(vec![1.0, 2.0, 3.0])?;

        let error = left
            .dot(&right)
            .expect_err("mismatched dot product should fail");
        assert!(matches!(error, ModelError::DimensionMismatch { .. }));
        Ok(())
    }

    #[test]
    fn dense_vector_add_reports_shape_mismatch() -> Result<(), ModelError> {
        let left = DenseVector::new(vec![1.0, 2.0])?;
        let right = DenseVector::new(vec![1.0])?;

        let error = left.add(&right).expect_err("mismatched add should fail");
        assert!(matches!(error, ModelError::DimensionMismatch { .. }));
        Ok(())
    }

    #[test]
    fn dense_vector_mean_and_variance_match_expected_values() -> Result<(), ModelError> {
        let vector = DenseVector::new(vec![1.0, 2.0, 3.0])?;

        assert!((vector.mean() - 2.0).abs() < 1e-6);
        assert!((vector.variance() - (2.0 / 3.0)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn dense_matrix_rejects_invalid_data_length() {
        let error =
            DenseMatrix::new(2, 2, vec![1.0, 2.0, 3.0]).expect_err("invalid matrix should fail");
        assert!(matches!(error, ModelError::InvalidMatrixData { .. }));
    }

    #[test]
    fn dense_matrix_from_rows_rejects_ragged_rows() {
        let error = DenseMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0]])
            .expect_err("ragged rows should fail");
        assert!(matches!(error, ModelError::InvalidMatrixData { .. }));
    }

    #[test]
    fn dense_matrix_mul_vec_matches_manual_computation() -> Result<(), ModelError> {
        let matrix = DenseMatrix::from_rows(vec![vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0]])?;
        let vector = DenseVector::new(vec![1.0, 2.0, 3.0])?;

        assert_eq!(matrix.mul_vec(&vector)?.as_slice(), &[7.0, 11.0]);
        Ok(())
    }

    #[test]
    fn dense_matrix_mul_vec_reports_shape_mismatch() -> Result<(), ModelError> {
        let matrix = DenseMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let vector = DenseVector::new(vec![1.0, 2.0, 3.0])?;

        let error = matrix
            .mul_vec(&vector)
            .expect_err("mismatched matrix-vector multiply should fail");
        assert!(matches!(error, ModelError::DimensionMismatch { .. }));
        Ok(())
    }
}
