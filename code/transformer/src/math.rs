//! Low-level dense math types used by the teaching crate.
//!
//! Raw learner literals are converted into [`ModelScalar`] at the boundary.
//! Public constructors and operations then use semantic wrappers such as
//! [`VectorLength`], [`VectorIndex`], and [`MatrixShape`].

use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

use crate::error::ModelError;

/// One finite scalar used by the Transformer teaching code.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ModelScalar(f32);

impl ModelScalar {
    pub(crate) fn from_raw(value: f32) -> Result<Self, ModelError> {
        if !value.is_finite() {
            return Err(ModelError::non_finite_value("model scalar", value));
        }

        Ok(Self(value))
    }

    pub(crate) fn as_f32(self) -> f32 {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn ensure_close_to(
        self,
        expected: ModelScalar,
        tolerance: ModelScalar,
    ) -> Result<(), ModelError> {
        let distance = (self.as_f32() - expected.as_f32()).abs();
        if distance <= tolerance.as_f32().abs() {
            return Ok(());
        }

        Err(ModelError::numerical_issue(
            "ModelScalar::ensure_close_to",
            "scalar was outside the requested tolerance",
        ))
    }

    #[cfg(test)]
    pub(crate) fn ensure_finite(self) -> Result<(), ModelError> {
        if self.as_f32().is_finite() {
            return Ok(());
        }

        Err(ModelError::non_finite_value("model scalar", self.as_f32()))
    }
}

impl TryFrom<f32> for ModelScalar {
    type Error = ModelError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for ModelScalar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl Add for ModelScalar {
    type Output = Result<ModelScalar, ModelError>;

    fn add(self, right: ModelScalar) -> Self::Output {
        ModelScalar::from_raw(self.as_f32() + right.as_f32())
    }
}

impl Sub for ModelScalar {
    type Output = Result<ModelScalar, ModelError>;

    fn sub(self, right: ModelScalar) -> Self::Output {
        ModelScalar::from_raw(self.as_f32() - right.as_f32())
    }
}

impl Mul for ModelScalar {
    type Output = Result<ModelScalar, ModelError>;

    fn mul(self, right: ModelScalar) -> Self::Output {
        ModelScalar::from_raw(self.as_f32() * right.as_f32())
    }
}

impl Div for ModelScalar {
    type Output = Result<ModelScalar, ModelError>;

    fn div(self, right: ModelScalar) -> Self::Output {
        ModelScalar::from_raw(self.as_f32() / right.as_f32())
    }
}

/// A non-zero vector length or model width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VectorLength(usize);

impl VectorLength {
    pub(crate) fn from_raw(value: usize, operation: &'static str) -> Result<Self, ModelError> {
        if value == 0 {
            return Err(ModelError::empty_input(
                operation,
                "length must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    pub(crate) fn from_known_nonzero(value: usize) -> Self {
        Self(value)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for VectorLength {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::from_raw(value, "VectorLength::try_from")
    }
}

impl fmt::Display for VectorLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A zero-based index into a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VectorIndex(usize);

impl VectorIndex {
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for VectorIndex {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for VectorIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A zero-based matrix row index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowIndex(usize);

impl RowIndex {
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for RowIndex {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

/// A zero-based matrix column index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColumnIndex(usize);

impl ColumnIndex {
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ColumnIndex {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

/// Matrix shape with non-zero row and column counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    rows: VectorLength,
    cols: VectorLength,
}

impl MatrixShape {
    /// Creates a matrix shape from checked non-zero dimensions.
    pub fn new(rows: VectorLength, cols: VectorLength) -> Self {
        Self { rows, cols }
    }

    /// Returns row count.
    pub fn rows(self) -> VectorLength {
        self.rows
    }

    /// Returns column count.
    pub fn cols(self) -> VectorLength {
        self.cols
    }
}

/// A printable view of scalar values that does not expose a raw slice type.
#[derive(Clone, Copy)]
pub struct ScalarValues<'a>(&'a [f32]);

impl ScalarValues<'_> {
    /// Iterates over checked scalar values.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = ModelScalar> + '_ {
        self.0.iter().copied().map(ModelScalar)
    }

    /// Returns the number of values.
    pub fn len(&self) -> VectorLength {
        VectorLength::from_known_nonzero(self.0.len())
    }
}

impl fmt::Debug for ScalarValues<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.0.iter()).finish()
    }
}

/// A dense one-dimensional vector of checked scalar values.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector(Vec<f32>);

impl DenseVector {
    /// Creates a vector from checked scalar values.
    pub fn new(data: impl IntoIterator<Item = ModelScalar>) -> Result<Self, ModelError> {
        let data = data
            .into_iter()
            .map(ModelScalar::as_f32)
            .collect::<Vec<_>>();
        Self::from_raw_values("DenseVector::new", data)
    }

    pub(crate) fn from_raw_values(
        operation: &'static str,
        data: impl IntoIterator<Item = f32>,
    ) -> Result<Self, ModelError> {
        let data = data.into_iter().collect::<Vec<_>>();
        if data.is_empty() {
            return Err(ModelError::empty_input(operation, "vector cannot be empty"));
        }

        for value in &data {
            if !value.is_finite() {
                return Err(ModelError::non_finite_value(
                    "dense vector component",
                    *value,
                ));
            }
        }

        Ok(Self(data))
    }

    /// Creates a zero vector.
    pub fn zeros(len: VectorLength) -> Result<Self, ModelError> {
        Self::zeros_raw(len.as_usize())
    }

    pub(crate) fn zeros_raw(len: usize) -> Result<Self, ModelError> {
        Self::from_raw_values("DenseVector::zeros", vec![0.0; len])
    }

    /// Creates a vector filled with ones.
    pub fn ones(len: VectorLength) -> Result<Self, ModelError> {
        Self::from_raw_values("DenseVector::ones", vec![1.0; len.as_usize()])
    }

    /// Returns the vector length.
    pub fn len(&self) -> VectorLength {
        VectorLength::from_known_nonzero(self.0.len())
    }

    pub(crate) fn len_usize(&self) -> usize {
        self.0.len()
    }

    /// Returns a printable scalar view.
    pub fn as_slice(&self) -> ScalarValues<'_> {
        ScalarValues(&self.0)
    }

    /// Iterates over checked scalar values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = ModelScalar> + '_ {
        self.0.iter().copied().map(ModelScalar)
    }

    pub(crate) fn as_raw_slice(&self) -> &[f32] {
        &self.0
    }

    /// Reads a single element.
    pub fn component(&self, index: VectorIndex) -> Result<ModelScalar, ModelError> {
        self.0
            .get(index.as_usize())
            .copied()
            .map(ModelScalar)
            .ok_or(ModelError::invalid_vector_index(
                "DenseVector::component",
                index.as_usize(),
                self.0.len(),
            ))
    }

    pub(crate) fn raw_at(&self, index: usize) -> f32 {
        self.0[index]
    }

    /// Overwrites a single element.
    pub fn set_component(
        &mut self,
        index: VectorIndex,
        value: ModelScalar,
    ) -> Result<(), ModelError> {
        let len = self.0.len();
        let slot = self
            .0
            .get_mut(index.as_usize())
            .ok_or(ModelError::invalid_vector_index(
                "DenseVector::set_component",
                index.as_usize(),
                len,
            ))?;
        *slot = value.as_f32();
        Ok(())
    }

    pub(crate) fn set_raw(&mut self, index: usize, value: f32) -> Result<(), ModelError> {
        self.set_component(VectorIndex(index), ModelScalar::from_raw(value)?)
    }

    /// Computes the dot product between two vectors.
    pub fn dot(&self, other: &DenseVector) -> Result<ModelScalar, ModelError> {
        if self.len() != other.len() {
            return Err(ModelError::dimension_mismatch(
                "DenseVector::dot",
                "left vector",
                vec![self.len_usize()],
                "right vector",
                vec![other.len_usize()],
                "dot product requires equal vector lengths",
            ));
        }

        ModelScalar::from_raw(self.dot_raw(other)?)
    }

    pub(crate) fn dot_raw(&self, other: &DenseVector) -> Result<f32, ModelError> {
        if self.len_usize() != other.len_usize() {
            return Err(ModelError::dimension_mismatch(
                "DenseVector::dot",
                "left vector",
                vec![self.len_usize()],
                "right vector",
                vec![other.len_usize()],
                "dot product requires equal vector lengths",
            ));
        }

        Ok(self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum())
    }

    fn elementwise_add(&self, other: &DenseVector) -> Result<DenseVector, ModelError> {
        if self.len() != other.len() {
            return Err(ModelError::dimension_mismatch(
                "DenseVector::add",
                "left vector",
                vec![self.len_usize()],
                "right vector",
                vec![other.len_usize()],
                "vector addition requires equal lengths",
            ));
        }

        DenseVector::from_raw_values(
            "DenseVector::add",
            self.0.iter().zip(other.0.iter()).map(|(a, b)| a + b),
        )
    }

    /// Scales a vector by a checked scalar.
    pub fn scale(&self, scalar: ModelScalar) -> DenseVector {
        self.scale_raw(scalar.as_f32())
    }

    pub(crate) fn scale_raw(&self, scalar: f32) -> DenseVector {
        DenseVector(self.0.iter().map(|value| value * scalar).collect())
    }

    pub(crate) fn map_raw<F>(&self, f: F) -> DenseVector
    where
        F: Fn(f32) -> f32,
    {
        DenseVector(self.0.iter().copied().map(f).collect())
    }

    /// Computes the mean of the vector.
    pub fn mean(&self) -> ModelScalar {
        ModelScalar(self.mean_raw())
    }

    pub(crate) fn mean_raw(&self) -> f32 {
        self.0.iter().sum::<f32>() / self.len_usize() as f32
    }

    /// Computes the population variance of the vector.
    pub fn variance(&self) -> ModelScalar {
        ModelScalar(self.variance_raw())
    }

    pub(crate) fn variance_raw(&self) -> f32 {
        let mean = self.mean_raw();

        self.0
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / self.len_usize() as f32
    }
}

impl<'b> Add<&'b DenseVector> for &DenseVector {
    type Output = Result<DenseVector, ModelError>;

    fn add(self, right: &'b DenseVector) -> Self::Output {
        self.elementwise_add(right)
    }
}

impl<'b> Mul<&'b DenseVector> for &DenseVector {
    type Output = Result<ModelScalar, ModelError>;

    fn mul(self, right: &'b DenseVector) -> Self::Output {
        self.dot(right)
    }
}

impl Mul<ModelScalar> for &DenseVector {
    type Output = DenseVector;

    fn mul(self, right: ModelScalar) -> Self::Output {
        self.scale(right)
    }
}

/// A dense row-major matrix of checked scalar values.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DenseMatrix {
    /// Creates a matrix from row-major checked scalar data.
    pub fn new(
        rows: VectorLength,
        cols: VectorLength,
        data: impl IntoIterator<Item = ModelScalar>,
    ) -> Result<Self, ModelError> {
        let data = data
            .into_iter()
            .map(ModelScalar::as_f32)
            .collect::<Vec<_>>();

        Self::from_raw_parts("DenseMatrix::new", rows.as_usize(), cols.as_usize(), data)
    }

    fn from_raw_parts(
        operation: &'static str,
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    ) -> Result<Self, ModelError> {
        VectorLength::from_raw(rows, operation)?;
        VectorLength::from_raw(cols, operation)?;

        if rows * cols != data.len() {
            return Err(ModelError::invalid_matrix_data(
                operation,
                rows,
                cols,
                data.len(),
            ));
        }

        for value in &data {
            if !value.is_finite() {
                return Err(ModelError::non_finite_value(
                    "dense matrix component",
                    *value,
                ));
            }
        }

        Ok(Self { rows, cols, data })
    }

    /// Creates a zero matrix.
    pub fn zeros(shape: MatrixShape) -> Result<Self, ModelError> {
        Self::zeros_raw(shape.rows().as_usize(), shape.cols().as_usize())
    }

    /// Creates a square identity matrix.
    pub fn identity(size: VectorLength) -> Result<Self, ModelError> {
        let size = size.as_usize();
        let data = (0..size)
            .flat_map(|row| (0..size).map(move |col| if row == col { 1.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        Self::from_raw_parts("DenseMatrix::identity", size, size, data)
    }

    pub(crate) fn zeros_raw(rows: usize, cols: usize) -> Result<Self, ModelError> {
        Self::from_raw_parts("DenseMatrix::zeros", rows, cols, vec![0.0; rows * cols])
    }

    /// Creates a matrix from nested row values.
    pub fn from_rows<I>(rows_data: I) -> Result<Self, ModelError>
    where
        I: IntoIterator,
        I::Item: IntoIterator<Item = ModelScalar>,
    {
        let rows_data = rows_data
            .into_iter()
            .map(|row| row.into_iter().map(ModelScalar::as_f32).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        if rows_data.is_empty() {
            return Err(ModelError::empty_input(
                "DenseMatrix::from_rows",
                "matrix cannot be empty",
            ));
        }

        let rows = rows_data.len();
        let cols = rows_data[0].len();

        if cols == 0 {
            return Err(ModelError::empty_input(
                "DenseMatrix::from_rows",
                "matrix rows cannot be empty",
            ));
        }

        for row in &rows_data {
            if row.len() != cols {
                return Err(ModelError::invalid_matrix_data(
                    "DenseMatrix::from_rows",
                    rows,
                    cols,
                    rows_data.iter().map(Vec::len).sum(),
                ));
            }
        }

        Self::from_raw_parts(
            "DenseMatrix::from_rows",
            rows,
            cols,
            rows_data.into_iter().flatten().collect(),
        )
    }

    /// Returns matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: VectorLength::from_known_nonzero(self.rows),
            cols: VectorLength::from_known_nonzero(self.cols),
        }
    }

    /// Returns the row count.
    pub fn rows(&self) -> VectorLength {
        self.shape().rows()
    }

    /// Returns the column count.
    pub fn cols(&self) -> VectorLength {
        self.shape().cols()
    }

    /// Reads one matrix element.
    pub fn component(&self, row: RowIndex, col: ColumnIndex) -> Result<ModelScalar, ModelError> {
        if row.as_usize() >= self.rows || col.as_usize() >= self.cols {
            return Err(ModelError::invalid_matrix_index(
                "DenseMatrix::component",
                row.as_usize(),
                col.as_usize(),
                self.rows,
                self.cols,
            ));
        }

        Ok(ModelScalar(self.raw_at(row.as_usize(), col.as_usize())))
    }

    pub(crate) fn raw_at(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Overwrites one matrix element.
    pub fn set_component(
        &mut self,
        row: RowIndex,
        col: ColumnIndex,
        value: ModelScalar,
    ) -> Result<(), ModelError> {
        if row.as_usize() >= self.rows || col.as_usize() >= self.cols {
            return Err(ModelError::invalid_matrix_index(
                "DenseMatrix::set_component",
                row.as_usize(),
                col.as_usize(),
                self.rows,
                self.cols,
            ));
        }

        self.data[row.as_usize() * self.cols + col.as_usize()] = value.as_f32();
        Ok(())
    }

    pub(crate) fn set_raw(&mut self, row: usize, col: usize, value: f32) -> Result<(), ModelError> {
        self.set_component(
            RowIndex(row),
            ColumnIndex(col),
            ModelScalar::from_raw(value)?,
        )
    }

    /// Multiplies the matrix by a vector.
    pub fn mul_vec(&self, vector: &DenseVector) -> Result<DenseVector, ModelError> {
        if self.cols != vector.len_usize() {
            return Err(ModelError::dimension_mismatch(
                "DenseMatrix::mul_vec",
                "matrix",
                vec![self.rows, self.cols],
                "vector",
                vec![vector.len_usize()],
                "matrix columns must equal vector length",
            ));
        }

        let mut out = vec![0.0; self.rows];

        for (row, slot) in out.iter_mut().enumerate().take(self.rows) {
            let mut sum = 0.0;

            for col in 0..self.cols {
                sum += self.raw_at(row, col) * vector.raw_at(col);
            }

            *slot = sum;
        }

        DenseVector::from_raw_values("DenseMatrix::mul_vec", out)
    }
}

impl<'b> Mul<&'b DenseVector> for &DenseMatrix {
    type Output = Result<DenseVector, ModelError>;

    fn mul(self, right: &'b DenseVector) -> Self::Output {
        self.mul_vec(right)
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMatrix, DenseVector, MatrixShape, ModelScalar, VectorLength};
    use crate::error::ModelError;

    fn assert_vector_values(vector: &DenseVector, expected: impl IntoIterator<Item = ModelScalar>) {
        let actual = vector
            .values()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn dense_vector_rejects_empty_input() {
        assert!(matches!(
            DenseVector::new(Vec::<ModelScalar>::new()),
            Err(ModelError::EmptyInput { .. })
        ));
    }

    #[test]
    fn dense_vector_mul_operator_computes_dot_product() -> Result<(), ModelError> {
        let left = DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;
        let right = DenseVector::new([
            ModelScalar::try_from(4.0)?,
            ModelScalar::try_from(5.0)?,
            ModelScalar::try_from(6.0)?,
        ])?;

        assert_eq!((&left * &right)?.to_string(), "32");
        Ok(())
    }

    #[test]
    fn dense_vector_mul_operator_reports_shape_mismatch() -> Result<(), ModelError> {
        let left = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
        let right = DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;

        assert!(matches!(
            &left * &right,
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn dense_vector_add_reports_shape_mismatch() -> Result<(), ModelError> {
        let left = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?;
        let right = DenseVector::new([ModelScalar::try_from(1.0)?])?;

        assert!(matches!(
            &left + &right,
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn dense_vector_mean_and_variance_match_expected_values() -> Result<(), ModelError> {
        let vector = DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;

        assert_eq!(vector.mean().to_string(), "2");
        assert_eq!(vector.variance().to_string(), "0.6666667");
        Ok(())
    }

    #[test]
    fn dense_matrix_rejects_invalid_data_length() -> Result<(), ModelError> {
        assert!(matches!(
            DenseMatrix::new(
                VectorLength::try_from(2)?,
                VectorLength::try_from(2)?,
                [
                    ModelScalar::try_from(1.0)?,
                    ModelScalar::try_from(2.0)?,
                    ModelScalar::try_from(3.0)?
                ],
            ),
            Err(ModelError::InvalidMatrixData { .. })
        ));
        Ok(())
    }

    #[test]
    fn dense_matrix_from_rows_rejects_ragged_rows() -> Result<(), ModelError> {
        assert!(matches!(
            DenseMatrix::from_rows([
                vec![ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?],
                vec![ModelScalar::try_from(3.0)?]
            ]),
            Err(ModelError::InvalidMatrixData { .. })
        ));
        Ok(())
    }

    #[test]
    fn dense_matrix_mul_operator_matches_manual_computation() -> Result<(), ModelError> {
        let matrix = DenseMatrix::from_rows([
            [
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(2.0)?,
            ],
            [
                ModelScalar::try_from(0.0)?,
                ModelScalar::try_from(1.0)?,
                ModelScalar::try_from(3.0)?,
            ],
        ])?;
        let vector = DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;

        assert_vector_values(
            &(&matrix * &vector)?,
            [ModelScalar::try_from(7.0)?, ModelScalar::try_from(11.0)?],
        );
        Ok(())
    }

    #[test]
    fn dense_matrix_mul_vec_reports_shape_mismatch() -> Result<(), ModelError> {
        let matrix = DenseMatrix::from_rows([
            [ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?],
            [ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?],
        ])?;
        let vector = DenseVector::new([
            ModelScalar::try_from(1.0)?,
            ModelScalar::try_from(2.0)?,
            ModelScalar::try_from(3.0)?,
        ])?;

        assert!(matches!(
            matrix.mul_vec(&vector),
            Err(ModelError::DimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn vector_length_try_from_rejects_zero_dimensions() {
        assert!(matches!(
            VectorLength::try_from(0_usize),
            Err(ModelError::EmptyInput { .. })
        ));
    }

    #[test]
    fn matrix_shape_keeps_dimensions_after_validation() -> Result<(), ModelError> {
        let shape = MatrixShape::new(VectorLength::try_from(2)?, VectorLength::try_from(3)?);

        assert_eq!(shape.rows(), VectorLength::try_from(2)?);
        assert_eq!(shape.cols(), VectorLength::try_from(3)?);
        Ok(())
    }
}
