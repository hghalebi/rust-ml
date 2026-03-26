//! Math primitives used by the tiny Transformer crate.

use crate::types::{ColumnCount, ColumnIndex, Dimension, RowCount, RowIndex, Scalar};

/// A dynamically sized vector of model scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<Scalar>,
}

impl Vector {
    /// Creates a vector from owned scalar data.
    pub fn new(data: Vec<Scalar>) -> Self {
        Self { data }
    }

    /// Creates a vector from primitive boundary values.
    pub fn from_f32s(data: Vec<f32>) -> Self {
        Self::new(data.into_iter().map(Scalar::from).collect())
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> Dimension {
        Dimension::new(self.data.len())
    }

    /// Returns true when the vector contains no values.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Computes the dot product with another vector of equal length.
    pub fn dot(&self, other: &Vector) -> Scalar {
        assert_eq!(self.len(), other.len(), "dot: dimension mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }

    /// Adds another vector element-wise.
    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len(), "add: dimension mismatch");
        Vector::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        )
    }

    /// Applies a scalar function to every element.
    pub fn map<F>(&self, f: F) -> Vector
    where
        F: Fn(Scalar) -> Scalar,
    {
        Vector::new(self.data.iter().copied().map(f).collect())
    }

    /// Scales every element by a scalar factor.
    pub fn scale(&self, scale: Scalar) -> Vector {
        self.map(|value| value * scale)
    }

    /// Borrows the underlying data as a slice.
    pub fn as_slice(&self) -> &[Scalar] {
        &self.data
    }
}

/// A row-major dynamic matrix of model scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: RowCount,
    cols: ColumnCount,
    data: Vec<Scalar>,
}

impl Matrix {
    /// Creates a matrix from row and column counts plus row-major scalar data.
    pub fn new(rows: RowCount, cols: ColumnCount, data: Vec<Scalar>) -> Self {
        assert_eq!(
            rows.get() * cols.get(),
            data.len(),
            "matrix: invalid data length"
        );
        Self { rows, cols, data }
    }

    /// Creates a matrix from primitive boundary values.
    pub fn from_f32s(rows: RowCount, cols: ColumnCount, data: Vec<f32>) -> Self {
        Self::new(rows, cols, data.into_iter().map(Scalar::from).collect())
    }

    /// Creates a zero matrix.
    pub fn zeros(rows: RowCount, cols: ColumnCount) -> Self {
        Self {
            rows,
            cols,
            data: vec![Scalar::ZERO; rows.get() * cols.get()],
        }
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> RowCount {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> ColumnCount {
        self.cols
    }

    /// Reads one matrix entry.
    pub fn get(&self, row: RowIndex, column: ColumnIndex) -> Scalar {
        self.data[row.get() * self.cols.get() + column.get()]
    }

    /// Writes one matrix entry.
    pub fn set(&mut self, row: RowIndex, column: ColumnIndex, value: Scalar) {
        self.data[row.get() * self.cols.get() + column.get()] = value;
    }

    /// Multiplies the matrix by a vector.
    pub fn mul_vec(&self, x: &Vector) -> Vector {
        assert_eq!(
            self.cols.get(),
            x.len().get(),
            "mul_vec: dimension mismatch"
        );

        let mut out = vec![Scalar::ZERO; self.rows.get()];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut sum = Scalar::ZERO;
            for column in 0..self.cols.get() {
                sum +=
                    self.get(RowIndex::new(row), ColumnIndex::new(column)) * x.as_slice()[column];
            }
            *slot = sum;
        }
        Vector::new(out)
    }

    /// Returns the transposed matrix.
    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(
            RowCount::new(self.cols.get()),
            ColumnCount::new(self.rows.get()),
        );
        for row in 0..self.rows.get() {
            for column in 0..self.cols.get() {
                out.set(
                    RowIndex::new(column),
                    ColumnIndex::new(row),
                    self.get(RowIndex::new(row), ColumnIndex::new(column)),
                );
            }
        }
        out
    }
}

/// A compile-time-sized vector used to demonstrate shape-safe math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorN<const N: usize> {
    /// Fixed-size vector data.
    pub data: [Scalar; N],
}

impl<const N: usize> VectorN<N> {
    /// Computes the dot product with another vector of the same size.
    pub fn dot(&self, other: &Self) -> Scalar {
        let mut sum = Scalar::ZERO;
        for i in 0..N {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Adds another vector of the same size.
    pub fn add(&self, other: &Self) -> Self {
        let mut out = [Scalar::ZERO; N];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.data[i] + other.data[i];
        }
        Self { data: out }
    }

    /// Applies a ReLU element-wise.
    pub fn relu(&self) -> Self {
        let mut out = [Scalar::ZERO; N];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.data[i].max(Scalar::ZERO);
        }
        Self { data: out }
    }
}

/// A compile-time-sized matrix used to demonstrate shape-safe products.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixMN<const R: usize, const C: usize> {
    /// Fixed-size matrix data in row-major nested-array form.
    pub data: [[Scalar; C]; R],
}

impl<const R: usize, const C: usize> MatrixMN<R, C> {
    /// Multiplies the matrix by a compile-time-sized vector.
    pub fn mul_vec(&self, x: &VectorN<C>) -> VectorN<R> {
        let mut out = [Scalar::ZERO; R];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut sum = Scalar::ZERO;
            for column in 0..C {
                sum += self.data[row][column] * x.data[column];
            }
            *slot = sum;
        }
        VectorN { data: out }
    }
}
