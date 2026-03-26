//! Math primitives used by the tiny Transformer crate.

/// A dynamically sized vector of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// Creates a vector from owned data.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true when the vector contains no values.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Computes the dot product with another vector of equal length.
    pub fn dot(&self, other: &Vector) -> f32 {
        assert_eq!(self.len(), other.len(), "dot: dimension mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Adds another vector element-wise.
    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len(), "add: dimension mismatch");
        Vector::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        )
    }

    /// Applies a scalar function to every element.
    pub fn map<F>(&self, f: F) -> Vector
    where
        F: Fn(f32) -> f32,
    {
        Vector::new(self.data.iter().copied().map(f).collect())
    }

    /// Scales every element by a scalar factor.
    pub fn scale(&self, s: f32) -> Vector {
        self.map(|x| x * s)
    }

    /// Borrows the underlying data as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// A row-major dynamic matrix of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Creates a matrix from row and column counts plus row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "matrix: invalid data length");
        Self { rows, cols, data }
    }

    /// Creates a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Reads one matrix entry.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Writes one matrix entry.
    pub fn set(&mut self, r: usize, c: usize, value: f32) {
        self.data[r * self.cols + c] = value;
    }

    /// Multiplies the matrix by a vector.
    pub fn mul_vec(&self, x: &Vector) -> Vector {
        assert_eq!(self.cols, x.len(), "mul_vec: dimension mismatch");

        let mut out = vec![0.0; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c) * x.as_slice()[c];
            }
            *slot = sum;
        }
        Vector::new(out)
    }

    /// Returns the transposed matrix.
    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }
}

/// A compile-time-sized vector used to demonstrate shape-safe math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorN<const N: usize> {
    /// Fixed-size vector data.
    pub data: [f32; N],
}

impl<const N: usize> VectorN<N> {
    /// Computes the dot product with another vector of the same size.
    pub fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Adds another vector of the same size.
    pub fn add(&self, other: &Self) -> Self {
        let mut out = [0.0; N];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.data[i] + other.data[i];
        }
        Self { data: out }
    }

    /// Applies a ReLU element-wise.
    pub fn relu(&self) -> Self {
        let mut out = [0.0; N];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.data[i].max(0.0);
        }
        Self { data: out }
    }
}

/// A compile-time-sized matrix used to demonstrate shape-safe products.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixMN<const R: usize, const C: usize> {
    /// Fixed-size matrix data in row-major nested-array form.
    pub data: [[f32; C]; R],
}

impl<const R: usize, const C: usize> MatrixMN<R, C> {
    /// Multiplies the matrix by a compile-time-sized vector.
    pub fn mul_vec(&self, x: &VectorN<C>) -> VectorN<R> {
        let mut out = [0.0; R];
        for (r, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for c in 0..C {
                sum += self.data[r][c] * x.data[c];
            }
            *slot = sum;
        }
        VectorN { data: out }
    }
}
