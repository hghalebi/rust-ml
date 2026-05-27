//! Typed kernel primitives for the CS336 Rust equivalent track.
//!
//! This crate keeps kernel ideas CPU-first and inspectable:
//!
//! ```text
//! MatrixRows * MatrixColumns -> ElementCount
//! ElementCount * ElementSize -> Bytes
//! ElementCount * FlopsPerElement -> FlopCount
//! Accumulator + KernelProduct -> Accumulator
//! ```
//!
//! Raw learner literals enter through `TryFrom` adapters. Public teaching APIs
//! then move through semantic values such as matrix shapes, tile shapes,
//! kernel scalars, byte counts, FLOP counts, and tiled execution traces.

pub mod error;

use std::{
    cmp, fmt,
    ops::{Add, Mul},
};

use error::KernelError;

pub use error::KernelError as Error;

fn nonzero_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, KernelError> {
    if value == 0 {
        return Err(KernelError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u64(
    role: &'static str,
    operation: &'static str,
    value: u64,
) -> Result<u64, KernelError> {
    if value == 0 {
        return Err(KernelError::empty_input(operation, role));
    }

    Ok(value)
}

fn finite(role: &'static str, value: f64) -> Result<f64, KernelError> {
    if !value.is_finite() {
        return Err(KernelError::non_finite_value(role, value));
    }

    Ok(value)
}

fn checked_usize_mul(
    operation: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, KernelError> {
    left.checked_mul(right).ok_or(KernelError::overflow(
        operation,
        "usize multiplication overflowed",
    ))
}

fn checked_u64_add(operation: &'static str, left: u64, right: u64) -> Result<u64, KernelError> {
    left.checked_add(right)
        .ok_or(KernelError::overflow(operation, "u64 addition overflowed"))
}

fn checked_u64_mul(operation: &'static str, left: u64, right: u64) -> Result<u64, KernelError> {
    left.checked_mul(right).ok_or(KernelError::overflow(
        operation,
        "u64 multiplication overflowed",
    ))
}

fn u64_from_usize(operation: &'static str, value: usize) -> Result<u64, KernelError> {
    u64::try_from(value).map_err(|_| KernelError::overflow(operation, "value exceeded u64"))
}

macro_rules! nonzero_count_type {
    ($name:ident, $doc:literal, $role:literal, $operation:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(usize);

        impl TryFrom<usize> for $name {
            type Error = KernelError;

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                Ok(Self(nonzero_usize($role, $operation, value)?))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, formatter)
            }
        }
    };
}

nonzero_count_type!(
    MatrixRows,
    "Number of rows in a matrix.",
    "matrix rows must be greater than zero",
    "MatrixRows::try_from"
);
nonzero_count_type!(
    MatrixColumns,
    "Number of columns in a matrix.",
    "matrix columns must be greater than zero",
    "MatrixColumns::try_from"
);
nonzero_count_type!(
    VectorLength,
    "Number of values in a vector.",
    "vector length must be greater than zero",
    "VectorLength::try_from"
);
nonzero_count_type!(
    TileRows,
    "Number of matrix rows processed by one tile.",
    "tile rows must be greater than zero",
    "TileRows::try_from"
);
nonzero_count_type!(
    TileColumns,
    "Number of matrix columns processed by one tile.",
    "tile columns must be greater than zero",
    "TileColumns::try_from"
);
nonzero_count_type!(
    TileRowSpan,
    "Actual number of rows covered by a tile window.",
    "tile row span must be greater than zero",
    "TileRowSpan::try_from"
);
nonzero_count_type!(
    TileColumnSpan,
    "Actual number of columns covered by a tile window.",
    "tile column span must be greater than zero",
    "TileColumnSpan::try_from"
);

impl MatrixRows {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl MatrixColumns {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl VectorLength {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TileRows {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TileColumns {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TileRowSpan {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TileColumnSpan {
    fn as_usize(self) -> usize {
        self.0
    }
}

/// Shape of a dense row-major matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    rows: MatrixRows,
    columns: MatrixColumns,
}

impl MatrixShape {
    /// Creates a matrix shape from validated dimensions.
    pub fn new(rows: MatrixRows, columns: MatrixColumns) -> Self {
        Self { rows, columns }
    }

    /// Returns the row count.
    pub fn rows(&self) -> MatrixRows {
        self.rows
    }

    /// Returns the column count.
    pub fn columns(&self) -> MatrixColumns {
        self.columns
    }

    /// Returns the total number of scalar matrix elements.
    pub fn element_count(&self) -> Result<ElementCount, KernelError> {
        self.rows * self.columns
    }
}

impl fmt::Display for MatrixShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}x{}", self.rows, self.columns)
    }
}

/// Shape of a teaching tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    rows: TileRows,
    columns: TileColumns,
}

impl TileShape {
    /// Creates a tile shape from validated tile dimensions.
    pub fn new(rows: TileRows, columns: TileColumns) -> Self {
        Self { rows, columns }
    }

    /// Returns the requested tile row count.
    pub fn rows(&self) -> TileRows {
        self.rows
    }

    /// Returns the requested tile column count.
    pub fn columns(&self) -> TileColumns {
        self.columns
    }
}

impl fmt::Display for TileShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}x{}", self.rows, self.columns)
    }
}

/// Zero-based row index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowIndex(usize);

impl RowIndex {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for RowIndex {
    type Error = KernelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for RowIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Zero-based column index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColumnIndex(usize);

impl ColumnIndex {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for ColumnIndex {
    type Error = KernelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for ColumnIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// One cell coordinate in a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixCell {
    row: RowIndex,
    column: ColumnIndex,
}

impl MatrixCell {
    /// Creates a matrix cell coordinate.
    pub fn new(row: RowIndex, column: ColumnIndex) -> Self {
        Self { row, column }
    }

    fn row(&self) -> RowIndex {
        self.row
    }

    fn column(&self) -> ColumnIndex {
        self.column
    }
}

/// Finite scalar used by teaching kernels.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct KernelScalar(f64);

impl KernelScalar {
    fn from_raw(role: &'static str, value: f64) -> Result<Self, KernelError> {
        Ok(Self(finite(role, value)?))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for KernelScalar {
    type Error = KernelError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw("kernel scalar", value)
    }
}

impl fmt::Display for KernelScalar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.4}", self.0)
    }
}

/// Product of two kernel scalars.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct KernelProduct(f64);

impl KernelProduct {
    fn from_raw(value: f64) -> Result<Self, KernelError> {
        Ok(Self(finite("kernel product", value)?))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl Mul for KernelScalar {
    type Output = Result<KernelProduct, KernelError>;

    fn mul(self, right: KernelScalar) -> Self::Output {
        KernelProduct::from_raw(self.as_f64() * right.as_f64())
    }
}

/// Running sum inside a reduction or matrix-vector tile.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Accumulator(f64);

impl Accumulator {
    fn zero() -> Self {
        Self(0.0)
    }

    fn into_scalar(self) -> Result<KernelScalar, KernelError> {
        KernelScalar::from_raw("accumulator", self.0)
    }
}

impl Add<KernelProduct> for Accumulator {
    type Output = Result<Accumulator, KernelError>;

    fn add(self, right: KernelProduct) -> Self::Output {
        Ok(Self(finite("accumulator", self.0 + right.as_f64())?))
    }
}

impl Add<KernelScalar> for Accumulator {
    type Output = Result<Accumulator, KernelError>;

    fn add(self, right: KernelScalar) -> Self::Output {
        Ok(Self(finite("accumulator", self.0 + right.as_f64())?))
    }
}

/// Non-empty vector of kernel scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelVector {
    values: Vec<KernelScalar>,
}

impl KernelVector {
    /// Builds a non-empty vector from validated scalars.
    pub fn from_values(
        values: impl IntoIterator<Item = KernelScalar>,
    ) -> Result<Self, KernelError> {
        let values = values.into_iter().collect::<Vec<_>>();
        VectorLength::try_from(values.len())?;
        Ok(Self { values })
    }

    /// Returns vector length.
    pub fn length(&self) -> VectorLength {
        VectorLength(self.values.len())
    }

    /// Iterates over values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = KernelScalar> + '_ {
        self.values.iter().copied()
    }

    fn value(&self, column: ColumnIndex) -> Result<KernelScalar, KernelError> {
        self.values
            .get(column.as_usize())
            .copied()
            .ok_or(KernelError::count_out_of_range(
                "vector index",
                "inside vector length",
                column.as_usize(),
            ))
    }
}

/// Dense row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelMatrix {
    shape: MatrixShape,
    values: Vec<KernelScalar>,
}

impl KernelMatrix {
    /// Builds a matrix from non-empty rows with a shared width.
    pub fn from_rows(rows: impl IntoIterator<Item = KernelVector>) -> Result<Self, KernelError> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let row_count = MatrixRows::try_from(rows.len())?;
        let first_width =
            rows.first()
                .map(KernelVector::length)
                .ok_or(KernelError::empty_input(
                    "KernelMatrix::from_rows",
                    "matrix must contain at least one row",
                ))?;

        let mut values = Vec::new();
        for row in rows {
            if row.length() != first_width {
                return Err(KernelError::shape_mismatch(
                    "KernelMatrix::from_rows",
                    "all matrix rows must have the same width",
                ));
            }
            values.extend(row.values());
        }

        Ok(Self {
            shape: MatrixShape::new(row_count, MatrixColumns::try_from(first_width.as_usize())?),
            values,
        })
    }

    /// Returns the matrix shape.
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn value(&self, cell: MatrixCell) -> Result<KernelScalar, KernelError> {
        if cell.row().as_usize() >= self.shape.rows().as_usize() {
            return Err(KernelError::count_out_of_range(
                "matrix row index",
                "inside matrix row count",
                cell.row().as_usize(),
            ));
        }
        if cell.column().as_usize() >= self.shape.columns().as_usize() {
            return Err(KernelError::count_out_of_range(
                "matrix column index",
                "inside matrix column count",
                cell.column().as_usize(),
            ));
        }

        let offset =
            cell.row().as_usize() * self.shape.columns().as_usize() + cell.column().as_usize();
        self.values
            .get(offset)
            .copied()
            .ok_or(KernelError::count_out_of_range(
                "matrix cell offset",
                "inside row-major storage",
                offset,
            ))
    }
}

/// Number of scalar elements involved in an estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElementCount(u64);

impl ElementCount {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, KernelError> {
        Ok(Self(nonzero_u64(
            "element count must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

impl Mul<MatrixColumns> for MatrixRows {
    type Output = Result<ElementCount, KernelError>;

    fn mul(self, right: MatrixColumns) -> Self::Output {
        ElementCount::from_raw(
            "MatrixRows::mul",
            u64_from_usize(
                "MatrixRows::mul",
                checked_usize_mul("MatrixRows::mul", self.as_usize(), right.as_usize())?,
            )?,
        )
    }
}

impl fmt::Display for ElementCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of bytes used by one scalar element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElementSize(Bytes);

impl ElementSize {
    /// Returns the size of one `f32` scalar.
    pub fn float32() -> Self {
        Self(Bytes(4))
    }

    /// Returns the size of one `f64` scalar.
    pub fn float64() -> Self {
        Self(Bytes(8))
    }

    fn bytes(self) -> Bytes {
        self.0
    }
}

impl fmt::Display for ElementSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of bytes moved or stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(u64);

impl Bytes {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, KernelError> {
        Ok(Self(nonzero_u64(
            "bytes must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for Bytes {
    type Error = KernelError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_raw("Bytes::try_from", value)
    }
}

impl Add for Bytes {
    type Output = Result<Bytes, KernelError>;

    fn add(self, right: Bytes) -> Self::Output {
        Bytes::from_raw(
            "Bytes::add",
            checked_u64_add("Bytes::add", self.as_u64(), right.as_u64())?,
        )
    }
}

impl Mul<ElementSize> for ElementCount {
    type Output = Result<Bytes, KernelError>;

    fn mul(self, right: ElementSize) -> Self::Output {
        Bytes::from_raw(
            "ElementCount::mul",
            checked_u64_mul("ElementCount::mul", self.as_u64(), right.bytes().as_u64())?,
        )
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} bytes", self.0)
    }
}

/// FLOPs performed per scalar element for a simple kernel estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FlopsPerElement(u64);

impl FlopsPerElement {
    /// One floating-point operation per element.
    pub fn one() -> Self {
        Self(1)
    }

    /// Two floating-point operations per element.
    pub fn two() -> Self {
        Self(2)
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

/// Count of floating-point operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FlopCount(u64);

impl FlopCount {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, KernelError> {
        Ok(Self(nonzero_u64(
            "FLOP count must be greater than zero",
            operation,
            value,
        )?))
    }
}

impl Mul<FlopsPerElement> for ElementCount {
    type Output = Result<FlopCount, KernelError>;

    fn mul(self, right: FlopsPerElement) -> Self::Output {
        FlopCount::from_raw(
            "ElementCount::mul",
            checked_u64_mul("ElementCount::mul", self.as_u64(), right.as_u64())?,
        )
    }
}

impl fmt::Display for FlopCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} FLOPs", self.0)
    }
}

/// One matrix tile window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileWindow {
    row_start: RowIndex,
    column_start: ColumnIndex,
    row_span: TileRowSpan,
    column_span: TileColumnSpan,
}

impl TileWindow {
    fn new(
        row_start: RowIndex,
        column_start: ColumnIndex,
        row_span: TileRowSpan,
        column_span: TileColumnSpan,
    ) -> Self {
        Self {
            row_start,
            column_start,
            row_span,
            column_span,
        }
    }

    /// Returns the first row covered by the tile.
    pub fn row_start(&self) -> RowIndex {
        self.row_start
    }

    /// Returns the first column covered by the tile.
    pub fn column_start(&self) -> ColumnIndex {
        self.column_start
    }

    /// Returns the actual number of rows covered by the tile.
    pub fn row_span(&self) -> TileRowSpan {
        self.row_span
    }

    /// Returns the actual number of columns covered by the tile.
    pub fn column_span(&self) -> TileColumnSpan {
        self.column_span
    }
}

impl fmt::Display for TileWindow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "tile rows {}..+{}, cols {}..+{}",
            self.row_start, self.row_span, self.column_start, self.column_span
        )
    }
}

/// Tiling plan for a matrix-shaped kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    matrix_shape: MatrixShape,
    tile_shape: TileShape,
    windows: Vec<TileWindow>,
}

impl TilePlan {
    /// Builds tile windows over a matrix shape.
    pub fn new(matrix_shape: MatrixShape, tile_shape: TileShape) -> Result<Self, KernelError> {
        let mut windows = Vec::new();
        let mut row_start = 0;

        while row_start < matrix_shape.rows().as_usize() {
            let row_span = cmp::min(
                tile_shape.rows().as_usize(),
                matrix_shape.rows().as_usize() - row_start,
            );
            let mut column_start = 0;

            while column_start < matrix_shape.columns().as_usize() {
                let column_span = cmp::min(
                    tile_shape.columns().as_usize(),
                    matrix_shape.columns().as_usize() - column_start,
                );
                windows.push(TileWindow::new(
                    RowIndex::from_raw(row_start),
                    ColumnIndex::from_raw(column_start),
                    TileRowSpan::try_from(row_span)?,
                    TileColumnSpan::try_from(column_span)?,
                ));
                column_start += tile_shape.columns().as_usize();
            }

            row_start += tile_shape.rows().as_usize();
        }

        Ok(Self {
            matrix_shape,
            tile_shape,
            windows,
        })
    }

    /// Returns the matrix shape being tiled.
    pub fn matrix_shape(&self) -> MatrixShape {
        self.matrix_shape
    }

    /// Returns the requested tile shape.
    pub fn tile_shape(&self) -> TileShape {
        self.tile_shape
    }

    /// Iterates over tile windows.
    pub fn windows(&self) -> impl ExactSizeIterator<Item = &TileWindow> + '_ {
        self.windows.iter()
    }
}

/// Trace for an elementwise GeLU-like teaching kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseTrace {
    output: KernelVector,
    element_count: ElementCount,
    flops: FlopCount,
    hbm_bytes: Bytes,
}

impl ElementwiseTrace {
    /// Applies a deterministic GeLU approximation to each vector element.
    pub fn gelu(input: KernelVector, element_size: ElementSize) -> Result<Self, KernelError> {
        let output = KernelVector::from_values(input.values().map(approximate_gelu))?;
        let element_count = ElementCount::from_raw(
            "ElementwiseTrace::gelu",
            u64_from_usize("ElementwiseTrace::gelu", input.length().as_usize())?,
        )?;
        let read_bytes = (element_count * element_size)?;
        let write_bytes = (element_count * element_size)?;
        let hbm_bytes = (read_bytes + write_bytes)?;
        let flops = (element_count * FlopsPerElement::two())?;

        Ok(Self {
            output,
            element_count,
            flops,
            hbm_bytes,
        })
    }

    /// Returns the output vector.
    pub fn output(&self) -> &KernelVector {
        &self.output
    }

    /// Returns how many elements were processed.
    pub fn element_count(&self) -> ElementCount {
        self.element_count
    }

    /// Returns the simplified FLOP estimate.
    pub fn flops(&self) -> FlopCount {
        self.flops
    }

    /// Returns the simplified HBM byte estimate.
    pub fn hbm_bytes(&self) -> Bytes {
        self.hbm_bytes
    }
}

fn approximate_gelu(value: KernelScalar) -> KernelScalar {
    let x = value.as_f64();
    let sigmoid_like = 1.0 / (1.0 + (-1.702 * x).exp());
    KernelScalar(x * sigmoid_like)
}

/// Trace for one row-wise reduction.
#[derive(Debug, Clone, PartialEq)]
pub struct RowReductionTrace {
    output: KernelScalar,
    element_count: ElementCount,
    flops: FlopCount,
}

impl RowReductionTrace {
    /// Sums one row through typed accumulation.
    pub fn sum(row: KernelVector) -> Result<Self, KernelError> {
        let output = row
            .values()
            .try_fold(Accumulator::zero(), |accumulator, value| {
                accumulator + value
            })?
            .into_scalar()?;
        let element_count = ElementCount::from_raw(
            "RowReductionTrace::sum",
            u64_from_usize("RowReductionTrace::sum", row.length().as_usize())?,
        )?;
        let flops = (element_count * FlopsPerElement::one())?;

        Ok(Self {
            output,
            element_count,
            flops,
        })
    }

    /// Returns the reduced scalar.
    pub fn output(&self) -> KernelScalar {
        self.output
    }

    /// Returns how many row elements were reduced.
    pub fn element_count(&self) -> ElementCount {
        self.element_count
    }

    /// Returns the simplified FLOP estimate.
    pub fn flops(&self) -> FlopCount {
        self.flops
    }
}

/// Trace for tiled matrix-vector multiplication.
#[derive(Debug, Clone, PartialEq)]
pub struct TiledMatVecTrace {
    output: KernelVector,
    tile_plan: TilePlan,
    element_count: ElementCount,
    flops: FlopCount,
    hbm_bytes: Bytes,
}

impl TiledMatVecTrace {
    /// Runs a tiny tiled matrix-vector multiplication.
    pub fn run(
        matrix: KernelMatrix,
        vector: KernelVector,
        tile_shape: TileShape,
        element_size: ElementSize,
    ) -> Result<Self, KernelError> {
        if matrix.shape().columns().as_usize() != vector.length().as_usize() {
            return Err(KernelError::shape_mismatch(
                "TiledMatVecTrace::run",
                "matrix columns must equal vector length",
            ));
        }

        let tile_plan = TilePlan::new(matrix.shape(), tile_shape)?;
        let mut accumulators = vec![Accumulator::zero(); matrix.shape().rows().as_usize()];

        for window in tile_plan.windows() {
            for row_offset in 0..window.row_span().as_usize() {
                let row = RowIndex::from_raw(window.row_start().as_usize() + row_offset);
                for column_offset in 0..window.column_span().as_usize() {
                    let column =
                        ColumnIndex::from_raw(window.column_start().as_usize() + column_offset);
                    let product =
                        (matrix.value(MatrixCell::new(row, column))? * vector.value(column)?)?;
                    let next = (accumulators[row.as_usize()] + product)?;
                    accumulators[row.as_usize()] = next;
                }
            }
        }

        let output = KernelVector::from_values(
            accumulators
                .into_iter()
                .map(Accumulator::into_scalar)
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let element_count = matrix.shape().element_count()?;
        let flops = (element_count * FlopsPerElement::two())?;
        let matrix_bytes = (element_count * element_size)?;
        let vector_bytes = (ElementCount::from_raw(
            "TiledMatVecTrace::run",
            u64_from_usize("TiledMatVecTrace::run", vector.length().as_usize())?,
        )? * element_size)?;
        let output_bytes = (ElementCount::from_raw(
            "TiledMatVecTrace::run",
            u64_from_usize("TiledMatVecTrace::run", output.length().as_usize())?,
        )? * element_size)?;
        let hbm_bytes = ((matrix_bytes + vector_bytes)? + output_bytes)?;

        Ok(Self {
            output,
            tile_plan,
            element_count,
            flops,
            hbm_bytes,
        })
    }

    /// Returns the output vector.
    pub fn output(&self) -> &KernelVector {
        &self.output
    }

    /// Returns the tile plan used by the kernel.
    pub fn tile_plan(&self) -> &TilePlan {
        &self.tile_plan
    }

    /// Returns how many matrix elements were multiplied.
    pub fn element_count(&self) -> ElementCount {
        self.element_count
    }

    /// Returns the simplified FLOP estimate.
    pub fn flops(&self) -> FlopCount {
        self.flops
    }

    /// Returns the simplified HBM byte estimate.
    pub fn hbm_bytes(&self) -> Bytes {
        self.hbm_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ElementSize, ElementwiseTrace, Error, KernelMatrix, KernelScalar, KernelVector,
        MatrixColumns, MatrixRows, MatrixShape, RowReductionTrace, TileColumns, TilePlan, TileRows,
        TileShape, TiledMatVecTrace,
    };

    fn vector(values: impl IntoIterator<Item = KernelScalar>) -> Result<KernelVector, Error> {
        KernelVector::from_values(values)
    }

    fn matrix() -> Result<KernelMatrix, Error> {
        KernelMatrix::from_rows([
            vector([
                KernelScalar::try_from(1.0)?,
                KernelScalar::try_from(2.0)?,
                KernelScalar::try_from(3.0)?,
            ])?,
            vector([
                KernelScalar::try_from(4.0)?,
                KernelScalar::try_from(5.0)?,
                KernelScalar::try_from(6.0)?,
            ])?,
        ])
    }

    #[test]
    fn matrix_shape_counts_elements_through_typed_multiplication() -> Result<(), Error> {
        let shape = MatrixShape::new(MatrixRows::try_from(2)?, MatrixColumns::try_from(3)?);

        assert_eq!(format!("{}", shape.element_count()?), "6");
        Ok(())
    }

    #[test]
    fn tile_plan_covers_edge_tiles() -> Result<(), Error> {
        let plan = TilePlan::new(
            MatrixShape::new(MatrixRows::try_from(3)?, MatrixColumns::try_from(5)?),
            TileShape::new(TileRows::try_from(2)?, TileColumns::try_from(2)?),
        )?;

        assert_eq!(plan.windows().count(), 6);
        Ok(())
    }

    #[test]
    fn row_reduction_sums_values_with_typed_accumulator() -> Result<(), Error> {
        let trace = RowReductionTrace::sum(vector([
            KernelScalar::try_from(1.0)?,
            KernelScalar::try_from(2.0)?,
            KernelScalar::try_from(3.5)?,
        ])?)?;

        assert_eq!(trace.output(), KernelScalar::try_from(6.5)?);
        Ok(())
    }

    #[test]
    fn tiled_matvec_matches_hand_calculation() -> Result<(), Error> {
        let trace = TiledMatVecTrace::run(
            matrix()?,
            vector([
                KernelScalar::try_from(1.0)?,
                KernelScalar::try_from(0.5)?,
                KernelScalar::try_from(2.0)?,
            ])?,
            TileShape::new(TileRows::try_from(1)?, TileColumns::try_from(2)?),
            ElementSize::float32(),
        )?;
        let values = trace
            .output()
            .values()
            .map(KernelScalar::as_f64)
            .collect::<Vec<_>>();

        assert_eq!(values, [8.0, 18.5]);
        Ok(())
    }

    #[test]
    fn tiled_matvec_rejects_vector_shape_mismatch() -> Result<(), Error> {
        let rejected = TiledMatVecTrace::run(
            matrix()?,
            vector([KernelScalar::try_from(1.0)?, KernelScalar::try_from(2.0)?])?,
            TileShape::new(TileRows::try_from(1)?, TileColumns::try_from(1)?),
            ElementSize::float32(),
        );

        assert!(matches!(rejected, Err(Error::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn elementwise_trace_reports_bytes_and_flops() -> Result<(), Error> {
        let trace = ElementwiseTrace::gelu(
            vector([
                KernelScalar::try_from(-1.0)?,
                KernelScalar::try_from(0.0)?,
                KernelScalar::try_from(1.0)?,
            ])?,
            ElementSize::float32(),
        )?;

        assert_eq!(format!("{}", trace.element_count()), "3");
        assert_eq!(format!("{}", trace.flops()), "6 FLOPs");
        assert_eq!(format!("{}", trace.hbm_bytes()), "24 bytes");
        Ok(())
    }
}
