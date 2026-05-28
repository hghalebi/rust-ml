//! Error types for the teaching Transformer crate.

use std::fmt;

use thiserror::Error;

/// Name of the operation that produced an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationName(&'static str);

impl OperationName {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for OperationName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Short learner-facing explanation attached to an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorDetails(&'static str);

impl ErrorDetails {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for ErrorDetails {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Name of a scalar role being validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRole(&'static str);

impl ValueRole {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for ValueRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Human-readable range expected for a scalar role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRange(&'static str);

impl ValueRange {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for ValueRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Rejected finite or non-finite scalar observed at a validation boundary.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RejectedScalar(f32);

impl RejectedScalar {
    pub(crate) fn new(value: f32) -> Self {
        Self(value)
    }
}

impl fmt::Display for RejectedScalar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Learner-facing label for one side of a shape comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeLabel(&'static str);

impl ShapeLabel {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for ShapeLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Shape observed while validating a Transformer operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeSignature(Vec<usize>);

impl ShapeSignature {
    pub(crate) fn new(value: Vec<usize>) -> Self {
        Self(value)
    }
}

impl fmt::Display for ShapeSignature {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

/// Matrix row count in a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatrixRowCount(usize);

impl MatrixRowCount {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for MatrixRowCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Matrix column count in a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatrixColumnCount(usize);

impl MatrixColumnCount {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for MatrixColumnCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Scalar data length observed while validating matrix data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatrixStorageLength(usize);

impl MatrixStorageLength {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for MatrixStorageLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Requested index in a vector, token sequence, or matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequestedIndex(usize);

impl RequestedIndex {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for RequestedIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Available length or dimension in an index/dimension diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AvailableLength(usize);

impl AvailableLength {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for AvailableLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Shape-aware failures for the teaching Transformer implementation.
#[derive(Debug, Error, PartialEq)]
pub enum ModelError {
    /// A scalar role received `NaN` or infinity.
    #[error("{role} must be finite, got {value}")]
    NonFiniteValue {
        /// The semantic role being constructed.
        role: ValueRole,
        /// The rejected raw value.
        value: RejectedScalar,
    },

    /// A scalar value violated its semantic range.
    #[error("{role} must be in range {range}, got {value}")]
    OutOfRange {
        /// The semantic role being constructed.
        role: ValueRole,
        /// Human-readable allowed range.
        range: ValueRange,
        /// The rejected raw value.
        value: RejectedScalar,
    },

    /// Two operands had incompatible shapes for an operation.
    #[error(
        "dimension mismatch in {operation}: {left_label} has shape {left_shape}, \
         {right_label} has shape {right_shape}. hint: {hint}"
    )]
    DimensionMismatch {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable label for the left operand.
        left_label: ShapeLabel,
        /// Shape of the left operand.
        left_shape: ShapeSignature,
        /// Human-readable label for the right operand.
        right_label: ShapeLabel,
        /// Shape of the right operand.
        right_shape: ShapeSignature,
        /// Guidance for fixing the mismatch.
        hint: ErrorDetails,
    },

    /// An operation requiring data received an empty input.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details about the missing data.
        details: ErrorDetails,
    },

    /// A matrix was built from incompatible dimensions and scalar data length.
    #[error(
        "invalid matrix data in {operation}: rows={rows}, cols={cols}, data_len={data_len}. \
         hint: rows * cols must equal data_len"
    )]
    InvalidMatrixData {
        /// The operation that failed.
        operation: OperationName,
        /// Requested matrix row count.
        rows: MatrixRowCount,
        /// Requested matrix column count.
        cols: MatrixColumnCount,
        /// Actual scalar element count.
        data_len: MatrixStorageLength,
    },

    /// A token sequence contained inconsistent token widths.
    #[error(
        "inconsistent token dimensions in {operation}: token at index {token_index} has dim {actual_dim}, \
         expected dim {expected_dim}. hint: all tokens in a sequence must share d_model"
    )]
    InconsistentTokenDimensions {
        /// The operation that failed.
        operation: OperationName,
        /// Index of the offending token.
        token_index: RequestedIndex,
        /// Expected token width.
        expected_dim: AvailableLength,
        /// Actual token width.
        actual_dim: AvailableLength,
    },

    /// A token index did not exist in a sequence.
    #[error("invalid token index in {operation}: index {index}, sequence length {len}")]
    InvalidTokenIndex {
        /// The operation that failed.
        operation: OperationName,
        /// Requested token index.
        index: RequestedIndex,
        /// Available token count.
        len: AvailableLength,
    },

    /// A vector index did not exist.
    #[error("invalid vector index in {operation}: index {index}, vector length {len}")]
    InvalidVectorIndex {
        /// The operation that failed.
        operation: OperationName,
        /// Requested vector index.
        index: RequestedIndex,
        /// Available vector length.
        len: AvailableLength,
    },

    /// A matrix index did not exist.
    #[error(
        "invalid matrix index in {operation}: row {row}, col {col}, matrix shape [{rows}, {cols}]"
    )]
    InvalidMatrixIndex {
        /// The operation that failed.
        operation: OperationName,
        /// Requested row index.
        row: RequestedIndex,
        /// Requested column index.
        col: RequestedIndex,
        /// Available row count.
        rows: MatrixRowCount,
        /// Available column count.
        cols: MatrixColumnCount,
    },

    /// A projection layer was configured with incompatible weights and bias.
    #[error("invalid projection in {operation}: {details}")]
    InvalidProjection {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation of the invalid configuration.
        details: ErrorDetails,
    },

    /// An attention or encoder configuration was internally inconsistent.
    #[error("invalid head configuration in {operation}: {details}")]
    InvalidHeadConfiguration {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation of the invalid configuration.
        details: ErrorDetails,
    },

    /// A numerical issue prevented a stable computation.
    #[error("numerical issue in {operation}: {details}")]
    NumericalIssue {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation of the numerical problem.
        details: ErrorDetails,
    },

    /// Restricted or private Transformer evidence tried to enter learner-facing public material.
    #[error("invalid public trace in {operation}: {details}")]
    InvalidPublicTrace {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation of the invalid release attempt.
        details: ErrorDetails,
    },
}

impl ModelError {
    pub(crate) fn non_finite_value(role: &'static str, value: f32) -> Self {
        Self::NonFiniteValue {
            role: ValueRole::new(role),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn out_of_range(role: &'static str, range: &'static str, value: f32) -> Self {
        Self::OutOfRange {
            role: ValueRole::new(role),
            range: ValueRange::new(range),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn dimension_mismatch(
        operation: &'static str,
        left_label: &'static str,
        left_shape: Vec<usize>,
        right_label: &'static str,
        right_shape: Vec<usize>,
        hint: &'static str,
    ) -> Self {
        Self::DimensionMismatch {
            operation: OperationName::new(operation),
            left_label: ShapeLabel::new(left_label),
            left_shape: ShapeSignature::new(left_shape),
            right_label: ShapeLabel::new(right_label),
            right_shape: ShapeSignature::new(right_shape),
            hint: ErrorDetails::new(hint),
        }
    }

    pub(crate) fn empty_input(operation: &'static str, details: &'static str) -> Self {
        Self::EmptyInput {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn invalid_matrix_data(
        operation: &'static str,
        rows: usize,
        cols: usize,
        data_len: usize,
    ) -> Self {
        Self::InvalidMatrixData {
            operation: OperationName::new(operation),
            rows: MatrixRowCount::new(rows),
            cols: MatrixColumnCount::new(cols),
            data_len: MatrixStorageLength::new(data_len),
        }
    }

    pub(crate) fn inconsistent_token_dimensions(
        operation: &'static str,
        token_index: usize,
        expected_dim: usize,
        actual_dim: usize,
    ) -> Self {
        Self::InconsistentTokenDimensions {
            operation: OperationName::new(operation),
            token_index: RequestedIndex::new(token_index),
            expected_dim: AvailableLength::new(expected_dim),
            actual_dim: AvailableLength::new(actual_dim),
        }
    }

    pub(crate) fn invalid_token_index(operation: &'static str, index: usize, len: usize) -> Self {
        Self::InvalidTokenIndex {
            operation: OperationName::new(operation),
            index: RequestedIndex::new(index),
            len: AvailableLength::new(len),
        }
    }

    pub(crate) fn invalid_vector_index(operation: &'static str, index: usize, len: usize) -> Self {
        Self::InvalidVectorIndex {
            operation: OperationName::new(operation),
            index: RequestedIndex::new(index),
            len: AvailableLength::new(len),
        }
    }

    pub(crate) fn invalid_matrix_index(
        operation: &'static str,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self::InvalidMatrixIndex {
            operation: OperationName::new(operation),
            row: RequestedIndex::new(row),
            col: RequestedIndex::new(col),
            rows: MatrixRowCount::new(rows),
            cols: MatrixColumnCount::new(cols),
        }
    }

    pub(crate) fn invalid_projection(operation: &'static str, details: &'static str) -> Self {
        Self::InvalidProjection {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn invalid_head_configuration(
        operation: &'static str,
        details: &'static str,
    ) -> Self {
        Self::InvalidHeadConfiguration {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn numerical_issue(operation: &'static str, details: &'static str) -> Self {
        Self::NumericalIssue {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn invalid_public_trace(operation: &'static str, details: &'static str) -> Self {
        Self::InvalidPublicTrace {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }
}
