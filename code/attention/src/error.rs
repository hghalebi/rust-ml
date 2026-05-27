//! Error types for the beginner attention teaching crate.

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
pub struct RejectedScalar(f64);

impl RejectedScalar {
    pub(crate) fn new(value: f64) -> Self {
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

/// Shape observed while validating an attention operation.
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

/// Requested token position in a learner-facing sequence error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequestedTokenIndex(usize);

impl RequestedTokenIndex {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for RequestedTokenIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Available token count in a learner-facing sequence error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SequenceLength(usize);

impl SequenceLength {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for SequenceLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Shape, range, and numerical failures for the attention examples.
#[derive(Debug, Error, PartialEq)]
pub enum AttentionError {
    /// A required vector, matrix, or sequence was empty.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },

    /// A scalar role received `NaN` or infinity.
    #[error("{role} must be finite, got {value}")]
    NonFiniteValue {
        /// The semantic role being constructed.
        role: ValueRole,
        /// The invalid raw value.
        value: RejectedScalar,
    },

    /// A scalar value violated its semantic range.
    #[error("{role} must be in range {range}, got {value}")]
    OutOfRange {
        /// The semantic role being constructed.
        role: ValueRole,
        /// Human-readable allowed range.
        range: ValueRange,
        /// The invalid raw value.
        value: RejectedScalar,
    },

    /// A matrix was built from rows with inconsistent lengths.
    #[error("invalid matrix data in {operation}: {details}")]
    InvalidMatrixData {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },

    /// Two values had incompatible dimensions.
    #[error(
        "dimension mismatch in {operation}: {left_label} has shape {left_shape}, \
         {right_label} has shape {right_shape}. hint: {hint}"
    )]
    DimensionMismatch {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable label for the left value.
        left_label: ShapeLabel,
        /// Shape of the left value.
        left_shape: ShapeSignature,
        /// Human-readable label for the right value.
        right_label: ShapeLabel,
        /// Shape of the right value.
        right_shape: ShapeSignature,
        /// Guidance for fixing the mismatch.
        hint: ErrorDetails,
    },

    /// A projection layer was configured incorrectly.
    #[error("invalid projection in {operation}: {details}")]
    InvalidProjection {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },

    /// A token index did not exist in the sequence.
    #[error("invalid token index in {operation}: index {index}, sequence length {len}")]
    InvalidTokenIndex {
        /// The operation that failed.
        operation: OperationName,
        /// Requested token index.
        index: RequestedTokenIndex,
        /// Available sequence length.
        len: SequenceLength,
    },

    /// A numerical issue prevented a stable computation.
    #[error("numerical issue in {operation}: {details}")]
    NumericalIssue {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },
}

impl AttentionError {
    pub(crate) fn empty_input(operation: &'static str, details: &'static str) -> Self {
        Self::EmptyInput {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn non_finite_value(role: &'static str, value: f64) -> Self {
        Self::NonFiniteValue {
            role: ValueRole::new(role),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn out_of_range(role: &'static str, range: &'static str, value: f64) -> Self {
        Self::OutOfRange {
            role: ValueRole::new(role),
            range: ValueRange::new(range),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn invalid_matrix_data(operation: &'static str, details: &'static str) -> Self {
        Self::InvalidMatrixData {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
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

    pub(crate) fn invalid_projection(operation: &'static str, details: &'static str) -> Self {
        Self::InvalidProjection {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn invalid_token_index(operation: &'static str, index: usize, len: usize) -> Self {
        Self::InvalidTokenIndex {
            operation: OperationName::new(operation),
            index: RequestedTokenIndex::new(index),
            len: SequenceLength::new(len),
        }
    }

    pub(crate) fn numerical_issue(operation: &'static str, details: &'static str) -> Self {
        Self::NumericalIssue {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }
}
