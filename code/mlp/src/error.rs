//! Error types for the tiny MLP teaching crate.

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

/// Shape observed while validating an MLP operation.
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

/// Shape and value failures for the MLP teaching implementation.
#[derive(Debug, Error, PartialEq)]
pub enum MlpError {
    /// A required vector or matrix was empty.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details about the missing data.
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

    /// The output layer was not configured as a one-logit binary classifier.
    #[error("invalid output layer in {operation}: {details}")]
    InvalidOutputLayer {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },

    /// Restricted or private MLP evidence tried to enter a learner-facing public trace.
    #[error("invalid public trace in {operation}: {details}")]
    InvalidPublicTrace {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },
}

impl MlpError {
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

    pub(crate) fn invalid_output_layer(operation: &'static str, details: &'static str) -> Self {
        Self::InvalidOutputLayer {
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
