//! Error types for the single-neuron teaching crate.

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

/// Learner-facing label for one side of a length comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthLabel(&'static str);

impl LengthLabel {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for LengthLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Length observed while validating an aligned vector operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VectorLength(usize);

impl VectorLength {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for VectorLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Failures that preserve the learner-facing reason for a neuron operation.
#[derive(Debug, Error, PartialEq)]
pub enum NeuronError {
    /// A collection that must contain at least one value was empty.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details about the missing data.
        details: ErrorDetails,
    },

    /// Two learner-visible vectors had incompatible lengths.
    #[error(
        "dimension mismatch in {operation}: {left_label} has length {left_len}, \
         {right_label} has length {right_len}. hint: {hint}"
    )]
    DimensionMismatch {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable label for the left value.
        left_label: LengthLabel,
        /// Length of the left value.
        left_len: VectorLength,
        /// Human-readable label for the right value.
        right_label: LengthLabel,
        /// Length of the right value.
        right_len: VectorLength,
        /// Guidance for fixing the mismatch.
        hint: ErrorDetails,
    },

    /// A scalar role received `NaN` or infinity.
    #[error("{role} must be finite, got {value}")]
    NonFiniteValue {
        /// The semantic role being constructed.
        role: ValueRole,
        /// The invalid raw value.
        value: RejectedScalar,
    },

    /// A probability-like target was outside the beginner lesson range.
    #[error("target must be between 0 and 1 inclusive, got {value}")]
    TargetOutOfRange {
        /// The invalid target value.
        value: RejectedScalar,
    },

    /// Learning rate construction failed.
    #[error("learning rate must be finite and greater than zero, got {value}")]
    InvalidLearningRate {
        /// The invalid learning-rate value.
        value: RejectedScalar,
    },

    /// A computed value became non-finite.
    #[error("numerical issue in {operation}: {details}")]
    NumericalIssue {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation of the numerical problem.
        details: ErrorDetails,
    },
}

impl NeuronError {
    pub(crate) fn empty_input(operation: &'static str, details: &'static str) -> Self {
        Self::EmptyInput {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn dimension_mismatch(
        operation: &'static str,
        left_label: &'static str,
        left_len: usize,
        right_label: &'static str,
        right_len: usize,
        hint: &'static str,
    ) -> Self {
        Self::DimensionMismatch {
            operation: OperationName::new(operation),
            left_label: LengthLabel::new(left_label),
            left_len: VectorLength::new(left_len),
            right_label: LengthLabel::new(right_label),
            right_len: VectorLength::new(right_len),
            hint: ErrorDetails::new(hint),
        }
    }

    pub(crate) fn non_finite_value(role: &'static str, value: f64) -> Self {
        Self::NonFiniteValue {
            role: ValueRole::new(role),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn target_out_of_range(value: f64) -> Self {
        Self::TargetOutOfRange {
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn invalid_learning_rate(value: f64) -> Self {
        Self::InvalidLearningRate {
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn numerical_issue(operation: &'static str, details: &'static str) -> Self {
        Self::NumericalIssue {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }
}
