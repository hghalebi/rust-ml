//! Error types for the evaluation-harness teaching crate.

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

/// Name of a value role being validated.
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

/// Failures for deterministic evaluation examples.
#[derive(Debug, Error, PartialEq)]
pub enum EvaluationError {
    /// A required value was empty.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details.
        details: ErrorDetails,
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

    /// A scalar role received `NaN` or infinity.
    #[error("{role} must be finite, got {value}")]
    NonFiniteValue {
        /// The semantic role being constructed.
        role: ValueRole,
        /// The rejected raw value.
        value: RejectedScalar,
    },

    /// A prediction was attached to the wrong example.
    #[error("example mismatch in {operation}: {details}")]
    ExampleMismatch {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details.
        details: ErrorDetails,
    },

    /// The same example appeared twice in a report.
    #[error("duplicate example in {operation}: {details}")]
    DuplicateExample {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details.
        details: ErrorDetails,
    },
}

impl EvaluationError {
    pub(crate) fn empty_input(operation: &'static str, details: &'static str) -> Self {
        Self::EmptyInput {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn out_of_range(role: &'static str, range: &'static str, value: f64) -> Self {
        Self::OutOfRange {
            role: ValueRole::new(role),
            range: ValueRange::new(range),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn non_finite_value(role: &'static str, value: f64) -> Self {
        Self::NonFiniteValue {
            role: ValueRole::new(role),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn example_mismatch(operation: &'static str, details: &'static str) -> Self {
        Self::ExampleMismatch {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn duplicate_example(operation: &'static str, details: &'static str) -> Self {
        Self::DuplicateExample {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }
}
