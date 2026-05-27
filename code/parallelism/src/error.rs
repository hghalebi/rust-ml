//! Error types for the parallelism teaching crate.

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

/// Human-readable range expected for a scalar or count role.
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

/// Rejected count-like value observed at a validation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RejectedCount(usize);

impl RejectedCount {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

impl fmt::Display for RejectedCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Validation and planning failures for typed parallelism examples.
#[derive(Debug, Error, PartialEq)]
pub enum ParallelismError {
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

    /// A count-like value violated its semantic range.
    #[error("{role} must be in range {range}, got {value}")]
    CountOutOfRange {
        /// The semantic role being constructed.
        role: ValueRole,
        /// Human-readable allowed range.
        range: ValueRange,
        /// The rejected raw value.
        value: RejectedCount,
    },

    /// A scalar role received `NaN` or infinity.
    #[error("{role} must be finite, got {value}")]
    NonFiniteValue {
        /// The semantic role being constructed.
        role: ValueRole,
        /// The rejected raw value.
        value: RejectedScalar,
    },

    /// A checked calculation overflowed.
    #[error("overflow in {operation}: {details}")]
    Overflow {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details.
        details: ErrorDetails,
    },

    /// A split requires exact divisibility but the counts do not divide.
    #[error("uneven split in {operation}: {details}")]
    UnevenSplit {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details.
        details: ErrorDetails,
    },

    /// A rank was used with a world that does not contain it.
    #[error("rank mismatch in {operation}: {details}")]
    RankMismatch {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable details.
        details: ErrorDetails,
    },
}

impl ParallelismError {
    pub(crate) fn empty_input(operation: &'static str, details: &'static str) -> Self {
        Self::EmptyInput {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn count_out_of_range(
        role: &'static str,
        range: &'static str,
        value: usize,
    ) -> Self {
        Self::CountOutOfRange {
            role: ValueRole::new(role),
            range: ValueRange::new(range),
            value: RejectedCount::new(value),
        }
    }

    pub(crate) fn non_finite_value(role: &'static str, value: f64) -> Self {
        Self::NonFiniteValue {
            role: ValueRole::new(role),
            value: RejectedScalar::new(value),
        }
    }

    pub(crate) fn overflow(operation: &'static str, details: &'static str) -> Self {
        Self::Overflow {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn uneven_split(operation: &'static str, details: &'static str) -> Self {
        Self::UnevenSplit {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }
}
