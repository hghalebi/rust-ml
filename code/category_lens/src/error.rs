//! Error types for the category-lens teaching crate.

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

/// Semantic role for a rejected name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameRole(&'static str);

impl NameRole {
    pub(crate) fn new(value: &'static str) -> Self {
        Self(value)
    }
}

impl fmt::Display for NameRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// A rejected object or map name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedName(String);

impl RejectedName {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for RejectedName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Failures that keep the object/map composition rule visible.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CategoryLensError {
    /// A learner supplied an empty object or map name.
    #[error("{role} name cannot be empty")]
    EmptyName {
        /// The semantic role being constructed.
        role: NameRole,
    },

    /// A learner supplied a name with unsupported control characters.
    #[error("{role} name contains unsupported characters: {value}")]
    InvalidName {
        /// The semantic role being constructed.
        role: NameRole,
        /// The rejected name.
        value: RejectedName,
    },

    /// Two maps cannot compose because the middle object does not match.
    #[error("composition mismatch in {operation}: {details}")]
    CompositionMismatch {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },

    /// A composition trace needs at least one map.
    #[error("empty composition in {operation}: {details}")]
    EmptyComposition {
        /// The operation that failed.
        operation: OperationName,
        /// Human-readable explanation.
        details: ErrorDetails,
    },
}

impl CategoryLensError {
    pub(crate) fn empty_name(role: &'static str) -> Self {
        Self::EmptyName {
            role: NameRole::new(role),
        }
    }

    pub(crate) fn invalid_name(role: &'static str, value: String) -> Self {
        Self::InvalidName {
            role: NameRole::new(role),
            value: RejectedName::new(value),
        }
    }

    pub(crate) fn composition_mismatch(operation: &'static str, details: &'static str) -> Self {
        Self::CompositionMismatch {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }

    pub(crate) fn empty_composition(operation: &'static str, details: &'static str) -> Self {
        Self::EmptyComposition {
            operation: OperationName::new(operation),
            details: ErrorDetails::new(details),
        }
    }
}
