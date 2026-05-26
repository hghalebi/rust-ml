//! Error types for the single-neuron teaching crate.

use thiserror::Error;

/// Failures that preserve the learner-facing reason for a neuron operation.
#[derive(Debug, Error, PartialEq)]
pub enum NeuronError {
    /// A collection that must contain at least one value was empty.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable details about the missing data.
        details: &'static str,
    },

    /// Two learner-visible vectors had incompatible lengths.
    #[error(
        "dimension mismatch in {operation}: {left_label} has length {left_len}, \
         {right_label} has length {right_len}. hint: {hint}"
    )]
    DimensionMismatch {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable label for the left value.
        left_label: &'static str,
        /// Length of the left value.
        left_len: usize,
        /// Human-readable label for the right value.
        right_label: &'static str,
        /// Length of the right value.
        right_len: usize,
        /// Guidance for fixing the mismatch.
        hint: &'static str,
    },

    /// A scalar role received `NaN` or infinity.
    #[error("{role} must be finite, got {value}")]
    NonFiniteValue {
        /// The semantic role being constructed.
        role: &'static str,
        /// The invalid raw value.
        value: f64,
    },

    /// A probability-like target was outside the beginner lesson range.
    #[error("target must be between 0 and 1 inclusive, got {value}")]
    TargetOutOfRange {
        /// The invalid target value.
        value: f64,
    },

    /// Learning rate construction failed.
    #[error("learning rate must be finite and greater than zero, got {value}")]
    InvalidLearningRate {
        /// The invalid learning-rate value.
        value: f64,
    },

    /// A computed value became non-finite.
    #[error("numerical issue in {operation}: {details}")]
    NumericalIssue {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable explanation of the numerical problem.
        details: &'static str,
    },
}
