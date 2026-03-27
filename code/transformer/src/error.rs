//! Error types for the teaching Transformer crate.

use thiserror::Error;

/// Shape-aware failures for the teaching Transformer implementation.
#[derive(Debug, Error)]
pub enum ModelError {
    /// Two operands had incompatible shapes for an operation.
    #[error(
        "dimension mismatch in {operation}: {left_label} has shape {left_shape:?}, \
         {right_label} has shape {right_shape:?}. hint: {hint}"
    )]
    DimensionMismatch {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable label for the left operand.
        left_label: &'static str,
        /// Shape of the left operand.
        left_shape: Vec<usize>,
        /// Human-readable label for the right operand.
        right_label: &'static str,
        /// Shape of the right operand.
        right_shape: Vec<usize>,
        /// Guidance for fixing the mismatch.
        hint: &'static str,
    },

    /// An operation requiring data received an empty input.
    #[error("empty input in {operation}: {details}")]
    EmptyInput {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable details about the missing data.
        details: &'static str,
    },

    /// A matrix was built from incompatible dimensions and raw storage.
    #[error(
        "invalid matrix data in {operation}: rows={rows}, cols={cols}, data_len={data_len}. \
         hint: rows * cols must equal data_len"
    )]
    InvalidMatrixData {
        /// The operation that failed.
        operation: &'static str,
        /// Requested matrix row count.
        rows: usize,
        /// Requested matrix column count.
        cols: usize,
        /// Actual raw element count.
        data_len: usize,
    },

    /// A token sequence contained inconsistent token widths.
    #[error(
        "inconsistent token dimensions in {operation}: token at index {token_index} has dim {actual_dim}, \
         expected dim {expected_dim}. hint: all tokens in a sequence must share d_model"
    )]
    InconsistentTokenDimensions {
        /// The operation that failed.
        operation: &'static str,
        /// Index of the offending token.
        token_index: usize,
        /// Expected token width.
        expected_dim: usize,
        /// Actual token width.
        actual_dim: usize,
    },

    /// A projection layer was configured with incompatible weights and bias.
    #[error("invalid projection in {operation}: {details}")]
    InvalidProjection {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable explanation of the invalid configuration.
        details: &'static str,
    },

    /// An attention or encoder configuration was internally inconsistent.
    #[error("invalid head configuration in {operation}: {details}")]
    InvalidHeadConfiguration {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable explanation of the invalid configuration.
        details: &'static str,
    },

    /// A numerical issue prevented a stable computation.
    #[error("numerical issue in {operation}: {details}")]
    NumericalIssue {
        /// The operation that failed.
        operation: &'static str,
        /// Human-readable explanation of the numerical problem.
        details: &'static str,
    },
}
