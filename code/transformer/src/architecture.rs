//! Typed Transformer architecture configuration.
//!
//! CS336-style architecture work starts with hyperparameters, but the learner
//! should not pass loose numbers through the model. This module turns those
//! numbers into semantic values before estimating shape and parameter budgets.

use std::{
    fmt,
    ops::{Add, Div, Mul},
};

use crate::{error::ModelError, math::VectorLength};

fn nonzero_count(
    operation: &'static str,
    role: &'static str,
    value: usize,
) -> Result<usize, ModelError> {
    if value == 0 {
        return Err(ModelError::empty_input(operation, role));
    }

    Ok(value)
}

fn checked_product(operation: &'static str, left: u128, right: u128) -> Result<u128, ModelError> {
    left.checked_mul(right).ok_or(ModelError::numerical_issue(
        operation,
        "parameter count exceeded u128",
    ))
}

/// Number of encoder blocks in the Transformer stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LayerCount(usize);

impl LayerCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for LayerCount {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_count(
            "LayerCount::try_from",
            "layer count must be greater than zero",
            value,
        )?))
    }
}

impl fmt::Display for LayerCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of attention heads in one multi-head attention block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HeadCount(usize);

impl HeadCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for HeadCount {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_count(
            "HeadCount::try_from",
            "head count must be greater than zero",
            value,
        )?))
    }
}

impl fmt::Display for HeadCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Width of the hidden feed-forward projection inside one Transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeedForwardWidth(VectorLength);

impl FeedForwardWidth {
    /// Creates a feed-forward width from a checked non-zero vector length.
    pub fn new(value: VectorLength) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0.as_usize()
    }
}

impl TryFrom<usize> for FeedForwardWidth {
    type Error = ModelError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(VectorLength::try_from(value)?))
    }
}

impl fmt::Display for FeedForwardWidth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl Div<HeadCount> for VectorLength {
    type Output = Result<VectorLength, ModelError>;

    fn div(self, right: HeadCount) -> Self::Output {
        let width = self.as_usize();
        let heads = right.as_usize();

        if !width.is_multiple_of(heads) {
            return Err(ModelError::invalid_head_configuration(
                "VectorLength::div",
                "model width must divide evenly across attention heads",
            ));
        }

        VectorLength::try_from(width / heads)
    }
}

/// Estimated number of learned scalar parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParameterCount(u128);

impl ParameterCount {
    fn from_raw(operation: &'static str, value: u128) -> Result<Self, ModelError> {
        if value == 0 {
            return Err(ModelError::empty_input(
                operation,
                "parameter count must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    fn as_u128(self) -> u128 {
        self.0
    }
}

impl TryFrom<u128> for ParameterCount {
    type Error = ModelError;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Self::from_raw("ParameterCount::try_from", value)
    }
}

impl fmt::Display for ParameterCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl Add for ParameterCount {
    type Output = Result<ParameterCount, ModelError>;

    fn add(self, right: ParameterCount) -> Self::Output {
        let total =
            self.as_u128()
                .checked_add(right.as_u128())
                .ok_or(ModelError::numerical_issue(
                    "ParameterCount::add",
                    "parameter count exceeded u128",
                ))?;

        ParameterCount::from_raw("ParameterCount::add", total)
    }
}

impl Mul<LayerCount> for ParameterCount {
    type Output = Result<ParameterCount, ModelError>;

    fn mul(self, right: LayerCount) -> Self::Output {
        let total = checked_product(
            "ParameterCount::mul",
            self.as_u128(),
            right.as_usize() as u128,
        )?;

        ParameterCount::from_raw("ParameterCount::mul", total)
    }
}

/// Encoder architecture hyperparameters after validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformerConfig {
    model_width: VectorLength,
    layer_count: LayerCount,
    head_count: HeadCount,
    feed_forward_width: FeedForwardWidth,
}

impl TransformerConfig {
    /// Creates a checked architecture configuration.
    ///
    /// The model width must divide evenly across attention heads. That
    /// invariant makes each head's query/key/value width a real typed value
    /// instead of a runtime accident.
    pub fn new(
        model_width: VectorLength,
        layer_count: LayerCount,
        head_count: HeadCount,
        feed_forward_width: FeedForwardWidth,
    ) -> Result<Self, ModelError> {
        let _head_width = (model_width / head_count)?;

        Ok(Self {
            model_width,
            layer_count,
            head_count,
            feed_forward_width,
        })
    }

    /// Returns the shared token width, usually called `d_model`.
    pub fn model_width(self) -> VectorLength {
        self.model_width
    }

    /// Returns the number of encoder blocks.
    pub fn layer_count(self) -> LayerCount {
        self.layer_count
    }

    /// Returns the number of attention heads per block.
    pub fn head_count(self) -> HeadCount {
        self.head_count
    }

    /// Returns the hidden width of the feed-forward map.
    pub fn feed_forward_width(self) -> FeedForwardWidth {
        self.feed_forward_width
    }

    /// Returns the per-head attention width.
    pub fn attention_head_width(self) -> Result<VectorLength, ModelError> {
        self.model_width / self.head_count
    }

    /// Estimates learned parameters for an encoder stack without embeddings.
    ///
    /// This intentionally omits token embeddings and task heads. The estimate
    /// focuses on one repeated encoder block:
    ///
    /// ```text
    /// q/k/v/out projections + feed-forward projections + two layer norms
    /// ```
    pub fn encoder_parameter_estimate(self) -> Result<ParameterCount, ModelError> {
        let model_width = self.model_width.as_usize() as u128;
        let feed_forward_width = self.feed_forward_width.as_usize() as u128;

        let attention_weights = ParameterCount::from_raw(
            "TransformerConfig::encoder_parameter_estimate",
            checked_product(
                "TransformerConfig::encoder_parameter_estimate",
                4,
                checked_product(
                    "TransformerConfig::encoder_parameter_estimate",
                    model_width,
                    model_width,
                )?,
            )?,
        )?;
        let attention_biases = ParameterCount::from_raw(
            "TransformerConfig::encoder_parameter_estimate",
            checked_product(
                "TransformerConfig::encoder_parameter_estimate",
                4,
                model_width,
            )?,
        )?;
        let feed_forward_weights = ParameterCount::from_raw(
            "TransformerConfig::encoder_parameter_estimate",
            checked_product(
                "TransformerConfig::encoder_parameter_estimate",
                2,
                checked_product(
                    "TransformerConfig::encoder_parameter_estimate",
                    model_width,
                    feed_forward_width,
                )?,
            )?,
        )?;
        let feed_forward_biases = ParameterCount::from_raw(
            "TransformerConfig::encoder_parameter_estimate",
            feed_forward_width
                .checked_add(model_width)
                .ok_or(ModelError::numerical_issue(
                    "TransformerConfig::encoder_parameter_estimate",
                    "parameter count exceeded u128",
                ))?,
        )?;
        let layer_norm_parameters = ParameterCount::from_raw(
            "TransformerConfig::encoder_parameter_estimate",
            checked_product(
                "TransformerConfig::encoder_parameter_estimate",
                4,
                model_width,
            )?,
        )?;

        let per_layer = (attention_weights + attention_biases)?;
        let per_layer = (per_layer + feed_forward_weights)?;
        let per_layer = (per_layer + feed_forward_biases)?;
        let per_layer = (per_layer + layer_norm_parameters)?;

        per_layer * self.layer_count
    }
}

#[cfg(test)]
mod tests {
    use super::{FeedForwardWidth, HeadCount, LayerCount, TransformerConfig};
    use crate::{ModelError, VectorLength};

    #[test]
    fn config_computes_head_width_and_parameter_estimate() -> Result<(), ModelError> {
        let config = TransformerConfig::new(
            VectorLength::try_from(8)?,
            LayerCount::try_from(2)?,
            HeadCount::try_from(2)?,
            FeedForwardWidth::try_from(32)?,
        )?;

        assert_eq!(config.attention_head_width()?.to_string(), "4");
        assert_eq!(config.encoder_parameter_estimate()?.to_string(), "1744");
        Ok(())
    }

    #[test]
    fn config_rejects_head_width_that_does_not_divide_model_width() -> Result<(), ModelError> {
        let result = TransformerConfig::new(
            VectorLength::try_from(10)?,
            LayerCount::try_from(1)?,
            HeadCount::try_from(3)?,
            FeedForwardWidth::try_from(40)?,
        );

        assert!(matches!(
            result,
            Err(ModelError::InvalidHeadConfiguration { .. })
        ));
        Ok(())
    }

    #[test]
    fn config_rejects_zero_count_adapters() {
        assert!(matches!(
            HeadCount::try_from(0),
            Err(ModelError::EmptyInput { .. })
        ));
        assert!(matches!(
            LayerCount::try_from(0),
            Err(ModelError::EmptyInput { .. })
        ));
        assert!(matches!(
            FeedForwardWidth::try_from(0),
            Err(ModelError::EmptyInput { .. })
        ));
    }
}
