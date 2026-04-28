//! Small helpers for explaining next-token targets and cross-entropy loss.

use thiserror::Error;

/// One vocabulary index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenId(pub usize);

/// One next-token training example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextTokenExample {
    /// The already-seen context tokens.
    pub context: Vec<TokenId>,
    /// The token that should come next.
    pub target: TokenId,
}

impl NextTokenExample {
    /// Creates one next-token example.
    pub fn new(context: Vec<TokenId>, target: TokenId) -> Self {
        Self { context, target }
    }
}

/// Errors returned by token-target utilities.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenTargetError {
    /// At least one logit is required to define a probability distribution.
    #[error("logits must not be empty")]
    EmptyLogits,
    /// The target token must refer to an existing class index.
    #[error("target token index {target} is out of range for {classes} logits")]
    TargetOutOfRange { target: usize, classes: usize },
}

fn validate_logits_and_target(logits: &[f64], target: TokenId) -> Result<(), TokenTargetError> {
    if logits.is_empty() {
        return Err(TokenTargetError::EmptyLogits);
    }

    if target.0 >= logits.len() {
        return Err(TokenTargetError::TargetOutOfRange {
            target: target.0,
            classes: logits.len(),
        });
    }

    Ok(())
}

/// Computes a stable softmax distribution from logits.
pub fn softmax(logits: &[f64]) -> Result<Vec<f64>, TokenTargetError> {
    if logits.is_empty() {
        return Err(TokenTargetError::EmptyLogits);
    }

    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exponentials.iter().sum();

    Ok(exponentials.into_iter().map(|value| value / sum).collect())
}

/// Computes the cross-entropy loss for the correct token.
pub fn cross_entropy_loss(logits: &[f64], target: TokenId) -> Result<f64, TokenTargetError> {
    validate_logits_and_target(logits, target)?;
    let probabilities = softmax(logits)?;
    Ok(-probabilities[target.0].ln())
}

/// Computes the logits gradient `p - one_hot(target)`.
pub fn cross_entropy_gradient(
    logits: &[f64],
    target: TokenId,
) -> Result<Vec<f64>, TokenTargetError> {
    validate_logits_and_target(logits, target)?;
    let mut probabilities = softmax(logits)?;
    probabilities[target.0] -= 1.0;
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        let diff = (left - right).abs();
        assert!(diff < 1e-9, "left={left}, right={right}, diff={diff}");
    }

    #[test]
    fn softmax_probabilities_sum_to_one() {
        let probabilities = match softmax(&[2.0, 1.0, 0.0]) {
            Ok(probabilities) => probabilities,
            Err(error) => panic!("unexpected error: {error}"),
        };

        let sum: f64 = probabilities.iter().sum();
        assert_close(sum, 1.0);
    }

    #[test]
    fn cross_entropy_prefers_better_target_logit() {
        let better = match cross_entropy_loss(&[2.5, 0.1, -0.3], TokenId(0)) {
            Ok(loss) => loss,
            Err(error) => panic!("unexpected error: {error}"),
        };
        let worse = match cross_entropy_loss(&[0.1, 2.5, -0.3], TokenId(0)) {
            Ok(loss) => loss,
            Err(error) => panic!("unexpected error: {error}"),
        };

        assert!(better < worse, "better={better}, worse={worse}");
    }

    #[test]
    fn cross_entropy_gradient_matches_probability_minus_one_hot() {
        let logits = [1.0, 0.0, -1.0];
        let probabilities = match softmax(&logits) {
            Ok(probabilities) => probabilities,
            Err(error) => panic!("unexpected error: {error}"),
        };
        let gradient = match cross_entropy_gradient(&logits, TokenId(0)) {
            Ok(gradient) => gradient,
            Err(error) => panic!("unexpected error: {error}"),
        };

        assert_close(gradient[0], probabilities[0] - 1.0);
        assert_close(gradient[1], probabilities[1]);
        assert_close(gradient[2], probabilities[2]);
    }

    #[test]
    fn token_target_errors_when_index_is_out_of_range() {
        let result = cross_entropy_loss(&[0.2, 0.8], TokenId(2));
        assert_eq!(
            result,
            Err(TokenTargetError::TargetOutOfRange {
                target: 2,
                classes: 2,
            })
        );
    }
}
