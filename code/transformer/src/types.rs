//! Semantic model types layered on top of the dense math primitives.

use crate::error::ModelError;
use crate::math::{DenseMatrix, DenseVector};

macro_rules! vector_role {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(pub DenseVector);

        impl $name {
            /// Returns the wrapped vector width.
            pub fn len(&self) -> usize {
                self.0.len()
            }

            /// Returns `true` when the wrapped vector is empty.
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Returns the wrapped vector as a slice.
            pub fn as_slice(&self) -> &[f32] {
                self.0.as_slice()
            }
        }
    };
}

vector_role!(TokenEmbedding, "A token representation inside the model.");
vector_role!(
    PositionEncoding,
    "A positional signal added to a token embedding."
);
vector_role!(Query, "A query vector used to score relevance.");
vector_role!(Key, "A key vector used as a match target for queries.");
vector_role!(
    Value,
    "A value vector that carries the information to be mixed."
);
vector_role!(
    AttentionOutput,
    "The output of one attention head for one token."
);
vector_role!(
    ConcatenatedHeadOutput,
    "The concatenation of several head outputs before output projection."
);
vector_role!(
    HiddenActivation,
    "The hidden activation inside the position-wise feed-forward network."
);
vector_role!(ProjectionBias, "A bias vector used by a projection layer.");
vector_role!(
    NormScale,
    "The learned scale parameter in layer normalization."
);
vector_role!(
    NormShift,
    "The learned shift parameter in layer normalization."
);

/// Raw attention scores before softmax normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionScores(pub Vec<f32>);

/// Attention weights after softmax normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionWeights(pub Vec<f32>);

/// A typed query projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryProjection(pub DenseMatrix);

/// A typed key projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyProjection(pub DenseMatrix);

/// A typed value projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueProjection(pub DenseMatrix);

/// A typed output projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputProjection(pub DenseMatrix);

/// The first feed-forward projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardProjection1(pub DenseMatrix);

/// The second feed-forward projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardProjection2(pub DenseMatrix);

/// A sequence of same-width token embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSequence {
    tokens: Vec<TokenEmbedding>,
    d_model: usize,
}

impl TokenSequence {
    /// Creates a sequence from token embeddings and checks a shared model width.
    pub fn new(tokens: Vec<TokenEmbedding>) -> Result<Self, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput {
                operation: "TokenSequence::new",
                details: "sequence cannot be empty",
            });
        }

        let d_model = tokens[0].len();

        if d_model == 0 {
            return Err(ModelError::EmptyInput {
                operation: "TokenSequence::new",
                details: "token embeddings cannot be empty",
            });
        }

        for (token_index, token) in tokens.iter().enumerate() {
            if token.len() != d_model {
                return Err(ModelError::InconsistentTokenDimensions {
                    operation: "TokenSequence::new",
                    token_index,
                    expected_dim: d_model,
                    actual_dim: token.len(),
                });
            }
        }

        Ok(Self { tokens, d_model })
    }

    /// Returns the token count.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` when the sequence has no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the shared token width.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Returns all tokens.
    pub fn tokens(&self) -> &[TokenEmbedding] {
        &self.tokens
    }

    /// Returns one token by index.
    pub fn token(&self, index: usize) -> &TokenEmbedding {
        &self.tokens[index]
    }

    /// Maps a fallible token transformation over the whole sequence.
    pub fn map_tokens<F>(&self, f: F) -> Result<TokenSequence, ModelError>
    where
        F: Fn(&TokenEmbedding) -> Result<TokenEmbedding, ModelError>,
    {
        let tokens = self.tokens.iter().map(f).collect::<Result<Vec<_>, _>>()?;
        TokenSequence::new(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{TokenEmbedding, TokenSequence};
    use crate::error::ModelError;
    use crate::math::DenseVector;

    #[test]
    fn token_sequence_rejects_empty_sequences() {
        let error = TokenSequence::new(vec![]).expect_err("empty sequence should fail");
        assert!(matches!(error, ModelError::EmptyInput { .. }));
    }

    #[test]
    fn token_sequence_rejects_inconsistent_widths() -> Result<(), ModelError> {
        let token_a = TokenEmbedding(DenseVector::new(vec![1.0, 2.0])?);
        let token_b = TokenEmbedding(DenseVector::new(vec![1.0, 2.0, 3.0])?);

        let error = TokenSequence::new(vec![token_a, token_b])
            .expect_err("inconsistent token widths should fail");
        assert!(matches!(
            error,
            ModelError::InconsistentTokenDimensions { .. }
        ));
        Ok(())
    }
}
