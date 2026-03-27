//! A small, typed Transformer teaching crate.
//!
//! The crate is intentionally explicit:
//! - raw math uses [`DenseVector`] and [`DenseMatrix`]
//! - model roles use semantic newtypes such as [`TokenEmbedding`], [`Query`], and [`Value`]
//! - fallible operations return [`ModelError`] with shape-aware diagnostics
//! - the encoder path stays readable enough to map directly onto the paper

pub mod attention;
pub mod error;
pub mod math;
pub mod transformer;
pub mod types;

pub use attention::{
    AttentionHead, KeyLayer, LinearAttentionHead, MultiHeadAttention, OutputLayer, QueryLayer,
    ValueLayer, concat_attention_outputs, scaled_attention_score, softmax, weighted_sum,
};
pub use error::ModelError;
pub use math::{DenseMatrix, DenseVector};
pub use transformer::{
    Encoder, FeedForward, FeedForwardLayer1, FeedForwardLayer2, LayerNorm, PositionalEncodingTable,
    TransformerEncoderBlock, add_sequences, add_token_embeddings,
};
pub use types::{
    AttentionOutput, AttentionScores, AttentionWeights, ConcatenatedHeadOutput,
    FeedForwardProjection1, FeedForwardProjection2, HiddenActivation, Key, KeyProjection,
    NormScale, NormShift, OutputProjection, PositionEncoding, ProjectionBias, Query,
    QueryProjection, TokenEmbedding, TokenSequence, Value, ValueProjection,
};
