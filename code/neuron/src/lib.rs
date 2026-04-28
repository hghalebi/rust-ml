//! A small, typed neuron-training teaching crate.
//!
//! The crate is intentionally explicit:
//! - one neuron is enough to show a complete forward and backward pass
//! - semantic wrappers keep training roles visible in the type names
//! - SGD and epoch loops make the training process concrete
//! - token-target helpers provide the first bridge from scalar labels to
//!   next-token losses

pub mod bigram;
pub mod dataset;
pub mod neuron;
pub mod optimizer;
pub mod token_targets;

pub use bigram::{BigramDataset, BigramEpochMetrics, BigramError, BigramExample, TinyBigramModel};
pub use dataset::{Dataset, DatasetError, TrainingExample};
pub use neuron::{
    Bias, Gradient, Input, Loss, Neuron, NeuronGradients, Prediction, Target, TrainingStep, Weight,
    sigmoid, squared_error,
};
pub use optimizer::{EpochMetrics, LearningRate, Sgd, average_loss, train_epoch, train_epochs};
pub use token_targets::{
    NextTokenExample, TokenId, TokenTargetError, cross_entropy_gradient, cross_entropy_loss,
    softmax,
};
