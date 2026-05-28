//! Small datasets used by the neuron training lessons.

use thiserror::Error;

use crate::neuron::{Input, Target};

/// One supervised scalar training example for the neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingExample {
    /// The first input feature.
    pub x1: Input,
    /// The second input feature.
    pub x2: Input,
    /// The desired scalar target.
    pub target: Target,
}

impl TrainingExample {
    /// Creates one training example.
    pub const fn new(x1: Input, x2: Input, target: Target) -> Self {
        Self { x1, x2, target }
    }
}

/// Errors returned when building datasets.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// The dataset must contain at least one example so average loss is defined.
    #[error("datasets must contain at least one example")]
    EmptyDataset,
}

/// A non-empty collection of training examples.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    examples: Vec<TrainingExample>,
}

impl Dataset {
    /// Creates a dataset from a vector of examples.
    pub fn new(examples: Vec<TrainingExample>) -> Result<Self, DatasetError> {
        if examples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }

        Ok(Self { examples })
    }

    /// Returns the classic AND gate dataset.
    pub fn and_gate() -> Self {
        Self {
            examples: vec![
                TrainingExample::new(Input(0.0), Input(0.0), Target(0.0)),
                TrainingExample::new(Input(0.0), Input(1.0), Target(0.0)),
                TrainingExample::new(Input(1.0), Input(0.0), Target(0.0)),
                TrainingExample::new(Input(1.0), Input(1.0), Target(1.0)),
            ],
        }
    }

    /// Returns the classic OR gate dataset.
    pub fn or_gate() -> Self {
        Self {
            examples: vec![
                TrainingExample::new(Input(0.0), Input(0.0), Target(0.0)),
                TrainingExample::new(Input(0.0), Input(1.0), Target(1.0)),
                TrainingExample::new(Input(1.0), Input(0.0), Target(1.0)),
                TrainingExample::new(Input(1.0), Input(1.0), Target(1.0)),
            ],
        }
    }

    /// Returns the number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Returns the examples as a slice.
    pub fn as_slice(&self) -> &[TrainingExample] {
        &self.examples
    }

    /// Iterates over the examples.
    pub fn iter(&self) -> impl Iterator<Item = &TrainingExample> {
        self.examples.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_new_rejects_empty_examples() {
        let result = Dataset::new(Vec::new());
        assert_eq!(result, Err(DatasetError::EmptyDataset));
    }

    #[test]
    fn or_gate_dataset_is_non_empty() {
        let dataset = Dataset::or_gate();
        assert_eq!(dataset.len(), 4);
        assert!(!dataset.is_empty());
    }
}
