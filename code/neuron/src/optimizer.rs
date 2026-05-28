//! Optimizer and training-loop helpers for the neuron crate.

use crate::dataset::Dataset;
use crate::neuron::{Loss, Neuron, NeuronGradients};

/// The scale applied to each gradient step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRate(pub f64);

/// Plain stochastic gradient descent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd {
    /// The learning rate used for each parameter update.
    pub learning_rate: LearningRate,
}

impl Sgd {
    /// Creates an SGD optimizer from an explicit learning rate.
    pub const fn new(learning_rate: LearningRate) -> Self {
        Self { learning_rate }
    }

    /// Applies one set of gradients to the neuron parameters.
    pub fn apply(&self, neuron: &mut Neuron, gradients: NeuronGradients) {
        neuron.apply_gradients(self.learning_rate.0, gradients);
    }
}

/// Summary metrics for one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    /// The 1-based epoch index.
    pub epoch: usize,
    /// The average loss after the epoch finishes.
    pub average_loss: Loss,
}

/// Computes the average loss across a dataset.
pub fn average_loss(neuron: &Neuron, dataset: &Dataset) -> Loss {
    let total_loss: f64 = dataset
        .iter()
        .map(|example| f64::from(neuron.loss(*example)))
        .sum();

    Loss(total_loss / dataset.len() as f64)
}

/// Trains the neuron for one full pass over the dataset.
pub fn train_epoch(neuron: &mut Neuron, dataset: &Dataset, optimizer: Sgd) -> EpochMetrics {
    for example in dataset.iter() {
        let gradients = neuron.backward(*example);
        optimizer.apply(neuron, gradients);
    }

    EpochMetrics {
        epoch: 1,
        average_loss: average_loss(neuron, dataset),
    }
}

/// Trains the neuron for a fixed number of epochs and records the average loss.
pub fn train_epochs(
    neuron: &mut Neuron,
    dataset: &Dataset,
    optimizer: Sgd,
    epochs: usize,
) -> Vec<EpochMetrics> {
    let mut metrics = Vec::with_capacity(epochs);

    for epoch_index in 0..epochs {
        for example in dataset.iter() {
            let gradients = neuron.backward(*example);
            optimizer.apply(neuron, gradients);
        }

        metrics.push(EpochMetrics {
            epoch: epoch_index + 1,
            average_loss: average_loss(neuron, dataset),
        });
    }

    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Bias, Input, Weight};

    fn assert_close(left: f64, right: f64) {
        let diff = (left - right).abs();
        assert!(diff < 1e-6, "left={left}, right={right}, diff={diff}");
    }

    #[test]
    fn train_epochs_reduces_average_loss_on_or_gate() {
        let dataset = Dataset::or_gate();
        let mut neuron = Neuron::new(Weight(0.0), Weight(0.0), Bias(0.0));
        let optimizer = Sgd::new(LearningRate(0.8));

        let before = f64::from(average_loss(&neuron, &dataset));
        let metrics = train_epochs(&mut neuron, &dataset, optimizer, 2_000);
        let after = f64::from(average_loss(&neuron, &dataset));

        assert_eq!(metrics.len(), 2_000);
        assert!(after < before, "before={before}, after={after}");

        let final_average = f64::from(metrics[metrics.len() - 1].average_loss);
        assert_close(final_average, after);

        for example in dataset.iter() {
            let prediction = f64::from(neuron.predict(example.x1, example.x2));
            let target = f64::from(example.target);
            if target > 0.5 {
                assert!(prediction > 0.7, "target={target}, prediction={prediction}");
            } else {
                assert!(prediction < 0.3, "target={target}, prediction={prediction}");
            }
        }
    }

    #[test]
    fn train_epoch_reports_average_loss_after_one_pass() {
        let dataset = Dataset::and_gate();
        let mut neuron = Neuron::new(Weight(0.2), Weight(0.2), Bias(-0.1));
        let optimizer = Sgd::new(LearningRate(0.5));

        let metrics = train_epoch(&mut neuron, &dataset, optimizer);

        assert_eq!(metrics.epoch, 1);
        let recomputed = f64::from(average_loss(&neuron, &dataset));
        assert_close(f64::from(metrics.average_loss), recomputed);
    }

    #[test]
    fn average_loss_handles_boolean_dataset() {
        let dataset = Dataset::and_gate();
        let neuron = Neuron::new(Weight(0.0), Weight(0.0), Bias(0.0));

        let loss = f64::from(average_loss(&neuron, &dataset));
        let baseline = 0.25;
        assert_close(loss, baseline);
    }

    #[test]
    fn trained_or_gate_prefers_positive_examples() {
        let dataset = Dataset::or_gate();
        let mut neuron = Neuron::new(Weight(0.0), Weight(0.0), Bias(0.0));
        let optimizer = Sgd::new(LearningRate(0.8));

        let _ = train_epochs(&mut neuron, &dataset, optimizer, 2_000);

        let zero_zero = f64::from(neuron.predict(Input(0.0), Input(0.0)));
        let one_one = f64::from(neuron.predict(Input(1.0), Input(1.0)));

        assert!(one_one > zero_zero);
        assert!(one_one > 0.7);
    }
}
