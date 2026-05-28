//! Typed single-neuron math plus a manual backward pass.

use crate::dataset::TrainingExample;

/// One scalar input value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input(pub f64);

/// One learnable weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weight(pub f64);

/// One learnable bias.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bias(pub f64);

/// A sigmoid output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction(pub f64);

/// A scalar supervision target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target(pub f64);

/// A scalar loss value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loss(pub f64);

/// One scalar gradient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient(pub f64);

/// All parameter gradients for the neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuronGradients {
    /// The loss gradient with respect to `w1`.
    pub w1: Gradient,
    /// The loss gradient with respect to `w2`.
    pub w2: Gradient,
    /// The loss gradient with respect to `b`.
    pub b: Gradient,
}

/// The full observable result of one training step before parameters move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingStep {
    /// The raw pre-activation score.
    pub z: f64,
    /// The post-sigmoid prediction.
    pub prediction: Prediction,
    /// The one-example loss.
    pub loss: Loss,
    /// The gradients needed for the optimizer step.
    pub gradients: NeuronGradients,
}

/// A two-input neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neuron {
    /// The weight applied to `x1`.
    pub w1: Weight,
    /// The weight applied to `x2`.
    pub w2: Weight,
    /// The additive bias.
    pub b: Bias,
}

impl Neuron {
    /// Creates a neuron from explicit parameters.
    pub const fn new(w1: Weight, w2: Weight, b: Bias) -> Self {
        Self { w1, w2, b }
    }

    /// Computes the raw weighted sum before the activation.
    pub fn pre_activation(&self, x1: Input, x2: Input) -> f64 {
        self.w1.0 * x1.0 + self.w2.0 * x2.0 + self.b.0
    }

    /// Computes the sigmoid prediction.
    pub fn predict(&self, x1: Input, x2: Input) -> Prediction {
        sigmoid(self.pre_activation(x1, x2))
    }

    /// Computes the one-example squared-error loss.
    pub fn loss(&self, example: TrainingExample) -> Loss {
        squared_error(self.predict(example.x1, example.x2), example.target)
    }

    /// Computes the full training step, including gradients.
    pub fn training_step(&self, example: TrainingExample) -> TrainingStep {
        let z = self.pre_activation(example.x1, example.x2);
        let prediction = sigmoid(z);
        let loss = squared_error(prediction, example.target);

        let d_loss_d_prediction = 2.0 * (prediction.0 - example.target.0);
        let d_prediction_d_z = prediction.0 * (1.0 - prediction.0);
        let upstream = d_loss_d_prediction * d_prediction_d_z;

        let gradients = NeuronGradients {
            w1: Gradient(upstream * example.x1.0),
            w2: Gradient(upstream * example.x2.0),
            b: Gradient(upstream),
        };

        TrainingStep {
            z,
            prediction,
            loss,
            gradients,
        }
    }

    /// Computes only the backward-pass gradients.
    pub fn backward(&self, example: TrainingExample) -> NeuronGradients {
        self.training_step(example).gradients
    }

    /// Applies one gradient update to the parameters.
    pub fn apply_gradients(&mut self, scale: f64, gradients: NeuronGradients) {
        self.w1 = Weight(self.w1.0 - scale * gradients.w1.0);
        self.w2 = Weight(self.w2.0 - scale * gradients.w2.0);
        self.b = Bias(self.b.0 - scale * gradients.b.0);
    }
}

impl From<Prediction> for f64 {
    fn from(value: Prediction) -> Self {
        value.0
    }
}

impl From<Target> for f64 {
    fn from(value: Target) -> Self {
        value.0
    }
}

impl From<Loss> for f64 {
    fn from(value: Loss) -> Self {
        value.0
    }
}

impl From<Gradient> for f64 {
    fn from(value: Gradient) -> Self {
        value.0
    }
}

/// Computes a sigmoid prediction from a raw score.
pub fn sigmoid(z: f64) -> Prediction {
    Prediction(1.0 / (1.0 + (-z).exp()))
}

/// Computes the squared-error loss.
pub fn squared_error(prediction: Prediction, target: Target) -> Loss {
    Loss((prediction.0 - target.0).powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        let diff = (left - right).abs();
        assert!(diff < 1e-8, "left={left}, right={right}, diff={diff}");
    }

    #[test]
    fn training_step_matches_manual_chain_rule() {
        let neuron = Neuron::new(Weight(0.8), Weight(-0.4), Bias(0.1));
        let example = TrainingExample::new(Input(1.0), Input(0.0), Target(1.0));

        let step = neuron.training_step(example);

        assert_close(step.z, 0.9);
        assert_close(f64::from(step.prediction), 0.7109495026250039);
        assert_close(f64::from(step.loss), 0.0835501957501944);
        assert_close(f64::from(step.gradients.w1), -0.11879993209599166);
        assert_close(f64::from(step.gradients.w2), 0.0);
        assert_close(f64::from(step.gradients.b), -0.11879993209599166);
    }

    #[test]
    fn apply_gradients_reduces_loss_for_one_example() {
        let example = TrainingExample::new(Input(1.0), Input(0.0), Target(1.0));
        let mut neuron = Neuron::new(Weight(0.8), Weight(-0.4), Bias(0.1));

        let before = f64::from(neuron.loss(example));
        let gradients = neuron.backward(example);
        neuron.apply_gradients(0.5, gradients);
        let after = f64::from(neuron.loss(example));

        assert!(after < before, "before={before}, after={after}");
    }
}
