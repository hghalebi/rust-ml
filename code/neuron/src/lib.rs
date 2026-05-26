//! A tiny, typed neuron crate for the beginner Rust ML lessons.
//!
//! The crate keeps the lesson translation visible:
//! - [`FeatureVector`] and [`WeightVector`] represent the inputs and knobs
//! - [`weighted_sum`] is the algebraic dot product
//! - [`TinyNeuron::predict`] is `mix -> squash`
//! - [`TinyNeuron::train_one_step`] is `blame -> trace -> adjust`

pub mod error;

use error::NeuronError;

pub use error::NeuronError as Error;

fn finite(role: &'static str, value: f64) -> Result<f64, NeuronError> {
    if !value.is_finite() {
        return Err(NeuronError::NonFiniteValue { role, value });
    }

    Ok(value)
}

macro_rules! finite_scalar {
    ($name:ident, $doc:literal, $role:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $name(f64);

        impl $name {
            /// Creates the scalar after checking that it is finite.
            pub fn new(value: f64) -> Result<Self, NeuronError> {
                Ok(Self(finite($role, value)?))
            }

            /// Returns the raw scalar value for printing, tests, and hand checks.
            pub fn value(self) -> f64 {
                self.0
            }
        }
    };
}

finite_scalar!(
    InputValue,
    "One scalar feature entering the neuron.",
    "input"
);
finite_scalar!(
    Weight,
    "One learned multiplier attached to one input.",
    "weight"
);
finite_scalar!(
    Bias,
    "The learned offset added after the weighted sum.",
    "bias"
);
finite_scalar!(
    WeightedSum,
    "The dot product before the bias is added.",
    "weighted sum"
);
finite_scalar!(
    PreActivation,
    "The raw score `z` before sigmoid.",
    "pre-activation"
);
finite_scalar!(
    Gradient,
    "A derivative that says how a parameter affects loss.",
    "gradient"
);
finite_scalar!(
    Adjustment,
    "A learning-rate-scaled parameter update.",
    "adjustment"
);
finite_scalar!(Loss, "Squared-error loss for one example.", "loss");

/// A probability-like label for the beginner binary examples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target(f64);

impl Target {
    /// Creates a target in the inclusive range `0..=1`.
    pub fn new(value: f64) -> Result<Self, NeuronError> {
        let value = finite("target", value)?;

        if !(0.0..=1.0).contains(&value) {
            return Err(NeuronError::TargetOutOfRange { value });
        }

        Ok(Self(value))
    }

    /// Returns the raw target value for printing, tests, and hand checks.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// A sigmoid output, always in `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction(f64);

impl Prediction {
    /// Creates a prediction when the caller has already produced a valid score.
    pub fn new(value: f64) -> Result<Self, NeuronError> {
        let value = finite("prediction", value)?;

        if !(0.0..=1.0).contains(&value) {
            return Err(NeuronError::NumericalIssue {
                operation: "Prediction::new",
                details: "prediction must stay between 0 and 1",
            });
        }

        Ok(Self(value))
    }

    /// Returns the raw prediction value for printing, tests, and hand checks.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// A positive step size for gradient descent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRate(f64);

impl LearningRate {
    /// Creates a learning rate that can actually move a parameter update.
    pub fn new(value: f64) -> Result<Self, NeuronError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(NeuronError::InvalidLearningRate { value });
        }

        Ok(Self(value))
    }

    /// Returns the scalar learning-rate value.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Input features for one example.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    values: Vec<InputValue>,
}

impl FeatureVector {
    /// Creates a non-empty feature vector.
    pub fn new(values: Vec<InputValue>) -> Result<Self, NeuronError> {
        if values.is_empty() {
            return Err(NeuronError::EmptyInput {
                operation: "FeatureVector::new",
                details: "a neuron needs at least one input feature",
            });
        }

        Ok(Self { values })
    }

    /// Convenience constructor for the two-input lessons.
    pub fn two(left: InputValue, right: InputValue) -> Self {
        Self {
            values: vec![left, right],
        }
    }

    /// Returns the number of input features.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when the vector has no features.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the input values.
    pub fn values(&self) -> &[InputValue] {
        &self.values
    }
}

/// Learned weights aligned with a [`FeatureVector`].
#[derive(Debug, Clone, PartialEq)]
pub struct WeightVector {
    values: Vec<Weight>,
}

impl WeightVector {
    /// Creates a non-empty weight vector.
    pub fn new(values: Vec<Weight>) -> Result<Self, NeuronError> {
        if values.is_empty() {
            return Err(NeuronError::EmptyInput {
                operation: "WeightVector::new",
                details: "a neuron needs at least one weight",
            });
        }

        Ok(Self { values })
    }

    /// Convenience constructor for the two-input lessons.
    pub fn two(left: Weight, right: Weight) -> Self {
        Self {
            values: vec![left, right],
        }
    }

    /// Returns the number of learned weights.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when the vector has no weights.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the learned weights.
    pub fn values(&self) -> &[Weight] {
        &self.values
    }
}

/// One labeled training row.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExample {
    features: FeatureVector,
    target: Target,
}

impl TrainingExample {
    /// Creates one labeled row.
    pub fn new(features: FeatureVector, target: Target) -> Self {
        Self { features, target }
    }

    /// Returns the input features.
    pub fn features(&self) -> &FeatureVector {
        &self.features
    }

    /// Returns the target label.
    pub fn target(&self) -> Target {
        self.target
    }
}

/// A non-empty group of examples for a training loop.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    examples: Vec<TrainingExample>,
}

impl Dataset {
    /// Creates a non-empty dataset.
    pub fn new(examples: Vec<TrainingExample>) -> Result<Self, NeuronError> {
        if examples.is_empty() {
            return Err(NeuronError::EmptyInput {
                operation: "Dataset::new",
                details: "training needs at least one example",
            });
        }

        Ok(Self { examples })
    }

    /// Builds the tiny AND dataset used in the neuron lessons.
    pub fn and_gate() -> Result<Self, NeuronError> {
        Ok(Self {
            examples: vec![
                TrainingExample::new(
                    FeatureVector::two(InputValue::new(0.0)?, InputValue::new(0.0)?),
                    Target::new(0.0)?,
                ),
                TrainingExample::new(
                    FeatureVector::two(InputValue::new(0.0)?, InputValue::new(1.0)?),
                    Target::new(0.0)?,
                ),
                TrainingExample::new(
                    FeatureVector::two(InputValue::new(1.0)?, InputValue::new(0.0)?),
                    Target::new(0.0)?,
                ),
                TrainingExample::new(
                    FeatureVector::two(InputValue::new(1.0)?, InputValue::new(1.0)?),
                    Target::new(1.0)?,
                ),
            ],
        })
    }

    /// Returns all examples.
    pub fn examples(&self) -> &[TrainingExample] {
        &self.examples
    }

    /// Returns the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns `true` when there are no examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// The gradients computed during one training step.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterGradients {
    weights: Vec<Gradient>,
    bias: Gradient,
}

impl ParameterGradients {
    /// Returns one gradient per weight.
    pub fn weights(&self) -> &[Gradient] {
        &self.weights
    }

    /// Returns the bias gradient.
    pub fn bias(&self) -> Gradient {
        self.bias
    }
}

/// Learner-visible values from one gradient update.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingStep {
    prediction_before: Prediction,
    loss_before: Loss,
    gradients: ParameterGradients,
    prediction_after: Prediction,
    loss_after: Loss,
}

impl TrainingStep {
    /// Prediction before the update.
    pub fn prediction_before(&self) -> Prediction {
        self.prediction_before
    }

    /// Loss before the update.
    pub fn loss_before(&self) -> Loss {
        self.loss_before
    }

    /// Gradients used for the update.
    pub fn gradients(&self) -> &ParameterGradients {
        &self.gradients
    }

    /// Prediction after the update.
    pub fn prediction_after(&self) -> Prediction {
        self.prediction_after
    }

    /// Loss after the update on the same example.
    pub fn loss_after(&self) -> Loss {
        self.loss_after
    }
}

/// A single sigmoid neuron.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyNeuron {
    weights: WeightVector,
    bias: Bias,
}

impl TinyNeuron {
    /// Creates a neuron whose weights define the expected feature count.
    pub fn new(weights: WeightVector, bias: Bias) -> Self {
        Self { weights, bias }
    }

    /// A deterministic starting point used by examples and tests.
    pub fn lesson_seed() -> Result<Self, NeuronError> {
        Ok(Self::new(
            WeightVector::two(Weight::new(0.5)?, Weight::new(-0.3)?),
            Bias::new(0.1)?,
        ))
    }

    /// Returns the learned weights.
    pub fn weights(&self) -> &WeightVector {
        &self.weights
    }

    /// Returns the learned bias.
    pub fn bias(&self) -> Bias {
        self.bias
    }

    /// Computes the raw score `z = w dot x + b`.
    pub fn raw_score(&self, features: &FeatureVector) -> Result<PreActivation, NeuronError> {
        let sum = weighted_sum(features, &self.weights)?;
        PreActivation::new(sum.value() + self.bias.value())
    }

    /// Runs the forward pass `mix -> squash`.
    pub fn predict(&self, features: &FeatureVector) -> Result<Prediction, NeuronError> {
        sigmoid(self.raw_score(features)?)
    }

    /// Computes squared-error loss for one example.
    pub fn loss(&self, example: &TrainingExample) -> Result<Loss, NeuronError> {
        squared_error(self.predict(example.features())?, example.target())
    }

    /// Runs one training update and returns the values a learner should inspect.
    pub fn train_one_step(
        &mut self,
        example: &TrainingExample,
        rate: LearningRate,
    ) -> Result<TrainingStep, NeuronError> {
        self.validate_features(example.features(), "TinyNeuron::train_one_step")?;

        let z = self.raw_score(example.features())?;
        let prediction_before = sigmoid(z)?;
        let loss_before = squared_error(prediction_before, example.target())?;

        let d_loss_d_prediction = 2.0 * (prediction_before.value() - example.target().value());
        let d_prediction_d_pre_activation =
            prediction_before.value() * (1.0 - prediction_before.value());
        let shared_blame = d_loss_d_prediction * d_prediction_d_pre_activation;

        let weight_gradients = example
            .features()
            .values()
            .iter()
            .map(|feature| Gradient::new(shared_blame * feature.value()))
            .collect::<Result<Vec<_>, _>>()?;
        let bias_gradient = Gradient::new(shared_blame)?;

        for (weight, gradient) in self.weights.values.iter_mut().zip(weight_gradients.iter()) {
            let adjustment = Adjustment::new(rate.value() * gradient.value())?;
            *weight = Weight::new(weight.value() - adjustment.value())?;
        }
        let bias_adjustment = Adjustment::new(rate.value() * bias_gradient.value())?;
        self.bias = Bias::new(self.bias.value() - bias_adjustment.value())?;

        let prediction_after = self.predict(example.features())?;
        let loss_after = squared_error(prediction_after, example.target())?;

        Ok(TrainingStep {
            prediction_before,
            loss_before,
            gradients: ParameterGradients {
                weights: weight_gradients,
                bias: bias_gradient,
            },
            prediction_after,
            loss_after,
        })
    }

    /// Runs one pass over the dataset and returns the average pre-update loss.
    pub fn train_epoch(
        &mut self,
        dataset: &Dataset,
        rate: LearningRate,
    ) -> Result<Loss, NeuronError> {
        let mut total = 0.0;

        for example in dataset.examples() {
            total += self.train_one_step(example, rate)?.loss_before().value();
        }

        Loss::new(total / dataset.len() as f64)
    }

    fn validate_features(
        &self,
        features: &FeatureVector,
        operation: &'static str,
    ) -> Result<(), NeuronError> {
        if features.len() != self.weights.len() {
            return Err(NeuronError::DimensionMismatch {
                operation,
                left_label: "features",
                left_len: features.len(),
                right_label: "weights",
                right_len: self.weights.len(),
                hint: "each input feature must have exactly one learned weight",
            });
        }

        Ok(())
    }
}

/// Computes the dot product between input features and weights.
pub fn weighted_sum(
    features: &FeatureVector,
    weights: &WeightVector,
) -> Result<WeightedSum, NeuronError> {
    if features.len() != weights.len() {
        return Err(NeuronError::DimensionMismatch {
            operation: "weighted_sum",
            left_label: "features",
            left_len: features.len(),
            right_label: "weights",
            right_len: weights.len(),
            hint: "dot product needs one weight per feature",
        });
    }

    WeightedSum::new(
        features
            .values()
            .iter()
            .zip(weights.values().iter())
            .map(|(feature, weight)| feature.value() * weight.value())
            .sum(),
    )
}

/// Applies sigmoid to a raw score.
pub fn sigmoid(z: PreActivation) -> Result<Prediction, NeuronError> {
    Prediction::new(1.0 / (1.0 + (-z.value()).exp()))
}

/// Computes squared-error loss.
pub fn squared_error(prediction: Prediction, target: Target) -> Result<Loss, NeuronError> {
    Loss::new((prediction.value() - target.value()).powi(2))
}

#[cfg(test)]
mod tests {
    use super::{
        Bias, Dataset, FeatureVector, InputValue, LearningRate, NeuronError, Target, TinyNeuron,
        TrainingExample, Weight, WeightVector, weighted_sum,
    };

    fn input(value: f64) -> InputValue {
        InputValue::new(value).expect("test input should be finite")
    }

    fn weight(value: f64) -> Weight {
        Weight::new(value).expect("test weight should be finite")
    }

    #[test]
    fn target_rejects_values_outside_probability_range() {
        let error = Target::new(1.2).expect_err("invalid target should fail");
        assert_eq!(error, NeuronError::TargetOutOfRange { value: 1.2 });
    }

    #[test]
    fn learning_rate_rejects_zero() {
        let error = LearningRate::new(0.0).expect_err("zero learning rate should fail");
        assert_eq!(error, NeuronError::InvalidLearningRate { value: 0.0 });
    }

    #[test]
    fn weighted_sum_matches_the_hand_calculation() -> Result<(), NeuronError> {
        let features = FeatureVector::two(input(1.0), input(0.5));
        let weights = WeightVector::two(weight(0.8), weight(-0.4));

        assert!((weighted_sum(&features, &weights)?.value() - 0.6).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn weighted_sum_reports_shape_mismatch() -> Result<(), NeuronError> {
        let features = FeatureVector::new(vec![input(1.0), input(2.0), input(3.0)])?;
        let weights = WeightVector::two(weight(0.1), weight(0.2));

        let error = weighted_sum(&features, &weights).expect_err("mismatch should fail");
        assert!(matches!(error, NeuronError::DimensionMismatch { .. }));
        Ok(())
    }

    #[test]
    fn forward_pass_matches_the_lesson_numbers() -> Result<(), NeuronError> {
        let neuron = TinyNeuron::new(
            WeightVector::two(weight(0.8), weight(-0.4)),
            Bias::new(0.1)?,
        );
        let features = FeatureVector::two(input(1.0), input(0.0));

        let raw_score = neuron.raw_score(&features)?;
        let prediction = neuron.predict(&features)?;

        assert!((raw_score.value() - 0.9).abs() < 1e-12);
        assert!((prediction.value() - 0.710_949_502_625).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn one_step_training_lowers_loss_for_the_same_example() -> Result<(), NeuronError> {
        let mut neuron = TinyNeuron::lesson_seed()?;
        let example = TrainingExample::new(
            FeatureVector::two(input(1.0), input(0.0)),
            Target::new(1.0)?,
        );
        let step = neuron.train_one_step(&example, LearningRate::new(0.5)?)?;

        assert!(step.loss_after().value() < step.loss_before().value());
        assert_eq!(step.gradients().weights().len(), 2);
        Ok(())
    }

    #[test]
    fn epoch_training_reduces_average_loss_on_and_gate() -> Result<(), NeuronError> {
        let dataset = Dataset::and_gate()?;
        let rate = LearningRate::new(0.8)?;
        let mut neuron = TinyNeuron::lesson_seed()?;
        let initial_loss = dataset
            .examples()
            .iter()
            .map(|example| neuron.loss(example).map(|loss| loss.value()))
            .sum::<Result<f64, _>>()?
            / dataset.len() as f64;

        for _ in 0..200 {
            neuron.train_epoch(&dataset, rate)?;
        }

        let final_loss = dataset
            .examples()
            .iter()
            .map(|example| neuron.loss(example).map(|loss| loss.value()))
            .sum::<Result<f64, _>>()?
            / dataset.len() as f64;

        assert!(
            final_loss < initial_loss,
            "expected final loss {final_loss} to be lower than initial loss {initial_loss}"
        );
        Ok(())
    }
}
