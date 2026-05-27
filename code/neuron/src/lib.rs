//! A tiny, typed neuron crate for the beginner Rust ML lessons.
//!
//! The crate keeps the lesson translation visible:
//! - [`FeatureVector`] and [`WeightVector`] represent the inputs and knobs
//! - [`weighted_sum`] is the algebraic dot product
//! - [`TinyNeuron::predict`] is `mix -> squash`
//! - [`TinyNeuron::train_one_step`] is `blame -> trace -> adjust`
//!
//! Raw learner literals enter through explicit `TryFrom` adapters. Public
//! teaching APIs then use semantic values such as [`InputValue`], [`Weight`],
//! [`Target`], [`LearningRate`], [`FeatureCount`], and [`Loss`].

pub mod error;

use std::{
    fmt,
    ops::{Add, Mul, Sub},
};

use error::NeuronError;

pub use error::NeuronError as Error;

fn finite(role: &'static str, value: f64) -> Result<f64, NeuronError> {
    if !value.is_finite() {
        return Err(NeuronError::non_finite_value(role, value));
    }

    Ok(value)
}

macro_rules! finite_scalar {
    ($name:ident, $doc:literal, $role:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            fn from_raw(value: f64) -> Result<Self, NeuronError> {
                Ok(Self(finite($role, value)?))
            }

            fn as_f64(self) -> f64 {
                self.0
            }
        }

        impl TryFrom<f64> for $name {
            type Error = NeuronError;

            fn try_from(value: f64) -> Result<Self, Self::Error> {
                Self::from_raw(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, formatter)
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
    WeightedProduct,
    "One input multiplied by its aligned weight.",
    "weighted product"
);
finite_scalar!(
    PreActivation,
    "The raw score `z` before sigmoid.",
    "pre-activation"
);
finite_scalar!(
    PredictionError,
    "Prediction minus target for one labeled example.",
    "prediction error"
);
finite_scalar!(
    SigmoidSlope,
    "Local slope of the sigmoid at the current prediction.",
    "sigmoid slope"
);
finite_scalar!(
    ParameterBlame,
    "Shared blame signal that flows into weight and bias gradients.",
    "parameter blame"
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

impl Add<WeightedProduct> for WeightedSum {
    type Output = Result<WeightedSum, NeuronError>;

    fn add(self, right: WeightedProduct) -> Self::Output {
        WeightedSum::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Add<Bias> for WeightedSum {
    type Output = Result<PreActivation, NeuronError>;

    fn add(self, right: Bias) -> Self::Output {
        PreActivation::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Mul<Weight> for InputValue {
    type Output = Result<WeightedProduct, NeuronError>;

    fn mul(self, right: Weight) -> Self::Output {
        WeightedProduct::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Sub<Target> for Prediction {
    type Output = Result<PredictionError, NeuronError>;

    fn sub(self, right: Target) -> Self::Output {
        PredictionError::from_raw(self.as_f64() - right.as_f64())
    }
}

impl Mul<SigmoidSlope> for PredictionError {
    type Output = Result<ParameterBlame, NeuronError>;

    fn mul(self, right: SigmoidSlope) -> Self::Output {
        ParameterBlame::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Mul<InputValue> for ParameterBlame {
    type Output = Result<Gradient, NeuronError>;

    fn mul(self, right: InputValue) -> Self::Output {
        Gradient::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Mul<Gradient> for LearningRate {
    type Output = Result<Adjustment, NeuronError>;

    fn mul(self, right: Gradient) -> Self::Output {
        Adjustment::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Sub<Adjustment> for Weight {
    type Output = Result<Weight, NeuronError>;

    fn sub(self, right: Adjustment) -> Self::Output {
        Weight::from_raw(self.as_f64() - right.as_f64())
    }
}

impl Sub<Adjustment> for Bias {
    type Output = Result<Bias, NeuronError>;

    fn sub(self, right: Adjustment) -> Self::Output {
        Bias::from_raw(self.as_f64() - right.as_f64())
    }
}

impl Mul<&WeightVector> for &FeatureVector {
    type Output = Result<WeightedSum, NeuronError>;

    fn mul(self, right: &WeightVector) -> Self::Output {
        weighted_sum_with_operation("FeatureVector * WeightVector", self, right)
    }
}

/// Number of input features or weights in a neuron vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeatureCount(usize);

impl FeatureCount {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for FeatureCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of examples in a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExampleCount(usize);

impl ExampleCount {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for ExampleCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A probability-like label for the beginner binary examples.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Target(f64);

impl Target {
    fn from_raw(value: f64) -> Result<Self, NeuronError> {
        let value = finite("target", value)?;

        if !(0.0..=1.0).contains(&value) {
            return Err(NeuronError::target_out_of_range(value));
        }

        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Target {
    type Error = NeuronError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A sigmoid output, always in `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Prediction(f64);

impl Prediction {
    fn from_raw(value: f64) -> Result<Self, NeuronError> {
        let value = finite("prediction", value)?;

        if !(0.0..=1.0).contains(&value) {
            return Err(NeuronError::numerical_issue(
                "Prediction::try_from",
                "prediction must stay between 0 and 1",
            ));
        }

        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }

    fn sigmoid_slope(self) -> Result<SigmoidSlope, NeuronError> {
        SigmoidSlope::from_raw(self.as_f64() * (1.0 - self.as_f64()))
    }
}

impl TryFrom<f64> for Prediction {
    type Error = NeuronError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for Prediction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A positive step size for gradient descent.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LearningRate(f64);

impl LearningRate {
    fn from_raw(value: f64) -> Result<Self, NeuronError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(NeuronError::invalid_learning_rate(value));
        }

        Ok(Self(value))
    }

    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for LearningRate {
    type Error = NeuronError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for LearningRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl PredictionError {
    fn squared_loss(self) -> Result<Loss, NeuronError> {
        Loss::from_raw(self.as_f64().powi(2))
    }

    fn through_sigmoid(self, prediction: Prediction) -> Result<ParameterBlame, NeuronError> {
        let local_blame = (self * prediction.sigmoid_slope()?)?;
        local_blame.double()
    }
}

impl ParameterBlame {
    fn double(self) -> Result<Self, NeuronError> {
        Self::from_raw(2.0 * self.as_f64())
    }
}

/// Input features for one example.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    values: Vec<InputValue>,
}

impl FeatureVector {
    /// Creates a non-empty feature vector.
    pub fn from_values(values: impl IntoIterator<Item = InputValue>) -> Result<Self, NeuronError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(NeuronError::empty_input(
                "FeatureVector::from_values",
                "a neuron needs at least one input feature",
            ));
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
    pub fn len(&self) -> FeatureCount {
        FeatureCount::from_raw(self.len_value())
    }

    /// Iterates over the input values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &InputValue> + '_ {
        self.values.iter()
    }

    fn len_value(&self) -> usize {
        self.values.len()
    }
}

/// Learned weights aligned with a [`FeatureVector`].
#[derive(Debug, Clone, PartialEq)]
pub struct WeightVector {
    values: Vec<Weight>,
}

impl WeightVector {
    /// Creates a non-empty weight vector.
    pub fn from_values(values: impl IntoIterator<Item = Weight>) -> Result<Self, NeuronError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(NeuronError::empty_input(
                "WeightVector::from_values",
                "a neuron needs at least one weight",
            ));
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
    pub fn len(&self) -> FeatureCount {
        FeatureCount::from_raw(self.len_value())
    }

    /// Iterates over the learned weights.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &Weight> + '_ {
        self.values.iter()
    }

    fn len_value(&self) -> usize {
        self.values.len()
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
    pub fn from_examples(
        examples: impl IntoIterator<Item = TrainingExample>,
    ) -> Result<Self, NeuronError> {
        let examples = examples.into_iter().collect::<Vec<_>>();
        if examples.is_empty() {
            return Err(NeuronError::empty_input(
                "Dataset::from_examples",
                "training needs at least one example",
            ));
        }

        Ok(Self { examples })
    }

    /// Builds the tiny AND dataset used in the neuron lessons.
    pub fn and_gate() -> Result<Self, NeuronError> {
        Self::from_examples([
            TrainingExample::new(
                FeatureVector::two(InputValue::try_from(0.0)?, InputValue::try_from(0.0)?),
                Target::try_from(0.0)?,
            ),
            TrainingExample::new(
                FeatureVector::two(InputValue::try_from(0.0)?, InputValue::try_from(1.0)?),
                Target::try_from(0.0)?,
            ),
            TrainingExample::new(
                FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
                Target::try_from(0.0)?,
            ),
            TrainingExample::new(
                FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(1.0)?),
                Target::try_from(1.0)?,
            ),
        ])
    }

    /// Iterates over all examples.
    pub fn examples(&self) -> impl ExactSizeIterator<Item = &TrainingExample> + '_ {
        self.examples.iter()
    }

    /// Returns the number of examples.
    pub fn len(&self) -> ExampleCount {
        ExampleCount::from_raw(self.len_value())
    }

    fn len_value(&self) -> usize {
        self.examples.len()
    }
}

/// The gradients computed during one training step.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterGradients {
    weights: Vec<Gradient>,
    bias: Gradient,
}

impl ParameterGradients {
    /// Iterates over one gradient per weight.
    pub fn weights(&self) -> impl ExactSizeIterator<Item = &Gradient> + '_ {
        self.weights.iter()
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
            WeightVector::two(Weight::try_from(0.5)?, Weight::try_from(-0.3)?),
            Bias::try_from(0.1)?,
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
        let sum = (features * &self.weights)?;
        sum + self.bias
    }

    /// Runs the forward pass `mix -> squash`.
    pub fn predict(&self, features: &FeatureVector) -> Result<Prediction, NeuronError> {
        sigmoid(self.raw_score(features)?)
    }

    /// Computes squared-error loss for one example.
    pub fn loss(&self, example: &TrainingExample) -> Result<Loss, NeuronError> {
        squared_error(self.predict(example.features())?, example.target())
    }

    /// Computes average loss without mutating the neuron.
    pub fn average_loss(&self, dataset: &Dataset) -> Result<Loss, NeuronError> {
        let total = dataset
            .examples()
            .map(|example| self.loss(example).map(Loss::as_f64))
            .sum::<Result<f64, _>>()?;

        Loss::from_raw(total / dataset.len().as_usize() as f64)
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

        let prediction_error = (prediction_before - example.target())?;
        let shared_blame = prediction_error.through_sigmoid(prediction_before)?;

        let weight_gradients = example
            .features()
            .values()
            .map(|feature| shared_blame * *feature)
            .collect::<Result<Vec<_>, _>>()?;
        let bias_gradient = Gradient::from_raw(shared_blame.as_f64())?;

        for (weight, gradient) in self.weights.values.iter_mut().zip(weight_gradients.iter()) {
            let adjustment = (rate * *gradient)?;
            *weight = (*weight - adjustment)?;
        }
        let bias_adjustment = (rate * bias_gradient)?;
        self.bias = (self.bias - bias_adjustment)?;

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
            total += self.train_one_step(example, rate)?.loss_before().as_f64();
        }

        Loss::from_raw(total / dataset.len().as_usize() as f64)
    }

    fn validate_features(
        &self,
        features: &FeatureVector,
        operation: &'static str,
    ) -> Result<(), NeuronError> {
        if features.len() != self.weights.len() {
            return Err(NeuronError::dimension_mismatch(
                operation,
                "features",
                features.len().as_usize(),
                "weights",
                self.weights.len().as_usize(),
                "each input feature must have exactly one learned weight",
            ));
        }

        Ok(())
    }
}

/// Computes the dot product between input features and weights.
///
/// This named helper is kept as a learner signpost. The operator form
/// `&features * &weights` is the same checked typed operation.
pub fn weighted_sum(
    features: &FeatureVector,
    weights: &WeightVector,
) -> Result<WeightedSum, NeuronError> {
    weighted_sum_with_operation("weighted_sum", features, weights)
}

fn weighted_sum_with_operation(
    operation: &'static str,
    features: &FeatureVector,
    weights: &WeightVector,
) -> Result<WeightedSum, NeuronError> {
    if features.len() != weights.len() {
        return Err(NeuronError::dimension_mismatch(
            operation,
            "features",
            features.len().as_usize(),
            "weights",
            weights.len().as_usize(),
            "dot product needs one weight per feature",
        ));
    }

    let mut total = WeightedSum::from_raw(0.0)?;

    for (feature, weight) in features.values().zip(weights.values()) {
        total = (total + (*feature * *weight)?)?;
    }

    Ok(total)
}

/// Applies sigmoid to a raw score.
pub fn sigmoid(z: PreActivation) -> Result<Prediction, NeuronError> {
    Prediction::from_raw(1.0 / (1.0 + (-z.as_f64()).exp()))
}

/// Computes squared-error loss.
pub fn squared_error(prediction: Prediction, target: Target) -> Result<Loss, NeuronError> {
    (prediction - target)?.squared_loss()
}

#[cfg(test)]
mod tests {
    use super::{
        Bias, Dataset, FeatureVector, InputValue, LearningRate, NeuronError, Target, TinyNeuron,
        TrainingExample, Weight, WeightVector, WeightedSum, weighted_sum,
    };

    fn features(left: InputValue, right: InputValue) -> FeatureVector {
        FeatureVector::two(left, right)
    }

    fn weights(left: Weight, right: Weight) -> WeightVector {
        WeightVector::two(left, right)
    }

    fn assert_weighted_sums_close(left: WeightedSum, right: WeightedSum) {
        assert!((left.as_f64() - right.as_f64()).abs() < 1e-12);
    }

    #[test]
    fn target_rejects_values_outside_probability_range() {
        let error = Target::try_from(1.2);
        assert!(matches!(error, Err(NeuronError::TargetOutOfRange { .. })));
        assert_eq!(
            error.err().map(|error| error.to_string()),
            Some("target must be between 0 and 1 inclusive, got 1.2".to_owned())
        );
    }

    #[test]
    fn learning_rate_rejects_zero() {
        let error = LearningRate::try_from(0.0);
        assert!(matches!(
            error,
            Err(NeuronError::InvalidLearningRate { .. })
        ));
        assert_eq!(
            error.err().map(|error| error.to_string()),
            Some("learning rate must be finite and greater than zero, got 0".to_owned())
        );
    }

    #[test]
    fn weighted_sum_matches_the_hand_calculation() -> Result<(), NeuronError> {
        let features = features(InputValue::try_from(1.0)?, InputValue::try_from(0.5)?);
        let weights = weights(Weight::try_from(0.8)?, Weight::try_from(-0.4)?);

        let expected = WeightedSum::try_from(0.6)?;
        let actual = weighted_sum(&features, &weights)?;

        assert_weighted_sums_close(actual, expected);
        Ok(())
    }

    #[test]
    fn feature_weight_vectors_multiply_into_a_weighted_sum() -> Result<(), NeuronError> {
        let features = features(InputValue::try_from(1.0)?, InputValue::try_from(0.5)?);
        let weights = weights(Weight::try_from(0.8)?, Weight::try_from(-0.4)?);

        let expected = WeightedSum::try_from(0.6)?;
        let actual = (&features * &weights)?;

        assert_weighted_sums_close(actual, expected);
        Ok(())
    }

    #[test]
    fn weighted_sum_reports_shape_mismatch() -> Result<(), NeuronError> {
        let features = FeatureVector::from_values([
            InputValue::try_from(1.0)?,
            InputValue::try_from(2.0)?,
            InputValue::try_from(3.0)?,
        ])?;
        let weights = weights(Weight::try_from(0.1)?, Weight::try_from(0.2)?);

        let error = weighted_sum(&features, &weights);
        assert!(matches!(error, Err(NeuronError::DimensionMismatch { .. })));
        Ok(())
    }

    #[test]
    fn forward_pass_matches_the_lesson_numbers() -> Result<(), NeuronError> {
        let neuron = TinyNeuron::new(
            weights(Weight::try_from(0.8)?, Weight::try_from(-0.4)?),
            Bias::try_from(0.1)?,
        );
        let features = features(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?);

        let raw_score = neuron.raw_score(&features)?;
        let prediction = neuron.predict(&features)?;

        assert!((raw_score.as_f64() - 0.9).abs() < 1e-12);
        assert!((prediction.as_f64() - 0.710_949_502_625).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn one_step_training_lowers_loss_for_the_same_example() -> Result<(), NeuronError> {
        let mut neuron = TinyNeuron::lesson_seed()?;
        let example = TrainingExample::new(
            features(InputValue::try_from(1.0)?, InputValue::try_from(0.0)?),
            Target::try_from(1.0)?,
        );
        let step = neuron.train_one_step(&example, LearningRate::try_from(0.5)?)?;

        assert!(step.loss_after() < step.loss_before());
        assert_eq!(step.gradients().weights().len(), 2);
        Ok(())
    }

    #[test]
    fn epoch_training_reduces_average_loss_on_and_gate() -> Result<(), NeuronError> {
        let dataset = Dataset::and_gate()?;
        let rate = LearningRate::try_from(0.8)?;
        let mut neuron = TinyNeuron::lesson_seed()?;
        let initial_loss = neuron.average_loss(&dataset)?;

        for _ in 0..200 {
            neuron.train_epoch(&dataset, rate)?;
        }

        let final_loss = neuron.average_loss(&dataset)?;

        assert!(
            final_loss < initial_loss,
            "expected final loss {final_loss} to be lower than initial loss {initial_loss}"
        );
        Ok(())
    }
}
