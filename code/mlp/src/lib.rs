//! A tiny, typed multi-layer perceptron teaching crate.
//!
//! The crate focuses on the first idea after a single neuron:
//! hidden layers create intermediate representations.
//!
//! It intentionally implements a small forward pass, not a full training
//! system. Module 04 teaches updates; this crate teaches shape flow and
//! representation flow.
//!
//! Raw learner literals enter through explicit `TryFrom` adapters. Public
//! teaching APIs then use semantic values such as [`InputValue`],
//! [`WeightValue`], [`BiasValue`], [`VectorWidth`], [`OutputLogit`], and
//! [`Prediction`].
//!
//! ```text
//! InputVector -> HiddenPreActivation
//! HiddenPreActivation -> HiddenActivation
//! HiddenActivation -> OutputLogit
//! OutputLogit -> Prediction
//! ReviewedForwardTrace -> PublicForwardTrace
//! ```

pub mod error;

use std::{
    fmt,
    ops::{Add, Mul},
};

use error::MlpError;

pub use error::MlpError as Error;

fn finite(role: &'static str, value: f64) -> Result<f64, MlpError> {
    if !value.is_finite() {
        return Err(MlpError::non_finite_value(role, value));
    }

    Ok(value)
}

macro_rules! finite_scalar {
    ($name:ident, $doc:literal, $role:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            fn from_raw(value: f64) -> Result<Self, MlpError> {
                Ok(Self(finite($role, value)?))
            }

            fn as_f64(self) -> f64 {
                self.0
            }
        }

        impl TryFrom<f64> for $name {
            type Error = MlpError;

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

finite_scalar!(InputValue, "One scalar input entering the MLP.", "input");
finite_scalar!(WeightValue, "One learned matrix weight.", "weight");
finite_scalar!(BiasValue, "One learned bias value.", "bias");
finite_scalar!(
    HiddenPreActivationValue,
    "One hidden-layer value before ReLU.",
    "hidden pre-activation"
);
finite_scalar!(
    HiddenActivationValue,
    "One hidden-layer value after ReLU.",
    "hidden activation"
);
finite_scalar!(
    WeightedProduct,
    "One signal multiplied by one aligned weight.",
    "weighted product"
);
finite_scalar!(
    WeightedSum,
    "One dot-product result before bias is added.",
    "weighted sum"
);
finite_scalar!(
    OutputLogit,
    "A single output logit before sigmoid.",
    "output logit"
);

impl WeightedSum {
    fn zero() -> Result<Self, MlpError> {
        Self::from_raw(0.0)
    }
}

impl HiddenPreActivationValue {
    fn relu(self) -> Result<HiddenActivationValue, MlpError> {
        HiddenActivationValue::from_raw(self.as_f64().max(0.0))
    }
}

impl From<HiddenPreActivationValue> for OutputLogit {
    fn from(value: HiddenPreActivationValue) -> Self {
        Self(value.as_f64())
    }
}

impl Add<WeightedProduct> for WeightedSum {
    type Output = Result<WeightedSum, MlpError>;

    fn add(self, right: WeightedProduct) -> Self::Output {
        WeightedSum::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Add<BiasValue> for WeightedSum {
    type Output = Result<HiddenPreActivationValue, MlpError>;

    fn add(self, right: BiasValue) -> Self::Output {
        HiddenPreActivationValue::from_raw(self.as_f64() + right.as_f64())
    }
}

impl Mul<WeightValue> for InputValue {
    type Output = Result<WeightedProduct, MlpError>;

    fn mul(self, right: WeightValue) -> Self::Output {
        WeightedProduct::from_raw(self.as_f64() * right.as_f64())
    }
}

impl Mul<WeightValue> for HiddenActivationValue {
    type Output = Result<WeightedProduct, MlpError>;

    fn mul(self, right: WeightValue) -> Self::Output {
        WeightedProduct::from_raw(self.as_f64() * right.as_f64())
    }
}

/// Width of a vector space in the teaching MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VectorWidth(usize);

impl VectorWidth {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for VectorWidth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of matrix rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowCount(usize);

impl RowCount {
    fn from_raw(value: usize) -> Result<Self, MlpError> {
        if value == 0 {
            return Err(MlpError::empty_input(
                "RowCount::from_raw",
                "row count must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for RowCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Shape of a row-major weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    rows: RowCount,
    cols: VectorWidth,
}

impl MatrixShape {
    fn new(rows: RowCount, cols: VectorWidth) -> Self {
        Self { rows, cols }
    }

    /// Returns row count.
    pub fn rows(self) -> RowCount {
        self.rows
    }

    /// Returns column count.
    pub fn cols(self) -> VectorWidth {
        self.cols
    }
}

impl fmt::Display for MatrixShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}x{}", self.rows, self.cols)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DenseVector<T> {
    values: Vec<T>,
}

impl<T> DenseVector<T> {
    fn from_values(
        operation: &'static str,
        values: impl IntoIterator<Item = T>,
    ) -> Result<Self, MlpError> {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(MlpError::empty_input(operation, "vector cannot be empty"));
        }

        Ok(Self { values })
    }

    fn width(&self) -> VectorWidth {
        VectorWidth(self.values.len())
    }

    fn values(&self) -> impl ExactSizeIterator<Item = &T> + '_ {
        self.values.iter()
    }
}

impl<T: Copy> DenseVector<T> {
    fn first(&self, operation: &'static str, details: &'static str) -> Result<T, MlpError> {
        self.values
            .first()
            .copied()
            .ok_or(MlpError::invalid_output_layer(operation, details))
    }
}

/// Input signals entering the MLP.
#[derive(Debug, Clone, PartialEq)]
pub struct InputVector(DenseVector<InputValue>);

impl InputVector {
    /// Creates a non-empty finite input vector.
    pub fn from_values(values: impl IntoIterator<Item = InputValue>) -> Result<Self, MlpError> {
        Ok(Self(DenseVector::from_values(
            "InputVector::from_values",
            values,
        )?))
    }

    /// Returns input width.
    pub fn width(&self) -> VectorWidth {
        self.0.width()
    }

    /// Iterates over input values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &InputValue> + '_ {
        self.0.values()
    }
}

/// Hidden-layer values before ReLU.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenPreActivation(DenseVector<HiddenPreActivationValue>);

impl HiddenPreActivation {
    fn from_values(
        values: impl IntoIterator<Item = HiddenPreActivationValue>,
    ) -> Result<Self, MlpError> {
        Ok(Self(DenseVector::from_values(
            "HiddenPreActivation::from_values",
            values,
        )?))
    }

    /// Returns hidden pre-activation width.
    pub fn width(&self) -> VectorWidth {
        self.0.width()
    }

    /// Iterates over hidden pre-activation values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &HiddenPreActivationValue> + '_ {
        self.0.values()
    }
}

/// Hidden representation after ReLU.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenActivation(DenseVector<HiddenActivationValue>);

impl HiddenActivation {
    fn from_values(
        values: impl IntoIterator<Item = HiddenActivationValue>,
    ) -> Result<Self, MlpError> {
        Ok(Self(DenseVector::from_values(
            "HiddenActivation::from_values",
            values,
        )?))
    }

    /// Returns hidden activation width.
    pub fn width(&self) -> VectorWidth {
        self.0.width()
    }

    /// Iterates over hidden activation values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &HiddenActivationValue> + '_ {
        self.0.values()
    }
}

/// Layer bias values.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasVector(DenseVector<BiasValue>);

impl BiasVector {
    /// Creates non-empty finite bias values.
    pub fn from_values(values: impl IntoIterator<Item = BiasValue>) -> Result<Self, MlpError> {
        Ok(Self(DenseVector::from_values(
            "BiasVector::from_values",
            values,
        )?))
    }

    /// Returns bias width.
    pub fn width(&self) -> VectorWidth {
        self.0.width()
    }

    fn values(&self) -> impl ExactSizeIterator<Item = &BiasValue> + '_ {
        self.0.values()
    }
}

/// One row of learned weights.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightRow(DenseVector<WeightValue>);

impl WeightRow {
    /// Creates one non-empty finite weight row.
    pub fn from_values(values: impl IntoIterator<Item = WeightValue>) -> Result<Self, MlpError> {
        Ok(Self(DenseVector::from_values(
            "WeightRow::from_values",
            values,
        )?))
    }

    fn width(&self) -> VectorWidth {
        self.0.width()
    }

    fn values(&self) -> impl ExactSizeIterator<Item = &WeightValue> + '_ {
        self.0.values()
    }
}

/// Learned weights for one linear layer.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightMatrix {
    rows: Vec<WeightRow>,
    shape: MatrixShape,
}

impl WeightMatrix {
    /// Creates a weight matrix from typed rows.
    pub fn from_rows(rows: impl IntoIterator<Item = WeightRow>) -> Result<Self, MlpError> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        if rows.is_empty() {
            return Err(MlpError::empty_input(
                "WeightMatrix::from_rows",
                "matrix cannot be empty",
            ));
        }

        let cols = rows
            .first()
            .map(WeightRow::width)
            .ok_or(MlpError::empty_input(
                "WeightMatrix::from_rows",
                "matrix cannot be empty",
            ))?;
        for row in &rows {
            if row.width() != cols {
                return Err(MlpError::invalid_matrix_data(
                    "WeightMatrix::from_rows",
                    "all rows must have the same length",
                ));
            }
        }

        let shape = MatrixShape::new(RowCount::from_raw(rows.len())?, cols);
        Ok(Self { rows, shape })
    }

    /// Returns matrix shape.
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Returns row count, which is the output width.
    pub fn rows(&self) -> RowCount {
        self.shape.rows()
    }

    /// Returns column count, which is the input width.
    pub fn cols(&self) -> VectorWidth {
        self.shape.cols()
    }

    fn multiply_input(&self, vector: &InputVector) -> Result<DenseVector<WeightedSum>, MlpError> {
        let values = vector.values().copied().collect::<Vec<_>>();
        self.multiply_values("WeightMatrix::multiply_input", vector.width(), &values)
    }

    fn multiply_hidden(
        &self,
        vector: &HiddenActivation,
    ) -> Result<DenseVector<WeightedSum>, MlpError> {
        let values = vector.values().copied().collect::<Vec<_>>();
        self.multiply_values("WeightMatrix::multiply_hidden", vector.width(), &values)
    }

    fn multiply_values<T>(
        &self,
        operation: &'static str,
        vector_width: VectorWidth,
        values: &[T],
    ) -> Result<DenseVector<WeightedSum>, MlpError>
    where
        T: Copy + Mul<WeightValue, Output = Result<WeightedProduct, MlpError>>,
    {
        if self.cols() != vector_width {
            return Err(MlpError::dimension_mismatch(
                operation,
                "weight matrix",
                vec![self.rows().as_usize(), self.cols().as_usize()],
                "input vector",
                vec![vector_width.as_usize()],
                "matrix columns must equal input vector width",
            ));
        }

        let mut output = Vec::with_capacity(self.rows().as_usize());
        for row in &self.rows {
            let mut sum = WeightedSum::zero()?;
            for (value, weight) in values.iter().zip(row.values()) {
                sum = (sum + (*value * *weight)?)?;
            }
            output.push(sum);
        }

        DenseVector::from_values(operation, output)
    }
}

/// A sigmoid prediction in the range `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Prediction(f64);

impl Prediction {
    fn from_raw(value: f64) -> Result<Self, MlpError> {
        let value = finite("prediction", value)?;
        if !(0.0..=1.0).contains(&value) {
            return Err(MlpError::out_of_range("prediction", "0..=1", value));
        }

        Ok(Self(value))
    }
}

impl TryFrom<f64> for Prediction {
    type Error = MlpError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for Prediction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// The first linear map from input space into hidden space.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenLayer {
    weights: WeightMatrix,
    bias: BiasVector,
}

impl HiddenLayer {
    /// Creates a hidden layer with compatible weights and bias.
    pub fn new(weights: WeightMatrix, bias: BiasVector) -> Result<Self, MlpError> {
        validate_layer("HiddenLayer::new", &weights, &bias)?;
        Ok(Self { weights, bias })
    }

    /// Returns expected input width.
    pub fn input_dim(&self) -> VectorWidth {
        self.weights.cols()
    }

    /// Returns hidden representation width.
    pub fn output_dim(&self) -> VectorWidth {
        VectorWidth(self.weights.rows().as_usize())
    }

    /// Applies the linear hidden layer before activation.
    pub fn forward(&self, input: &InputVector) -> Result<HiddenPreActivation, MlpError> {
        let z = self.weights.multiply_input(input)?;
        add_bias("HiddenLayer::forward", z, &self.bias)
    }
}

/// ReLU activation for hidden representations.
pub fn relu(values: &HiddenPreActivation) -> Result<HiddenActivation, MlpError> {
    HiddenActivation::from_values(
        values
            .values()
            .map(|value| value.relu())
            .collect::<Result<Vec<_>, _>>()?,
    )
}

/// The final linear layer from hidden space to one binary-classification logit.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputLayer {
    weights: WeightMatrix,
    bias: BiasVector,
}

impl OutputLayer {
    /// Creates a one-logit output layer.
    pub fn new(weights: WeightMatrix, bias: BiasVector) -> Result<Self, MlpError> {
        validate_layer("OutputLayer::new", &weights, &bias)?;

        if weights.rows().as_usize() != 1 || bias.width().as_usize() != 1 {
            return Err(MlpError::invalid_output_layer(
                "OutputLayer::new",
                "this teaching output layer must produce exactly one logit",
            ));
        }

        Ok(Self { weights, bias })
    }

    /// Returns expected hidden width.
    pub fn input_dim(&self) -> VectorWidth {
        self.weights.cols()
    }

    /// Applies the output layer.
    pub fn forward(&self, hidden: &HiddenActivation) -> Result<OutputLogit, MlpError> {
        let z = self.weights.multiply_hidden(hidden)?;
        let z = add_bias("OutputLayer::forward", z, &self.bias)?;
        Ok(OutputLogit::from(z.0.first(
            "OutputLayer::forward",
            "one-logit output layer produced no output value",
        )?))
    }
}

/// Full learner-visible trace for one MLP forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardTrace {
    hidden_pre_activation: HiddenPreActivation,
    hidden_activation: HiddenActivation,
    output_logit: OutputLogit,
    prediction: Prediction,
}

impl ForwardTrace {
    /// Hidden-layer values before ReLU.
    pub fn hidden_pre_activation(&self) -> &HiddenPreActivation {
        &self.hidden_pre_activation
    }

    /// Hidden representation after ReLU.
    pub fn hidden_activation(&self) -> &HiddenActivation {
        &self.hidden_activation
    }

    /// Output logit before sigmoid.
    pub fn output_logit(&self) -> OutputLogit {
        self.output_logit
    }

    /// Final sigmoid prediction.
    pub fn prediction(&self) -> Prediction {
        self.prediction
    }
}

/// Publication class attached to an MLP forward trace before public release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpTraceVisibility {
    /// Safe to include in learner-facing public MLP material.
    Public,
    /// Useful for restricted study, but not public learner-facing material.
    ResearchRestricted,
    /// Must stay out of public learner-facing material.
    Private,
}

impl fmt::Display for MlpTraceVisibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Public => "public",
            Self::ResearchRestricted => "research-restricted",
            Self::Private => "private",
        };
        formatter.write_str(label)
    }
}

/// Typed decision at the MLP-trace publication boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicMlpDecision {
    /// The trace can appear in public learner-facing material.
    Publishable,
    /// The trace must stay out of public learner-facing material.
    Blocked,
}

/// MLP forward trace plus explicit publication review evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewedForwardTrace {
    trace: ForwardTrace,
    visibility: MlpTraceVisibility,
}

impl ReviewedForwardTrace {
    /// Creates a reviewed MLP forward trace.
    pub fn new(trace: ForwardTrace, visibility: MlpTraceVisibility) -> Self {
        Self { trace, visibility }
    }

    /// Returns the reviewed MLP forward trace.
    pub fn trace(&self) -> &ForwardTrace {
        &self.trace
    }

    /// Returns the publication class.
    pub fn visibility(&self) -> MlpTraceVisibility {
        self.visibility
    }

    /// Classifies whether this trace can enter public learner-facing material.
    pub fn release_decision(&self) -> PublicMlpDecision {
        match self.visibility {
            MlpTraceVisibility::Public => PublicMlpDecision::Publishable,
            MlpTraceVisibility::ResearchRestricted | MlpTraceVisibility::Private => {
                PublicMlpDecision::Blocked
            }
        }
    }

    fn into_trace(self) -> ForwardTrace {
        self.trace
    }
}

/// MLP forward trace checked for learner-facing public release.
#[derive(Debug, Clone, PartialEq)]
pub struct PublicForwardTrace(ForwardTrace);

impl PublicForwardTrace {
    /// Builds a public MLP trace only from a reviewed public trace.
    pub fn from_reviewed_trace(reviewed: ReviewedForwardTrace) -> Result<Self, MlpError> {
        if reviewed.release_decision() == PublicMlpDecision::Blocked {
            return Err(MlpError::invalid_public_trace(
                "PublicForwardTrace::from_reviewed_trace",
                "public MLP traces cannot include restricted or private representation evidence",
            ));
        }

        Ok(Self(reviewed.into_trace()))
    }

    /// Returns the checked MLP forward trace.
    pub fn trace(&self) -> &ForwardTrace {
        &self.0
    }

    /// Returns hidden-layer values before ReLU.
    pub fn hidden_pre_activation(&self) -> &HiddenPreActivation {
        self.0.hidden_pre_activation()
    }

    /// Returns hidden representation after ReLU.
    pub fn hidden_activation(&self) -> &HiddenActivation {
        self.0.hidden_activation()
    }

    /// Returns output logit before sigmoid.
    pub fn output_logit(&self) -> OutputLogit {
        self.0.output_logit()
    }

    /// Returns final sigmoid prediction.
    pub fn prediction(&self) -> Prediction {
        self.0.prediction()
    }
}

/// A two-layer MLP for binary teaching examples.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMlp {
    hidden: HiddenLayer,
    output: OutputLayer,
}

impl TinyMlp {
    /// Creates a tiny MLP with compatible hidden/output dimensions.
    pub fn new(hidden: HiddenLayer, output: OutputLayer) -> Result<Self, MlpError> {
        if hidden.output_dim() != output.input_dim() {
            return Err(MlpError::dimension_mismatch(
                "TinyMlp::new",
                "hidden layer output",
                vec![hidden.output_dim().as_usize()],
                "output layer input",
                vec![output.input_dim().as_usize()],
                "output layer must accept the hidden activation width",
            ));
        }

        Ok(Self { hidden, output })
    }

    /// A deterministic XOR-shaped network used by examples.
    pub fn xor_seed() -> Result<Self, MlpError> {
        let hidden = HiddenLayer::new(
            WeightMatrix::from_rows([
                WeightRow::from_values([
                    WeightValue::try_from(1.0)?,
                    WeightValue::try_from(-1.0)?,
                ])?,
                WeightRow::from_values([
                    WeightValue::try_from(-1.0)?,
                    WeightValue::try_from(1.0)?,
                ])?,
            ])?,
            BiasVector::from_values([BiasValue::try_from(0.0)?, BiasValue::try_from(0.0)?])?,
        )?;
        let output = OutputLayer::new(
            WeightMatrix::from_rows([WeightRow::from_values([
                WeightValue::try_from(5.0)?,
                WeightValue::try_from(5.0)?,
            ])?])?,
            BiasVector::from_values([BiasValue::try_from(-2.5)?])?,
        )?;
        Self::new(hidden, output)
    }

    /// Returns expected input width.
    pub fn input_dim(&self) -> VectorWidth {
        self.hidden.input_dim()
    }

    /// Returns hidden representation width.
    pub fn hidden_dim(&self) -> VectorWidth {
        self.hidden.output_dim()
    }

    /// Runs the full forward pass and returns learner-visible intermediate values.
    pub fn forward(&self, input: &InputVector) -> Result<ForwardTrace, MlpError> {
        let hidden_pre_activation = self.hidden.forward(input)?;
        let hidden_activation = relu(&hidden_pre_activation)?;
        let output_logit = self.output.forward(&hidden_activation)?;
        let prediction = sigmoid(output_logit)?;

        Ok(ForwardTrace {
            hidden_pre_activation,
            hidden_activation,
            output_logit,
            prediction,
        })
    }
}

/// Applies sigmoid to one logit.
pub fn sigmoid(logit: OutputLogit) -> Result<Prediction, MlpError> {
    Prediction::from_raw(1.0 / (1.0 + (-logit.as_f64()).exp()))
}

fn validate_layer(
    operation: &'static str,
    weights: &WeightMatrix,
    bias: &BiasVector,
) -> Result<(), MlpError> {
    if weights.rows().as_usize() != bias.width().as_usize() {
        return Err(MlpError::dimension_mismatch(
            operation,
            "weight matrix rows",
            vec![weights.rows().as_usize()],
            "bias vector",
            vec![bias.width().as_usize()],
            "each output unit needs exactly one bias",
        ));
    }

    Ok(())
}

fn add_bias(
    operation: &'static str,
    vector: DenseVector<WeightedSum>,
    bias: &BiasVector,
) -> Result<HiddenPreActivation, MlpError> {
    if vector.width() != bias.width() {
        return Err(MlpError::dimension_mismatch(
            operation,
            "linear output",
            vec![vector.width().as_usize()],
            "bias",
            vec![bias.width().as_usize()],
            "bias length must match layer output width",
        ));
    }

    HiddenPreActivation::from_values(
        vector
            .values()
            .zip(bias.values())
            .map(|(left, right)| *left + *right)
            .collect::<Result<Vec<_>, _>>()?,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        BiasValue, BiasVector, HiddenActivationValue, HiddenLayer, HiddenPreActivationValue,
        InputValue, InputVector, MlpError, MlpTraceVisibility, OutputLayer, Prediction,
        PublicForwardTrace, ReviewedForwardTrace, TinyMlp, WeightMatrix, WeightRow, WeightValue,
        relu,
    };

    fn input(left: InputValue, right: InputValue) -> Result<InputVector, MlpError> {
        InputVector::from_values([left, right])
    }

    fn weights(left: WeightValue, right: WeightValue) -> Result<WeightRow, MlpError> {
        WeightRow::from_values([left, right])
    }

    fn bias(values: impl IntoIterator<Item = BiasValue>) -> Result<BiasVector, MlpError> {
        BiasVector::from_values(values)
    }

    fn forward_trace() -> Result<super::ForwardTrace, MlpError> {
        TinyMlp::xor_seed()?.forward(&input(
            InputValue::try_from(1.0)?,
            InputValue::try_from(0.0)?,
        )?)
    }

    #[test]
    fn hidden_layer_finds_directional_features() -> Result<(), MlpError> {
        let hidden = HiddenLayer::new(
            WeightMatrix::from_rows([
                weights(WeightValue::try_from(1.0)?, WeightValue::try_from(-1.0)?)?,
                weights(WeightValue::try_from(-1.0)?, WeightValue::try_from(1.0)?)?,
            ])?,
            bias([BiasValue::try_from(0.0)?, BiasValue::try_from(0.0)?])?,
        )?;

        let pre = hidden.forward(&input(
            InputValue::try_from(1.0)?,
            InputValue::try_from(0.0)?,
        )?)?;
        let activation = relu(&pre)?;
        let pre_values = pre.values().copied().collect::<Vec<_>>();
        let activation_values = activation.values().copied().collect::<Vec<_>>();

        assert_eq!(
            pre_values,
            vec![
                HiddenPreActivationValue::try_from(1.0)?,
                HiddenPreActivationValue::try_from(-1.0)?
            ]
        );
        assert_eq!(
            activation_values,
            vec![
                HiddenActivationValue::try_from(1.0)?,
                HiddenActivationValue::try_from(0.0)?
            ]
        );
        Ok(())
    }

    #[test]
    fn tiny_mlp_separates_xor_corners() -> Result<(), MlpError> {
        let mlp = TinyMlp::xor_seed()?;

        let false_low = mlp
            .forward(&input(
                InputValue::try_from(0.0)?,
                InputValue::try_from(0.0)?,
            )?)?
            .prediction();
        let true_left = mlp
            .forward(&input(
                InputValue::try_from(1.0)?,
                InputValue::try_from(0.0)?,
            )?)?
            .prediction();
        let true_right = mlp
            .forward(&input(
                InputValue::try_from(0.0)?,
                InputValue::try_from(1.0)?,
            )?)?
            .prediction();
        let false_high = mlp
            .forward(&input(
                InputValue::try_from(1.0)?,
                InputValue::try_from(1.0)?,
            )?)?
            .prediction();

        assert!(false_low < Prediction::try_from(0.1)?);
        assert!(true_left > Prediction::try_from(0.9)?);
        assert!(true_right > Prediction::try_from(0.9)?);
        assert!(false_high < Prediction::try_from(0.1)?);
        Ok(())
    }

    #[test]
    fn hidden_layer_rejects_bias_shape_mismatch() -> Result<(), MlpError> {
        let error = HiddenLayer::new(
            WeightMatrix::from_rows([
                weights(WeightValue::try_from(1.0)?, WeightValue::try_from(0.0)?)?,
                weights(WeightValue::try_from(0.0)?, WeightValue::try_from(1.0)?)?,
            ])?,
            bias([BiasValue::try_from(0.0)?])?,
        );

        assert!(matches!(error, Err(MlpError::DimensionMismatch { .. })));
        Ok(())
    }

    #[test]
    fn output_layer_rejects_multiple_logits() -> Result<(), MlpError> {
        let error = OutputLayer::new(
            WeightMatrix::from_rows([
                weights(WeightValue::try_from(1.0)?, WeightValue::try_from(0.0)?)?,
                weights(WeightValue::try_from(0.0)?, WeightValue::try_from(1.0)?)?,
            ])?,
            bias([BiasValue::try_from(0.0)?, BiasValue::try_from(0.0)?])?,
        );

        assert!(matches!(error, Err(MlpError::InvalidOutputLayer { .. })));
        Ok(())
    }

    #[test]
    fn tiny_mlp_rejects_output_input_width_mismatch() -> Result<(), MlpError> {
        let hidden = HiddenLayer::new(
            WeightMatrix::from_rows([
                weights(WeightValue::try_from(1.0)?, WeightValue::try_from(-1.0)?)?,
                weights(WeightValue::try_from(-1.0)?, WeightValue::try_from(1.0)?)?,
            ])?,
            bias([BiasValue::try_from(0.0)?, BiasValue::try_from(0.0)?])?,
        )?;
        let output = OutputLayer::new(
            WeightMatrix::from_rows([WeightRow::from_values([
                WeightValue::try_from(1.0)?,
                WeightValue::try_from(1.0)?,
                WeightValue::try_from(1.0)?,
            ])?])?,
            bias([BiasValue::try_from(0.0)?])?,
        )?;

        let error = TinyMlp::new(hidden, output);
        assert!(matches!(error, Err(MlpError::DimensionMismatch { .. })));
        Ok(())
    }

    #[test]
    fn prediction_rejects_values_outside_probability_range() {
        let too_high = Prediction::try_from(1.2);
        let too_low = Prediction::try_from(-0.1);

        assert!(matches!(too_high, Err(MlpError::OutOfRange { .. })));
        assert!(matches!(too_low, Err(MlpError::OutOfRange { .. })));
    }

    #[test]
    fn public_forward_trace_accepts_public_reviewed_trace() -> Result<(), MlpError> {
        let public_trace = PublicForwardTrace::from_reviewed_trace(ReviewedForwardTrace::new(
            forward_trace()?,
            MlpTraceVisibility::Public,
        ))?;

        assert!(public_trace.prediction() > Prediction::try_from(0.9)?);
        assert_eq!(
            public_trace.hidden_activation().width(),
            TinyMlp::xor_seed()?.hidden_dim()
        );
        Ok(())
    }

    #[test]
    fn public_forward_trace_blocks_restricted_and_private_traces() -> Result<(), MlpError> {
        let restricted = PublicForwardTrace::from_reviewed_trace(ReviewedForwardTrace::new(
            forward_trace()?,
            MlpTraceVisibility::ResearchRestricted,
        ));
        let private = PublicForwardTrace::from_reviewed_trace(ReviewedForwardTrace::new(
            forward_trace()?,
            MlpTraceVisibility::Private,
        ));

        assert!(matches!(
            restricted.err(),
            Some(MlpError::InvalidPublicTrace { .. })
        ));
        assert!(matches!(
            private.err(),
            Some(MlpError::InvalidPublicTrace { .. })
        ));
        Ok(())
    }
}
