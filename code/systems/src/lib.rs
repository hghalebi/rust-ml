//! Typed systems measurements for the CS336 Rust equivalent track.
//!
//! This crate keeps the R2 systems work small and concrete. It does not try to
//! benchmark a full model. It teaches the resource vocabulary first:
//!
//! ```text
//! shape -> elements -> bytes
//! shape -> operations -> FLOPs
//! repeated timings -> median timing
//! FLOPs / bytes -> arithmetic intensity
//! bytes / bandwidth -> transfer time
//! ```
//!
//! Raw learner literals enter through `TryFrom` adapters. Public teaching APIs
//! then move through semantic values such as [`BatchSize`], [`SequenceLength`],
//! [`ModelWidth`], [`Bytes`], [`BytesPerSecond`], [`Flops`], and
//! [`ElapsedNanos`].

pub mod error;

use std::{
    fmt,
    ops::{Add, Div, Mul},
};

use error::SystemsError;

pub use error::SystemsError as Error;

const NANOS_PER_SECOND: u128 = 1_000_000_000;

fn nonzero_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, SystemsError> {
    if value == 0 {
        return Err(SystemsError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u64(
    role: &'static str,
    operation: &'static str,
    value: u64,
) -> Result<u64, SystemsError> {
    if value == 0 {
        return Err(SystemsError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u128(
    role: &'static str,
    operation: &'static str,
    value: u128,
) -> Result<u128, SystemsError> {
    if value == 0 {
        return Err(SystemsError::empty_input(operation, role));
    }

    Ok(value)
}

fn checked_mul(operation: &'static str, left: usize, right: usize) -> Result<usize, SystemsError> {
    left.checked_mul(right).ok_or(SystemsError::overflow(
        operation,
        "multiplication exceeded usize",
    ))
}

fn checked_u64_from_usize(operation: &'static str, value: usize) -> Result<u64, SystemsError> {
    u64::try_from(value).map_err(|_| SystemsError::overflow(operation, "value exceeded u64"))
}

macro_rules! count_type {
    ($name:ident, $doc:literal, $role:literal, $operation:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(usize);

        impl $name {
            fn as_usize(self) -> usize {
                self.0
            }
        }

        impl TryFrom<usize> for $name {
            type Error = SystemsError;

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                Ok(Self(nonzero_usize($role, $operation, value)?))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, formatter)
            }
        }
    };
}

count_type!(
    BatchSize,
    "Number of examples processed together.",
    "batch size must be greater than zero",
    "BatchSize::try_from"
);
count_type!(
    SequenceLength,
    "Number of token positions in one sequence.",
    "sequence length must be greater than zero",
    "SequenceLength::try_from"
);
count_type!(
    ModelWidth,
    "Number of channels in one token representation.",
    "model width must be greater than zero",
    "ModelWidth::try_from"
);
count_type!(
    RowCount,
    "Number of rows in a matrix-shaped operation.",
    "row count must be greater than zero",
    "RowCount::try_from"
);
count_type!(
    ColumnCount,
    "Number of columns in a matrix-shaped operation.",
    "column count must be greater than zero",
    "ColumnCount::try_from"
);

/// Number of scalar elements in a tensor-like value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElementCount(usize);

impl ElementCount {
    fn from_raw(operation: &'static str, value: usize) -> Result<Self, SystemsError> {
        Ok(Self(nonzero_usize(
            "element count must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for ElementCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of bytes occupied or moved by a computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(u64);

impl Bytes {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, SystemsError> {
        Ok(Self(nonzero_u64(
            "bytes must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for Bytes {
    type Error = SystemsError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_raw("Bytes::try_from", value)
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} bytes", self.0)
    }
}

impl Add for Bytes {
    type Output = Result<Bytes, SystemsError>;

    fn add(self, right: Bytes) -> Self::Output {
        let total = self
            .as_u64()
            .checked_add(right.as_u64())
            .ok_or(SystemsError::overflow(
                "Bytes::add",
                "byte count exceeded u64",
            ))?;
        Bytes::from_raw("Bytes::add", total)
    }
}

impl Div<BytesPerSecond> for Bytes {
    type Output = Result<ElapsedNanos, SystemsError>;

    fn div(self, bandwidth: BytesPerSecond) -> Self::Output {
        let numerator = u128::from(self.as_u64())
            .checked_mul(NANOS_PER_SECOND)
            .ok_or(SystemsError::overflow(
                "Bytes::div<BytesPerSecond>",
                "byte-nanosecond product exceeded u128",
            ))?;
        let denominator = bandwidth.as_u128();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;
        let elapsed = if remainder == 0 {
            quotient
        } else {
            quotient.checked_add(1).ok_or(SystemsError::overflow(
                "Bytes::div<BytesPerSecond>",
                "rounded transfer-time exceeded u128",
            ))?
        };

        ElapsedNanos::from_raw("Bytes::div<BytesPerSecond>", elapsed)
    }
}

/// Sustained memory bandwidth in bytes per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BytesPerSecond(u128);

impl BytesPerSecond {
    fn from_raw(operation: &'static str, value: u128) -> Result<Self, SystemsError> {
        Ok(Self(nonzero_u128(
            "bandwidth must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u128(self) -> u128 {
        self.0
    }
}

impl TryFrom<u128> for BytesPerSecond {
    type Error = SystemsError;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Self::from_raw("BytesPerSecond::try_from", value)
    }
}

impl fmt::Display for BytesPerSecond {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} bytes/s", self.0)
    }
}

/// Number of bytes used by one scalar element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElementSize(Bytes);

impl ElementSize {
    /// Returns the size of one `f32` scalar.
    pub fn float32() -> Self {
        Self(Bytes(4))
    }

    /// Returns the size of one `f64` scalar.
    pub fn float64() -> Self {
        Self(Bytes(8))
    }

    fn bytes(self) -> Bytes {
        self.0
    }
}

impl fmt::Display for ElementSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl Mul<ElementSize> for ElementCount {
    type Output = Result<Bytes, SystemsError>;

    fn mul(self, element_size: ElementSize) -> Self::Output {
        let elements = checked_u64_from_usize("ElementCount::mul", self.as_usize())?;
        let bytes =
            elements
                .checked_mul(element_size.bytes().as_u64())
                .ok_or(SystemsError::overflow(
                    "ElementCount::mul",
                    "byte count exceeded u64",
                ))?;

        Bytes::from_raw("ElementCount::mul", bytes)
    }
}

/// Number of floating-point operations in a teaching estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Flops(u64);

impl Flops {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, SystemsError> {
        Ok(Self(nonzero_u64(
            "FLOP count must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for Flops {
    type Error = SystemsError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_raw("Flops::try_from", value)
    }
}

impl fmt::Display for Flops {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} FLOPs", self.0)
    }
}

impl Add for Flops {
    type Output = Result<Flops, SystemsError>;

    fn add(self, right: Flops) -> Self::Output {
        let total = self
            .as_u64()
            .checked_add(right.as_u64())
            .ok_or(SystemsError::overflow(
                "Flops::add",
                "FLOP count exceeded u64",
            ))?;
        Flops::from_raw("Flops::add", total)
    }
}

impl Div<Bytes> for Flops {
    type Output = Result<ArithmeticIntensity, SystemsError>;

    fn div(self, bytes: Bytes) -> Self::Output {
        ArithmeticIntensity::from_raw(self.as_u64() as f64 / bytes.as_u64() as f64)
    }
}

/// Elapsed time for one measured run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElapsedNanos(u128);

impl ElapsedNanos {
    fn from_raw(operation: &'static str, value: u128) -> Result<Self, SystemsError> {
        Ok(Self(nonzero_u128(
            "elapsed time must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u128(self) -> u128 {
        self.0
    }
}

impl TryFrom<u128> for ElapsedNanos {
    type Error = SystemsError;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Self::from_raw("ElapsedNanos::try_from", value)
    }
}

impl fmt::Display for ElapsedNanos {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ns", self.0)
    }
}

/// A finite FLOPs-per-byte ratio.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ArithmeticIntensity(f64);

impl ArithmeticIntensity {
    fn from_raw(value: f64) -> Result<Self, SystemsError> {
        if !value.is_finite() {
            return Err(SystemsError::non_finite_value(
                "arithmetic intensity",
                value,
            ));
        }
        if value < 0.0 {
            return Err(SystemsError::out_of_range(
                "arithmetic intensity",
                ">= 0",
                value,
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for ArithmeticIntensity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.4} FLOPs/byte", self.0)
    }
}

/// A named memory tier used for accelerator mental models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLevel {
    /// Registers closest to the arithmetic units.
    RegisterFile,
    /// Shared or local memory visible to a small group of execution lanes.
    SharedMemory,
    /// Device memory with high bandwidth but longer access paths than local memory.
    HighBandwidthMemory,
    /// Host memory reached across a slower device boundary.
    HostMemory,
}

impl fmt::Display for MemoryLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::RegisterFile => "register file",
            Self::SharedMemory => "shared memory",
            Self::HighBandwidthMemory => "high-bandwidth memory",
            Self::HostMemory => "host memory",
        };
        formatter.write_str(name)
    }
}

/// Estimated movement of bytes through one memory tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryTransfer {
    level: MemoryLevel,
    bytes_moved: Bytes,
    bandwidth: BytesPerSecond,
}

impl MemoryTransfer {
    /// Creates a typed memory-transfer estimate.
    pub fn new(level: MemoryLevel, bytes_moved: Bytes, bandwidth: BytesPerSecond) -> Self {
        Self {
            level,
            bytes_moved,
            bandwidth,
        }
    }

    /// Returns the memory tier.
    pub fn level(&self) -> MemoryLevel {
        self.level
    }

    /// Returns the transferred bytes.
    pub fn bytes_moved(&self) -> Bytes {
        self.bytes_moved
    }

    /// Returns the assumed sustained bandwidth.
    pub fn bandwidth(&self) -> BytesPerSecond {
        self.bandwidth
    }

    /// Estimates transfer time from bytes and sustained bandwidth.
    pub fn estimated_elapsed(&self) -> Result<ElapsedNanos, SystemsError> {
        self.bytes_moved / self.bandwidth
    }
}

/// A non-empty human-readable stage label.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageName(String);

impl StageName {
    fn from_owned(value: String) -> Result<Self, SystemsError> {
        if value.trim().is_empty() {
            return Err(SystemsError::empty_input(
                "StageName::try_from",
                "stage name cannot be empty",
            ));
        }
        Ok(Self(value))
    }
}

impl TryFrom<&str> for StageName {
    type Error = SystemsError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_owned(value.to_owned())
    }
}

impl TryFrom<String> for StageName {
    type Error = SystemsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_owned(value)
    }
}

impl fmt::Display for StageName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Shape of an activation tensor in the tiny systems examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationShape {
    batch_size: BatchSize,
    sequence_length: SequenceLength,
    model_width: ModelWidth,
}

impl ActivationShape {
    /// Creates an activation shape from semantic dimensions.
    pub fn new(
        batch_size: BatchSize,
        sequence_length: SequenceLength,
        model_width: ModelWidth,
    ) -> Self {
        Self {
            batch_size,
            sequence_length,
            model_width,
        }
    }

    /// Returns the number of scalar elements.
    pub fn elements(&self) -> Result<ElementCount, SystemsError> {
        let batch_sequence = checked_mul(
            "ActivationShape::elements",
            self.batch_size.as_usize(),
            self.sequence_length.as_usize(),
        )?;
        let elements = checked_mul(
            "ActivationShape::elements",
            batch_sequence,
            self.model_width.as_usize(),
        )?;
        ElementCount::from_raw("ActivationShape::elements", elements)
    }

    /// Estimates activation memory for the given element size.
    pub fn activation_bytes(&self, element_size: ElementSize) -> Result<Bytes, SystemsError> {
        self.elements()? * element_size
    }
}

/// Shape of a matrix-vector multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixVectorShape {
    rows: RowCount,
    cols: ColumnCount,
}

impl MatrixVectorShape {
    /// Creates a matrix-vector shape.
    pub fn new(rows: RowCount, cols: ColumnCount) -> Self {
        Self { rows, cols }
    }

    /// Estimates multiply-add FLOPs for one matrix-vector product.
    pub fn multiply_add_flops(&self) -> Result<Flops, SystemsError> {
        let pairs = checked_mul(
            "MatrixVectorShape::multiply_add_flops",
            self.rows.as_usize(),
            self.cols.as_usize(),
        )?;
        let flops = checked_mul("MatrixVectorShape::multiply_add_flops", pairs, 2)?;
        Flops::from_raw(
            "MatrixVectorShape::multiply_add_flops",
            checked_u64_from_usize("MatrixVectorShape::multiply_add_flops", flops)?,
        )
    }

    /// Estimates bytes read for weights and input plus bytes written for output.
    pub fn bytes_moved(&self, element_size: ElementSize) -> Result<Bytes, SystemsError> {
        let weight_elements = checked_mul(
            "MatrixVectorShape::bytes_moved",
            self.rows.as_usize(),
            self.cols.as_usize(),
        )?;
        let total_elements = weight_elements
            .checked_add(self.cols.as_usize())
            .and_then(|value| value.checked_add(self.rows.as_usize()))
            .ok_or(SystemsError::overflow(
                "MatrixVectorShape::bytes_moved",
                "element count exceeded usize",
            ))?;
        ElementCount::from_raw("MatrixVectorShape::bytes_moved", total_elements)? * element_size
    }
}

/// Teaching estimate for one dense self-attention pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionEstimate {
    sequence_length: SequenceLength,
    model_width: ModelWidth,
}

impl AttentionEstimate {
    /// Creates an attention estimate from semantic dimensions.
    pub fn new(sequence_length: SequenceLength, model_width: ModelWidth) -> Self {
        Self {
            sequence_length,
            model_width,
        }
    }

    /// Estimates query-key score FLOPs for all token pairs.
    pub fn score_flops(&self) -> Result<Flops, SystemsError> {
        self.token_pair_width_flops("AttentionEstimate::score_flops")
    }

    /// Estimates value mixing FLOPs for all token pairs.
    pub fn value_mix_flops(&self) -> Result<Flops, SystemsError> {
        self.token_pair_width_flops("AttentionEstimate::value_mix_flops")
    }

    /// Estimates score plus value-mixing FLOPs.
    pub fn total_flops(&self) -> Result<Flops, SystemsError> {
        self.score_flops()? + self.value_mix_flops()?
    }

    /// Estimates bytes required to materialize the dense score matrix.
    pub fn score_matrix_bytes(&self, element_size: ElementSize) -> Result<Bytes, SystemsError> {
        let entries = checked_mul(
            "AttentionEstimate::score_matrix_bytes",
            self.sequence_length.as_usize(),
            self.sequence_length.as_usize(),
        )?;
        ElementCount::from_raw("AttentionEstimate::score_matrix_bytes", entries)? * element_size
    }

    fn token_pair_width_flops(&self, operation: &'static str) -> Result<Flops, SystemsError> {
        let token_pairs = checked_mul(
            operation,
            self.sequence_length.as_usize(),
            self.sequence_length.as_usize(),
        )?;
        let multiply_add_pairs = checked_mul(operation, token_pairs, self.model_width.as_usize())?;
        let flops = checked_mul(operation, multiply_add_pairs, 2)?;
        Flops::from_raw(operation, checked_u64_from_usize(operation, flops)?)
    }
}

/// One measured or estimated stage in a systems trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageMeasurement {
    name: StageName,
    elapsed: ElapsedNanos,
    flops: Flops,
    bytes_moved: Bytes,
}

impl StageMeasurement {
    /// Creates a typed stage measurement.
    pub fn new(name: StageName, elapsed: ElapsedNanos, flops: Flops, bytes_moved: Bytes) -> Self {
        Self {
            name,
            elapsed,
            flops,
            bytes_moved,
        }
    }

    /// Returns the stage label.
    pub fn name(&self) -> &StageName {
        &self.name
    }

    /// Returns elapsed time.
    pub fn elapsed(&self) -> ElapsedNanos {
        self.elapsed
    }

    /// Returns FLOPs.
    pub fn flops(&self) -> Flops {
        self.flops
    }

    /// Returns moved bytes.
    pub fn bytes_moved(&self) -> Bytes {
        self.bytes_moved
    }

    /// Computes FLOPs per byte moved.
    pub fn arithmetic_intensity(&self) -> Result<ArithmeticIntensity, SystemsError> {
        self.flops / self.bytes_moved
    }
}

/// Non-empty repeated measurements for one stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageMeasurements(Vec<StageMeasurement>);

impl StageMeasurements {
    /// Creates a checked non-empty measurement set.
    pub fn from_measurements(
        measurements: impl IntoIterator<Item = StageMeasurement>,
    ) -> Result<Self, SystemsError> {
        let measurements = measurements.into_iter().collect::<Vec<_>>();
        if measurements.is_empty() {
            return Err(SystemsError::empty_input(
                "StageMeasurements::from_measurements",
                "measurement set cannot be empty",
            ));
        }
        Ok(Self(measurements))
    }

    /// Iterates over measurements.
    pub fn measurements(&self) -> impl ExactSizeIterator<Item = &StageMeasurement> + '_ {
        self.0.iter()
    }

    /// Returns the median elapsed time.
    pub fn median_elapsed(&self) -> Result<ElapsedNanos, SystemsError> {
        let mut elapsed = self
            .0
            .iter()
            .map(|measurement| measurement.elapsed.as_u128())
            .collect::<Vec<_>>();
        elapsed.sort_unstable();
        let midpoint = elapsed.len() / 2;
        let median = if elapsed.len() % 2 == 0 {
            let total = elapsed[midpoint - 1].checked_add(elapsed[midpoint]).ok_or(
                SystemsError::overflow(
                    "StageMeasurements::median_elapsed",
                    "elapsed time exceeded u128",
                ),
            )?;
            total / 2
        } else {
            elapsed[midpoint]
        };

        ElapsedNanos::from_raw("StageMeasurements::median_elapsed", median)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActivationShape, AttentionEstimate, BatchSize, Bytes, BytesPerSecond, ColumnCount,
        ElapsedNanos, ElementSize, Flops, MatrixVectorShape, MemoryLevel, MemoryTransfer,
        ModelWidth, RowCount, SequenceLength, StageMeasurement, StageMeasurements, StageName,
    };
    use crate::error::SystemsError;

    #[test]
    fn activation_shape_counts_elements_and_bytes() -> Result<(), SystemsError> {
        let shape = ActivationShape::new(
            BatchSize::try_from(2)?,
            SequenceLength::try_from(3)?,
            ModelWidth::try_from(4)?,
        );

        assert_eq!(shape.elements()?.to_string(), "24");
        assert_eq!(
            shape.activation_bytes(ElementSize::float32())?.to_string(),
            "96 bytes"
        );
        Ok(())
    }

    #[test]
    fn attention_estimate_counts_score_and_value_mix_flops() -> Result<(), SystemsError> {
        let estimate =
            AttentionEstimate::new(SequenceLength::try_from(3)?, ModelWidth::try_from(4)?);

        assert_eq!(estimate.score_flops()?.to_string(), "72 FLOPs");
        assert_eq!(estimate.value_mix_flops()?.to_string(), "72 FLOPs");
        assert_eq!(estimate.total_flops()?.to_string(), "144 FLOPs");
        assert_eq!(
            estimate
                .score_matrix_bytes(ElementSize::float32())?
                .to_string(),
            "36 bytes"
        );
        Ok(())
    }

    #[test]
    fn matrix_vector_shape_estimates_flops_and_bytes() -> Result<(), SystemsError> {
        let shape = MatrixVectorShape::new(RowCount::try_from(3)?, ColumnCount::try_from(4)?);

        assert_eq!(shape.multiply_add_flops()?.to_string(), "24 FLOPs");
        assert_eq!(
            shape.bytes_moved(ElementSize::float32())?.to_string(),
            "76 bytes"
        );
        Ok(())
    }

    #[test]
    fn stage_measurement_reports_arithmetic_intensity() -> Result<(), SystemsError> {
        let measurement = StageMeasurement::new(
            StageName::try_from("matvec")?,
            ElapsedNanos::try_from(1_000_u128)?,
            Flops::try_from(200_u64)?,
            Bytes::try_from(100_u64)?,
        );

        assert_eq!(
            measurement.arithmetic_intensity()?.to_string(),
            "2.0000 FLOPs/byte"
        );
        Ok(())
    }

    #[test]
    fn resource_ops_keep_units_readable() -> Result<(), SystemsError> {
        let activation_shape = ActivationShape::new(
            BatchSize::try_from(1)?,
            SequenceLength::try_from(3)?,
            ModelWidth::try_from(4)?,
        );
        let activation_memory = activation_shape.elements()? * ElementSize::float32();
        let total_flops = Flops::try_from(40_u64)? + Flops::try_from(20_u64)?;
        let intensity = total_flops? / activation_memory?;

        assert_eq!(intensity?.to_string(), "1.2500 FLOPs/byte");
        Ok(())
    }

    #[test]
    fn memory_transfer_estimates_time_from_bandwidth() -> Result<(), SystemsError> {
        let transfer = MemoryTransfer::new(
            MemoryLevel::HighBandwidthMemory,
            Bytes::try_from(1_024_u64)?,
            BytesPerSecond::try_from(512_u128)?,
        );

        assert_eq!(transfer.level().to_string(), "high-bandwidth memory");
        assert_eq!(transfer.bytes_moved().to_string(), "1024 bytes");
        assert_eq!(transfer.bandwidth().to_string(), "512 bytes/s");
        assert_eq!(transfer.estimated_elapsed()?.to_string(), "2000000000 ns");
        Ok(())
    }

    #[test]
    fn bandwidth_rounds_nonzero_transfer_up_to_one_nanosecond() -> Result<(), SystemsError> {
        let elapsed = Bytes::try_from(1_u64)? / BytesPerSecond::try_from(2_000_000_000_u128)?;

        assert_eq!(elapsed?.to_string(), "1 ns");
        Ok(())
    }

    #[test]
    fn bandwidth_rounding_does_not_overflow_for_large_denominators() -> Result<(), SystemsError> {
        let elapsed = Bytes::try_from(1_u64)? / BytesPerSecond::try_from(u128::MAX)?;

        assert_eq!(elapsed?.to_string(), "1 ns");
        Ok(())
    }

    #[test]
    fn repeated_measurements_report_median_elapsed() -> Result<(), SystemsError> {
        let measurements = StageMeasurements::from_measurements([
            StageMeasurement::new(
                StageName::try_from("run-a")?,
                ElapsedNanos::try_from(30_u128)?,
                Flops::try_from(10_u64)?,
                Bytes::try_from(5_u64)?,
            ),
            StageMeasurement::new(
                StageName::try_from("run-b")?,
                ElapsedNanos::try_from(10_u128)?,
                Flops::try_from(10_u64)?,
                Bytes::try_from(5_u64)?,
            ),
            StageMeasurement::new(
                StageName::try_from("run-c")?,
                ElapsedNanos::try_from(20_u128)?,
                Flops::try_from(10_u64)?,
                Bytes::try_from(5_u64)?,
            ),
        ])?;

        assert_eq!(measurements.median_elapsed()?.to_string(), "20 ns");
        Ok(())
    }

    #[test]
    fn semantic_counts_reject_zero() {
        let error = BatchSize::try_from(0).err();
        assert!(matches!(error, Some(SystemsError::EmptyInput { .. })));
    }

    #[test]
    fn bandwidth_rejects_zero() {
        let error = BytesPerSecond::try_from(0_u128).err();
        assert!(matches!(error, Some(SystemsError::EmptyInput { .. })));
    }
}
