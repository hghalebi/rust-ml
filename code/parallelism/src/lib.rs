//! Typed parallelism primitives for the CS336 Rust equivalent track.
//!
//! This crate keeps distributed-training ideas small and inspectable:
//!
//! ```text
//! WorldSize + RankIndex -> RankId
//! GlobalBatchSize / WorldSize -> LocalBatchSize
//! ModelWidth / WorldSize -> ShardWidth
//! LayerCount / WorldSize -> LayersPerRank
//! CollectiveTrace + ParallelTraceVisibility -> PublicParallelismReport
//! CommunicationBytes + CommunicationBytes -> CommunicationBytes
//! ```
//!
//! Raw learner literals enter through `TryFrom` adapters. Public teaching APIs
//! then move through semantic values such as ranks, worlds, shard sizes,
//! pipeline stages, communication rounds, byte budgets, and public-report
//! review boundaries.

pub mod error;

use std::{
    fmt,
    ops::{Add, Div, Mul},
};

use error::ParallelismError;

pub use error::ParallelismError as Error;

fn nonzero_usize(
    role: &'static str,
    operation: &'static str,
    value: usize,
) -> Result<usize, ParallelismError> {
    if value == 0 {
        return Err(ParallelismError::empty_input(operation, role));
    }

    Ok(value)
}

fn nonzero_u64(
    role: &'static str,
    operation: &'static str,
    value: u64,
) -> Result<u64, ParallelismError> {
    if value == 0 {
        return Err(ParallelismError::empty_input(operation, role));
    }

    Ok(value)
}

fn finite(role: &'static str, value: f64) -> Result<f64, ParallelismError> {
    if !value.is_finite() {
        return Err(ParallelismError::non_finite_value(role, value));
    }

    Ok(value)
}

fn checked_u64_add(
    operation: &'static str,
    left: u64,
    right: u64,
) -> Result<u64, ParallelismError> {
    left.checked_add(right).ok_or(ParallelismError::overflow(
        operation,
        "u64 addition overflowed",
    ))
}

fn checked_u64_mul(
    operation: &'static str,
    left: u64,
    right: u64,
) -> Result<u64, ParallelismError> {
    left.checked_mul(right).ok_or(ParallelismError::overflow(
        operation,
        "u64 multiplication overflowed",
    ))
}

fn exact_divide(
    operation: &'static str,
    numerator: usize,
    denominator: usize,
) -> Result<usize, ParallelismError> {
    if denominator == 0 {
        return Err(ParallelismError::empty_input(
            operation,
            "denominator must be greater than zero",
        ));
    }
    if !numerator.is_multiple_of(denominator) {
        return Err(ParallelismError::uneven_split(
            operation,
            "the value must divide evenly by world size for this teaching example",
        ));
    }

    Ok(numerator / denominator)
}

macro_rules! nonzero_count_type {
    ($name:ident, $doc:literal, $role:literal, $operation:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(usize);

        impl TryFrom<usize> for $name {
            type Error = ParallelismError;

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

nonzero_count_type!(
    WorldSize,
    "Number of ranks participating in one distributed job.",
    "world size must be greater than zero",
    "WorldSize::try_from"
);
nonzero_count_type!(
    GlobalBatchSize,
    "Number of examples before data parallel sharding.",
    "global batch size must be greater than zero",
    "GlobalBatchSize::try_from"
);
nonzero_count_type!(
    LocalBatchSize,
    "Number of examples owned by one data-parallel rank.",
    "local batch size must be greater than zero",
    "LocalBatchSize::try_from"
);
nonzero_count_type!(
    ModelWidth,
    "Width of a model representation before tensor parallel sharding.",
    "model width must be greater than zero",
    "ModelWidth::try_from"
);
nonzero_count_type!(
    ShardWidth,
    "Width owned by one tensor-parallel rank.",
    "shard width must be greater than zero",
    "ShardWidth::try_from"
);
nonzero_count_type!(
    LayerCount,
    "Number of layers before pipeline parallel sharding.",
    "layer count must be greater than zero",
    "LayerCount::try_from"
);
nonzero_count_type!(
    LayersPerRank,
    "Number of layers owned by one pipeline stage.",
    "layers per rank must be greater than zero",
    "LayersPerRank::try_from"
);
nonzero_count_type!(
    MicroBatchCount,
    "Number of micro-batches used to fill a pipeline.",
    "micro-batch count must be greater than zero",
    "MicroBatchCount::try_from"
);

impl WorldSize {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl GlobalBatchSize {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl ModelWidth {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl LayerCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl MicroBatchCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

/// Raw rank index before it is checked against a world size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankIndex(usize);

impl RankIndex {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for RankIndex {
    type Error = ParallelismError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

impl fmt::Display for RankIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Rank identity after validation against a world size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankId(RankIndex);

impl RankId {
    /// Creates a rank identity after checking that it belongs to the world.
    pub fn new(index: RankIndex, world_size: WorldSize) -> Result<Self, ParallelismError> {
        if index.as_usize() >= world_size.as_usize() {
            return Err(ParallelismError::count_out_of_range(
                "rank index",
                "0..world_size",
                index.as_usize(),
            ));
        }

        Ok(Self(index))
    }

    fn as_usize(self) -> usize {
        self.0.as_usize()
    }
}

impl fmt::Display for RankId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "rank {}", self.as_usize())
    }
}

/// Finite scalar stored in a tiny teaching tensor.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TensorElement(f64);

impl TensorElement {
    fn as_f64(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for TensorElement {
    type Error = ParallelismError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Ok(Self(finite("tensor element", value)?))
    }
}

impl fmt::Display for TensorElement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:.1}", self.0)
    }
}

/// Non-empty one-dimensional teaching tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorLine {
    values: Vec<TensorElement>,
}

impl TensorLine {
    /// Builds a non-empty tensor line from validated scalar values.
    pub fn from_values(
        values: impl IntoIterator<Item = TensorElement>,
    ) -> Result<Self, ParallelismError> {
        let values = values.into_iter().collect::<Vec<_>>();
        nonzero_usize(
            "tensor line length must be greater than zero",
            "TensorLine::from_values",
            values.len(),
        )?;
        Ok(Self { values })
    }

    /// Returns the number of scalar values.
    pub fn length(&self) -> TensorLength {
        TensorLength(self.values.len())
    }

    /// Iterates over scalar values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = TensorElement> + '_ {
        self.values.iter().copied()
    }

    fn segment(
        &self,
        start: ShardStart,
        length: ShardLength,
    ) -> Result<TensorLine, ParallelismError> {
        let end = start.as_usize() + length.as_usize();
        let values = self
            .values
            .get(start.as_usize()..end)
            .ok_or(ParallelismError::count_out_of_range(
                "tensor segment",
                "inside tensor bounds",
                end,
            ))?
            .to_vec();

        TensorLine::from_values(values)
    }

    fn sum(&self) -> Result<TensorElement, ParallelismError> {
        let total = self
            .values()
            .try_fold(0.0, |sum, value| finite("tensor sum", sum + value.as_f64()))?;
        TensorElement::try_from(total)
    }
}

/// Non-zero tensor length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorLength(usize);

impl TensorLength {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for TensorLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Zero-based segment start in a tensor line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShardStart(usize);

impl ShardStart {
    fn from_raw(value: usize) -> Self {
        Self(value)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for ShardStart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Non-zero segment length in a tensor line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShardLength(usize);

impl ShardLength {
    fn from_raw(value: usize) -> Result<Self, ParallelismError> {
        Ok(Self(nonzero_usize(
            "shard length must be greater than zero",
            "ShardLength::from_raw",
            value,
        )?))
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for ShardLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl Div<WorldSize> for GlobalBatchSize {
    type Output = Result<LocalBatchSize, ParallelismError>;

    fn div(self, right: WorldSize) -> Self::Output {
        LocalBatchSize::try_from(exact_divide(
            "GlobalBatchSize::div",
            self.as_usize(),
            right.as_usize(),
        )?)
    }
}

impl Div<WorldSize> for ModelWidth {
    type Output = Result<ShardWidth, ParallelismError>;

    fn div(self, right: WorldSize) -> Self::Output {
        ShardWidth::try_from(exact_divide(
            "ModelWidth::div",
            self.as_usize(),
            right.as_usize(),
        )?)
    }
}

impl Div<WorldSize> for LayerCount {
    type Output = Result<LayersPerRank, ParallelismError>;

    fn div(self, right: WorldSize) -> Self::Output {
        LayersPerRank::try_from(exact_divide(
            "LayerCount::div",
            self.as_usize(),
            right.as_usize(),
        )?)
    }
}

impl Div<WorldSize> for TensorLength {
    type Output = Result<ShardLength, ParallelismError>;

    fn div(self, right: WorldSize) -> Self::Output {
        ShardLength::from_raw(exact_divide(
            "TensorLength::div",
            self.as_usize(),
            right.as_usize(),
        )?)
    }
}

/// One rank's owned shard of a tensor-like teaching value.
#[derive(Debug, Clone, PartialEq)]
pub struct RankShard {
    rank: RankId,
    start: ShardStart,
    values: TensorLine,
}

impl RankShard {
    fn new(rank: RankId, start: ShardStart, values: TensorLine) -> Self {
        Self {
            rank,
            start,
            values,
        }
    }

    /// Returns the rank that owns this shard.
    pub fn rank(&self) -> RankId {
        self.rank
    }

    /// Returns the start offset in the original tensor line.
    pub fn start(&self) -> ShardStart {
        self.start
    }

    /// Returns the local tensor values.
    pub fn values(&self) -> &TensorLine {
        &self.values
    }
}

/// A complete sharding plan over all ranks.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardPlan {
    strategy: ParallelStrategy,
    shards: Vec<RankShard>,
}

impl ShardPlan {
    fn new(strategy: ParallelStrategy, shards: Vec<RankShard>) -> Result<Self, ParallelismError> {
        nonzero_usize(
            "shard plan must contain at least one shard",
            "ShardPlan::new",
            shards.len(),
        )?;
        Ok(Self { strategy, shards })
    }

    /// Returns the parallel strategy that produced this plan.
    pub fn strategy(&self) -> ParallelStrategy {
        self.strategy
    }

    /// Iterates over rank shards.
    pub fn shards(&self) -> impl ExactSizeIterator<Item = &RankShard> + '_ {
        self.shards.iter()
    }
}

/// The parallel axis used by one teaching plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Split the batch axis and replicate model parameters.
    DataParallel,
    /// Split the model-width axis and communicate activations.
    TensorParallel,
    /// Split the layer-depth axis and communicate activations between stages.
    PipelineParallel,
}

impl fmt::Display for ParallelStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataParallel => formatter.write_str("data-parallel"),
            Self::TensorParallel => formatter.write_str("tensor-parallel"),
            Self::PipelineParallel => formatter.write_str("pipeline-parallel"),
        }
    }
}

/// Splits a tensor line evenly across ranks.
pub fn shard_tensor_line(
    values: TensorLine,
    world_size: WorldSize,
    strategy: ParallelStrategy,
) -> Result<ShardPlan, ParallelismError> {
    let shard_length = (values.length() / world_size)?;
    let mut shards = Vec::new();

    for rank_index in 0..world_size.as_usize() {
        let rank = RankId::new(RankIndex::try_from(rank_index)?, world_size)?;
        let start = ShardStart::from_raw(rank_index * shard_length.as_usize());
        let segment = values.segment(start, shard_length)?;
        shards.push(RankShard::new(rank, start, segment));
    }

    ShardPlan::new(strategy, shards)
}

/// Data-parallel shape summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataParallelLayout {
    global_batch: GlobalBatchSize,
    world_size: WorldSize,
    local_batch: LocalBatchSize,
}

impl DataParallelLayout {
    /// Creates a layout by splitting the global batch across ranks.
    pub fn new(
        global_batch: GlobalBatchSize,
        world_size: WorldSize,
    ) -> Result<Self, ParallelismError> {
        let local_batch = (global_batch / world_size)?;
        Ok(Self {
            global_batch,
            world_size,
            local_batch,
        })
    }

    /// Returns the original global batch size.
    pub fn global_batch(&self) -> GlobalBatchSize {
        self.global_batch
    }

    /// Returns the number of ranks.
    pub fn world_size(&self) -> WorldSize {
        self.world_size
    }

    /// Returns the local batch size per rank.
    pub fn local_batch(&self) -> LocalBatchSize {
        self.local_batch
    }
}

/// Tensor-parallel shape summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelLayout {
    model_width: ModelWidth,
    world_size: WorldSize,
    shard_width: ShardWidth,
}

impl TensorParallelLayout {
    /// Creates a layout by splitting model width across ranks.
    pub fn new(model_width: ModelWidth, world_size: WorldSize) -> Result<Self, ParallelismError> {
        let shard_width = (model_width / world_size)?;
        Ok(Self {
            model_width,
            world_size,
            shard_width,
        })
    }

    /// Returns the original model width.
    pub fn model_width(&self) -> ModelWidth {
        self.model_width
    }

    /// Returns the number of ranks.
    pub fn world_size(&self) -> WorldSize {
        self.world_size
    }

    /// Returns the local model width per rank.
    pub fn shard_width(&self) -> ShardWidth {
        self.shard_width
    }
}

/// Pipeline-parallel shape summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineLayout {
    layer_count: LayerCount,
    world_size: WorldSize,
    layers_per_rank: LayersPerRank,
}

impl PipelineLayout {
    /// Creates a layout by splitting layers across ranks.
    pub fn new(layer_count: LayerCount, world_size: WorldSize) -> Result<Self, ParallelismError> {
        let layers_per_rank = (layer_count / world_size)?;
        Ok(Self {
            layer_count,
            world_size,
            layers_per_rank,
        })
    }

    /// Returns the original layer count.
    pub fn layer_count(&self) -> LayerCount {
        self.layer_count
    }

    /// Returns the number of pipeline stages.
    pub fn stage_count(&self) -> PipelineStageCount {
        PipelineStageCount(self.world_size.as_usize())
    }

    /// Returns the number of layers per rank.
    pub fn layers_per_rank(&self) -> LayersPerRank {
        self.layers_per_rank
    }
}

/// Number of pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PipelineStageCount(usize);

impl PipelineStageCount {
    fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for PipelineStageCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Number of forward schedule slots in a simple pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PipelineScheduleLength(usize);

impl PipelineScheduleLength {
    fn from_raw(value: usize) -> Result<Self, ParallelismError> {
        Ok(Self(nonzero_usize(
            "pipeline schedule length must be greater than zero",
            "PipelineScheduleLength::from_raw",
            value,
        )?))
    }
}

impl Add<MicroBatchCount> for PipelineStageCount {
    type Output = Result<PipelineScheduleLength, ParallelismError>;

    fn add(self, right: MicroBatchCount) -> Self::Output {
        let ramp = self
            .as_usize()
            .checked_sub(1)
            .ok_or(ParallelismError::empty_input(
                "PipelineStageCount::add",
                "pipeline stage count must be greater than zero",
            ))?;
        let total = ramp
            .checked_add(right.as_usize())
            .ok_or(ParallelismError::overflow(
                "PipelineStageCount::add",
                "pipeline schedule length overflowed usize",
            ))?;

        PipelineScheduleLength::from_raw(total)
    }
}

impl fmt::Display for PipelineScheduleLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} schedule slots", self.0)
    }
}

/// Number of bytes communicated by a teaching collective estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommunicationBytes(u64);

impl CommunicationBytes {
    fn from_raw(operation: &'static str, value: u64) -> Result<Self, ParallelismError> {
        Ok(Self(nonzero_u64(
            "communication bytes must be greater than zero",
            operation,
            value,
        )?))
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for CommunicationBytes {
    type Error = ParallelismError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::from_raw("CommunicationBytes::try_from", value)
    }
}

impl Add for CommunicationBytes {
    type Output = Result<CommunicationBytes, ParallelismError>;

    fn add(self, right: CommunicationBytes) -> Self::Output {
        CommunicationBytes::from_raw(
            "CommunicationBytes::add",
            checked_u64_add("CommunicationBytes::add", self.as_u64(), right.as_u64())?,
        )
    }
}

impl fmt::Display for CommunicationBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} bytes", self.0)
    }
}

/// Non-zero number of communication rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommunicationRounds(u64);

impl CommunicationRounds {
    fn as_u64(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for CommunicationRounds {
    type Error = ParallelismError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(Self(nonzero_u64(
            "communication rounds must be greater than zero",
            "CommunicationRounds::try_from",
            value,
        )?))
    }
}

impl Mul<CommunicationRounds> for CommunicationBytes {
    type Output = Result<CommunicationBytes, ParallelismError>;

    fn mul(self, right: CommunicationRounds) -> Self::Output {
        CommunicationBytes::from_raw(
            "CommunicationBytes::mul",
            checked_u64_mul("CommunicationBytes::mul", self.as_u64(), right.as_u64())?,
        )
    }
}

/// Collective communication pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveKind {
    /// Sum local values and replicate the result on every rank.
    AllReduce,
    /// Sum local values and distribute one reduced shard per rank.
    ReduceScatter,
    /// Gather local shards so every rank sees the full value.
    AllGather,
}

impl fmt::Display for CollectiveKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllReduce => formatter.write_str("all-reduce"),
            Self::ReduceScatter => formatter.write_str("reduce-scatter"),
            Self::AllGather => formatter.write_str("all-gather"),
        }
    }
}

/// Result of one collective operation over teaching shards.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectiveTrace {
    kind: CollectiveKind,
    input: ShardPlan,
    reduced_value: TensorElement,
    communication: CommunicationBytes,
}

impl CollectiveTrace {
    /// Runs a tiny all-reduce over scalar shard sums.
    pub fn all_reduce(
        input: ShardPlan,
        bytes_per_rank: CommunicationBytes,
    ) -> Result<Self, ParallelismError> {
        let reduced_value = input
            .shards()
            .try_fold(TensorElement::try_from(0.0)?, |sum, shard| {
                TensorElement::try_from(sum.as_f64() + shard.values().sum()?.as_f64())
            })?;
        let communication = (bytes_per_rank * CommunicationRounds::try_from(2)?)?;

        Ok(Self {
            kind: CollectiveKind::AllReduce,
            input,
            reduced_value,
            communication,
        })
    }

    /// Returns the collective kind.
    pub fn kind(&self) -> CollectiveKind {
        self.kind
    }

    /// Returns the input sharding plan.
    pub fn input(&self) -> &ShardPlan {
        &self.input
    }

    /// Returns the reduced scalar value for the teaching example.
    pub fn reduced_value(&self) -> TensorElement {
        self.reduced_value
    }

    /// Returns the communication estimate.
    pub fn communication(&self) -> CommunicationBytes {
        self.communication
    }
}

/// Publication class attached to a parallelism trace before public report release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelTraceVisibility {
    /// Safe to include in learner-facing public parallelism reports.
    Public,
    /// Useful for restricted study, but not public learner-facing material.
    ResearchRestricted,
    /// Must stay out of public learner-facing material.
    Private,
}

impl fmt::Display for ParallelTraceVisibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Public => "public",
            Self::ResearchRestricted => "research-restricted",
            Self::Private => "private",
        };
        formatter.write_str(label)
    }
}

/// Typed decision at the parallelism-report publication boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicParallelismDecision {
    /// The trace can appear in a public learner-facing report.
    Publishable,
    /// The trace must stay out of public learner-facing reports.
    Blocked,
}

/// Collective trace plus explicit publication review evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewedCollectiveTrace {
    trace: CollectiveTrace,
    visibility: ParallelTraceVisibility,
}

impl ReviewedCollectiveTrace {
    /// Creates a reviewed collective trace.
    pub fn new(trace: CollectiveTrace, visibility: ParallelTraceVisibility) -> Self {
        Self { trace, visibility }
    }

    /// Returns the reviewed collective trace.
    pub fn trace(&self) -> &CollectiveTrace {
        &self.trace
    }

    /// Returns the publication class.
    pub fn visibility(&self) -> ParallelTraceVisibility {
        self.visibility
    }

    /// Classifies whether this trace can enter a public learner-facing report.
    pub fn release_decision(&self) -> PublicParallelismDecision {
        match self.visibility {
            ParallelTraceVisibility::Public => PublicParallelismDecision::Publishable,
            ParallelTraceVisibility::ResearchRestricted | ParallelTraceVisibility::Private => {
                PublicParallelismDecision::Blocked
            }
        }
    }

    fn into_trace(self) -> CollectiveTrace {
        self.trace
    }
}

/// Collective trace checked for learner-facing public release.
#[derive(Debug, Clone, PartialEq)]
pub struct PublicParallelismReport(CollectiveTrace);

impl PublicParallelismReport {
    /// Builds a public parallelism report only from a reviewed public trace.
    pub fn from_reviewed_trace(
        reviewed: ReviewedCollectiveTrace,
    ) -> Result<Self, ParallelismError> {
        if reviewed.release_decision() == PublicParallelismDecision::Blocked {
            return Err(ParallelismError::invalid_public_report(
                "PublicParallelismReport::from_reviewed_trace",
                "public parallelism reports cannot include restricted or private collective traces",
            ));
        }

        Ok(Self(reviewed.into_trace()))
    }

    /// Returns the checked collective trace.
    pub fn trace(&self) -> &CollectiveTrace {
        &self.0
    }

    /// Returns the public collective kind.
    pub fn kind(&self) -> CollectiveKind {
        self.0.kind()
    }

    /// Returns the public communication estimate.
    pub fn communication(&self) -> CommunicationBytes {
        self.0.communication()
    }

    /// Returns the public reduced scalar value.
    pub fn reduced_value(&self) -> TensorElement {
        self.0.reduced_value()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CollectiveTrace, CommunicationBytes, DataParallelLayout, Error, GlobalBatchSize,
        LayerCount, MicroBatchCount, ModelWidth, ParallelStrategy, ParallelTraceVisibility,
        PipelineLayout, PublicParallelismReport, ReviewedCollectiveTrace, TensorElement,
        TensorLine, TensorParallelLayout, WorldSize, shard_tensor_line,
    };

    fn values() -> Result<TensorLine, Error> {
        TensorLine::from_values([
            TensorElement::try_from(1.0)?,
            TensorElement::try_from(2.0)?,
            TensorElement::try_from(3.0)?,
            TensorElement::try_from(4.0)?,
        ])
    }

    fn collective_trace() -> Result<CollectiveTrace, Error> {
        let plan = shard_tensor_line(
            values()?,
            WorldSize::try_from(2)?,
            ParallelStrategy::DataParallel,
        )?;

        CollectiveTrace::all_reduce(plan, CommunicationBytes::try_from(64)?)
    }

    #[test]
    fn rank_layouts_reject_uneven_splits() -> Result<(), Error> {
        let rejected =
            DataParallelLayout::new(GlobalBatchSize::try_from(10)?, WorldSize::try_from(4)?);

        assert!(matches!(rejected, Err(Error::UnevenSplit { .. })));
        Ok(())
    }

    #[test]
    fn data_parallel_layout_splits_batch_axis() -> Result<(), Error> {
        let layout =
            DataParallelLayout::new(GlobalBatchSize::try_from(128)?, WorldSize::try_from(4)?)?;

        assert_eq!(layout.local_batch(), super::LocalBatchSize::try_from(32)?);
        Ok(())
    }

    #[test]
    fn tensor_parallel_layout_splits_model_width() -> Result<(), Error> {
        let layout =
            TensorParallelLayout::new(ModelWidth::try_from(1024)?, WorldSize::try_from(4)?)?;

        assert_eq!(layout.shard_width(), super::ShardWidth::try_from(256)?);
        Ok(())
    }

    #[test]
    fn pipeline_layout_splits_layers_and_computes_schedule() -> Result<(), Error> {
        let layout = PipelineLayout::new(LayerCount::try_from(12)?, WorldSize::try_from(3)?)?;
        let schedule = (layout.stage_count() + MicroBatchCount::try_from(4)?)?;

        assert_eq!(layout.layers_per_rank(), super::LayersPerRank::try_from(4)?);
        assert_eq!(format!("{schedule}"), "6 schedule slots");
        Ok(())
    }

    #[test]
    fn sharding_preserves_rank_owned_segments() -> Result<(), Error> {
        let plan = shard_tensor_line(
            values()?,
            WorldSize::try_from(2)?,
            ParallelStrategy::DataParallel,
        )?;
        let shard_sums = plan
            .shards()
            .map(|shard| shard.values().sum())
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(
            shard_sums,
            [TensorElement::try_from(3.0)?, TensorElement::try_from(7.0)?]
        );
        Ok(())
    }

    #[test]
    fn all_reduce_reports_reduced_value_and_typed_communication() -> Result<(), Error> {
        let trace = collective_trace()?;

        assert_eq!(trace.reduced_value(), TensorElement::try_from(10.0)?);
        assert_eq!(trace.communication(), CommunicationBytes::try_from(128)?);
        Ok(())
    }

    #[test]
    fn public_parallelism_report_accepts_public_collective_trace() -> Result<(), Error> {
        let report = PublicParallelismReport::from_reviewed_trace(ReviewedCollectiveTrace::new(
            collective_trace()?,
            ParallelTraceVisibility::Public,
        ))?;

        assert_eq!(report.reduced_value(), TensorElement::try_from(10.0)?);
        assert_eq!(report.communication(), CommunicationBytes::try_from(128)?);
        Ok(())
    }

    #[test]
    fn public_parallelism_report_blocks_restricted_and_private_collective_traces()
    -> Result<(), Error> {
        let restricted =
            PublicParallelismReport::from_reviewed_trace(ReviewedCollectiveTrace::new(
                collective_trace()?,
                ParallelTraceVisibility::ResearchRestricted,
            ));
        let private = PublicParallelismReport::from_reviewed_trace(ReviewedCollectiveTrace::new(
            collective_trace()?,
            ParallelTraceVisibility::Private,
        ));

        assert!(matches!(
            restricted.err(),
            Some(Error::InvalidPublicReport { .. })
        ));
        assert!(matches!(
            private.err(),
            Some(Error::InvalidPublicReport { .. })
        ));
        Ok(())
    }
}
