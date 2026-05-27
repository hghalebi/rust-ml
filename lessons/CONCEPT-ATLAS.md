# Concept Atlas

This atlas shows the same learning path from several angles:

```text
ML idea -> meaningful Rust type -> checked map -> runnable proof
```

Use it when you feel that the lessons, code crates, and CS336 Rust assignments
are separate pieces. They are meant to be one progression.

## How To Read The Atlas

Each row has four learner questions:

- **Object:** what kind of value is this?
- **Invariant:** what must be true before downstream code may use it?
- **Map:** what transformation owns the change?
- **Proof:** which local example or test makes the idea executable?

In the category-theory lens, an object is a meaningful space of values and a
map is a function between those spaces. The Rust newtype is the learner-visible
name for the object.

## Core Object Ladder

| Stage | Objects | Invariants | Starts in |
| --- | --- | --- | --- |
| Category lens | `TypedObject`, `TypedMap`, `CompositionTrace` | adjacent maps compose only when the middle object matches | [`code/category_lens`](../code/category_lens/README.md) |
| Scalar meaning | `InputValue`, `Weight`, `Bias`, `Target` | finite values, probability ranges where needed | [`code/neuron`](../code/neuron/README.md) |
| Vector meaning | `FeatureVector`, `WeightVector`, `InputVector` | matching widths, non-empty vectors | [`lessons/02-vectors`](02-vectors/README.md) |
| Prediction path | `WeightedSum`, `PreActivation`, `Prediction`, `Loss` | finite arithmetic, valid prediction range | [`code/neuron`](../code/neuron/README.md) |
| Learning path | `PredictionError`, `Gradient`, `Adjustment`, `LearningRate` | positive learning rate, checked update arithmetic | [`lessons/04-learning`](04-learning/README.md) |
| Hidden representation | `HiddenActivation`, `OutputLogit`, `MatrixShape` | layer widths agree before composition | [`code/mlp`](../code/mlp/README.md) |
| Sequence attention | `TokenEmbedding`, `Query`, `Key`, `Value`, `PublicAttentionTrace` | shared model width, role-specific vectors, and public trace review | [`code/attention`](../code/attention/README.md) |
| Architecture choices | `TransformerConfig`, `LayerCount`, `HeadCount`, `FeedForwardWidth` | model width divides evenly into attention heads | [`code/transformer`](../code/transformer/README.md) |
| Expert routing | `ExpertScores`, `ExpertChoice`, `ExpertRoute`, `ExpertBank` | one score per expert, then an existing expert applies the token map | [`code/transformer`](../code/transformer/README.md) |
| Transformer encoder | `PositionEncoding`, `TokenSequence`, `AttentionOutputSequence` | same sequence length and `d_model` for residual maps | [`code/transformer`](../code/transformer/README.md) |
| Language modeling | `RawText`, `ReviewedRawText`, `Token`, `TokenId`, `NextTokenBatch`, `PublicLanguageModelingExample` | known vocabulary, aligned input and target lengths, and public text review | [`code/lm_basics`](../code/lm_basics/README.md) |
| Systems evidence | `Bytes`, `BytesPerSecond`, `MemoryLevel`, `Flops`, `ElapsedNanos`, `ArithmeticIntensity`, `PublicSystemsReport` | units, memory tiers, and public-report class stay separate during arithmetic | [`code/systems`](../code/systems/README.md) |
| Kernel tiling | `MatrixShape`, `TileShape`, `TilePlan`, `FlopCount`, `PublicKernelReport` | tile windows, resource units, and public-report class stay explicit | [`code/kernels`](../code/kernels/README.md) |
| Scaling evidence | `TrainingRun`, `MetricRecord`, `ComputeBudgetFlops`, `ValidationLoss`, `ScalingTradeoff`, `PublicScalingReport` | every loss, recommendation, and public report keeps the run evidence that produced it | [`code/scaling`](../code/scaling/README.md) |
| Data preparation | `RawDocument`, `NormalizedDocument`, `DedupKey`, `CorpusShard`, `PublicCorpusManifest` | provenance, filter reasons, mixture weights, licenses, and public/private boundaries stay visible | [`code/data`](../code/data/README.md) |
| Evaluation report | `EvalExample`, `ScoredPrediction`, `EvalReport`, `PublicEvalReport` | prompts, answers, run IDs, metrics, and public-release review stay visible | [`code/evaluation`](../code/evaluation/README.md) |
| Inference trace | `DecodeRequest`, `SamplingMode`, `KvCacheEntry`, `LatencyBudget`, `PublicDecodeTrace` | context, cache, generated tokens, and public-release review advance together | [`code/inference`](../code/inference/README.md) |
| Parallel training | `WorldSize`, `RankId`, `RankShard`, `CommunicationBytes`, `PublicParallelismReport` | every shard keeps its rank, origin, communication units, and public-report class | [`code/parallelism`](../code/parallelism/README.md) |
| Post-training signals | `PreferencePair`, `RewardScore`, `VerifierFeedback`, `AuditRecord`, `AlignmentWorkflow`, `PublicAlignmentRelease` | signal source, failure meaning, lifecycle stage, and public-release class are preserved | [`code/alignment`](../code/alignment/README.md) |

## Map Ladder

The course repeatedly asks you to name the map before trusting the code.

| Map | Plain-English reading | Executable anchor |
| --- | --- | --- |
| `TypedObject -> TypedMap -> CompositionTrace` | name the object/map rule before applying it to ML | `rust_ml_category_lens --example 02_compose_neuron_forward` |
| `FeatureVector * WeightVector -> WeightedSum` | mix input evidence with learned importance | `rust_ml_neuron --example 01_weighted_sum` |
| `WeightedSum + Bias -> PreActivation -> Prediction` | shift the score, then squash it into a prediction | `rust_ml_neuron --example 02_forward_pass` |
| `Prediction - Target -> PredictionError -> Adjustment` | compare, blame, and update one parameter path | `rust_ml_neuron --example 03_one_step_training` |
| `InputVector -> HiddenActivation -> Prediction` | build a representation before the final judgment | `rust_ml_mlp --example 03_forward_trace` |
| `Query * Key -> AttentionScore -> AttentionWeight` | decide which token should influence this token | `rust_ml_attention --example 02_softmax_focus` |
| `AttentionWeight * Value -> AttentionOutput` | mix value vectors according to the attention distribution | `rust_ml_attention --example 03_weighted_sum` |
| `ReviewedAttentionTrace -> PublicAttentionTrace` | keep restricted or private attention evidence out of learner-facing traces | `rust_ml_attention --example 05_public_trace` |
| `VectorLength / HeadCount -> VectorLength` | make each attention head's width a checked value | `rust_ml_transformer --example architecture_config` |
| `ExpertScores -> ExpertChoice -> ExpertRoute -> TokenEmbedding` | route one token to one selected expert, then apply that expert map | `rust_ml_transformer --example expert_routing` |
| `TokenEmbedding + PositionEncoding -> TokenEmbedding` | add position without changing token width | `rust_ml_transformer --example encoder_demo` |
| `RawText -> TokenTextSequence -> TokenIdSequence` | turn text into checked language-model input | `rust_ml_lm_basics --example 01_tokenize_and_encode` |
| `NextTokenBatch -> Logits -> Loss -> Update` | make the smallest complete language-model training loop | `rust_ml_lm_basics --example 04_training_step` |
| `ReviewedRawText -> PublicLanguageModelingExample` | keep restricted or private text out of learner-facing language-model examples | `rust_ml_lm_basics --example 05_public_training_example` |
| `ActivationShape -> ElementCount -> Bytes` | convert model shape into memory evidence | `rust_ml_systems --example 01_memory_accounting` |
| `Bytes + BytesPerSecond + MemoryLevel -> ElapsedNanos` | estimate why the same bytes cost different time at different memory tiers | `rust_ml_systems --example 05_memory_hierarchy` |
| `ReviewedStageMeasurement* -> PublicSystemsReport` | keep restricted or private benchmark measurements out of learner-facing systems reports | `rust_ml_systems --example 06_public_report` |
| `MatrixRows * MatrixColumns -> ElementCount` | convert shape into compute and memory accounting | `rust_ml_kernels --example 04_kernel_estimate` |
| `ReviewedTiledMatVecTrace -> PublicKernelReport` | keep restricted or private kernel traces out of learner-facing public reports | `rust_ml_kernels --example 05_public_report` |
| `MetricRecord -> ScalingFit -> ForecastLoss` | turn runs into a limited, inspectable scaling claim | `rust_ml_scaling --example 03_forecast_loss` |
| `ScalingCandidate + ScalingCandidate -> ScalingTradeoff` | compare model/data/compute choices without dropping units | `rust_ml_scaling --example 05_tradeoff_decision` |
| `ReviewedMetricRecord* -> PublicScalingReport` | keep restricted or private experiment evidence out of learner-facing scaling reports | `rust_ml_scaling --example 06_public_report` |
| `RawDocument -> NormalizedDocument -> FilterDecision` | make data quality a typed part of the model path | `rust_ml_data --example 02_filter_and_dedup` |
| `DatasetCard -> PublicCorpusManifest` | keep restricted or private source evidence out of learner-facing public content | `rust_ml_data --example 05_public_manifest` |
| `ReviewedScoredPrediction -> PublicEvalReport` | keep restricted or private evaluation examples out of learner-facing public reports | `rust_ml_evaluation --example 05_public_report` |
| `ContextTokens + TokenId -> ContextTokens` | grow one autoregressive trace without losing the cache boundary | `rust_ml_inference --example 03_kv_cache_trace` |
| `ReviewedDecodeTrace -> PublicDecodeTrace` | keep restricted or private generated traces out of learner-facing public content | `rust_ml_inference --example 05_public_trace` |
| `GlobalBatchSize / WorldSize -> LocalBatchSize` | split a global object into rank-owned local work | `rust_ml_parallelism --example 01_data_parallel_batch` |
| `ReviewedCollectiveTrace -> PublicParallelismReport` | keep restricted or private distributed-training traces out of learner-facing public reports | `rust_ml_parallelism --example 05_public_report` |
| `PreferenceSignal -> UpdateSignal -> AuditRecord` | keep post-training feedback auditable | `rust_ml_alignment --example 04_audit_record` |
| `AuditRecord -> AlignmentWorkflow -> AlignmentTransition` | prevent an alignment update from skipping the audit gate | `rust_ml_alignment --example 05_alignment_workflow` |
| `ReviewedAlignmentWorkflow -> PublicAlignmentRelease` | keep restricted or private post-training feedback out of learner-facing public material | `rust_ml_alignment --example 06_public_release` |

## Runnable Proofs

Every important idea should have a local proof:

- a lesson explanation that names the idea
- a Rust type that protects the meaning
- a test that catches the broken case
- an example that prints or exposes the trace

For example, "a token ID must belong to the vocabulary" is not only prose. It
is checked by `TokenId`, tested by `code/lm_basics`, and exercised by the R1
assignment.

Run the full proof set with:

```bash
python3 scripts/check_course_content.py
python3 scripts/check_public_content.py
python3 scripts/check_rust_teaching_contract.py
python3 scripts/check_teaching_crates.py
python3 scripts/check_teaching_examples.py
cargo test --manifest-path code/Cargo.toml --workspace --all-targets
```

## CS336 Extension

The CS336 Rust equivalent track extends the same atlas into a larger systems
journey:

```text
R1 Basics -> R2 Systems -> R3 Scaling -> R4 Data -> R5 Alignment
```

Each assignment should still answer the same questions:

- What objects are introduced?
- Which raw literals enter through validation adapters?
- Which maps compose into the model or system?
- Which failure signal proves the learner has not hidden the hard part?

That is why the assignments include expected deliverables, required checks,
assessment rubrics, and failure signals.

## Mastery Trace

A learner is ready to move on when they can trace one value across the course.

Example trace:

```text
RawText
  -> TokenTextSequence
  -> TokenIdSequence
  -> NextTokenBatch
  -> Logits
  -> Loss
  -> Adjustment
  -> TilePlan
  -> MetricRecord
  -> ScalingFit
  -> DecodeTrace
  -> RankShard
```

The trace is not just a chain of names. At each arrow, the learner should be
able to say:

- what changed
- what stayed invariant
- which type prevents confusion
- which test would fail if the map were wrong

That is the standard for deep intuition in this repo.
