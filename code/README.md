# Code

This directory is reserved for runnable examples that follow the lesson progression.

It is the executable companion layer for the course, not the canonical teaching layer.

## Strategy

- `lessons/` stays canonical for authored teaching content.
- `code/` is the executable companion layer once a topic earns real runnable examples.
- Active crates must have tests and examples that line up with the lesson sequence.
- Planned topic directories stay as honest roadmap placeholders until they are ready.
- New active crates must expose semantic public APIs. Raw literals belong at explicit validation adapters, not in domain-level function signatures.
- Newtype operations should use readable `std::ops` implementations when that makes the mathematical composition clearer to learners.

## Packaging Rule

When executable lessons start, each topic directory becomes a Cargo workspace crate.

Do not use loose standalone `.rs` files as the long-term structure.

## Workspace Commands

Run all active teaching crates:

```bash
cargo test --manifest-path code/Cargo.toml --workspace --all-targets
python3 scripts/check_rust_teaching_contract.py
python3 scripts/check_teaching_crates.py
python3 scripts/check_teaching_examples.py
```

Run the category-theory lens examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 01_objects_and_maps
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 02_compose_neuron_forward
cargo run --manifest-path code/Cargo.toml -p rust_ml_category_lens --example 03_composition_failure
```

Run the beginner neuron examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 02_forward_pass
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 03_one_step_training
cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 04_and_gate_epoch
```

Run the MLP bridge examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 01_hidden_features
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 02_shape_flow
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 03_forward_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_mlp --example 04_xor_table
```

Run the attention bridge examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 02_softmax_focus
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 03_weighted_sum
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace
```

Run the CS336 Rust R1 basics examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 02_next_token_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 03_uniform_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example
```

Run the CS336 Rust R2 systems examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 01_memory_accounting
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 02_attention_flops
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 03_median_timing
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 04_arithmetic_intensity
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 05_memory_hierarchy
cargo run --manifest-path code/Cargo.toml -p rust_ml_systems --example 06_public_report
```

Run the CS336 Rust kernels examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 01_elementwise_gelu
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 02_row_sum_reduction
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 03_tiled_matvec
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 04_kernel_estimate
cargo run --manifest-path code/Cargo.toml -p rust_ml_kernels --example 05_public_report
```

Run the CS336 Rust R3 scaling examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 01_record_runs
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 02_fit_power_law
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 03_forecast_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 04_report_limitations
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 05_tradeoff_decision
cargo run --manifest-path code/Cargo.toml -p rust_ml_scaling --example 06_public_report
```

Run the CS336 Rust R4 data examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 01_normalize_documents
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 02_filter_and_dedup
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 03_build_shard
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 04_source_mixture
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 05_public_manifest
```

Run the CS336 Rust evaluation examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 01_score_prediction
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 02_accuracy_report
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 03_reject_mismatched_ids
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 04_compare_runs
cargo run --manifest-path code/Cargo.toml -p rust_ml_evaluation --example 05_public_report
```

Run the CS336 Rust inference examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 01_greedy_decode
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 02_sampling_controls
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 03_kv_cache_trace
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 04_latency_budget
cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 05_public_trace
```

Run the CS336 Rust parallelism examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 01_data_parallel_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 02_tensor_parallel_width
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 03_collective_all_reduce
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 04_pipeline_schedule
cargo run --manifest-path code/Cargo.toml -p rust_ml_parallelism --example 05_public_report
```

Run the CS336 Rust R5 alignment examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 01_instruction_example
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 02_preference_signal
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 03_verifier_feedback
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 04_audit_record
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 05_alignment_workflow
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 06_public_release
```

Run the advanced Transformer example:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example architecture_config
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example expert_routing
cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example encoder_demo
```

## Topics

| Topic | Status | Purpose |
| --- | --- | --- |
| [category_lens](category_lens/README.md) | Active crate | The executable object/map/composition lens used across the course. |
| [neuron](neuron/README.md) | Active crate | The first typed trainable model companion crate. |
| [mlp](mlp/README.md) | Active crate | Tiny typed hidden-layer companion crate for shape and representation flow. |
| [attention](attention/README.md) | Active crate | Beginner scaled dot-product attention companion crate with typed Q/K/V roles, ops-based newtype arithmetic, and public trace review. |
| [transformer](transformer/README.md) | Active crate | Real tested encoder-path teaching crate with typed architecture configuration, expert routing, and ops-based positional/residual arithmetic. |
| [lm_basics](lm_basics/README.md) | Active crate | First CS336 Rust language-modeling artifact: tokenization, checked IDs, next-token batches, loss, one update, and public text review. |
| [systems](systems/README.md) | Active crate | First CS336 Rust systems artifact: typed memory, bandwidth, hierarchy, FLOP, timing, arithmetic-intensity estimates, and public report review. |
| [kernels](kernels/README.md) | Active crate | First CS336 Rust kernels artifact: typed tiling, reductions, matrix-vector traces, resource estimates, and public report review. |
| [scaling](scaling/README.md) | Active crate | First CS336 Rust scaling artifact: typed experiment records, power-law fitting, forecasts, tradeoffs, limitations, and public report review. |
| [data](data/README.md) | Active crate | First CS336 Rust data artifact: typed normalization, filtering, deduplication, shards, source mixtures, and public corpus manifests. |
| [evaluation](evaluation/README.md) | Active crate | First CS336 Rust evaluation artifact: typed examples, predictions, reports, exact-match accuracy, run comparison, and public report review. |
| [inference](inference/README.md) | Active crate | First CS336 Rust inference artifact: typed decoding controls, KV-cache traces, latency budgets, and public trace review. |
| [parallelism](parallelism/README.md) | Active crate | First CS336 Rust parallelism artifact: typed ranks, sharding plans, collective traces, pipeline schedules, and public report review. |
| [alignment](alignment/README.md) | Active crate | First CS336 Rust alignment artifact: typed instruction examples, preference signals, verifier feedback, audit records, workflow transitions, and public release review. |
