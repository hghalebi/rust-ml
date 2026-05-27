# data

Status: active.

This crate is the first executable companion for [R4 Data](../../assignments/cs336-rust/04-data.md) in the CS336 Rust equivalent track.

It teaches data preparation as a typed corpus pipeline:

```text
RawDocument -> NormalizedDocument -> FilterDecision -> CorpusShard
DatasetCard -> PublicCorpusManifest
```

## Owns

- assignment: [R4 Data](../../assignments/cs336-rust/04-data.md)
- track: [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)

## Current State

- active teaching crate
- typed document IDs, source names, raw text, normalized text, token counts, dedup keys, filter reasons, corpus shards, and mixture weights
- deterministic lowercase and whitespace normalization
- explicit filtering decisions with durable rejection reasons
- deterministic duplicate keys for normalized text
- corpus shard construction from accepted filter decisions
- source mixture validation with non-negative weights and positive total weight
- public corpus manifests that reject restricted or private source cards before publication
- ops-based checked addition for document counts and token budgets

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_normalize_documents.rs
  02_filter_and_dedup.rs
  03_build_shard.rs
  04_source_mixture.rs
  05_public_manifest.rs
```

## Learning Ladder

1. `01_normalize_documents` turns one raw public-safe document into normalized text and a dedup key.
2. `02_filter_and_dedup` shows accepted, duplicate, and too-short decisions with explicit reasons.
3. `03_build_shard` builds a tiny shard from accepted decisions.
4. `04_source_mixture` validates a source mixture.
5. `05_public_manifest` makes the public/private corpus boundary executable.

## Category Lens

Read the pipeline as a composition of data-quality maps:

```text
RawDocument -> NormalizedDocument
NormalizedDocument -> DedupKey
NormalizedDocument -> FilterDecision
AcceptedDocuments -> CorpusShard
SourceWeights -> SourceMixture
DatasetCard -> PublicCorpusManifest
```

The composition rule is evidence preservation. A document should not become
training data unless its source, normalized text, dedup key, and filtering
decision remain visible. A source should not become public learner-facing
material unless its license and visibility evidence pass the manifest boundary.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_data --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 01_normalize_documents
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 02_filter_and_dedup
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 03_build_shard
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 04_source_mixture
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 05_public_manifest
```

## Scope

This crate intentionally uses tiny public-safe strings, not real web-scale corpora.

The goal is to teach the invariants first: every document has a source, every transformation is deterministic, every rejection has a reason, every source mixture has a meaningful total, and every public manifest excludes restricted or private source cards.
