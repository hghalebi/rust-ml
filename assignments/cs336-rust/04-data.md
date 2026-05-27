# R4 Data: From Raw Text To Training Corpus

## Goal

Build a reproducible data path.

Language models learn from data, so data quality is part of the model, not a separate chore.

## What You Build

Create a Rust data pipeline that:

- reads raw text documents from local files
- assigns stable document IDs
- normalizes text with explicit rules
- filters documents with typed reasons
- deduplicates near-identical records with a simple key
- samples a mixture of sources for training

## Active Starter Crate

The first executable artifact is [`code/data`](../../code/data/README.md).

It starts with typed corpus preparation:

```text
RawDocument -> NormalizedDocument -> FilterDecision -> CorpusShard
```

Run it with:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 01_normalize_documents
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 02_filter_and_dedup
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 03_build_shard
cargo run --manifest-path code/Cargo.toml -p rust_ml_data --example 04_source_mixture
```

## Object/Map Preflight

Before implementation, write this preflight in your assignment notes:

- **Objects:** `SourceName`, `RawDocument`, `NormalizedDocument`, `DedupKey`, `FilterDecision`, `CorpusShard`, `MixtureWeight`.
- **Maps:** ingest document, normalize text, derive dedup key, filter with reason, insert into shard, sample source mixture.
- **Composition path:** `RawDocument -> NormalizedDocument -> DedupKey -> FilterDecision -> CorpusShard -> SourceMixture`.
- **Invariant to protect with newtypes:** public sample text, source identity, dedup keys, and filter reasons must remain separately named.

## Expected Deliverables

- a tiny public-safe sample corpus
- a deterministic normalization example
- a filter report that preserves each rejection reason
- a deduplication fixture with two near-identical records
- a source-mixture example with typed weights

## Newtype And Category-Theory Lens

Use newtypes for:

- `DocumentId`
- `SourceName`
- `RawDocument`
- `NormalizedDocument`
- `DedupKey`
- `FilterReason`
- `MixtureWeight`
- `CorpusShard`

The core composition is:

```text
RawDocument -> NormalizedDocument -> FilterDecision -> CorpusShard
```

## Required Checks

- test that normalization is deterministic
- test that filter reasons are recorded, not silently discarded
- test that duplicate documents map to the same dedup key
- test that mixture weights are non-negative and sum to a meaningful total

## Assessment Rubric

- **Boundary hygiene:** raw text is validated and named before downstream corpus logic uses it.
- **Data accountability:** every filtered document has an explicit, learner-visible reason.
- **Dedup clarity:** duplicate detection is simple enough to inspect and typed enough to avoid mixing keys with text.
- **Public safety:** examples contain only tiny synthetic or public-safe text and no local/private paths.

## Failure Signals

- normalization changes between runs for the same input
- rejected documents disappear without a recorded reason
- deduplication compares raw strings in scattered call sites instead of using a typed key
- sample data includes private text, credentials, large corpora, or machine-specific paths

## Suggested Repo Integration

Start from the active `code/data` crate. Keep raw data samples tiny and public-safe.

Do not commit large corpora, private text, credentials, or local machine paths.
