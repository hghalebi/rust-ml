# The Public Decode Boundary And Typed Latency

## Overview

The loop in inference produces a private runtime trace. Before learners can
consume it, we must apply two review boundaries:

1. publication visibility for generated content,
2. typed latency accounting for public reporting.

This lesson shows both boundaries with exact constructors.

## Learning Goals

- explain why `PublicDecodeTrace` requires `ReviewedDecodeTrace`,
- state which visibility values are allowed for public release,
- build a `LatencyReport` from prefill, per-token, and budget units,
- interpret `TraceVisibility` as a domain concept, not a rendering choice,
- describe where validation should happen: constructor time, not at print time.

## Plain-English Explanation

Inference traces include details that are not always safe for public teaching
material. A `ReviewedDecodeTrace` carries an explicit review outcome (`Public`,
`ResearchRestricted`, `Private`).

Only `Public` traces can be converted into a learner-facing `PublicDecodeTrace`.

Latency is the same pattern: you should not mix raw milliseconds and token
counts. You should first represent each piece with typed units and then combine
them in one explicit constructor.

## Algebra Form

Release boundary map:

```text
ReviewedDecodeTrace -> PublicDecodeTrace
```
Latency model:

```text
DecodeTrace + LatencyMillis + LatencyMillis + LatencyBudget -> LatencyReport
```

Budget decision:

```text
total <= LatencyBudget -> LatencyStatus
```

## Rust Form

```rust
use rust_ml_inference::{
    ContextWindow,
    DecodeRequest,
    LatencyBudget,
    LatencyMillis,
    LatencyReport,
    Logit,
    MaxNewTokens,
    NextTokenRule,
    PromptTokens,
    PublicDecodeTrace,
    RankedToken,
    ReviewedDecodeTrace,
    SamplingMode,
    TokenId,
    TokenIndex,
    TokenRankings,
    TokenText,
    TraceVisibility,
    ToyNextTokenModel,
    ToyVocabulary,
    VocabularyEntry,
    VocabularySize,
    decode,
};

fn token(
    index: TokenIndex,
    vocabulary_size: VocabularySize,
) -> Result<TokenId, rust_ml_inference::Error> {
    TokenId::new(index, vocabulary_size)
}

fn build_model(
    vocabulary_size: VocabularySize,
) -> Result<(ToyVocabulary, ToyNextTokenModel), rust_ml_inference::Error> {
    let start = token(TokenIndex::try_from(0)?, vocabulary_size)?;
    let first = token(TokenIndex::try_from(1)?, vocabulary_size)?;
    let second = token(TokenIndex::try_from(2)?, vocabulary_size)?;
    let end = token(TokenIndex::try_from(3)?, vocabulary_size)?;

    let vocabulary = ToyVocabulary::new(
        vocabulary_size,
        [
            VocabularyEntry::new(start, TokenText::try_from("<prompt>")?),
            VocabularyEntry::new(first, TokenText::try_from("safe")?),
            VocabularyEntry::new(second, TokenText::try_from("review")?),
            VocabularyEntry::new(end, TokenText::try_from("<end>")?),
        ],
    )?;

    let model = ToyNextTokenModel::new(
        vocabulary_size,
        TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(1.0)?)])?,
        [
            NextTokenRule::new(
                start,
                TokenRankings::from_candidates([
                    RankedToken::new(first, Logit::try_from(4.0)?),
                    RankedToken::new(second, Logit::try_from(1.5)?),
                ])?,
            ),
            NextTokenRule::new(
                first,
                TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(3.5)?)])?,
            ),
        ],
    )?;

    Ok((vocabulary, model))
}

fn reviewed_trace(
    model: &ToyNextTokenModel,
    vocabulary_size: VocabularySize,
    visibility: TraceVisibility,
) -> Result<ReviewedDecodeTrace, rust_ml_inference::Error> {
    let prompt = PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
    let request = DecodeRequest::new(
        prompt,
        ContextWindow::try_from(3)?,
        MaxNewTokens::try_from(2)?,
        SamplingMode::TopK {
            k: rust_ml_inference::TopK::try_from(2)?,
            temperature: rust_ml_inference::Temperature::try_from(0.9)?,
        },
    )?;

    Ok(ReviewedDecodeTrace::new(decode(model, request)?, visibility))
}

fn main() -> Result<(), rust_ml_inference::Error> {
    let vocabulary_size = VocabularySize::try_from(4)?;
    let (vocabulary, model) = build_model(vocabulary_size)?;
    let public_reviewed = reviewed_trace(&model, vocabulary_size, TraceVisibility::Public)?;
    let public_trace = PublicDecodeTrace::from_reviewed_trace(public_reviewed)?;

    let report = LatencyReport::from_trace(
        public_trace.trace(),
        LatencyMillis::try_from(18)?,
        LatencyMillis::try_from(12)?,
        LatencyBudget::new(LatencyMillis::try_from(40)?),
    )?;

    println!(
        "public generated = {}",
        vocabulary.decode(public_trace.generated_tokens())?
    );
    println!("total latency = {}", report.total());
    println!("budget status = {}", report.status());

    let blocked = reviewed_trace(&model, vocabulary_size, TraceVisibility::ResearchRestricted)
        .and_then(PublicDecodeTrace::from_reviewed_trace);

    println!(
        "blocked from public trace: {}",
        match blocked {
            Ok(_) => "unexpectedly allowed".to_owned(),
            Err(error) => error.to_string(),
        }
    );

    Ok(())
}
```

This snippet has two boundaries in one place:

- `ReviewedDecodeTrace -> PublicDecodeTrace`,
- `LatencyReport::from_trace` for typed timing composition.

## Why This Matters

The public boundary is a semantic boundary. If visibility is restricted, the
same decoded content must not flow into learner-facing output.

Latency is also semantic: it is not a display string, but a typed composition of
timed parts with a budget decision.

## Concept Trace

- **Object/newtype:** `TraceVisibility`, `ReviewedDecodeTrace`, `PublicDecodeTrace`, `LatencyMillis`, `LatencyBudget`, `LatencyReport`.
- **Invariant:** only traces labeled `Public` can become `PublicDecodeTrace`; budget checks happen against typed latencies.
- **Map:** `TraceVisibility + DecodeTrace -> ReviewedDecodeTrace -> PublicDecodeTrace` and `DecodeTrace + LatencyMillis + LatencyMillis + LatencyBudget -> LatencyReport`.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 05_public_trace`.
- **Failure signal:** restricted traces are printed as public or a raw number is used in budget arithmetic instead of typed milliseconds.

## Short Practice

1. Why is `TraceVisibility::ResearchRestricted` rejected by
   `PublicDecodeTrace::from_reviewed_trace`?
2. Why is `LatencyReport::from_trace` safer than printing raw arithmetic on numbers?
3. Which part of the example guarantees that cache visibility and token decisions were reviewed before public use?
4. What does `LatencyBudget::new` protect against in learner-facing material?
