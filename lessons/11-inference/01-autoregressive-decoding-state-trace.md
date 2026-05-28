# Autoregressive Decoding as a Typed State Trace

## Overview

Inference is not one magic call that jumps from prompt to full output.

It is a typed state machine that repeats one step:

1. choose one next token,
2. append it to generated output,
3. append the same token to the context,
4. append one cache entry tied to that generation step,
5. repeat until the token budget is used.

That loop is the core insight. Typed modeling makes it impossible to update
one lane without the others.

## Learning Goals

- name the typed state objects that participate in one decode step,
- explain the contract behind `DecodeRequest`,
- describe why context, generated tokens, and cache entries must stay aligned,
- connect `decode` to a typed repeated-map structure,
- read a typed latency boundary from decode outputs.

## Plain-English Explanation

Think of the decode loop as a production line with three synchronized tracks.

At each step:

- a token is chosen from the model's ranked list,
- the chosen token extends the generated sequence,
- the chosen token is also valid context for the next step,
- the cache records which token was used and at what cache position.

If any track moves without the others, the trace no longer describes a valid
autoregressive state.

## Algebra Form

Valid requests are built by typed conversion then validated construction:

```text
PromptTokens + ContextWindow + MaxNewTokens + SamplingMode -> DecodeRequest
```

Each decode step updates three lanes with the same chosen token:

```text
ContextTokens + TokenId -> ContextTokens
GeneratedTokens + TokenId -> GeneratedTokens
KvCache + KvCacheEntry -> KvCache
```

The loop is a fold over the model state:

```text
DecodeState -> DecodeState
```

## Rust Form

```rust
use rust_ml_inference::{
    ContextWindow,
    DecodeRequest,
    Logit,
    MaxNewTokens,
    NextTokenRule,
    PromptTokens,
    RankedToken,
    SamplingMode,
    TokenId,
    TokenIndex,
    TokenRankings,
    TokenText,
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

fn main() -> Result<(), rust_ml_inference::Error> {
    let vocabulary_size = VocabularySize::try_from(4)?;

    let start = token(TokenIndex::try_from(0)?, vocabulary_size)?;
    let typed = token(TokenIndex::try_from(1)?, vocabulary_size)?;
    let rust = token(TokenIndex::try_from(2)?, vocabulary_size)?;
    let end = token(TokenIndex::try_from(3)?, vocabulary_size)?;

    let vocabulary = ToyVocabulary::new(
        vocabulary_size,
        [
            VocabularyEntry::new(start, TokenText::try_from("<prompt>")?),
            VocabularyEntry::new(typed, TokenText::try_from("typed")?),
            VocabularyEntry::new(rust, TokenText::try_from("rust")?),
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
                    RankedToken::new(typed, Logit::try_from(4.0)?),
                    RankedToken::new(rust, Logit::try_from(2.0)?),
                ])?,
            ),
            NextTokenRule::new(
                typed,
                TokenRankings::from_candidates([RankedToken::new(rust, Logit::try_from(4.0)?)])?,
            ),
            NextTokenRule::new(
                rust,
                TokenRankings::from_candidates([RankedToken::new(end, Logit::try_from(4.0)?)])?,
            ),
        ],
    )?;

    let prompt = PromptTokens::from_tokens([token(TokenIndex::try_from(0)?, vocabulary_size)?])?;
    let trace = decode(
        &model,
        DecodeRequest::new(
            prompt,
            ContextWindow::try_from(4)?,
            MaxNewTokens::try_from(3)?,
            SamplingMode::Greedy,
        )?,
    )?;

    println!(
        "generated tokens: {}",
        trace.generated_tokens().count()
    );
    println!(
        "decoded text: {}",
        vocabulary.decode(trace.generated_tokens())?
    );
    println!("cache entries: {}", trace.cache().entry_count());
    for entry in trace.cache().entries() {
        println!("{} {} {}", entry.position(), entry.role(), entry.token_id());
    }

    Ok(())
}
```

The snippet shows how one decoded step changes all three lanes together.

## Why This Matters

`decode` teaches learners how to compose type-safe state transitions.

This is where invalid intermediate states become hard to represent:

- generated sequence length and cache entries stay synchronized,
- each generated token remains in-vocabulary,
- context-window pressure and request constraints are enforced before decoding,
- invalid public traces are screened before entering learner-facing material.

This makes the process inspectable for pedagogy.

## Concept Trace

- **Object/newtype:** `TokenIndex`, `TokenId`, `VocabularySize`, `TokenRankings`, `PromptTokens`, `ContextTokens`, `GeneratedTokens`, `DecodeRequest`, `KvCacheEntry`, `KvCache`, `DecodeStepRecord`, `DecodeTrace`, `SamplingMode`.
- **Invariant:** each decode step updates context, generated tokens, and cache entry together; generated token counts and cache steps remain aligned.
- **Map:** `ContextTokens + TokenId -> ContextTokens`, `GeneratedTokens + TokenId -> GeneratedTokens`, `KvCache + KvCacheEntry -> KvCache`.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_inference --example 01_greedy_decode`.
- **Failure signal:** the generated-token count diverges from cache-generated-entry count in any traced step.

## Short Practice

1. In one sentence, explain why `DecodeRequest::new` is safer than a raw tuple.
2. What three typed lanes must update together in each loop step?
3. Which object protects that selected token IDs stay inside vocabulary?
4. Why does the cache track step index and role instead of only token IDs?
