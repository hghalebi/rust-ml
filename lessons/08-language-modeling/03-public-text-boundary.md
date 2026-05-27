# The Public Text Boundary

## Overview

A tokenizer can process many kinds of text. A public course cannot publish every
kind of text.

That is why this module has two separate paths:

```text
RawText -> TokenTextSequence -> Vocabulary -> TokenIdSequence
ReviewedRawText -> PublicLanguageModelingExample
```

The first path is about computation. The second path is about public learner
material.

## Learning Goals

- distinguish valid text processing from public release eligibility
- explain why the public boundary comes before tokenization
- read `ReviewedRawText -> PublicLanguageModelingExample` as a typed map
- explain the difference between `Public`, `ResearchRestricted`, and `Private`
- predict why restricted or private text is rejected

## Plain-English Explanation

If text is private, the correct behavior is not "tokenize it carefully." The
correct behavior is to keep it out of public examples.

The repo models that rule directly:

```text
RawText + TextVisibility -> ReviewedRawText
ReviewedRawText + tokenizer -> PublicLanguageModelingExample
```

Only reviewed public text can become a public example. Restricted or private
text receives a typed error at the public boundary.

## Algebra Form

The computation map is:

```text
RawText -> TokenTextSequence -> Vocabulary -> TokenIdSequence -> NextTokenBatch
```

The public-release map is:

```text
ReviewedRawText -> PublicLanguageModelingExample
```

The release invariant is:

```text
visibility == Public
```

If the visibility is `ResearchRestricted` or `Private`, the map does not produce
a public example.

## Rust Form

```rust
use rust_ml_lm_basics::{
    PublicLanguageModelingExample, RawText, ReviewedRawText, TextVisibility,
    WhitespaceTokenizer,
};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let public_example = PublicLanguageModelingExample::from_reviewed_text(
        ReviewedRawText::new(
            RawText::try_from("red blue red")?,
            TextVisibility::Public,
        ),
        WhitespaceTokenizer,
    )?;

    println!("public token count = {}", public_example.tokens().len());
    println!("public vocabulary size = {}", public_example.vocabulary().size());

    let private_result = PublicLanguageModelingExample::from_reviewed_text(
        ReviewedRawText::new(
            RawText::try_from("red blue red")?,
            TextVisibility::Private,
        ),
        WhitespaceTokenizer,
    );

    match private_result {
        Ok(_) => println!("unexpected public example"),
        Err(error) => println!("blocked from public example: {error}"),
    }

    Ok(())
}
```

The same toy string is used twice so the learner sees that the string contents
are not the only question. The review class changes the meaning of the boundary.

## Why This Matters

The repo is public learner material. It must not let private or restricted
source text leak into examples, traces, or reports.

This is also a type-design lesson. A boolean such as `is_public` would hide too
much meaning. `TextVisibility` names the possible states, and
`PublicLanguageModelingExample::from_reviewed_text` owns the invariant.

## Concept Trace

- **Object/newtype:** `RawText`, `ReviewedRawText`, `TextVisibility`, and `PublicLanguageModelingExample`.
- **Invariant:** learner-facing language-modeling examples can use only reviewed public text.
- **Map:** reviewed text -> public language-modeling example.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example`.
- **Failure signal:** you treat text that can be tokenized as text that can be published.

## Short Practice

1. Why does the public boundary come before tokenization?
2. Which type carries the visibility decision?
3. Which constructor rejects restricted or private text?
4. Why is `TextVisibility` clearer than a boolean flag?
