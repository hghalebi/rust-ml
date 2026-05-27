# 08 Language Modeling

Status: active.

This folder maps to course Module 7.

This module turns the Transformer path into the first language-modeling loop:
public text becomes tokens, token IDs become next-token pairs, and a tiny model
turns those pairs into loss and an update.

## Outcomes

After this module, you should be able to:

- explain language modeling as "predict the next token"
- trace `RawText -> TokenTextSequence -> Vocabulary -> TokenIdSequence`
- explain why token IDs belong to the vocabulary that created them
- build a `NextTokenBatch` from adjacent token IDs
- read `TinyBigramLanguageModel * NextTokenBatch -> Loss` as typed composition
- explain why one update can lower loss on the same tiny batch
- explain why public learner text must cross `PublicLanguageModelingExample`
  before tokenization
- run the companion language-modeling examples and predict their outputs

## Lessons

1. [From Text To Token IDs](01-text-to-token-ids.md)
2. [Next-Token Batches, Loss, And Updates](02-next-token-batches-loss-and-update.md)
3. [The Public Text Boundary](03-public-text-boundary.md)

## Practice

- [Language-modeling exercises](exercises.md)
- [Language-modeling solutions](solutions.md)

## Code Artifact

- Active crate: [`code/lm_basics`](../../code/lm_basics/README.md)
- Assignment bridge: [R1 Basics](../../assignments/cs336-rust/01-basics.md)

Run the examples:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 02_next_token_batch
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 03_uniform_loss
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example
```

Keep this translation in view:

```text
reviewed public text -> checked tokens -> checked ids -> next-token batch -> loss -> update
```

The public boundary comes before tokenization. That is intentional: a private or
restricted source should not become public training material just because the
tokenizer and model can process it.

## Prerequisite

- Complete [07 Transformer](../07-transformer/README.md)

## Before You Move On

You are ready for the larger CS336 Rust assignment path when you can explain this
chain without treating any step as "just data":

```text
RawText -> ReviewedRawText -> PublicLanguageModelingExample
PublicLanguageModelingExample -> TokenTextSequence -> Vocabulary -> TokenIdSequence
TokenIdSequence -> NextTokenBatch -> Loss -> TrainingStepTrace
```

You should also be able to say which type rejects a token ID outside the
vocabulary and which type rejects restricted or private text before it reaches
learner-facing examples.
