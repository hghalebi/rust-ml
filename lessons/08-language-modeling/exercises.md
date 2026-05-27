# Language-Modeling Exercises

## Exercise 1: Tokenize and encode

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode
```

Write down:

1. the token sequence
2. the token ID sequence
3. the vocabulary size

Then explain why the two `red` tokens receive the same ID.

## Exercise 2: Build next-token pairs

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 02_next_token_batch
```

For the ID sequence `[0, 1, 0]`, write the input IDs and target IDs.

Then answer: why is the context length `2`, not `3`?

## Exercise 3: Interpret uniform loss

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 03_uniform_loss
```

Explain why the loss is close to:

```text
ln(2)
```

Use the words `VocabularySize`, `Logit`, and `Loss` in your explanation.

## Exercise 4: Read one update

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 04_training_step
```

Write down:

1. baseline loss
2. loss before the update
3. loss after the update

Then answer: what changed, the batch or the model?

## Exercise 5: Check the public text gate

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 05_public_training_example
```

Write down:

1. the public token count
2. the public vocabulary size
3. the public loss before and after one update
4. the boundary that rejects restricted or private text

Then answer: why should this rejection happen before tokenization?

## Failure Signals

- You treat token text and token IDs as the same object.
- You reuse a token ID without asking which vocabulary produced it.
- You build next-token inputs and targets with different lengths.
- You expect loss to fall without changing model parameters.
- You treat tokenizable text as automatically safe for public learner material.

## Debugging Hints

- Label each object by role: raw text, token text, vocabulary, token ID, batch, logit, loss, update, or public example.
- For a sequence of length `n`, next-token training creates `n - 1` input-target pairs.
- Check vocabulary size before interpreting a token ID.
- Compare `loss_before` and `loss_after` on the same batch.
- Ask the public-content question before the tokenizer question.
