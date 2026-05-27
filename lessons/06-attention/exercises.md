# Attention Exercises

## Exercise 1: Separate sequence shape

A sequence has four tokens. Each token vector has width eight.

Answer:

1. What is the sequence length?
2. What is the token width?
3. Which one changes if we add one more token?
4. Which one changes if we use wider embeddings?

## Exercise 2: Compute one score

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 01_score_one_pair
```

Then compute the same score by hand:

```text
query = [1, 1]
key   = [1, 0]
score = (query dot key) / sqrt(2)
```

## Exercise 3: Explain softmax focus

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 02_softmax_focus
```

Explain why the largest score gets the largest weight, but all weights still add up to one.

## Exercise 4: Mix values

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 03_weighted_sum
```

Explain the printed vector using:

```text
0.75 * [2, 0] + 0.25 * [0, 4]
```

## Exercise 5: Read the full trace

Run:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 04_attention_trace
```

Write down:

1. raw scores
2. normalized weights
3. mixed output

Then answer: which token did query token `0` focus on most?

## Failure Signals

- You confuse sequence length with token width.
- You compute attention scores but cannot say which query/key pair produced each score.
- You treat softmax as choosing one token instead of assigning a distribution of weights.
- You mix values before normalizing scores into weights.

## Debugging Hints

- Label every vector by role: token, query, key, value, score, weight, or output.
- Compute one score by hand before reading a full trace.
- Check that attention weights add up to one before mixing values.
- When reading an output vector, decompose it into weighted value-vector contributions.
