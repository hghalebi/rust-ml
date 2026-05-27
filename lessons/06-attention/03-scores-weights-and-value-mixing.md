# Scores, Weights, and Value Mixing

## Overview

Attention has three mechanical steps:

```text
score -> normalize -> mix
```

First a query scores every key. Then softmax turns those scores into weights. Finally those weights mix the value vectors.

## Learning Goals

- compute scaled attention scores
- explain softmax as normalized focus
- explain why weights must line up with values
- trace one attention output by hand
- read attention as a composition of small maps

## Plain-English Explanation

### Scores are not weights yet

A score is a raw compatibility number.

Higher means "this key matches the query more strongly", but scores do not yet say how much of each value should be used.

Softmax turns scores into weights:

```text
raw scores -> positive weights that sum to one
```

### Weights mix values

Once we have weights, each value vector contributes in proportion to its weight.

If the weights are:

```text
[0.75, 0.25]
```

and the values are:

```text
[2.0, 0.0]
[0.0, 4.0]
```

then the mixed output is:

```text
0.75 * [2.0, 0.0] + 0.25 * [0.0, 4.0] = [1.5, 1.0]
```

## Algebra Form

For query token `i`, score every key:

```math
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
```

Normalize the scores:

```math
a_{ij} = softmax(s_i)_j
```

Mix the values:

```math
out_i = \sum_j a_{ij} v_j
```

The composition is:

```text
Query * Keys -> AttentionScores
AttentionScores -> AttentionWeights
AttentionWeights * Values -> AttentionOutput
```

## Rust Form

```rust
use rust_ml_attention::{AttentionWeight, AttentionWeights, Value, ValueComponent, ValueSequence};

fn main() -> Result<(), rust_ml_attention::Error> {
    let weights = AttentionWeights::from_weights([
        AttentionWeight::try_from(0.75)?,
        AttentionWeight::try_from(0.25)?,
    ])?;
    let values = ValueSequence::from_values([
        Value::from_values([
            ValueComponent::try_from(2.0)?,
            ValueComponent::try_from(0.0)?,
        ])?,
        Value::from_values([
            ValueComponent::try_from(0.0)?,
            ValueComponent::try_from(4.0)?,
        ])?,
    ])?;
    let output = (&weights * &values)?;

    for component in output.values() {
        println!("{component:.4}");
    }
    Ok(())
}
```

The companion crate makes this safer with `AttentionWeights`, `Value`, and `AttentionOutput`.

## Why This Matters

This is the heart of attention.

The model is not simply choosing one token. It is building a weighted mixture of information from the sequence.

That is why attention can preserve multiple pieces of context at once, and why its shape and numerical checks matter.

## Worked Example: Public Attention Traces

The attention trace proves what happened during computation. A public attention
trace proves something different: the trace was reviewed and is allowed to become
learner-facing evidence.

### English

`AttentionTrace` answers a mathematical question:

```text
Which scores, weights, and output did this query produce?
```

`PublicAttentionTrace` answers a publication question:

```text
May this trace be shown in the public course material?
```

These questions should not share one raw flag. They are different meanings, so
the crate gives them different types.

### Algebra

The computation map is:

```text
Query * Keys -> AttentionScores
AttentionScores -> AttentionWeights
AttentionWeights * Values -> AttentionOutput
```

The public-release map is:

```text
AttentionTrace + public review -> PublicAttentionTrace
```

The first map checks attention mechanics. The second map checks whether the
evidence belongs in the public learning path.

### Rust

Run the reviewed trace example:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace
```

It prints the public query token, its normalized weights, and its mixed output:

```text
public query token = 0
public weights     = [0.4011, 0.1978, 0.4011]
public output      = [0.8022, 0.5989]
```

Then it tries to publish a non-public trace and receives a typed error:

```text
blocked from public attention trace: invalid public trace in PublicAttentionTrace::from_reviewed_trace: public attention traces cannot include restricted or private sequence evidence
```

The important lesson is not only that one branch failed. The important lesson is
where it failed: at `PublicAttentionTrace::from_reviewed_trace`, the boundary
that owns the publication invariant.

### Category-Theory Lens

Read each arrow as a meaning-preserving map between typed objects. Attention
math maps typed vectors into a typed output. Public review maps reviewed evidence
into publishable evidence. The second map is not a numerical layer; it is a
boundary in the learning system.

### Checkpoint

If someone says "the trace is valid, so it can be public", what type do they
forget? They forget `PublicAttentionTrace`. A valid computation is not the same
object as reviewed public evidence.

## Concept Trace

- **Object/newtype:** `AttentionScore`, `AttentionWeight`, `Value`, `AttentionOutput`, and `PublicAttentionTrace`.
- **Invariant:** scores become normalized weights before values are mixed; a trace becomes public only after review.
- **Map:** query-key score -> softmax weight -> weighted value mixture -> reviewed public evidence.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 03_weighted_sum` and `cargo run --manifest-path code/Cargo.toml -p rust_ml_attention --example 05_public_trace`.
- **Failure signal:** you mix values before checking that weights form a distribution.
- **Public-content failure signal:** you treat a valid `AttentionTrace` as automatically publishable.

## Short Practice

1. What is the difference between an attention score and an attention weight?
2. Why should attention weights sum to one?
3. If there are three values, how many weights are required?
4. Which map creates the final `AttentionOutput`?
5. Which type proves that an attention trace has crossed the public-content boundary?

## Retrieval Practice

Close the lesson and answer from memory:

1. Recall: what are the three mechanical attention steps?
2. Explain: why is softmax needed before value mixing?
3. Apply: where should the crate reject a trace that is computationally valid but not publishable?
