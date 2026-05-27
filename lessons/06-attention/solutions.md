# Attention Solutions

## Solution 1: Separate sequence shape

The sequence length is `4`.

The token width is `8`.

Adding one more token changes the sequence length.

Using wider embeddings changes the token width.

## Solution 2: Compute one score

The dot product is:

```text
1 * 1 + 1 * 0 = 1
```

The width is `2`, so the scaled score is:

```text
1 / sqrt(2) = 0.7071...
```

That matches the example output rounded to four decimal places.

## Solution 3: Explain softmax focus

Softmax exponentiates the scores and divides each exponentiated value by the total.

The largest score gets the largest exponentiated value, so it gets the largest weight.

All weights add up to one because each exponentiated value is divided by the same total.

## Solution 4: Mix values

The first value contributes:

```text
0.75 * [2, 0] = [1.5, 0.0]
```

The second value contributes:

```text
0.25 * [0, 4] = [0.0, 1.0]
```

Adding them gives:

```text
[1.5, 1.0]
```

## Solution 5: Read the full trace

The exact weights are smooth softmax values, not hard selections.

For the example sequence, query token `0` scores token `0` and token `2` higher than token `1`, because token `0` matches the first dimension.

The output mixes value vectors according to those weights. Query token `0` focuses least on token `1`.

There is a tie for strongest focus: token `0` and token `2` receive the same highest weight.

## Solution 6: Review a trace for public release

The public trace example prints:

```text
public query token = 0
public weights     = [0.4011, 0.1978, 0.4011]
public output      = [0.8022, 0.5989]
```

`AttentionTrace` is enough to show the computed scores, weights, and output. It
is not enough to prove that the trace belongs in public course material.

The boundary is `PublicAttentionTrace::from_reviewed_trace`. That constructor
accepts reviewed public evidence and rejects reviewed evidence with non-public
visibility.

This is separate from attention math because the two checks protect different
invariants. Attention math checks whether scores normalize into weights and
weights mix values. Public review checks whether the trace is appropriate for a
learner-facing artifact.

## Self-Check

- You can keep sequence length separate from token width.
- You can say which query/key pair produced each attention score.
- You can explain softmax as a distribution over tokens.
- You can decompose the output as weighted value-vector contributions.
- You can explain why `AttentionTrace` proves computation while `PublicAttentionTrace` proves public-release eligibility.
