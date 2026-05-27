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

## Self-Check

- You can keep sequence length separate from token width.
- You can say which query/key pair produced each attention score.
- You can explain softmax as a distribution over tokens.
- You can decompose the output as weighted value-vector contributions.
