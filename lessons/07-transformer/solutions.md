# Transformer Solutions

## Solution 1: Trigger a useful shape error

The error should identify:

- `DenseMatrix::mul_vec` as the failing operation
- matrix shape `[2, 2]`
- vector shape `[3]`

The hint should say that matrix columns must equal vector length.

That is exactly the kind of failure message you want when debugging model wiring.

## Solution 2: Build a `TokenSequence`

Every token in a `TokenSequence` needs the same width because the model treats the sequence as one matrix:

```math
X \in \mathbb{R}^{n \times d_{model}}
```

If one token has width `3` and another has width `4`, there is no single `d_model`.

`TokenSequence::new` rejects that immediately.

## Solution 3: Project one token into query, key, and value

Across `QueryLayer`, `KeyLayer`, and `ValueLayer`, the input token embedding is the same.

What changes is the learned projection:

- different weight matrix
- different bias
- different semantic output type

That is why the newtype pattern matters. A `Query` and a `Value` are not interchangeable just because both wrap a `DenseVector`.

## Solution 4: Print attention outputs for one head

The head output tells you the token after it has looked across the sequence.

If you want to inspect the internal reasoning more closely, the right things to print are:

- projected queries
- projected keys
- raw scaled scores
- softmax weights
- final weighted sums

That is the shortest path from "attention feels magical" to "attention is just traceable math."

## Solution 5: Add positional encodings

After positional encoding:

- the token count stays the same
- `d_model` stays the same
- the numeric values change

That is the whole point.

The model keeps "what this token is" while also gaining "where this token is."

Without that signal, attention alone does not know whether a token came first or last.

## Solution 6: Run one full encoder block

The output shape must match the input shape:

```math
n \times d_{model}
```

That is required because the block adds residual connections:

```math
X + \mathrm{Attention}(X)
```

and later:

```math
A + \mathrm{FFN}(A)
```

Residual addition only makes sense when both sides have the same width.

## Solution 7: Standard attention versus linear attention

The architectural slot stays the same:

```text
encoder block -> attention module
```

What changes is the math inside that slot.

Standard attention forms the exact scaled dot-product interaction pattern.
Linear attention rewrites the computation through feature-map-based summaries.

That is why linear attention is best understood as a later efficient attention family, not as the definition of the original Transformer paper.

## Self-Check

- You can read a shape error as an object/map mismatch.
- You can build `TokenSequence` only when token widths agree.
- You can explain why positional encoding and residual addition preserve `d_model`.
- You can distinguish the original attention definition from efficiency variants.
