# Language-Modeling Solutions

## Solution 1: Tokenize and encode

The example prints:

```text
tokens = [red, blue, red]
ids    = [0, 1, 0]
vocab size = 2
```

The two `red` tokens receive the same ID because the vocabulary preserves one
entry per unique token. The first `red` creates ID `0`; the later `red` reuses
that checked vocabulary entry.

## Solution 2: Build next-token pairs

For:

```text
[0, 1, 0]
```

the next-token batch is:

```text
inputs  = [0, 1]
targets = [1, 0]
```

The context length is `2` because there are two supervised pairs:

```text
position 0 predicts position 1
position 1 predicts position 2
```

The final token has no next token inside this tiny sequence.

## Solution 3: Interpret uniform loss

A uniform model over two vocabulary entries gives each target probability `0.5`.

Cross-entropy for the correct target is:

```text
-ln(0.5) = ln(2) = 0.6931...
```

`VocabularySize` tells us there are two choices. Equal `Logit` values make the
softmax distribution uniform. `Loss` records the non-negative penalty for the
observed target.

## Solution 4: Read one update

The example prints:

```text
baseline loss = 0.6931
loss before   = 0.6931
loss after    = 0.4741
```

The batch stayed the same. The model changed because
`TinyBigramLanguageModel::train_one_step` updated the logit table using the
checked `LearningRate`.

## Solution 5: Check the public text gate

The public example prints:

```text
public token count = 3
public vocabulary size = 2
public loss before = 0.6931
public loss after = 0.4741
```

The rejecting boundary is:

```text
PublicLanguageModelingExample::from_reviewed_text
```

The rejection happens before tokenization because public safety is not a token
processing question. Restricted or private text should not become public tokens,
IDs, batches, losses, traces, or examples.

## Self-Check

- You can trace public text into token IDs without losing the vocabulary boundary.
- You can explain why a length-three token sequence creates two next-token pairs.
- You can explain uniform loss as `ln(vocabulary_size)` for equal logits.
- You can name what changes during one training step.
- You can explain why `PublicLanguageModelingExample` is a public-content boundary, not a model layer.
