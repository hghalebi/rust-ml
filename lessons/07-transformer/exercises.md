# Transformer Exercises

## Exercise 1: Trigger a useful shape error

Run this and read the error carefully:

```rust
use rust_ml_transformer::{ModelScalar, DenseMatrix, DenseVector, ModelError};

fn main() -> Result<(), ModelError> {
    let matrix = DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?], [ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?]])?;
    let vector = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?;

    let result = &matrix * &vector;
    println!("{result:?}");
    Ok(())
}
```

Questions:

- which operation failed?
- what were the two shapes?
- what hint did the error give you?

## Exercise 2: Build a `TokenSequence`

Create a three-token sequence with model width `4`.

Questions:

- why does every token need the same width?
- what would `TokenSequence::new` reject?

## Exercise 3: Project one token into query, key, and value

Start from one `TokenEmbedding` and build:

- one `QueryLayer`
- one `KeyLayer`
- one `ValueLayer`

Questions:

- what stays the same across all three layers?
- what changes?

## Exercise 4: Print attention weights for one head

Use a tiny `AttentionHead` and a two-token sequence. Print the attention outputs.

Questions:

- which token seems to matter more for the first output?
- how would you inspect the scores and weights if you wanted more visibility?

## Exercise 5: Add positional encodings

Take a small `TokenSequence` and run `PositionalEncodingTable::add_to_sequence`.

Questions:

- what stays the same after adding positions?
- what changes?
- why does the model need this signal if it no longer uses recurrence?

## Exercise 6: Run one full encoder block

Build:

- `MultiHeadAttention`
- `LayerNorm`
- `FeedForward`
- `TransformerEncoderBlock`

and run one forward pass.

Questions:

- what shape does the output have?
- why does the block require the attention output width to match the input width?

## Exercise 7: Standard attention versus linear attention

Replace `AttentionHead` with `LinearAttentionHead` in a tiny experiment.

Questions:

- what architectural slot stayed the same?
- what math changed?
- why is that an efficiency discussion rather than a definition of the 2017 paper?

## Failure Signals

- You can build a type but cannot explain the invariant its constructor checks.
- You ignore shape errors instead of reading the operation, shapes, and hint.
- You add positional encodings but cannot say which dimensions stay unchanged.
- You describe linear attention as "the Transformer" instead of a comparison point in the same architectural slot.

## Debugging Hints

- Start every Transformer experiment by naming sequence length and `d_model`.
- When a constructor fails, read the error as teaching material before changing the code.
- Use typed addition for residual and positional maps so the code mirrors the algebra.
- Keep the 2017 architecture separate from later efficiency variants when explaining attention.
