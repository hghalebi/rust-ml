# Transformer Encoder in Small Chunks

## Overview

This lesson is the low-cognitive-load version.

Rule:

- one idea
- one tiny code snippet
- one small win

Every chunk uses the same rhythm:

`English -> Algebra -> Rust`

Do not try to hold the whole Transformer in your head at once. Stack tiny truths until the architecture becomes ordinary.

## Learning Goals

- move from vectors and matrices to one encoder block without losing the plot
- connect each Transformer concept to one compact algebra form
- use the companion crate as the stable Rust vocabulary for every chunk
- understand where linear attention fits without mixing it up with the 2017 paper

## Chunk 0: A neural network is just input, math, output

### English

At first principles level, a neural network is:

input -> some math -> output

The math is mostly:

- multiply
- add
- apply a small function like ReLU

That is already enough to start.

### Algebra

```math
\mathrm{input} \rightarrow \mathrm{some\ math} \rightarrow \mathrm{output}
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError};

fn main() -> Result<(), ModelError> {
    let input = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?;
    println!("{:?}", input.as_slice());
    Ok(())
}
```

## Chunk 1: A vector is just a list of numbers

### English

A token embedding starts life as a vector.

Nothing magical yet.

### Algebra

```math
x = [x_1, x_2, \ldots, x_n]
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError};

fn main() -> Result<(), ModelError> {
    let x = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?;
    println!("len = {}", x.len());
    println!("{:?}", x.as_slice());
    Ok(())
}
```

## Chunk 2: Newtypes stop the architecture from becoming a pile of vectors

### English

`TokenEmbedding`, `Query`, `Key`, and `Value` are all vectors underneath.

They are **not** the same idea.

The newtype pattern preserves that meaning.

### Algebra

```math
x \neq q \neq k \neq v
```

even if each object lives in some vector space.

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError, Query, TokenEmbedding};

fn main() -> Result<(), ModelError> {
    let token = TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?);
    let query = Query::from_vector(DenseVector::new([ModelScalar::try_from(0.2)?, ModelScalar::try_from(0.5)?])?);

    println!("token width = {}", token.len());
    println!("query width = {}", query.len());
    Ok(())
}
```

## Chunk 3: Dot product measures alignment

### English

Two vectors go in.

One number comes out.

That number tells us how much the vectors align.

### Algebra

```math
a \cdot b = \sum_i a_i b_i
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError};

fn main() -> Result<(), ModelError> {
    let a = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?;
    let b = DenseVector::new([ModelScalar::try_from(4.0)?, ModelScalar::try_from(5.0)?, ModelScalar::try_from(6.0)?])?;

    println!("{}", a.dot(&b)?);
    Ok(())
}
```

## Chunk 4: A matrix is a table of learned weights

### English

A matrix transforms one vector into another.

This is the heart of linear layers.

### Algebra

```math
y = Wx
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseMatrix, ModelError};

fn main() -> Result<(), ModelError> {
    let matrix = DenseMatrix::from_rows([
        [ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?],
        [ModelScalar::try_from(4.0)?, ModelScalar::try_from(5.0)?, ModelScalar::try_from(6.0)?],
    ])?;

    println!("rows = {}", matrix.rows());
    println!("cols = {}", matrix.cols());
    Ok(())
}
```

## Chunk 5: Matrix times vector is the basic neural-network move

### English

Every output row takes a weighted sum of the input vector.

### Algebra

```math
y_r = \sum_c W_{r,c} x_c
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseMatrix, DenseVector, ModelError};

fn main() -> Result<(), ModelError> {
    let matrix = DenseMatrix::from_rows([
        [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(2.0)?],
        [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(3.0)?],
    ])?;
    let vector = DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?;

    let output = (&matrix * &vector)?;
    println!("{:?}", output.as_slice());
    Ok(())
}
```

## Chunk 6: The Transformer starts with a sequence

### English

A Transformer does not process one vector.

It processes a sequence of token embeddings.

### Algebra

```math
X \in \mathbb{R}^{n \times d_{model}}
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError, TokenEmbedding, TokenSequence};

fn main() -> Result<(), ModelError> {
    let seq = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
    ])?;

    println!("tokens = {}", seq.len());
    println!("d_model = {}", seq.d_model());
    Ok(())
}
```

## Chunk 7: Query, key, and value are learned projections

### English

For each token embedding `x`, we make:

- query: what am I looking for?
- key: what do I offer?
- value: what information do I carry?

### Algebra

```math
q = W_Q x + b_Q,\quad
k = W_K x + b_K,\quad
v = W_V x + b_V
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseMatrix, DenseVector, ModelError, ProjectionBias, QueryLayer, QueryProjection,
    TokenEmbedding,
};

fn main() -> Result<(), ModelError> {
    let layer = QueryLayer::new(
        QueryProjection::from_matrix(DenseMatrix::from_rows([
            [ModelScalar::try_from(0.2)?, ModelScalar::try_from(0.1)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.3)?],
            [ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.4)?, ModelScalar::try_from(0.1)?, ModelScalar::try_from(0.2)?],
        ])?),
        ProjectionBias::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?),
    )?;

    let token = TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?);
    let query = layer.forward(&token)?;
    println!("{:?}", query.as_slice());
    Ok(())
}
```

## Chunk 8: One attention score is one query-key comparison

### English

We compare one query against one key.

Higher score means stronger match.

### Algebra

```math
\mathrm{score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, Key, ModelError, Query};

fn main() -> Result<(), ModelError> {
    let query = Query::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?);
    let key = Key::from_vector(DenseVector::new([ModelScalar::try_from(3.0)?, ModelScalar::try_from(4.0)?])?);

    println!("{}", (&query * &key)?);
    Ok(())
}
```

## Chunk 9: Softmax turns scores into weights

### English

Raw scores can be any numbers.

Softmax turns them into weights that:

- are positive
- sum to 1

### Algebra

```math
\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
```

### Rust

```rust
use rust_ml_transformer::{softmax, AttentionScore, AttentionScores, ModelError};

fn main() -> Result<(), ModelError> {
    let scores = AttentionScores::from_scores([
        AttentionScore::try_from(2.0)?,
        AttentionScore::try_from(1.0)?,
        AttentionScore::try_from(0.1)?,
    ])?;
    let weights = softmax(&scores)?;

    println!("{:?}", weights.values().collect::<Vec<_>>());
    Ok(())
}
```

## Chunk 10: Weighted sum mixes the values

### English

After we have attention weights, we mix the value vectors.

This creates the new contextualized representation for one token.

### Algebra

```math
\mathrm{output} = \sum_i \alpha_i v_i
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    AttentionWeight, AttentionWeights, DenseVector, ModelError, Value, ValueSequence,
};

fn main() -> Result<(), ModelError> {
    let weights = AttentionWeights::from_weights([
        AttentionWeight::try_from(0.25)?,
        AttentionWeight::try_from(0.75)?,
    ])?;
    let values = ValueSequence::from_values([
        Value::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        Value::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(2.0)?])?),
    ])?;

    let output = (&weights * &values)?;
    println!("{:?}", output.as_slice());
    Ok(())
}
```

## Chunk 11: One attention head is the whole self-attention recipe

### English

One attention head:

1. projects every token to query, key, value
2. compares each query with all keys
3. mixes the values

That is self-attention.

### Algebra

```math
\mathrm{head}(X) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    AttentionHead, DenseMatrix, DenseVector, KeyLayer, KeyProjection, ModelError,
    ProjectionBias, QueryLayer, QueryProjection, TokenEmbedding, TokenSequence, ValueLayer,
    TokenIndex, ValueProjection, VectorLength,
};

fn identity_projection() -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?], [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?]])
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn main() -> Result<(), ModelError> {
    let width = VectorLength::try_from(2)?;
    let head = AttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;

    let seq = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;

    let outputs = head.forward(&seq)?;
    println!("{:?}", outputs.output(TokenIndex::try_from(0)?)?.as_slice());
    Ok(())
}
```

## Chunk 12: Multi-head attention means several learned views in parallel

### English

One head gives one way to compare tokens.

Multiple heads let the model learn multiple views at once.

Then the head outputs are concatenated and projected back.

### Algebra

```math
\mathrm{MultiHead}(Q, K, V) =
\mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)W_O
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    AttentionHead, DenseMatrix, DenseVector, KeyLayer, KeyProjection, ModelError,
    MultiHeadAttention, OutputLayer, OutputProjection, ProjectionBias, QueryLayer,
    QueryProjection, TokenEmbedding, TokenIndex, TokenSequence, ValueLayer, ValueProjection,
    VectorLength,
};

fn identity_projection() -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?], [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?]])
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn main() -> Result<(), ModelError> {
    let width = VectorLength::try_from(2)?;
    let head_a = AttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;
    let head_b = AttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;
    let mha = MultiHeadAttention::new(
        vec![head_a, head_b],
        OutputLayer::new(
            OutputProjection::from_matrix(DenseMatrix::from_rows([
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?],
                [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?],
            ])?),
            zero_bias(width)?,
        )?,
    )?;

    let seq = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;

    let output = mha.forward(&seq)?;
    println!("{:?}", output.token(TokenIndex::try_from(0)?)?.as_slice());
    Ok(())
}
```

## Chunk 13: Positional encoding tells the model where a token is

### English

Attention alone does not know order.

So the paper adds positional information to token embeddings.

### Algebra

```math
x_{\mathrm{with\ position}} = x_{\mathrm{token}} + x_{\mathrm{position}}
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseVector, ModelError, PositionalEncodingTable, TokenEmbedding, TokenIndex, TokenSequence,
};

fn main() -> Result<(), ModelError> {
    let seq = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;
    let positions = PositionalEncodingTable::new(rust_ml_transformer::VectorLength::try_from(4)?);

    let with_position = positions.add_to_sequence(&seq)?;
    println!("{:?}", with_position.token(TokenIndex::try_from(0)?)?.as_slice());
    Ok(())
}
```

## Chunk 14: Residual connections keep the old signal alive

### English

Instead of replacing `x` with `f(x)`, we add them:

`x + f(x)`

That gives the model an easy path to preserve earlier information.

### Algebra

```math
\mathrm{residual}(x) = x + f(x)
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, ModelError, TokenEmbedding, TokenIndex, TokenSequence};

fn main() -> Result<(), ModelError> {
    let left = TokenSequence::new(vec![TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?])?)])?;
    let right = TokenSequence::new(vec![TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.5)?, ModelScalar::try_from(-1.0)?])?)])?;

    let sum = (&left + &right)?;
    println!("{:?}", sum.token(TokenIndex::try_from(0)?)?.as_slice());
    Ok(())
}
```

## Chunk 15: Layer normalization stabilizes each token

### English

After attention or feed-forward, values can get messy.

Layer normalization rescales each token so the model behaves more predictably.

### Algebra

```math
\mathrm{LayerNorm}(x)_i =
\gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
```

### Rust

```rust
use rust_ml_transformer::{ModelScalar, DenseVector, LayerNorm, ModelError, TokenEmbedding};

fn main() -> Result<(), ModelError> {
    let norm = LayerNorm::new(rust_ml_transformer::VectorLength::try_from(3)?)?;
    let token = TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(2.0)?, ModelScalar::try_from(3.0)?])?);

    let normalized = norm.forward_token(&token)?;
    println!("{:?}", normalized.as_slice());
    Ok(())
}
```

## Chunk 16: Feed-forward transforms each token on its own

### English

Attention mixes information across tokens.

Feed-forward transforms each token independently.

That division is one of the core architectural ideas.

### Algebra

```math
\mathrm{FFN}(x) = W_2(\mathrm{ReLU}(W_1x + b_1)) + b_2
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseMatrix, DenseVector, FeedForward, FeedForwardLayer1, FeedForwardLayer2,
    FeedForwardProjection1, FeedForwardProjection2, ModelError, ProjectionBias, TokenEmbedding,
};

fn main() -> Result<(), ModelError> {
    let feed_forward = FeedForward::new(
        FeedForwardLayer1::new(
            FeedForwardProjection1::from_matrix(DenseMatrix::from_rows([
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?],
                [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?],
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(1.0)?],
            ])?),
            ProjectionBias::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?),
        )?,
        FeedForwardLayer2::new(
            FeedForwardProjection2::from_matrix(DenseMatrix::from_rows([
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?],
                [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?],
            ])?),
            ProjectionBias::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?),
        )?,
    )?;

    let token = TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(-2.0)?])?);
    let output = feed_forward.forward_token(&token)?;
    println!("{:?}", output.as_slice());
    Ok(())
}
```

## Chunk 17: One encoder block is the full heartbeat

### English

One encoder block does:

1. multi-head attention
2. residual add
3. layer norm
4. feed-forward
5. residual add
6. layer norm

That is the heartbeat of the encoder.

### Algebra

```math
A = \mathrm{LayerNorm}(X + \mathrm{MHA}(X))
```

```math
Y = \mathrm{LayerNorm}(A + \mathrm{FFN}(A))
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    AttentionHead, DenseMatrix, DenseVector, FeedForward, FeedForwardLayer1, FeedForwardLayer2,
    FeedForwardProjection1, FeedForwardProjection2, KeyLayer, KeyProjection, LayerNorm,
    ModelError, MultiHeadAttention, OutputLayer, OutputProjection, ProjectionBias, QueryLayer,
    QueryProjection, TokenEmbedding, TokenIndex, TokenSequence, TransformerEncoderBlock, ValueLayer,
    ValueProjection, VectorLength,
};

fn identity_projection() -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?], [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?]])
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn main() -> Result<(), ModelError> {
    let width = VectorLength::try_from(2)?;
    let head = AttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;
    let attention = MultiHeadAttention::new(
        vec![head],
        OutputLayer::new(OutputProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;
    let feed_forward = FeedForward::new(
        FeedForwardLayer1::new(
            FeedForwardProjection1::from_matrix(DenseMatrix::from_rows([
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?],
                [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?],
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(1.0)?],
            ])?),
            ProjectionBias::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?),
        )?,
        FeedForwardLayer2::new(
            FeedForwardProjection2::from_matrix(DenseMatrix::from_rows([
                [ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?],
                [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?],
            ])?),
            ProjectionBias::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(0.0)?])?),
        )?,
    )?;
    let block = TransformerEncoderBlock::new(
        attention,
        LayerNorm::new(width)?,
        feed_forward,
        LayerNorm::new(width)?,
    )?;

    let seq = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;

    let output = block.forward(&seq)?;
    println!("{:?}", output.token(TokenIndex::try_from(0)?)?.as_slice());
    Ok(())
}
```

## Chunk 18: Linear attention fits in the same slot, not the same paper

### English

Keep this category line clean:

- original paper: scaled dot-product multi-head attention
- later efficient family: linear attention

Same architecture slot.

Different attention math.

### Algebra

```math
\mathrm{Transformer\ block}
\rightarrow
\mathrm{attention\ module}
```

Original paper:

```math
\mathrm{attention\ module} =
\mathrm{scaled\ dot\ product\ multihead\ attention}
```

Later variant:

```math
\mathrm{attention\ module} =
\mathrm{linear\ attention}
```

### Rust

```rust
use rust_ml_transformer::{
    ModelScalar,
    DenseMatrix, DenseVector, KeyLayer, KeyProjection, LinearAttentionHead, ModelError,
    ProjectionBias, QueryLayer, QueryProjection, TokenEmbedding, TokenIndex, TokenSequence, ValueLayer,
    ValueProjection, VectorLength,
};

fn identity_projection() -> Result<DenseMatrix, ModelError> {
    DenseMatrix::from_rows([[ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?], [ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?]])
}

fn zero_bias(width: VectorLength) -> Result<ProjectionBias, ModelError> {
    Ok(ProjectionBias::from_vector(DenseVector::zeros(width)?))
}

fn main() -> Result<(), ModelError> {
    let width = VectorLength::try_from(2)?;
    let head = LinearAttentionHead::new(
        QueryLayer::new(QueryProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        KeyLayer::new(KeyProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
        ValueLayer::new(ValueProjection::from_matrix(identity_projection()?), zero_bias(width)?)?,
    )?;

    let seq = TokenSequence::new(vec![
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(1.0)?, ModelScalar::try_from(0.0)?])?),
        TokenEmbedding::from_vector(DenseVector::new([ModelScalar::try_from(0.0)?, ModelScalar::try_from(1.0)?])?),
    ])?;

    let outputs = head.forward(&seq)?;
    println!("{:?}", outputs.output(TokenIndex::try_from(0)?)?.as_slice());
    Ok(())
}
```

## Concept Trace

- **Object/newtype:** `ModelScalar`, `DenseVector`, `TokenEmbedding`, `TokenSequence`, `AttentionHead`, and `Encoder`.
- **Invariant:** each chunk preserves the role needed by the next chunk before composing.
- **Map:** small typed chunks -> attention head -> multi-head attention -> encoder block -> encoder.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_transformer --example encoder_demo`.
- **Failure signal:** you can follow one chunk in isolation but cannot explain how its output object becomes the next chunk's input object.

## Final memory card

Keep this in your head:

```text
TokenEmbedding = what the token is
Query          = what the token is looking for
Key            = what the token offers
Value          = what information the token carries
```

Then:

```text
attention = compare query with keys, then mix values
```

Then:

```text
encoder block = attention + residual + norm + feed-forward + residual + norm
```

That is the spine of the whole paper.
