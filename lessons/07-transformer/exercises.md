# Transformer Exercises

## Exercise 1: Print attention weights

Add a print inside the attention loop:

```rust
println!("{:?}", weights);
```

Questions:

- which tokens attend most strongly to which other tokens?
- do the rows behave like probability distributions?

## Exercise 2: Change the embeddings

Try a sharper input such as:

```rust
vec![1.0, 0.0, 0.0, 0.0]
```

Questions:

- how do the attention scores change?
- does the output become more concentrated or more mixed?

## Exercise 3: Remove the residual connections

Delete both vector additions and pass only the transformed outputs forward.

Questions:

- what changes conceptually?
- why do deep models usually keep residual paths?

## Exercise 4: Replace ReLU with the identity function

Turn the feed-forward nonlinearity into:

```rust
fn identity_vec(x: &[f64]) -> Vec<f64> {
    x.to_vec()
}
```

Questions:

- what happens to the expressiveness of the feed-forward sublayer?
- why is a nonlinearity useful here?

## Exercise 5: Explain one token end to end

Choose one token $x_i$ and explain, in English, what happens to it through:

1. attention
2. first residual
3. feed-forward
4. final residual

## Exercise 6: Standard attention vs linear attention

Explain the difference between these two computation styles:

- compare every token with every other token directly
- precompute key-value summaries and let each query use them

Questions:

- why is the first pattern quadratic in sequence length?
- why can the second pattern be cheaper?

## Exercise 7: Why bother with types?

Suppose you model everything as nested `Vec<Vec<f32>>`.

Questions:

- what becomes harder to understand?
- what does a type like `Sequence` tell the reader immediately?
- why might const generics be useful in a teaching implementation?
