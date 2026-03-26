# Transformer Solutions

## Solution 1: Print attention weights

The weights show how strongly each token uses information from the others. Each row should sum to 1 because the weights come from a softmax. Large values indicate the tokens that contribute most to the contextualized output for that row.

## Solution 2: Change the embeddings

Sharper embeddings usually make some dot products stand out more strongly, which can make the attention distribution more concentrated. If the embeddings are more similar to each other, the attention tends to spread more evenly.

## Solution 3: Remove the residual connections

Without residuals, each stage must completely replace the previous representation instead of adding an improvement to it. Conceptually, that means the model loses an easy path for preserving the original signal. In deeper models this usually makes learning harder and less stable.

## Solution 4: Replace ReLU with the identity function

If ReLU becomes the identity, the feed-forward block becomes only a stack of linear maps. That reduces expressiveness because compositions of linear maps collapse into another linear map. The nonlinearity is what lets the block represent richer transformations.

## Solution 5: Explain one token end to end

For one token $x_i$:

1. attention builds a contextualized version of the token by mixing information from other tokens
2. the first residual adds the original token back so the model keeps its earlier signal
3. the feed-forward layer transforms that token representation independently of the other tokens
4. the final residual combines the pre-feed-forward representation with the nonlinear update

This is why a Transformer block can both gather context and preserve identity.

## Solution 6: Standard attention vs linear attention

Standard attention compares each token with every other token explicitly, so with $n$ tokens you get roughly $n^2$ pairwise interactions. Linear attention tries to avoid materializing that full token-to-token matrix by precomputing summaries such as key-value accumulations and then letting each query consult those summaries. That can reduce the sequence-length cost for fixed hidden size.

## Solution 7: Why bother with types?

If everything is just nested `Vec<Vec<f32>>`, it becomes harder to see whether a value is meant to be:

- one token vector
- a whole sequence
- a weight matrix
- a projected representation

A type like `Sequence` tells the reader that the value is a list of same-width token vectors, not just an arbitrary nested list. Const generics go one step further by letting some dimension mistakes fail at compile time instead of later at runtime.
