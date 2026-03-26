# Attention is all you need (Transformer) - Model explanation (including math), Inference and Training

Type: user-provided transcript  
Intended modules: [06 Attention](../../lessons/06-attention/README.md), [07 Transformer](../../lessons/07-transformer/README.md)

## Note

This file is a lightly formatted transcription of the video text provided to the repo. Section headings were normalized for readability, but the substance follows the supplied transcript.

## Introduction and Recurrent Neural Networks

Hello guys, welcome to my video about the Transformer. This is actually the version 2.0 of my series on the Transformer. I had a previous video in which I talked about the Transformer, but the audio quality was not good. As suggested by my viewers, because the video had huge success, they asked me to improve the audio quality. That is why I am doing this video.

You do not have to watch the previous series because I will be doing basically the same things but with some improvements. I am compensating for some mistakes I made and adding some improvements. After watching this video, I suggest watching my other video about how to code a Transformer model from scratch: how to code the model itself, how to train it on data, and how to run inference.

Stick with me because it is going to be a little long journey, but for sure it will be worth it.

Before we talk about the Transformer, I want to first talk about recurrent neural networks, the networks that were used before the Transformer for most sequence-to-sequence tasks. Let us review them.

Recurrent neural networks existed a long time before the Transformer and allowed us to map one sequence of input to another sequence of output. In this case our input is $X$ and we want an output sequence $Y$. What we did before is that we split the sequence into single items, so we gave the recurrent neural network the first item as input, $X_1$, along with an initial state usually made of only zeros, and the recurrent neural network produced an output, let us call it $Y_1$. This happened at the first time step.

Then we took the hidden state of the network from the previous time step along with the next input token, $X_2$, and the network had to produce the second output token, $Y_2$. Then we repeated the same procedure at the third time step, in which we took the hidden state of the previous time step along with the input token at time step 3, and the network had to produce the next output token, which is $Y_3$.

If you have $n$ tokens, you need $n$ time steps to map an input sequence into an output sequence. This worked fine for a lot of tasks, but it had some problems.

## Problems with RNNs

The first problem with recurrent neural networks is that they are slow for long sequences. Think of the process we just described: we have kind of like a `for` loop in which we do the same operation for every token in the input. The longer the sequence, the longer this computation, and this made the network hard to train for long sequences.

The second problem was vanishing or exploding gradients. You may have heard these terms before, but I will try to give a brief practical explanation.

Frameworks like PyTorch convert our networks into a computation graph. Imagine we have two inputs, $x$ and $y$. Our graph first multiplies these numbers:

```math
f(x, y) = x \cdot y
```

Then the result, call it $z$, is sent to another function:

```math
g(z) = z^2
```

What PyTorch wants to calculate is usually the derivative of the loss with respect to each weight. In this simple example, we calculate the derivative of the output function with respect to all its inputs:

```math
\frac{dg}{df} \cdot \frac{df}{dx}
```

This is the chain rule.

The longer the chain of computation, the longer this multiplication chain becomes. If we have many nodes one after another, the product can become very small or very large. For example, if two numbers are $0.5$ and $0.5$, their product is $0.25$, which is smaller. If we multiply many numbers smaller than one, we get an even smaller number. If we multiply many numbers bigger than one, we get an even bigger number.

This is not desirable because the CPU or GPU can only represent numbers up to a certain precision, and because the contribution of that product to the output may become too small or too large. If it becomes very small, the framework will adjust the weights very slowly because the gradient contribution is tiny. This is the vanishing gradient. In the other case it can explode into very big numbers, and that is also a problem.

The next problem is difficulty in accessing information from long ago. The first input token is given to the recurrent neural network with the first state. The network produces a new hidden state; then we use that hidden state along with the next token to produce the next output. If we have a very long input sequence, the final token will have a hidden state whose contribution from the first token has nearly gone because of this long chain of multiplication. That means the last token will not depend much on the first token.

This is not good because, as humans, we know that in a long text the context from 200 words before can still be relevant to the current word. This is something the RNN could not represent well, and this is why we have the Transformer.

## Transformer Architecture Overview

The Transformer solves these RNN problems. We can divide the structure of the Transformer into two macro blocks:

- an encoder
- a decoder

At the top there is also a linear layer. The encoder and decoder are connected, and some output from the encoder is sent as input to the decoder.

Before we continue, the speaker introduces some notation, especially matrix multiplication.

Imagine we have an input matrix of shape sequence by $d_{\text{model}}$, for example $6 \times 512$. Each row is a word, and each word is represented by 512 numbers. If we multiply this matrix by its transpose, we get a new matrix:

```math
(6 \times 512) \cdot (512 \times 6) = 6 \times 6
```

Each value in this output matrix is the dot product of one row with one column. The first value is the dot product of the first row with the first column. The second value is the dot product of the first row with the second column, and so on.

The dot product means taking matching positions, multiplying them, and summing them:

```math
a \cdot b = \sum_i a_i b_i
```

The speaker emphasizes that this notation will be used throughout the rest of the explanation.

## The Encoder: Input Embeddings and Positional Encoding

The encoder starts with input embeddings. We begin with a sentence of six words, tokenize it, transform the sentence into tokens, and map the resulting input IDs into vectors of size 512.

Each word becomes an embedding vector of size 512. The same vocabulary item always maps to the same row in the embedding table, but the embedding values themselves are learned parameters. The vocabulary ID stays fixed, while the actual embedding numbers change during training according to the needs of the loss function.

The quantity 512 is called $d_{\text{model}}$, following the notation from the paper.

The next layer of the encoder is positional encoding. The point of positional encoding is that each word should carry information about its position in the sentence. The embeddings alone do not tell the model whether a word is at the beginning, middle, or end.

The speaker describes positional encoding as a fixed vector of size 512 that is added to each token embedding. This vector is not learned. It is computed once and then reused for every sentence during training and inference.

The positional encodings follow the formulas from the paper:

```math
\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
```

```math
\text{PE}(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
```

The even dimensions use sine, and the odd dimensions use cosine. The speaker explains that these functions create a visible pattern over positions and dimensions, and the hope is that the model can learn from that pattern.

## Self-Attention Mechanism

The next layer of the encoder is multi-head attention, but first the speaker visualizes single-head self-attention.

Self-attention allows the model to relate words to each other. Input embeddings capture meaning. Positional encodings add information about where words are in the sentence. Self-attention then lets each word interact with all the other words.

For a sentence with six words and embedding size 512, we can view the input as a matrix of shape $6 \times 512$. The speaker refers to the input as $Q$, $K$, and $V$, initially all equal to the same sentence representation in the self-attention case.

The attention formula is:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

If we multiply a $6 \times 512$ matrix by a $512 \times 6$ matrix, we obtain a $6 \times 6$ matrix. Each entry is the dot product between one word and another. After the softmax, each row sums to 1.

The speaker explains that each value in this matrix represents the intensity of the relationship between one word and another. When we multiply this attention matrix by $V$, we obtain a new matrix of shape $6 \times 512$.

Now each output embedding captures:

- the meaning of the word
- the position of the word
- the relationship of that word with all the other words

The speaker also mentions several useful properties:

- permutation invariance in the self-attention computation
- no new parameters in the plain self-attention formula itself
- large values often appear along the diagonal because a word is strongly related to itself
- if an interaction should be blocked, we can replace a score with $-\infty$ before the softmax so that it becomes 0 afterward

That last property becomes important in the decoder.

## Multi-Head Attention

The speaker next explains how self-attention becomes multi-head attention.

We start with an input of shape sequence by $d_{\text{model}}$, for example $6 \times 512$. We make three copies of it:

- query
- key
- value

Then we multiply them by learned parameter matrices:

```math
Q' = QW_Q
```

```math
K' = KW_K
```

```math
V' = VW_V
```

Each of these matrices keeps shape sequence by $d_{\text{model}}$.

Next, the matrices are split along the model dimension into multiple heads. If there are $h = 4$ heads, then:

```math
d_k = d_{\text{model}} / h
```

Each head sees the full sentence, but only a slice of each word embedding.

The attention for each head is computed independently:

```math
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
```

Then the heads are concatenated:

```math
\text{Concat}(\text{head}_1, \ldots, \text{head}_h)
```

Finally, the concatenated result is projected with another learned matrix:

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
```

The speaker emphasizes that different heads may focus on different relationships between words. One head may relate "making" to "difficult," while another may focus on a different dependency because it sees a different slice of the embedding space.

The speaker also offers an intuitive interpretation of the names query, key, and value using a database analogy: a query like "love" is compared with category keys such as "romantic" or "comedy," and the resulting similarity scores determine which values should be emphasized.

## Add and Norm

The next layer is Add and Norm, which uses layer normalization.

The speaker contrasts layer normalization with batch normalization. In layer normalization, we treat each item in the batch independently. For each item, we compute its own mean and variance over its features, normalize it, and then apply two learned parameters:

```math
\gamma \cdot \hat{x} + \beta
```

The point is not only to normalize values but also to let the model learn how strongly to scale and shift them afterward.

The speaker points out that batch normalization mixes across batch items, while layer normalization handles each example independently, which is appropriate for Transformers.

## The Decoder and Masked Multi-Head Attention

On the decoder side, we again have output embeddings and positional encodings. Then we have masked multi-head attention, followed by cross-attention with the encoder output.

The decoder self-attention is called masked because the model must be causal: the output at position $t$ can only depend on tokens up to position $t$, not future tokens.

The speaker explains this with the attention score matrix. All positions above the main diagonal are replaced with $-\infty$ before the softmax. After softmax, those values become 0. This prevents future tokens from influencing the current token.

Then comes the cross-attention block:

- queries come from the decoder side
- keys and values come from the encoder output

So this block is not self-attention anymore. It is cross-attention between the decoder representation and the encoder representation.

After that, there is a feed-forward layer, add-and-norm again, and finally a linear layer that projects the decoder output from sequence by $d_{\text{model}}$ to sequence by vocabulary size.

That last projection lets us interpret each output vector as scores over the vocabulary.

## Training the Transformer

The speaker uses translation as the main example and goes from the English sentence "I love you very much" to an Italian sentence.

First, the encoder input is prepared by adding two special tokens:

- start of sentence (SOS)
- end of sentence (EOS)

So the encoder sees:

- SOS
- the English sentence tokens
- EOS

The decoder input is the target sentence shifted right. In the example, the decoder starts with SOS, followed by padding as needed to fit the fixed sequence length.

The encoder produces an output of shape sequence by $d_{\text{model}}$. This output encodes meaning, position, and contextual interaction between words.

The decoder takes:

- its own shifted-right input as the masked self-attention input
- the encoder output as keys and values for cross-attention

After the decoder, a linear layer maps the decoder output into vocabulary logits. A softmax converts those logits into probabilities. The label is the expected translated sentence ending with EOS.

The loss used is cross-entropy loss, and then that loss is backpropagated through all the weights.

The reason for SOS and EOS is also explained:

- when the model sees SOS, it should output the first target token
- when it sees the last real token, it should output EOS

This tells the model when the translation is complete.

The speaker highlights one major advantage of Transformer training over RNN training: training happens in one pass. We give an input sequence to the encoder, a shifted-right target sequence to the decoder, compute the output, compare to the label, and backpropagate. There is no token-by-token recurrent loop during training.

## Inference

Inference works differently. The encoder side is run once on the source sentence:

- add SOS and EOS
- convert to embeddings
- add positional encodings
- run through the encoder

The decoder starts with only SOS. That token is embedded, positionally encoded, and sent through the decoder together with the encoder output.

The decoder output is projected to vocabulary logits, a softmax is applied, and the highest-scoring token is selected. If the model was trained correctly, the first predicted token should be the first token of the translation.

At the next time step:

- the encoder output is reused
- the newly predicted token is appended to the decoder input
- the decoder runs again

This repeats until the model outputs EOS.

The speaker stresses that unlike training, inference happens token by token. We cannot generate the full translated sentence in one pass because each next token depends on the tokens generated so far.

The speaker also mentions greedy decoding, where at each step we choose the token with the maximum softmax value, and beam search, where we keep the top $B$ candidate sequences and expand them in parallel, retaining only the most probable ones.

## Conclusion

The speaker closes by thanking the audience and emphasizing that the video was long but worthwhile because it walked through each aspect of the Transformer. The viewer is encouraged to watch the companion coding video about implementing a Transformer model from scratch, training it on a dataset, running inference, and using the provided GitHub repository and Colab notebook.

The speaker asks viewers to subscribe, to mention anything they did not understand so that further explanations can be added, and to point out ways to improve future videos.
