# From Text To Token IDs

## Overview

A language model does not read words directly. It reads token IDs.

The first language-modeling task is therefore a translation:

```text
text -> tokens -> vocabulary -> token IDs
```

In this repo, each arrow is a typed map. Raw learner text becomes `RawText`,
tokens become `TokenTextSequence`, the vocabulary owns the mapping, and token IDs
are checked against that vocabulary.

## Learning Goals

- explain why tokenization is a map, not a string trick
- distinguish token text from token IDs
- explain why `TokenId` needs a `VocabularySize`
- read `Vocabulary * TokenTextSequence -> TokenIdSequence` as typed composition
- predict the output of the first language-modeling example

## Plain-English Explanation

Text is human-readable. Token IDs are model-readable.

The vocabulary is the dictionary between the two worlds. It records which token
gets which ID, then every later model step uses those checked IDs.

For the tiny public text:

```text
red blue red
```

the tokenizer produces:

```text
red, blue, red
```

The vocabulary sees two unique tokens:

```text
red  -> 0
blue -> 1
```

So the text becomes:

```text
0, 1, 0
```

The repeated `red` token gets the same ID both times because the vocabulary owns
the identity map.

## Algebra Form

Let `T` be the tokenization map and `V` be the vocabulary encoding map:

```text
T(text) = [red, blue, red]
V([red, blue, red]) = [0, 1, 0]
```

The composition is:

```text
RawText -> TokenTextSequence
TokenTextSequence + Vocabulary -> TokenIdSequence
```

The invariant is vocabulary membership:

```text
0 <= token_id < vocabulary_size
```

A token ID has meaning only inside the vocabulary that produced it.

## Rust Form

```rust
use rust_ml_lm_basics::{RawText, Vocabulary, WhitespaceTokenizer};

fn main() -> Result<(), rust_ml_lm_basics::Error> {
    let text = RawText::try_from("red blue red")?;
    let tokens = WhitespaceTokenizer.tokenize(&text)?;
    let vocabulary = Vocabulary::from_tokens(&tokens)?;
    let ids = (&vocabulary * &tokens)?;

    println!("token count = {}", tokens.len());
    println!("vocabulary size = {}", vocabulary.size());
    for token_id in ids.ids() {
        println!("token id = {token_id}");
    }

    Ok(())
}
```

The raw string appears only at the boundary. After `RawText::try_from`, the path
uses semantic objects.

The expression:

```text
&vocabulary * &tokens
```

is not ordinary number multiplication. It is a readable `std::ops` spelling for
the checked map from token text to token IDs.

## Why This Matters

Most language-modeling bugs start when IDs are treated as plain integers.

The type model prevents that shortcut. A `TokenId` is not "some number"; it is a
checked position inside a vocabulary. That is why the next lesson can build
input and target pairs without guessing what each number means.

## Concept Trace

- **Object/newtype:** `RawText`, `Token`, `TokenTextSequence`, `Vocabulary`, `TokenId`, and `TokenIdSequence`.
- **Invariant:** a token ID must be inside the vocabulary that produced it.
- **Map:** raw text -> token text -> vocabulary-owned token IDs.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_lm_basics --example 01_tokenize_and_encode`.
- **Failure signal:** you treat a token ID as a reusable raw integer instead of a vocabulary-bound object.

## Short Practice

1. In `red blue red`, why are there three tokens but only two vocabulary entries?
2. Which object owns the mapping from token text to token ID?
3. Why is `TokenId` not enough without a vocabulary size?
4. Which typed operation encodes tokens into IDs?
