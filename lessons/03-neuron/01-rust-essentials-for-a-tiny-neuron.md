# Lesson 4: Rust Essentials for a Tiny Neuron

## Overview

This lesson gives you just enough Rust to read the neuron code as a real system instead of as decorative punctuation. The goal is not to "finish Rust first." The goal is to learn the specific Rust tools that make the math readable.

## Learning Goals

- explain why `f64`, functions, structs, and loops are enough for a first neuron
- use the newtype pattern to distinguish inputs from weights and targets
- read `.0`, `mut`, `&`, `*`, `Vec`, and `impl` without stopping the lesson flow
- know where to go in the official Rust documentation when you want more depth

## Plain-English Explanation

### Rust as honest math

For this module, Rust is a strict way to write down math as code.

That means:

- numbers become `f64`
- formulas become functions
- model parameters become struct fields
- training data becomes a `Vec`
- repeated updates become a loop

### Newtypes are labels, not decoration

The neuron uses small tuple structs such as `Input(f64)` and `Weight(f64)`.

That is the newtype pattern.

It matters because a prediction and a target may both be floating-point numbers, but they do not mean the same thing. Rust lets us preserve that meaning in the type system instead of hoping we stay disciplined forever.

### The small Rust moves that matter here

- `struct` groups related data
- `fn` turns one mathematical step into a named operation
- `impl` attaches behavior to a type
- `mut` makes change explicit
- `&` borrows a value instead of copying it
- `*` reads the value behind a reference when needed
- `Vec` stores datasets and dynamic sequences of examples

### Official Rust references that actually matter for this module

- Rust book: [The Rust Programming Language](https://doc.rust-lang.org/book/)
- numbers and `f64`: [Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html)
- functions: [How Functions Work](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html)
- structs: [Defining and Instantiating Structs](https://doc.rust-lang.org/book/ch05-01-defining-structs.html)
- tuple structs and the newtype pattern: [Tuple Structs Without Named Fields](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#tuple-structs-without-named-fields)
- methods and `impl`: [Method Syntax](https://doc.rust-lang.org/book/ch05-03-method-syntax.html)
- mutability: [Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)
- borrowing: [References and Borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)
- vectors: [Storing Lists of Values with Vectors](https://doc.rust-lang.org/book/ch08-01-vectors.html)
- loops: [Control Flow](https://doc.rust-lang.org/book/ch03-05-control-flow.html)
- patterns in loops: [All the Places Patterns Can Be Used](https://doc.rust-lang.org/book/ch18-01-all-the-places-for-patterns.html)
- derived traits such as `Debug`, `Clone`, and `Copy`: [Traits: Defining Shared Behavior](https://doc.rust-lang.org/book/ch10-02-traits.html)
- `f64` math methods such as `.exp()` and `.powi()`: [`f64` primitive docs](https://doc.rust-lang.org/std/primitive.f64.html)
- formatted printing: [`println!`](https://doc.rust-lang.org/std/macro.println.html)

## Algebra Form

The neuron material uses a small set of real-valued symbols:

```math
x, w, b, y, \hat{y} \in \mathbb{R}
```

The parameter set is just:

```math
\text{Neuron} = \{w_1, w_2, b\}
```

The deeper mapping for this module is:

```text
math -> functions -> struct -> loop -> system
```

That is the real translation exercise.

## Rust Form

```rust
#[derive(Debug, Clone, Copy)]
struct Input(f64);

#[derive(Debug, Clone, Copy)]
struct Weight(f64);

#[derive(Debug, Clone, Copy)]
struct Bias(f64);

#[derive(Debug, Clone, Copy)]
struct Neuron {
    w1: Weight,
    w2: Weight,
    b: Bias,
}

fn pre_activation(x1: Input, x2: Input, w1: Weight, w2: Weight, b: Bias) -> f64 {
    w1.0 * x1.0 + w2.0 * x2.0 + b.0
}

impl Neuron {
    fn raw_score(&self, x1: Input, x2: Input) -> f64 {
        pre_activation(x1, x2, self.w1, self.w2, self.b)
    }
}

fn main() {
    let neuron = Neuron {
        w1: Weight(0.8),
        w2: Weight(-0.4),
        b: Bias(0.1),
    };

    let z = neuron.raw_score(Input(1.0), Input(0.5));
    println!("raw score z = {:.3}", z);
}
```

```rust
#[derive(Debug, Clone, Copy)]
struct Input(f64);

fn main() {
    let dataset = vec![
        (Input(0.0), Input(0.0), 0.0_f64),
        (Input(0.0), Input(1.0), 0.0_f64),
        (Input(1.0), Input(0.0), 0.0_f64),
        (Input(1.0), Input(1.0), 1.0_f64),
    ];

    for (x1, x2, target) in &dataset {
        println!("x1={:.1}, x2={:.1}, target={:.1}", x1.0, x2.0, *target);
    }
}
```

## Why This Matters

If you understand the Rust pieces above, the rest of the neuron lesson stops being "Rust syntax" and becomes what it should be: a small mathematical system encoded honestly.

This is the same architectural instinct you use in larger systems:

- separate roles clearly
- give values meaningful types
- make mutation explicit
- let the structure explain the behavior

## Short Practice

1. In one sentence, what does the newtype pattern buy us in this module?
2. Why is `mut` required for parameter updates but not for a plain forward pass?
3. When you see `for (x1, x2, target) in &dataset`, what do `&` and `*target` mean?
