# Rust Essentials for a Tiny Neuron

## Overview

This lesson gives you just enough Rust to read the neuron code as a real system instead of as decorative punctuation. The goal is not to "finish Rust first." The goal is to learn the specific Rust tools that make the math readable.

## Learning Goals

- explain how raw numeric literals become typed neuron values at the boundary
- use the newtype pattern to distinguish inputs from weights and targets
- explain when to overload operators instead of peeling values out with `.0`
- read `mut`, `&`, `*`, and `impl` without stopping the lesson flow
- know where to go in the official Rust documentation when you want more depth

## Plain-English Explanation

### Rust as honest math

For this module, Rust is a strict way to write down math as code.

That means:

- raw numeric literals are validated into semantic types
- formulas become functions
- model parameters become struct fields
- training data becomes a `Dataset`
- repeated updates become a loop

### Newtypes are labels, not decoration

The neuron uses semantic values such as `InputValue`, `Weight`, and `Target`.

That is the newtype pattern.

It matters because a prediction and a target may both be floating-point numbers, but they do not mean the same thing. Rust lets us preserve that meaning in the type system instead of hoping we stay disciplined forever.

### Better than `value.0` everywhere

Inside a newtype implementation, you can always reach the inner number. In
public learner code, that storage detail should stay hidden.

For this kind of teaching code, the better move is usually:

- keep the newtypes
- do not implement `Deref<Target = f64>`
- implement only the math operators that make semantic sense

That lets the code read like:

- `&features * &weights`
- `prediction - target`
- `self.w1 = self.w1 - lr * gradient`

without erasing the meaning of the types.

### The small Rust moves that matter here

- `struct` groups related data
- `fn` turns one mathematical step into a named operation
- `impl` attaches behavior to a type
- `mut` makes change explicit
- `&` borrows a value instead of copying it
- `*` reads the value behind a reference when needed
- `Dataset` stores a checked sequence of examples

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
use rust_ml_neuron::{Bias, FeatureVector, InputValue, Weight, WeightVector};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let features = FeatureVector::two(InputValue::try_from(1.0)?, InputValue::try_from(0.5)?);
    let weights = WeightVector::two(Weight::try_from(0.8)?, Weight::try_from(-0.4)?);
    let bias = Bias::try_from(0.1)?;

    let weighted_evidence = (&features * &weights)?;
    let z = (weighted_evidence + bias)?;

    println!("raw score z = {z:.3}");
    Ok(())
}
```

```rust
use rust_ml_neuron::Dataset;

fn main() -> Result<(), rust_ml_neuron::Error> {
    let dataset = Dataset::and_gate()?;

    for example in dataset.examples() {
        print!("features:");
        for value in example.features().values() {
            print!(" {value}");
        }
        println!(" -> target {}", example.target());
    }

    Ok(())
}
```

## Why This Matters

If you understand the Rust pieces above, the rest of the neuron lesson stops being "Rust syntax" and becomes what it should be: a small mathematical system encoded honestly.

This is the same architectural instinct you use in larger systems:

- separate roles clearly
- give values meaningful types
- make mutation explicit
- let the structure explain the behavior

## Concept Trace

- **Object/newtype:** `InputValue`, `Weight`, `Bias`, `Target`, and `LearningRate`.
- **Invariant:** each role has a different meaning even when the stored representation is numeric.
- **Map:** typed fields and inputs -> checked forward-pass values.
- **Runnable proof:** `cargo run --manifest-path code/Cargo.toml -p rust_ml_neuron --example 01_weighted_sum`.
- **Failure signal:** you remove the newtypes and the compiler can no longer help distinguish inputs, weights, and targets.

## Short Practice

1. In one sentence, what does the newtype pattern buy us in this module?
2. Why is operator overloading a better fit here than implementing `Deref<Target = f64>` for every numeric role?
3. Why is `mut` required for parameter updates but not for a plain forward pass?
4. When you see `for (x1, x2, target) in &dataset`, what do `&` and `*x1` mean?
