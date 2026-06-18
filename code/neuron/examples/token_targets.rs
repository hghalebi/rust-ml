use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClassCount(usize);

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenContext {
    tokens: Vec<TokenId>,
}

impl TokenContext {
    fn two_tokens(first: TokenId, second: TokenId) -> Self {
        Self {
            tokens: vec![first, second],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NextTokenExample {
    context: TokenContext,
    target: TokenId,
}

impl NextTokenExample {
    fn from_context(context: TokenContext, target: TokenId) -> Self {
        Self { context, target }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Logits {
    values: Vec<f64>,
}

impl Logits {
    fn lesson_scores() -> Self {
        Self {
            values: vec![0.8, -0.1, 1.6],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Probabilities {
    values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct Gradient {
    values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct Loss(f64);

#[derive(Debug)]
enum Error {
    EmptyLogits,
    TargetOutOfRange {
        target: TokenId,
        class_count: ClassCount,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => write!(formatter, "logits must not be empty"),
            Self::TargetOutOfRange {
                target,
                class_count,
            } => {
                write!(
                    formatter,
                    "target token index {} is out of range for {} logits",
                    target.0, class_count.0
                )
            }
        }
    }
}

fn main() {
    let example = NextTokenExample::from_context(
        TokenContext::two_tokens(TokenId(0), TokenId(1)),
        TokenId(2),
    );
    let logits = Logits::lesson_scores();

    let loss = match cross_entropy_loss(&logits, example.target) {
        Ok(loss) => loss,
        Err(error) => {
            eprintln!("failed to compute token loss: {error}");
            return;
        }
    };

    let gradient = match cross_entropy_gradient(&logits, example.target) {
        Ok(gradient) => gradient,
        Err(error) => {
            eprintln!("failed to compute token gradient: {error}");
            return;
        }
    };

    println!(
        "context_len={} target={} loss={:.4} gradient={:?}",
        example.context.tokens.len(),
        example.target.0,
        loss.0,
        gradient.values
    );
}

fn softmax(logits: &Logits) -> Result<Probabilities, Error> {
    if logits.values.is_empty() {
        return Err(Error::EmptyLogits);
    }

    let max = logits
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits
        .values
        .iter()
        .map(|logit| (logit - max).exp())
        .collect();
    let sum: f64 = exponentials.iter().sum();

    Ok(Probabilities {
        values: exponentials.into_iter().map(|value| value / sum).collect(),
    })
}

fn cross_entropy_loss(logits: &Logits, target: TokenId) -> Result<Loss, Error> {
    let probabilities = softmax(logits)?;
    probabilities
        .values
        .get(target.0)
        .map(|probability| Loss(-probability.ln()))
        .ok_or(Error::TargetOutOfRange {
            target,
            class_count: ClassCount(logits.values.len()),
        })
}

fn cross_entropy_gradient(logits: &Logits, target: TokenId) -> Result<Gradient, Error> {
    let mut probabilities = softmax(logits)?;
    let gradient = probabilities
        .values
        .get_mut(target.0)
        .ok_or(Error::TargetOutOfRange {
            target,
            class_count: ClassCount(logits.values.len()),
        })?;
    *gradient -= 1.0;
    Ok(Gradient {
        values: probabilities.values,
    })
}
