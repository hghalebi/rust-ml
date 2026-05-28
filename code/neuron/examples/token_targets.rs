use std::fmt;
use std::num::TryFromIntError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenId(usize);

#[derive(Debug, Clone, PartialEq, Eq)]
struct NextTokenExample {
    context: Vec<TokenId>,
    target: TokenId,
}

impl NextTokenExample {
    fn new(context: Vec<TokenId>, target: TokenId) -> Self {
        Self { context, target }
    }
}

#[derive(Debug)]
enum Error {
    EmptyLogits,
    TargetOutOfRange { target: usize, classes: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => write!(formatter, "logits must not be empty"),
            Self::TargetOutOfRange { target, classes } => {
                write!(
                    formatter,
                    "target token index {target} is out of range for {classes} logits"
                )
            }
        }
    }
}

fn main() {
    let example = NextTokenExample::new(vec![TokenId(0), TokenId(1)], TokenId(2));
    let logits = vec![0.8, -0.1, 1.6];

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
        "context_len={} target={} loss={loss:.4} gradient={gradient:?}",
        example.context.len(),
        example.target.0
    );
}

fn softmax(logits: &[f64]) -> Result<Vec<f64>, Error> {
    if logits.is_empty() {
        return Err(Error::EmptyLogits);
    }

    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exponentials.iter().sum();

    Ok(exponentials.into_iter().map(|value| value / sum).collect())
}

fn cross_entropy_loss(logits: &[f64], target: TokenId) -> Result<f64, Error> {
    let probabilities = softmax(logits)?;
    probabilities
        .get(target.0)
        .map(|probability| -probability.ln())
        .ok_or(Error::TargetOutOfRange {
            target: target.0,
            classes: logits.len(),
        })
}

fn cross_entropy_gradient(logits: &[f64], target: TokenId) -> Result<Vec<f64>, Error> {
    let mut probabilities = softmax(logits)?;
    let gradient = probabilities
        .get_mut(target.0)
        .ok_or(Error::TargetOutOfRange {
            target: target.0,
            classes: logits.len(),
        })?;
    *gradient -= 1.0;
    Ok(probabilities)
}

impl TryFrom<usize> for TokenId {
    type Error = TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}
