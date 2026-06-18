use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VocabSize(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelWidth(usize);

impl ModelWidth {
    fn lesson_width() -> Self {
        Self(8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DimensionCount(usize);

impl From<VocabSize> for DimensionCount {
    fn from(value: VocabSize) -> Self {
        Self(value.0)
    }
}

impl From<ModelWidth> for DimensionCount {
    fn from(value: ModelWidth) -> Self {
        Self(value.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LearningRate(f64);

impl LearningRate {
    fn lesson_rate() -> Self {
        Self(0.1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EpochCount(usize);

impl EpochCount {
    fn workshop_epochs() -> Self {
        Self(300)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct InitScale(f64);

impl InitScale {
    fn small_weights() -> Self {
        Self(0.2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InitSeed(usize);

impl InitSeed {
    fn embedding_seed() -> Self {
        Self(1)
    }

    fn head_seed() -> Self {
        Self(2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MatrixShape {
    rows: DimensionCount,
    cols: DimensionCount,
}

impl MatrixShape {
    fn new(rows: DimensionCount, cols: DimensionCount) -> Self {
        Self { rows, cols }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct AverageLoss(f64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BigramEpoch(usize);

#[derive(Debug, Clone, Copy, PartialEq)]
struct BigramEpochMetrics {
    epoch: BigramEpoch,
    average_loss: AverageLoss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BigramExample {
    input: TokenId,
    target: TokenId,
}

impl BigramExample {
    fn new(input: TokenId, target: TokenId) -> Self {
        Self { input, target }
    }
}

#[derive(Debug)]
enum Error {
    EmptyVocabulary,
    ZeroModelWidth,
    TooShortTokenStream,
    TokenOutOfRange {
        observed: TokenId,
        vocab_size: VocabSize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVocabulary => write!(formatter, "vocabulary size must be greater than zero"),
            Self::ZeroModelWidth => write!(formatter, "model width must be greater than zero"),
            Self::TooShortTokenStream => {
                write!(formatter, "at least two tokens are required for bigrams")
            }
            Self::TokenOutOfRange {
                observed,
                vocab_size,
            } => {
                write!(
                    formatter,
                    "token {} is outside vocabulary size {}",
                    observed.0, vocab_size.0
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Vocabulary {
    names: Vec<&'static str>,
}

impl Vocabulary {
    fn lesson_vocab() -> Self {
        Self {
            names: vec!["<pad>", "I", "love", "Rust", "."],
        }
    }

    fn size(&self) -> VocabSize {
        VocabSize(self.names.len())
    }
}

#[derive(Debug, Clone)]
struct TokenStream {
    tokens: Vec<TokenId>,
}

impl TokenStream {
    fn repeated_sentence_cycle() -> Self {
        Self {
            tokens: vec![
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(4),
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(4),
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(4),
            ],
        }
    }
}

#[derive(Debug, Clone)]
struct BigramDataset {
    examples: Vec<BigramExample>,
    vocab_size: VocabSize,
}

impl BigramDataset {
    fn from_stream(stream: &TokenStream, vocabulary: &Vocabulary) -> Result<Self, Error> {
        let vocab_size = vocabulary.size();
        if vocab_size.0 == 0 {
            return Err(Error::EmptyVocabulary);
        }
        if stream.tokens.len() < 2 {
            return Err(Error::TooShortTokenStream);
        }

        for observed in &stream.tokens {
            if observed.0 >= vocab_size.0 {
                return Err(Error::TokenOutOfRange {
                    observed: *observed,
                    vocab_size,
                });
            }
        }

        let examples = stream
            .tokens
            .windows(2)
            .map(|window| BigramExample::new(window[0], window[1]))
            .collect();

        Ok(Self {
            examples,
            vocab_size,
        })
    }

    fn iter(&self) -> impl Iterator<Item = &BigramExample> {
        self.examples.iter()
    }
}

#[derive(Debug, Clone)]
struct BigramEpochTrace {
    metrics: Vec<BigramEpochMetrics>,
}

impl BigramEpochTrace {
    fn empty() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    fn push(&mut self, metric: BigramEpochMetrics) {
        self.metrics.push(metric);
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    rows: Vec<Vec<f64>>,
}

impl Matrix {
    fn deterministic(shape: MatrixShape, scale: InitScale, seed: InitSeed) -> Self {
        let mut rows = vec![vec![0.0; shape.cols.0]; shape.rows.0];
        for (row_index, row) in rows.iter_mut().enumerate() {
            for (col_index, slot) in row.iter_mut().enumerate() {
                let linear_index = row_index * shape.cols.0 + col_index;
                let raw = (linear_index.wrapping_mul(37) + seed.0.wrapping_mul(101)) % 1000;
                let unit = raw as f64 / 1000.0;
                *slot = (unit - 0.5) * scale.0;
            }
        }
        Self { rows }
    }
}

#[derive(Debug, Clone)]
struct BiasVector {
    values: Vec<f64>,
}

impl BiasVector {
    fn zeroed(vocab_size: VocabSize) -> Self {
        Self {
            values: vec![0.0; vocab_size.0],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Logits {
    values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct Probabilities {
    values: Vec<f64>,
}

#[derive(Debug)]
struct TinyBigramModel {
    embedding: Matrix,
    lm_head: Matrix,
    bias: BiasVector,
}

impl TinyBigramModel {
    fn new(vocab_size: VocabSize, model_width: ModelWidth) -> Result<Self, Error> {
        if vocab_size.0 == 0 {
            return Err(Error::EmptyVocabulary);
        }
        if model_width.0 == 0 {
            return Err(Error::ZeroModelWidth);
        }

        Ok(Self {
            embedding: Matrix::deterministic(
                MatrixShape::new(vocab_size.into(), model_width.into()),
                InitScale::small_weights(),
                InitSeed::embedding_seed(),
            ),
            lm_head: Matrix::deterministic(
                MatrixShape::new(model_width.into(), vocab_size.into()),
                InitScale::small_weights(),
                InitSeed::head_seed(),
            ),
            bias: BiasVector::zeroed(vocab_size),
        })
    }

    fn vocab_size(&self) -> VocabSize {
        VocabSize(self.bias.values.len())
    }

    fn validate_token(&self, candidate: TokenId) -> Result<(), Error> {
        let vocab_size = self.vocab_size();
        if candidate.0 >= vocab_size.0 {
            return Err(Error::TokenOutOfRange {
                observed: candidate,
                vocab_size,
            });
        }
        Ok(())
    }

    fn logits_for_token(&self, input: TokenId) -> Result<Logits, Error> {
        self.validate_token(input)?;

        let hidden = &self.embedding.rows[input.0];
        let mut logits = self.bias.values.clone();

        for (token_id, logit) in logits.iter_mut().enumerate() {
            for (dimension, hidden_component) in hidden.iter().enumerate() {
                *logit += hidden_component * self.lm_head.rows[dimension][token_id];
            }
        }

        Ok(Logits { values: logits })
    }

    fn probabilities_for_token(&self, input: TokenId) -> Result<Probabilities, Error> {
        let logits = self.logits_for_token(input)?;
        Ok(softmax(&logits))
    }

    fn predict_next(&self, input: TokenId) -> Result<TokenId, Error> {
        let probabilities = self.probabilities_for_token(input)?;
        Ok(argmax(&probabilities))
    }

    fn average_loss(&self, dataset: &BigramDataset) -> Result<AverageLoss, Error> {
        let mut total = 0.0;
        for example in dataset.iter() {
            let logits = self.logits_for_token(example.input)?;
            let probabilities = softmax(&logits);
            total += -probabilities.values[example.target.0].ln();
        }
        Ok(AverageLoss(total / dataset.examples.len() as f64))
    }

    fn train_one_example(
        &mut self,
        example: BigramExample,
        learning_rate: LearningRate,
    ) -> Result<(), Error> {
        self.validate_token(example.input)?;
        self.validate_token(example.target)?;

        let logits = self.logits_for_token(example.input)?;
        let model_width = self.embedding.rows[example.input.0].len();
        let hidden = self.embedding.rows[example.input.0].clone();
        let mut probabilities = softmax(&logits);

        probabilities.values[example.target.0] -= 1.0;

        for (token_id, grad) in probabilities.values.iter().copied().enumerate() {
            for (dimension, hidden_component) in hidden.iter().enumerate().take(model_width) {
                let update = grad * hidden_component;
                self.lm_head.rows[dimension][token_id] -= learning_rate.0 * update;
            }
            self.bias.values[token_id] -= learning_rate.0 * grad;
        }

        for dimension in 0..model_width {
            let mut embedding_grad = 0.0;
            for (probability, weight) in probabilities
                .values
                .iter()
                .zip(self.lm_head.rows[dimension].iter())
            {
                embedding_grad += probability * weight;
            }
            self.embedding.rows[example.input.0][dimension] -= learning_rate.0 * embedding_grad;
        }

        Ok(())
    }

    fn train_epochs(
        &mut self,
        dataset: &BigramDataset,
        learning_rate: LearningRate,
        epochs: EpochCount,
    ) -> Result<BigramEpochTrace, Error> {
        let mut trace = BigramEpochTrace::empty();
        for epoch in 1..=epochs.0 {
            for example in dataset.iter() {
                self.train_one_example(*example, learning_rate)?;
            }
            let average_loss = self.average_loss(dataset)?;
            trace.push(BigramEpochMetrics {
                epoch: BigramEpoch(epoch),
                average_loss,
            });
        }
        Ok(trace)
    }
}

fn softmax(logits: &Logits) -> Probabilities {
    let max_logit = logits
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits
        .values
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();
    let sum: f64 = exponentials.iter().sum();

    Probabilities {
        values: exponentials.into_iter().map(|value| value / sum).collect(),
    }
}

fn argmax(probabilities: &Probabilities) -> TokenId {
    let mut best_index = 0;
    let mut best_value = probabilities.values[0];

    for (index, value) in probabilities.values.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }

    TokenId(best_index)
}

fn main() {
    let vocabulary = Vocabulary::lesson_vocab();
    let token_stream = TokenStream::repeated_sentence_cycle();
    let dataset = match BigramDataset::from_stream(&token_stream, &vocabulary) {
        Ok(dataset) => dataset,
        Err(error) => {
            println!("failed to build dataset: {error}");
            return;
        }
    };

    let mut model = match TinyBigramModel::new(dataset.vocab_size, ModelWidth::lesson_width()) {
        Ok(model) => model,
        Err(error) => {
            println!("failed to create model: {error}");
            return;
        }
    };

    let before = match model.average_loss(&dataset) {
        Ok(loss) => loss,
        Err(error) => {
            println!("failed to measure initial loss: {error}");
            return;
        }
    };
    let metrics = match model.train_epochs(
        &dataset,
        LearningRate::lesson_rate(),
        EpochCount::workshop_epochs(),
    ) {
        Ok(metrics) => metrics,
        Err(error) => {
            println!("training failed: {error}");
            return;
        }
    };
    let after = match model.average_loss(&dataset) {
        Ok(loss) => loss,
        Err(error) => {
            println!("failed to measure final loss: {error}");
            return;
        }
    };

    println!("average loss: before={:.4}, after={:.4}", before.0, after.0);

    for input_id in 1..vocabulary.names.len() {
        let input = TokenId(input_id);
        let predicted = match model.predict_next(input) {
            Ok(token_id) => token_id,
            Err(error) => {
                println!("prediction failed for input {input_id}: {error}");
                continue;
            }
        };
        let probabilities = match model.probabilities_for_token(input) {
            Ok(probabilities) => probabilities,
            Err(error) => {
                println!("probabilities failed for input {input_id}: {error}");
                continue;
            }
        };
        println!(
            "{:>4} -> {:>4}   confidence {:.3}",
            vocabulary.names[input_id],
            vocabulary.names[predicted.0],
            probabilities.values[predicted.0]
        );
    }

    if let Some(last_epoch) = metrics.metrics.last() {
        println!(
            "last epoch={} average_loss={:.4}",
            last_epoch.epoch.0, last_epoch.average_loss.0
        );
    }
}
