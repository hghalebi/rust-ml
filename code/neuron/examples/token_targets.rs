use rust_ml_neuron::{NextTokenExample, TokenId, cross_entropy_gradient, cross_entropy_loss};

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
