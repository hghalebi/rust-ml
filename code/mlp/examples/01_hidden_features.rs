use rust_ml_mlp::{InputValue, InputVector, TinyMlp};

fn main() -> Result<(), rust_ml_mlp::Error> {
    let mlp = TinyMlp::xor_seed()?;

    for (left, right) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let input =
            InputVector::from_values([InputValue::try_from(left)?, InputValue::try_from(right)?])?;
        let trace = mlp.forward(&input)?;
        let hidden = trace
            .hidden_activation()
            .values()
            .map(|value| format!("{value:.1}"))
            .collect::<Vec<_>>()
            .join(", ");

        println!("input=({left:.0}, {right:.0}) hidden=[{hidden}]");
    }

    Ok(())
}
