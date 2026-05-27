use rust_ml_mlp::{InputValue, InputVector, TinyMlp};

fn main() -> Result<(), rust_ml_mlp::Error> {
    let mlp = TinyMlp::xor_seed()?;

    println!("x1 x2 | prediction");
    println!("------+-----------");

    for (left, right) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let input =
            InputVector::from_values([InputValue::try_from(left)?, InputValue::try_from(right)?])?;
        let prediction = mlp.forward(&input)?.prediction();
        println!("{left:.0}  {right:.0}  | {prediction:.4}");
    }

    Ok(())
}
