use rust_ml_mlp::TinyMlp;

fn main() -> Result<(), rust_ml_mlp::Error> {
    let mlp = TinyMlp::xor_seed()?;

    println!("InputVector width       = {}", mlp.input_dim());
    println!("HiddenActivation width  = {}", mlp.hidden_dim());
    println!("OutputLogit width       = 1");
    println!("Map: InputVector -> HiddenActivation -> OutputLogit -> Prediction");

    Ok(())
}
