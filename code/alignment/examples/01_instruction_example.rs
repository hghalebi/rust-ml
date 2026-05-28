use rust_ml_alignment::{Instruction, InstructionExample, Response, SignalSource};

fn main() -> Result<(), rust_ml_alignment::Error> {
    let example = InstructionExample::new(
        Instruction::try_from("solve 2 + 2 with a visible check")?,
        Response::try_from("2 + 2 = 4 because two pairs make four")?,
        SignalSource::try_from("public-toy-fixture")?,
    );

    println!("instruction = {}", example.instruction());
    println!("response    = {}", example.response());
    println!("source      = {}", example.source());

    Ok(())
}
