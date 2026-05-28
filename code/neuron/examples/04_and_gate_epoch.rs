use rust_ml_neuron::{Dataset, LearningRate, TinyNeuron};

fn main() -> Result<(), rust_ml_neuron::Error> {
    let dataset = Dataset::and_gate()?;
    let mut neuron = TinyNeuron::lesson_seed()?;
    let rate = LearningRate::try_from(0.8)?;

    println!(
        "epoch 0 average loss = {:.4}",
        neuron.average_loss(&dataset)?
    );

    for epoch in 1..=20 {
        neuron.train_epoch(&dataset, rate)?;

        if epoch == 1 || epoch == 5 || epoch == 10 || epoch == 20 {
            println!(
                "epoch {epoch} average loss = {:.4}",
                neuron.average_loss(&dataset)?
            );
        }
    }

    Ok(())
}
